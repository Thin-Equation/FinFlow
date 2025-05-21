"""
Parallel Agent for workflow execution.

This module provides a ParallelAgent that executes workflow tasks in parallel
when dependencies allow it, using a thread pool executor.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
import threading

from workflow.workflow_definitions import (
    WorkflowDefinition, 
    WorkflowExecutionContext,
    WorkflowResult,
    WorkflowStatus,
    WorkflowTask,
    TaskStatus,
    TaskId
)

from utils.session_state import get_or_create_session_state

class ParallelAgent:
    """Agent for parallel execution of workflow tasks."""
    
    def __init__(
        self,
        name: str = "ParallelAgent",
        model: str = "gemini-2.0-flash",
        description: str = "Executes workflow tasks in parallel",
        instruction: str = "",
        temperature: float = 0.2,
        max_workers: int = 10
    ):
        """Initialize the parallel agent."""
        # Initialize properties
        self.name = name
        self.model = model
        self.description = description
        self.instruction = instruction or self._default_instruction()
        self.temperature = temperature
        self.max_workers = max_workers
        self.current_workflow = None
        self.execution_context = None
        self._context_lock = threading.RLock()
        
        # Set up logger
        self.logger = logging.getLogger(f"finflow.agents.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"ParallelAgent '{name}' initialized with {max_workers} workers")
    
    def _default_instruction(self) -> str:
        """Default instruction for the agent."""
        return (
            "You are a workflow execution agent specialized in parallel processing. "
            "Your role is to execute financial process workflows efficiently, running "
            "independent tasks in parallel while respecting dependencies. You monitor "
            "execution, manage thread safety, and properly handle errors."
        )
    
    def _thread_safe_context_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation on the execution context in a thread-safe manner."""
        with self._context_lock:
            return operation(*args, **kwargs)
    
    def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a workflow in parallel.
        
        Args:
            workflow: The workflow definition to execute
            context: The agent context
            parameters: Optional parameters for the workflow
            
        Returns:
            WorkflowResult: The result of the workflow execution
        """
        self.current_workflow = workflow
        
        # Setup workflow execution context
        session_state = get_or_create_session_state(context)
        
        self.execution_context = WorkflowExecutionContext(
            workflow_id=workflow.id,
            parameters=parameters or {},
            session=session_state.to_dict()
        )
        
        # Log workflow execution start
        self.logger.info(f"Starting parallel execution of workflow '{workflow.name}' ({workflow.id})")
        start_time = datetime.now()
        
        # Get execution context
        execution_context = self.execution_context
        context_lock = self._context_lock
        max_workers = self.max_workers
        
        # Validate workflow
        is_valid, errors = workflow.validate()
        if not is_valid:
            error_msg = f"Workflow validation failed: {', '.join(errors)}"
            self.logger.error(error_msg)
            return WorkflowResult(
                workflow_id=workflow.id,
                status=WorkflowStatus.FAILED,
                error=ValueError(error_msg)
            )
        
        # Track task state
        pending_tasks: Set[TaskId] = set(workflow.tasks.keys())
        completed_tasks: Set[TaskId] = set()
        failed_tasks: Set[TaskId] = set()
        running_tasks: Set[TaskId] = set()
        task_futures: Dict[TaskId, Future] = {}
        
        # Use a thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process tasks until all are processed or a terminal failure occurs
            while pending_tasks or running_tasks:
                # Find tasks that can be executed
                executable_tasks = self._find_executable_tasks(workflow, pending_tasks, completed_tasks, running_tasks)
                
                # Submit executable tasks to the thread pool
                for task_id in executable_tasks:
                    task = workflow.tasks[task_id]
                    self.logger.info(f"Submitting task '{task.name}' ({task.id}) for execution")
                    
                    # Mark task as running and move to running set
                    task.mark_running()
                    pending_tasks.remove(task.id)
                    running_tasks.add(task.id)
                    
                    # Submit task to executor
                    future = executor.submit(self._execute_task, task)
                    task_futures[task.id] = future
                
                # Wait for at least one task to complete if we have running tasks
                if running_tasks:
                    done_futures = []
                    try:
                        # Wait for the first task to complete with a timeout
                        done_futures, _ = wait(
                            [task_futures[tid] for tid in running_tasks],
                            return_when=FIRST_COMPLETED,
                            timeout=0.1  # Small timeout to allow checking for deadlock
                        )
                    except Exception as e:
                        self.logger.error(f"Error waiting for tasks to complete: {str(e)}")
                    
                    # Process completed tasks
                    for future in done_futures:
                        for task_id, task_future in task_futures.items():
                            if task_future == future:
                                task = workflow.tasks[task_id]
                                try:
                                    # Get result and handle task completion
                                    result = future.result()
                                    
                                    with context_lock:
                                        execution_context.set_result(task.id, result)
                                        task.mark_completed(result)
                                    
                                    # Move task from running to completed
                                    running_tasks.remove(task.id)
                                    completed_tasks.add(task.id)
                                    
                                    self.logger.info(f"Task '{task.name}' completed successfully")
                                    
                                except Exception as e:
                                    # Handle task failure
                                    self.logger.error(f"Task '{task.name}' failed with error: {str(e)}")
                                    
                                    with context_lock:
                                        task.mark_failed(e)
                                    
                                    # Move task from running to failed
                                    running_tasks.remove(task.id)
                                    failed_tasks.add(task.id)
                                    
                                    # Mark dependent tasks as skipped
                                    dependent_tasks = workflow.get_dependent_tasks(task.id)
                                    for dep_task in dependent_tasks:
                                        if dep_task.id in pending_tasks:
                                            self.logger.warning(f"Skipping task '{dep_task.name}' due to dependency failure")
                                            dep_task.status = TaskStatus.SKIPPED
                                            pending_tasks.remove(dep_task.id)
                                    
                                    # Early termination if configured
                                    with context_lock:
                                        if not execution_context.get_state("continue_on_error", False):
                                            self.logger.error("Workflow execution will terminate due to task failure")
                                            # Cancel all running tasks
                                            for running_id in running_tasks.copy():
                                                if running_id != task_id:
                                                    task_futures[running_id].cancel()
                                                    workflow.tasks[running_id].status = TaskStatus.CANCELED
                                                    running_tasks.remove(running_id)
                                            
                                            # Clear pending tasks
                                            pending_tasks.clear()
                                
                                # We found the completed future, break the inner loop
                                break
                
                # Check for deadlock (no executable or running tasks but still have pending)
                if not executable_tasks and not running_tasks and pending_tasks:
                    error_msg = f"Deadlock detected in workflow execution. Pending tasks: {pending_tasks}"
                    logger.error(error_msg)
                    return WorkflowResult(
                        workflow_id=workflow.id,
                        status=WorkflowStatus.FAILED,
                        error=RuntimeError(error_msg),
                        task_results=execution_context.results,
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                
                # Small sleep to prevent CPU hogging when no tasks are ready
                if not executable_tasks and not done_futures:
                    time.sleep(0.01)
        
        # Check for overall workflow status
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if failed_tasks:
            return WorkflowResult(
                workflow_id=workflow.id,
                status=WorkflowStatus.FAILED,
                task_results=execution_context.results,
                execution_time=execution_time,
                error=RuntimeError(f"Workflow completed with {len(failed_tasks)} failed tasks")
            )
        
        # Extract the final result if specified
        final_result = None
        with context_lock:
            output_task_id = execution_context.get_state("output_task_id")
            if output_task_id:
                final_result = execution_context.get_result(output_task_id)
        
        return WorkflowResult(
            workflow_id=workflow.id,
            status=WorkflowStatus.COMPLETED,
            result=final_result,
            task_results=execution_context.results,
            execution_time=execution_time
        )
    
    def _execute_task(self, task: WorkflowTask) -> Any:
        """
        Execute a single task in a thread-safe manner.
        
        Args:
            task: The task to execute
            
        Returns:
            Any: The result of task execution
            
        Raises:
            Exception: If task execution fails
        """
        try:
            # Get execution context and lock from __dict__
            execution_context = self.__dict__["execution_context"]
            context_lock = self.__dict__["_context_lock"]
            
            with context_lock:
                context_copy = WorkflowExecutionContext(
                    workflow_id=execution_context.workflow_id,
                    parameters=execution_context.parameters.copy(),
                    state=execution_context.state.copy(),
                    results=execution_context.results.copy(),
                    session=execution_context.session.copy() if execution_context.session else None,
                    start_time=execution_context.start_time
                )
            
            # Execute the task with a copy of the context to prevent race conditions
            result = task.execute(context_copy)
            
            # Merge state changes back to the main context
            with context_lock:
                for key, value in context_copy.state.items():
                    if key not in execution_context.state or execution_context.state[key] != value:
                        execution_context.state[key] = value
            
            return result
            
        except Exception as e:
            logger = self.__dict__["logger"]
            logger.error(f"Error executing task '{task.name}': {str(e)}")
            raise
    
    def _find_executable_tasks(
        self,
        workflow: WorkflowDefinition,
        pending_tasks: Set[TaskId],
        completed_tasks: Set[TaskId],
        running_tasks: Set[TaskId]
    ) -> List[TaskId]:
        """
        Find tasks that can be executed based on dependency completion.
        
        Args:
            workflow: The workflow definition
            pending_tasks: Set of pending task IDs
            completed_tasks: Set of completed task IDs
            running_tasks: Set of running task IDs
            
        Returns:
            List[TaskId]: List of task IDs that can be executed
        """
        executable_tasks = []
        
        for task_id in list(pending_tasks):
            task = workflow.tasks[task_id]
            dependencies = set(task.dependencies)
            
            # Check if all dependencies are completed (not just running)
            if dependencies.issubset(completed_tasks):
                executable_tasks.append(task_id)
        
        return executable_tasks
    
    def run_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        context: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        wait: bool = True
    ) -> WorkflowResult:
        """
        Run a workflow with the given context and parameters.
        
        Args:
            workflow_definition: The workflow definition to run
            context: The agent context
            parameters: Parameters for the workflow execution
            wait: Whether to wait for workflow completion
            
        Returns:
            WorkflowResult: The result of the workflow execution
        """
        # Register the workflow in the session state
        session_state = get_or_create_session_state(context)
        workflows = session_state.get("workflows", {})
        
        if workflow_definition.id in workflows:
            logger = self.__dict__["logger"]
            logger.warning(f"Workflow {workflow_definition.id} already exists in session state")
        
        # Register the workflow
        workflows[workflow_definition.id] = {
            "id": workflow_definition.id,
            "name": workflow_definition.name,
            "description": workflow_definition.description,
            "status": WorkflowStatus.PENDING.value,
            "start_time": datetime.now().isoformat(),
            "parameters": parameters or {}
        }
        
        session_state.set("workflows", workflows)
        
        # Execute the workflow
        try:
            workflows[workflow_definition.id]["status"] = WorkflowStatus.RUNNING.value
            session_state.set("workflows", workflows)
            
            result = self.execute_workflow(workflow_definition, context, parameters)
            
            # Update workflow status
            workflows[workflow_definition.id]["status"] = result.status.value
            workflows[workflow_definition.id]["end_time"] = datetime.now().isoformat()
            workflows[workflow_definition.id]["execution_time"] = result.execution_time
            
            if result.error:
                workflows[workflow_definition.id]["error"] = str(result.error)
            
            session_state.set("workflows", workflows)
            context["session_state"] = session_state.to_dict()
            
            return result
            
        except Exception as e:
            logger = self.__dict__["logger"]
            logger.error(f"Error executing workflow: {str(e)}")
            
            # Update workflow status
            workflows[workflow_definition.id]["status"] = WorkflowStatus.FAILED.value
            workflows[workflow_definition.id]["end_time"] = datetime.now().isoformat()
            workflows[workflow_definition.id]["error"] = str(e)
            
            session_state.set("workflows", workflows)
            context["session_state"] = session_state.to_dict()
            
            return WorkflowResult(
                workflow_id=workflow_definition.id,
                status=WorkflowStatus.FAILED,
                error=e
            )
