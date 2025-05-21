"""
Sequential Agent for workflow execution.

This module provides a SequentialAgent that executes workflow tasks in sequence,
respecting dependencies between tasks.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from workflow.workflow_definitions import (
    WorkflowDefinition, 
    WorkflowExecutionContext,
    WorkflowResult,
    WorkflowStatus,
    TaskStatus,
    TaskId
)

from utils.session_state import get_or_create_session_state

class SequentialAgent:
    """Agent for sequential execution of workflow tasks."""
    
    def __init__(
        self,
        name: str = "SequentialAgent",
        model: str = "gemini-2.0-flash",
        description: str = "Executes workflow tasks in sequence",
        instruction: str = "",
        temperature: float = 0.2,
    ):
        """Initialize the sequential agent."""
        # Initialize properties
        self.name = name
        self.model = model
        self.description = description
        self.instruction = instruction or self._default_instruction()
        self.temperature = temperature
        self.current_workflow = None
        self.execution_context = None
        
        # Set up logger
        self.logger = logging.getLogger(f"finflow.agents.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"SequentialAgent '{name}' initialized")
    
    def _default_instruction(self) -> str:
        """Default instruction for the agent."""
        return (
            "You are a workflow execution agent specialized in sequential processing. "
            "Your role is to execute financial process workflows step by step, ensuring "
            "that each task is completed successfully before moving to dependent tasks. "
            "Always monitor for errors and handle them appropriately."
        )
    
    def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a workflow sequentially.
        
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
        self.logger.info(f"Starting sequential execution of workflow '{workflow.name}' ({workflow.id})")
        start_time = datetime.now()
        
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
        
        # Get execution context
        execution_context = self.execution_context
        
        # Track task dependencies and completion
        pending_tasks: Set[TaskId] = set(workflow.tasks.keys())
        completed_tasks: Set[TaskId] = set()
        failed_tasks: Set[TaskId] = set()
        
        # Execute tasks in topological order
        while pending_tasks:
            # Find tasks that can be executed
            executable_tasks = self._find_executable_tasks(workflow, pending_tasks, completed_tasks)
            
            if not executable_tasks:
                # Check for deadlock
                if pending_tasks and not failed_tasks:
                    error_msg = f"Deadlock detected in workflow execution. Pending tasks: {pending_tasks}"
                    self.logger.error(error_msg)
                    return WorkflowResult(
                        workflow_id=workflow.id,
                        status=WorkflowStatus.FAILED,
                        error=RuntimeError(error_msg),
                        task_results=execution_context.results,
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                break
            
            # Execute each task
            for task_id in executable_tasks:
                task = workflow.tasks[task_id]
                
                try:
                    # Mark task as running
                    task.mark_running()
                    self.logger.info(f"Executing task '{task.name}' ({task.id})")
                    
                    # Execute the task
                    result = task.execute(execution_context)
                    
                    # Store the result
                    execution_context.set_result(task.id, result)
                    task.mark_completed(result)
                    
                    # Move task from pending to completed
                    pending_tasks.remove(task.id)
                    completed_tasks.add(task.id)
                    
                    self.logger.info(f"Task '{task.name}' completed successfully")
                    
                except Exception as e:
                    # Handle task failure
                    self.logger.error(f"Task '{task.name}' failed with error: {str(e)}")
                    task.mark_failed(e)
                    
                    # Move task from pending to failed
                    pending_tasks.remove(task.id)
                    failed_tasks.add(task.id)
                    
                    # Stop depending tasks from execution
                    dependent_tasks = workflow.get_dependent_tasks(task.id)
                    for dep_task in dependent_tasks:
                        if dep_task.id in pending_tasks:
                            self.logger.warning(f"Skipping task '{dep_task.name}' due to dependency failure")
                            dep_task.status = TaskStatus.SKIPPED
                            pending_tasks.remove(dep_task.id)
                    
                    # Early termination if configured
                    if not execution_context.get_state("continue_on_error", False):
                        error_msg = f"Workflow execution stopped due to task failure: {str(e)}"
                        self.logger.error(error_msg)
                        return WorkflowResult(
                            workflow_id=workflow.id,
                            status=WorkflowStatus.FAILED,
                            error=e,
                            task_results=execution_context.results,
                            execution_time=(datetime.now() - start_time).total_seconds()
                        )
        
        # Check for overall workflow status
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Get execution context from __dict__
        execution_context = self.__dict__["execution_context"]
        
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
    
    def _find_executable_tasks(
        self,
        workflow: WorkflowDefinition,
        pending_tasks: Set[TaskId],
        completed_tasks: Set[TaskId]
    ) -> List[TaskId]:
        """
        Find tasks that can be executed based on dependency completion.
        
        Args:
            workflow: The workflow definition
            pending_tasks: Set of pending task IDs
            completed_tasks: Set of completed task IDs
            
        Returns:
            List[TaskId]: List of task IDs that can be executed
        """
        executable_tasks = []
        
        for task_id in pending_tasks:
            task = workflow.tasks[task_id]
            dependencies = set(task.dependencies)
            
            # Check if all dependencies are completed
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
            self.logger.warning(f"Workflow {workflow_definition.id} already exists in session state")
        
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
            self.logger.error(f"Error executing workflow: {str(e)}")
            
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
