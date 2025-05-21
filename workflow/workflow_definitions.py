"""
Core workflow definitions for financial processes.

This module defines the core classes and interfaces for workflow definitions and execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Set, Tuple
import uuid
import logging

# Type definitions for improved type safety
T = TypeVar('T')
WorkflowId = str
TaskId = str
AgentId = str

class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    SUSPENDED = "suspended"

class TaskStatus(str, Enum):
    """Status of a workflow task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"

@dataclass
class WorkflowExecutionContext:
    """Context for workflow execution with shared state and parameters."""
    workflow_id: WorkflowId
    parameters: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    session: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize the execution context with default values if not provided."""
        if not self.start_time:
            self.start_time = datetime.now()

    def get_result(self, task_id: TaskId, default: Optional[Any] = None) -> Any:
        """Get the result of a specific task."""
        return self.results.get(task_id, default)
    
    def set_result(self, task_id: TaskId, result: Any) -> None:
        """Set the result of a specific task."""
        self.results[task_id] = result
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a value in the shared state."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value from the shared state."""
        return self.state.get(key, default)
    
    def has_state(self, key: str) -> bool:
        """Check if a key exists in the state."""
        return key in self.state
    
    def merge_parameters(self, params: Dict[str, Any]) -> None:
        """Merge additional parameters into the context."""
        self.parameters.update(params)

@dataclass
class WorkflowTask:
    """Definition of a workflow task."""
    id: TaskId
    name: str
    description: str
    execute: Callable[[WorkflowExecutionContext], Any]
    dependencies: List[TaskId] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    retries: int = 0
    retry_delay_seconds: float = 5.0
    status: TaskStatus = field(default=TaskStatus.PENDING)
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the task with a unique ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def mark_running(self) -> None:
        """Mark the task as running."""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now()
        if result is not None:
            self.result = result
    
    def mark_failed(self, error: Optional[Exception] = None) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.end_time = datetime.now()
        if error is not None:
            self.error = error
    
    def is_completed(self) -> bool:
        """Check if the task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if the task is failed."""
        return self.status == TaskStatus.FAILED
    
    def is_skipped(self) -> bool:
        """Check if the task is skipped."""
        return self.status == TaskStatus.SKIPPED
    
    def is_pending(self) -> bool:
        """Check if the task is pending."""
        return self.status == TaskStatus.PENDING
    
    def reset(self) -> None:
        """Reset the task to its initial state."""
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None

@dataclass
class WorkflowResult(Generic[T]):
    """Result of a workflow execution."""
    workflow_id: WorkflowId
    status: WorkflowStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time: Optional[float] = None
    task_results: Dict[TaskId, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if the workflow executed successfully."""
        return self.status == WorkflowStatus.COMPLETED and self.error is None

class WorkflowDefinition(ABC):
    """Abstract base class for workflow definitions."""
    
    def __init__(self, name: str, description: str):
        self.id: WorkflowId = str(uuid.uuid4())
        self.name: str = name
        self.description: str = description
        self.tasks: Dict[TaskId, WorkflowTask] = {}
        self.dependencies: Dict[TaskId, Set[TaskId]] = {}
        self.logger = logging.getLogger(f"finflow.workflow.{name}")
    
    def add_task(self, task: WorkflowTask) -> 'WorkflowDefinition':
        """Add a task to the workflow."""
        self.tasks[task.id] = task
        self.dependencies[task.id] = set(task.dependencies)
        return self
    
    def get_task(self, task_id: TaskId) -> Optional[WorkflowTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks(self) -> List[WorkflowTask]:
        """Get all tasks in the workflow."""
        return list(self.tasks.values())
    
    def get_initial_tasks(self) -> List[WorkflowTask]:
        """Get tasks with no dependencies."""
        return [task for task in self.tasks.values() if not task.dependencies]
    
    def get_dependent_tasks(self, task_id: TaskId) -> List[WorkflowTask]:
        """Get tasks that depend on the given task."""
        dependent_tasks = []
        for tid, deps in self.dependencies.items():
            if task_id in deps:
                dependent_tasks.append(self.tasks[tid])
        return dependent_tasks
    
    def are_dependencies_completed(self, task_id: TaskId, context: WorkflowExecutionContext) -> bool:
        """Check if all dependencies of a task are completed."""
        for dep_id in self.dependencies.get(task_id, set()):
            dep_result = context.get_result(dep_id)
            if dep_result is None or self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the workflow definition."""
        errors = []
        
        # Check for circular dependencies
        visited = set()
        temp = set()
        
        def has_cycle(node: TaskId) -> bool:
            """Detect cycle in directed graph using DFS."""
            if node in temp:
                return True
            if node in visited:
                return False
            
            temp.add(node)
            
            for dep in self.get_dependent_tasks(node):
                if has_cycle(dep.id):
                    return True
            
            temp.remove(node)
            visited.add(node)
            return False
        
        # Check each node for cycles
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Circular dependency detected involving task {task_id}")
        
        # Check for missing dependencies
        for task_id, deps in self.dependencies.items():
            for dep_id in deps:
                if dep_id not in self.tasks:
                    errors.append(f"Task {task_id} depends on non-existent task {dep_id}")
        
        return len(errors) == 0, errors
    
    @abstractmethod
    def execute(self, context: Optional[WorkflowExecutionContext] = None) -> WorkflowResult:
        """Execute the workflow."""
        pass

class FinancialProcess(WorkflowDefinition):
    """Base class for financial process workflows."""
    
    def __init__(self, name: str, description: str, process_type: str):
        super().__init__(name, description)
        self.process_type = process_type
        self.metadata: Dict[str, Any] = {}
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the financial process."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any:
        """Get metadata for the financial process."""
        return self.metadata.get(key, default)
    
    def execute(self, context: Optional[WorkflowExecutionContext] = None) -> WorkflowResult:
        """
        Execute the financial process workflow.
        
        This implementation delegates to sequential execution logic but can be overridden.
        
        Args:
            context: Execution context, created if not provided
            
        Returns:
            WorkflowResult: The result of workflow execution
        """
        # Create a default context if none provided
        if context is None:
            context = WorkflowExecutionContext(workflow_id=self.id)
        
        # Track task state
        pending_tasks = set(self.tasks.keys())
        completed_tasks = set()
        failed_tasks = set()
        
        # Save start time
        start_time = datetime.now()
        
        # Process tasks in topological order
        while pending_tasks:
            # Find tasks that can be executed
            executable_tasks = []
            for task_id in list(pending_tasks):
                dependencies = set(self.tasks[task_id].dependencies)
                if dependencies.issubset(completed_tasks):
                    executable_tasks.append(task_id)
            
            # Check for deadlock
            if not executable_tasks and pending_tasks:
                error_msg = f"Deadlock detected in workflow execution. Pending tasks: {pending_tasks}"
                self.logger.error(error_msg)
                return WorkflowResult(
                    workflow_id=self.id,
                    status=WorkflowStatus.FAILED,
                    error=RuntimeError(error_msg),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute tasks
            for task_id in executable_tasks:
                task = self.tasks[task_id]
                
                try:
                    # Execute the task
                    self.logger.info(f"Executing task '{task.name}' ({task.id})")
                    task.mark_running()
                    result = task.execute(context)
                    
                    # Store result
                    context.set_result(task.id, result)
                    task.mark_completed(result)
                    
                    # Update task sets
                    pending_tasks.remove(task.id)
                    completed_tasks.add(task.id)
                    
                except Exception as e:
                    # Handle failure
                    self.logger.error(f"Task '{task.name}' failed: {str(e)}")
                    task.mark_failed(e)
                    
                    pending_tasks.remove(task.id)
                    failed_tasks.add(task.id)
                    
                    # Stop depending tasks
                    for dep_task in self.get_dependent_tasks(task.id):
                        if dep_task.id in pending_tasks:
                            dep_task.status = TaskStatus.SKIPPED
                            pending_tasks.remove(dep_task.id)
                    
                    # Early termination unless continue_on_error is set
                    if not context.get_state("continue_on_error", False):
                        return WorkflowResult(
                            workflow_id=self.id,
                            status=WorkflowStatus.FAILED,
                            error=e,
                            task_results=context.results,
                            execution_time=(datetime.now() - start_time).total_seconds()
                        )
        
        # All tasks processed, determine final status
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if failed_tasks:
            return WorkflowResult(
                workflow_id=self.id,
                status=WorkflowStatus.FAILED,
                error=RuntimeError(f"Workflow completed with {len(failed_tasks)} failed tasks"),
                task_results=context.results,
                execution_time=execution_time
            )
        
        # Get final result if specified
        final_result = None
        output_task_id = context.get_state("output_task_id")
        if output_task_id and output_task_id in context.results:
            final_result = context.get_result(output_task_id)
        
        return WorkflowResult(
            workflow_id=self.id,
            status=WorkflowStatus.COMPLETED,
            result=final_result,
            task_results=context.results,
            execution_time=execution_time
        )

class Workflow:
    """Workflow factory and utilities."""
    
    @staticmethod
    def create(name: str, description: str, process_type: str = "generic") -> FinancialProcess:
        """Create a new financial process workflow."""
        return FinancialProcess(name, description, process_type)
