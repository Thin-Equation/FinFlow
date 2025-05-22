"""
Recovery mechanisms for FinFlow system.

This module provides:
1. Workflow recovery
2. Automatic retry capabilities
3. Checkpointing and resumption
4. Fallback strategies
"""

import logging
import json
import time
import os
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from pathlib import Path

from utils.error_handling import (
    FinFlowError, ErrorSeverity
)

# For type hints
T = TypeVar('T')
WorkflowState = Dict[str, Any]
RecoveryFn = Callable[[WorkflowState], None]


class RecoveryStrategy(Enum):
    """Strategies for recovering from failures."""
    RESTART = "restart"           # Restart the workflow from the beginning
    RESUME = "resume"             # Resume from the last checkpoint
    SKIP_STEP = "skip_step"       # Skip the failed step and continue
    USE_FALLBACK = "use_fallback" # Use a fallback implementation
    MANUAL = "manual"             # Require manual intervention


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"     # Not yet started
    RUNNING = "running"     # Currently executing
    COMPLETED = "completed" # Successfully completed
    FAILED = "failed"       # Failed execution
    RECOVERED = "recovered" # Recovered from failure
    CANCELED = "canceled"   # Manually canceled


@dataclass
class RecoveryPoint:
    """A recovery point for resuming workflows."""
    workflow_id: str
    step_name: str
    timestamp: float = field(default_factory=time.time)
    state: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.RUNNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "state": self.state,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecoveryPoint':
        """Create from dictionary representation."""
        status_str = data.get("status", WorkflowStatus.FAILED.value)
        try:
            status = WorkflowStatus(status_str)
        except ValueError:
            status = WorkflowStatus.FAILED
            
        return cls(
            workflow_id=data["workflow_id"],
            step_name=data["step_name"],
            timestamp=data.get("timestamp", time.time()),
            state=data.get("state", {}),
            status=status
        )


class RecoveryManager:
    """
    Manager for workflow recovery mechanisms.
    
    This class provides functionality for:
    1. Creating and managing recovery points
    2. Automatic resumption of failed workflows
    3. Fallback implementations for critical operations
    """
    
    _instance: Optional['RecoveryManager'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'RecoveryManager':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self, recovery_dir: Optional[str] = None):
        """
        Initialize the recovery manager.
        
        Args:
            recovery_dir: Directory to store recovery points
        """
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
            
        self.logger = logging.getLogger("finflow.recovery")
        
        # Recovery directory
        self.recovery_dir = recovery_dir or os.path.join(os.getcwd(), "recovery_data")
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Recovery points in memory
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        
        # Registered recovery handlers
        self.recovery_handlers: Dict[str, Dict[str, RecoveryFn]] = {}
        
        # Load existing recovery points
        self._load_recovery_points()
        
        self.logger.info(f"Recovery manager initialized with {len(self.recovery_points)} recovery points")
    
    def _load_recovery_points(self) -> None:
        """Load recovery points from disk."""
        recovery_path = Path(self.recovery_dir)
        
        if not recovery_path.exists():
            self.logger.debug("No recovery directory found, skipping load")
            return
            
        try:
            for file_path in recovery_path.glob("*.recovery.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        recovery_point = RecoveryPoint.from_dict(data)
                        self.recovery_points[recovery_point.workflow_id] = recovery_point
                except Exception as e:
                    self.logger.warning(f"Failed to load recovery point from {file_path}: {e}")
                    
            self.logger.debug(f"Loaded {len(self.recovery_points)} recovery points")
        except Exception as e:
            self.logger.error(f"Error loading recovery points: {e}")
    
    def create_recovery_point(
        self, 
        workflow_id: str, 
        step_name: str, 
        state: Dict[str, Any]
    ) -> RecoveryPoint:
        """
        Create a new recovery point.
        
        Args:
            workflow_id: Unique identifier for the workflow
            step_name: Current step in the workflow
            state: Workflow state to save
            
        Returns:
            The created recovery point
        """
        recovery_point = RecoveryPoint(
            workflow_id=workflow_id,
            step_name=step_name,
            state=state,
            status=WorkflowStatus.RUNNING
        )
        
        # Store in memory
        self.recovery_points[workflow_id] = recovery_point
        
        # Save to disk
        self._save_recovery_point(recovery_point)
        
        self.logger.debug(f"Created recovery point for workflow {workflow_id} at step {step_name}")
        return recovery_point
    
    def update_recovery_point(
        self, 
        workflow_id: str,
        step_name: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        status: Optional[WorkflowStatus] = None
    ) -> Optional[RecoveryPoint]:
        """
        Update an existing recovery point.
        
        Args:
            workflow_id: Workflow identifier
            step_name: New step name (if changed)
            state: Updated state (if changed)
            status: New status (if changed)
            
        Returns:
            Updated recovery point or None if not found
        """
        if workflow_id not in self.recovery_points:
            self.logger.warning(f"Cannot update recovery point: workflow {workflow_id} not found")
            return None
            
        recovery_point = self.recovery_points[workflow_id]
        
        # Update fields
        if step_name is not None:
            recovery_point.step_name = step_name
            
        if state is not None:
            recovery_point.state = state
            
        if status is not None:
            recovery_point.status = status
            
        # Update timestamp
        recovery_point.timestamp = time.time()
        
        # Save to disk
        self._save_recovery_point(recovery_point)
        
        self.logger.debug(f"Updated recovery point for workflow {workflow_id}")
        return recovery_point
    
    def _save_recovery_point(self, recovery_point: RecoveryPoint) -> None:
        """
        Save a recovery point to disk.
        
        Args:
            recovery_point: The recovery point to save
        """
        file_path = os.path.join(
            self.recovery_dir, 
            f"{recovery_point.workflow_id}.recovery.json"
        )
        
        try:
            with open(file_path, "w") as f:
                json.dump(recovery_point.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save recovery point to {file_path}: {e}")
    
    def complete_workflow(self, workflow_id: str, success: bool = True) -> None:
        """
        Mark a workflow as completed.
        
        Args:
            workflow_id: Workflow identifier
            success: Whether the workflow completed successfully
        """
        status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
        
        if workflow_id in self.recovery_points:
            self.update_recovery_point(
                workflow_id=workflow_id,
                status=status
            )
            
            # If successful, clean up the recovery file
            if success:
                self._cleanup_recovery_point(workflow_id)
                
            self.logger.info(f"Marked workflow {workflow_id} as {status.value}")
        else:
            self.logger.warning(f"Cannot complete workflow {workflow_id}: no recovery point found")
    
    def _cleanup_recovery_point(self, workflow_id: str) -> None:
        """
        Clean up a recovery point file.
        
        Args:
            workflow_id: Workflow identifier
        """
        file_path = os.path.join(
            self.recovery_dir, 
            f"{workflow_id}.recovery.json"
        )
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.debug(f"Cleaned up recovery point file for {workflow_id}")
                
            # Remove from memory
            if workflow_id in self.recovery_points:
                del self.recovery_points[workflow_id]
                
        except Exception as e:
            self.logger.warning(f"Failed to clean up recovery point for {workflow_id}: {e}")
    
    def register_recovery_handler(
        self, 
        workflow_type: str, 
        step_name: str,
        handler: RecoveryFn
    ) -> None:
        """
        Register a recovery handler for a specific workflow step.
        
        Args:
            workflow_type: Type of workflow
            step_name: Name of the workflow step
            handler: Recovery function to call
        """
        if workflow_type not in self.recovery_handlers:
            self.recovery_handlers[workflow_type] = {}
            
        self.recovery_handlers[workflow_type][step_name] = handler
        self.logger.debug(f"Registered recovery handler for {workflow_type}.{step_name}")
    
    def get_recovery_points(
        self, 
        status: Optional[WorkflowStatus] = None,
        after: Optional[float] = None,
        before: Optional[float] = None
    ) -> List[RecoveryPoint]:
        """
        Get recovery points matching criteria.
        
        Args:
            status: Filter by status
            after: Only include points after this timestamp
            before: Only include points before this timestamp
            
        Returns:
            List of matching recovery points
        """
        result = []
        
        for point in self.recovery_points.values():
            # Apply filters
            if status is not None and point.status != status:
                continue
                
            if after is not None and point.timestamp < after:
                continue
                
            if before is not None and point.timestamp > before:
                continue
                
            result.append(point)
            
        return result
    
    def get_failed_workflows(self, max_age_hours: float = 24.0) -> List[RecoveryPoint]:
        """
        Get failed workflows for potential recovery.
        
        Args:
            max_age_hours: Maximum age in hours to consider
            
        Returns:
            List of failed workflow recovery points
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        return self.get_recovery_points(
            status=WorkflowStatus.FAILED,
            after=cutoff_time
        )
    
    def attempt_recovery(
        self, 
        workflow_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.RESUME
    ) -> bool:
        """
        Attempt to recover a failed workflow.
        
        Args:
            workflow_id: Workflow identifier
            strategy: Recovery strategy to use
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if workflow_id not in self.recovery_points:
            self.logger.warning(f"Cannot recover workflow {workflow_id}: no recovery point found")
            return False
            
        recovery_point = self.recovery_points[workflow_id]
        
        # Only attempt recovery for failed workflows
        if recovery_point.status != WorkflowStatus.FAILED:
            self.logger.warning(f"Cannot recover workflow {workflow_id}: status is {recovery_point.status.value}")
            return False
            
        # Get workflow type from state
        workflow_type = recovery_point.state.get("workflow_type", "unknown")
        step_name = recovery_point.step_name
        
        # Check if we have a handler
        handlers = self.recovery_handlers.get(workflow_type, {})
        handler = handlers.get(step_name)
        
        if handler is not None:
            try:
                self.logger.info(f"Attempting to recover workflow {workflow_id} at step {step_name}")
                
                # Update status to recovering
                self.update_recovery_point(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.RUNNING
                )
                
                # Call the recovery handler
                handler(recovery_point.state)
                
                # If we get here, recovery was successful
                self.update_recovery_point(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.RECOVERED
                )
                
                self.logger.info(f"Successfully recovered workflow {workflow_id}")
                return True
                
            except Exception as e:
                self.logger.error(
                    f"Failed to recover workflow {workflow_id}: {e}",
                    exc_info=True
                )
                
                # Update status back to failed
                self.update_recovery_point(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.FAILED
                )
                
                return False
        else:
            self.logger.warning(
                f"No recovery handler found for {workflow_type}.{step_name}"
            )
            return False
    
    def automatic_recovery_loop(self, interval: float = 300.0) -> None:
        """
        Start automatic recovery loop in a background thread.
        
        Args:
            interval: Check interval in seconds
        """
        def recovery_loop() -> None:
            """Background thread function for periodic recovery."""
            while True:
                try:
                    # Get failed workflows in the last 24 hours
                    failed_workflows = self.get_failed_workflows()
                    
                    if failed_workflows:
                        self.logger.info(f"Found {len(failed_workflows)} failed workflows to attempt recovery")
                        
                        for point in failed_workflows:
                            self.attempt_recovery(point.workflow_id)
                    
                except Exception as e:
                    self.logger.error(f"Error in recovery loop: {e}")
                
                time.sleep(interval)
        
        thread = threading.Thread(
            target=recovery_loop, 
            name="recovery-loop",
            daemon=True
        )
        thread.start()
        self.logger.info(f"Started automatic recovery loop with {interval}s interval")


class WorkflowCheckpointer:
    """
    Helper for checkpointing workflows.
    
    This class provides a convenient interface for creating recovery points
    at each step of a workflow.
    """
    
    def __init__(self, workflow_id: str, workflow_type: str):
        """
        Initialize the workflow checkpointer.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_type: Type of workflow
        """
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.recovery_manager = RecoveryManager.get_instance()
        self.logger = logging.getLogger(f"finflow.recovery.workflow.{workflow_id}")
        
        # Current workflow state
        self.state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "start_time": time.time(),
            "steps_completed": []
        }
    
    def checkpoint(self, step_name: str, state_updates: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a checkpoint at the current workflow step.
        
        Args:
            step_name: Name of the current step
            state_updates: Updates to the workflow state
        """
        # Update state
        if state_updates:
            self.state.update(state_updates)
            
        # Record step completion
        if step_name not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step_name)
            
        # Create or update recovery point
        self.recovery_manager.create_recovery_point(
            workflow_id=self.workflow_id,
            step_name=step_name,
            state=self.state
        )
    
    def complete(self, success: bool = True, final_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the workflow as completed.
        
        Args:
            success: Whether the workflow completed successfully
            final_state: Final state updates
        """
        # Update final state
        if final_state:
            self.state.update(final_state)
            
        self.state["end_time"] = time.time()
        self.state["success"] = success
        
        # Mark as completed in recovery manager
        self.recovery_manager.complete_workflow(
            workflow_id=self.workflow_id,
            success=success
        )
        
        status = "successfully" if success else "with failure"
        self.logger.info(f"Workflow {self.workflow_id} completed {status}")


def create_workflow_checkpointer(workflow_type: str, context: Dict[str, Any]) -> WorkflowCheckpointer:
    """
    Create a workflow checkpointer for a new workflow.
    
    Args:
        workflow_type: Type of workflow
        context: Workflow context with workflow_id
        
    Returns:
        Configured WorkflowCheckpointer
    """
    # Generate a workflow ID if not present
    if "workflow_id" not in context:
        import uuid
        context["workflow_id"] = f"workflow-{uuid.uuid4().hex[:8]}"
        
    workflow_id = context["workflow_id"]
    
    return WorkflowCheckpointer(workflow_id, workflow_type)


class StepHandler(Generic[T]):
    """
    Wrapper for workflow step handlers with automatic recovery.
    
    This class provides a way to wrap workflow step functions with
    automatic checkpointing and error handling.
    """
    
    def __init__(
        self,
        step_name: str,
        handler_fn: Callable[[Dict[str, Any]], T],
        workflow_type: str,
        fallback_fn: Optional[Callable[[Dict[str, Any]], T]] = None
    ):
        """
        Initialize the step handler.
        
        Args:
            step_name: Name of the step
            handler_fn: Function that implements the step
            workflow_type: Type of workflow
            fallback_fn: Optional fallback implementation
        """
        self.step_name = step_name
        self.handler_fn = handler_fn
        self.workflow_type = workflow_type
        self.fallback_fn = fallback_fn
        self.logger = logging.getLogger(f"finflow.workflow.step.{step_name}")
        
        # Register with recovery manager
        recovery_manager = RecoveryManager.get_instance()
        recovery_manager.register_recovery_handler(
            workflow_type=workflow_type,
            step_name=step_name,
            handler=self._recovery_handler
        )
    
    def execute(
        self, 
        checkpointer: WorkflowCheckpointer, 
        context: Dict[str, Any]
    ) -> T:
        """
        Execute the step with checkpointing.
        
        Args:
            checkpointer: Workflow checkpointer
            context: Step execution context
            
        Returns:
            Step result
            
        Raises:
            FinFlowError: If the step fails and cannot be recovered
        """
        try:
            # Checkpoint at start of step
            checkpointer.checkpoint(
                step_name=f"{self.step_name}_start", 
                state_updates={"current_step": self.step_name}
            )
            
            # Execute the handler
            self.logger.debug(f"Executing step {self.step_name}")
            result = self.handler_fn(context)
            
            # Checkpoint after successful step
            checkpointer.checkpoint(
                step_name=f"{self.step_name}_complete",
                state_updates={
                    f"{self.step_name}_result": "success",
                    "last_successful_step": self.step_name
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in step {self.step_name}: {e}", exc_info=True)
            
            # Checkpoint the failure
            checkpointer.checkpoint(
                step_name=f"{self.step_name}_failed",
                state_updates={
                    f"{self.step_name}_result": "error",
                    f"{self.step_name}_error": str(e),
                    "last_error": str(e),
                    "last_error_traceback": traceback.format_exc()
                }
            )
            
            # Try fallback if available
            if self.fallback_fn is not None:
                try:
                    self.logger.info(f"Attempting fallback for step {self.step_name}")
                    result = self.fallback_fn(context)
                    
                    # Checkpoint fallback success
                    checkpointer.checkpoint(
                        step_name=f"{self.step_name}_fallback_complete",
                        state_updates={
                            f"{self.step_name}_result": "fallback_success",
                            "last_successful_step": f"{self.step_name}_fallback"
                        }
                    )
                    
                    return result
                    
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback also failed for step {self.step_name}: {fallback_error}",
                        exc_info=True
                    )
                    
                    # Checkpoint fallback failure
                    checkpointer.checkpoint(
                        step_name=f"{self.step_name}_fallback_failed",
                        state_updates={
                            f"{self.step_name}_fallback_result": "error",
                            f"{self.step_name}_fallback_error": str(fallback_error)
                        }
                    )
            
            # If we get here, both main and fallback failed
            if isinstance(e, FinFlowError):
                raise
            else:
                raise FinFlowError(
                    f"Step {self.step_name} failed: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    error_code="ERR_WORKFLOW_STEP_FAILED",
                    details={"step": self.step_name, "workflow_type": self.workflow_type},
                    cause=e
                ) from e
    
    def _recovery_handler(self, state: Dict[str, Any]) -> None:
        """
        Handler for recovery attempts.
        
        Args:
            state: Workflow state from recovery point
        """
        self.logger.info(f"Recovery handler triggered for step {self.step_name}")
        
        # This would be implemented based on the specific step
        # For now, we'll just log the recovery attempt
        self.logger.info(f"Would recover workflow step {self.step_name} with state: {state}")
        
        # In a real implementation, this would:
        # 1. Analyze the state to determine what failed
        # 2. Execute recovery logic specific to this step
        # 3. Either retry the step or skip to the next one
        pass