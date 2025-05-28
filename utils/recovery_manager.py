"""
Recovery manager for the FinFlow system.

This module provides functionality for recovering from failures in document processing
and workflow execution, implementing checkpoint-based recovery, retry strategies,
and partial result handling.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import threading
import uuid
from pathlib import Path

from utils.metrics import AppMetricsCollector, time_function, MetricType, Timer

# Create module logger
logger = logging.getLogger(__name__)

# Define recovery types
class RecoveryStrategy(str, Enum):
    """Types of recovery strategies."""
    RETRY = "retry"               # Simple retry of the failed operation
    CHECKPOINT = "checkpoint"     # Resume from last known good state
    ALTERNATE = "alternate"       # Use an alternate method or pathway
    SKIP = "skip"                 # Skip the failed component and continue
    PARTIAL = "partial"           # Use partial results and continue
    FALLBACK = "fallback"         # Use fallback or cached data


class RecoveryState(str, Enum):
    """State of recovery process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded" 
    FAILED = "failed"
    ABANDONED = "abandoned"


class RecoveryCheckpoint:
    """
    Represents a recovery checkpoint with workflow state that can be resumed.
    """
    
    def __init__(
        self, 
        workflow_id: str, 
        checkpoint_id: Optional[str] = None,
        state: Dict[str, Any] = None,
        timestamp: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize a recovery checkpoint.
        
        Args:
            workflow_id: ID of the workflow this checkpoint belongs to
            checkpoint_id: Optional ID for this checkpoint (auto-generated if not provided)
            state: State to save in the checkpoint
            timestamp: Optional timestamp (defaults to current time)
            metadata: Optional metadata dictionary
        """
        self.workflow_id = workflow_id
        self.checkpoint_id = checkpoint_id or f"cp_{uuid.uuid4().hex[:8]}"
        self.state = state or {}
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "metadata": self.metadata,
            "state": self.state
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecoveryCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            checkpoint_id=data["checkpoint_id"],
            state=data["state"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )
    

class RecoveryPlan:
    """
    Defines a plan for recovery from a failure.
    """
    
    def __init__(
        self,
        entity_id: str,
        entity_type: str,
        strategy: RecoveryStrategy,
        state: RecoveryState = RecoveryState.PENDING,
        checkpoint: Optional[RecoveryCheckpoint] = None,
        max_attempts: int = 3,
        details: Dict[str, Any] = None
    ):
        """Initialize a recovery plan.
        
        Args:
            entity_id: ID of the entity to recover (workflow, document, etc.)
            entity_type: Type of entity (workflow, document, etc.)
            strategy: Recovery strategy to use
            state: Current state of recovery
            checkpoint: Optional checkpoint to use for recovery
            max_attempts: Maximum number of recovery attempts
            details: Additional details for recovery
        """
        self.recovery_id = f"recovery_{uuid.uuid4().hex}"
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.strategy = strategy
        self.state = state
        self.checkpoint = checkpoint
        self.max_attempts = max_attempts
        self.attempts = 0
        self.details = details or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        self.completed_at: Optional[float] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery plan to dictionary."""
        result = {
            "recovery_id": self.recovery_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "strategy": self.strategy,
            "state": self.state,
            "max_attempts": self.max_attempts,
            "attempts": self.attempts,
            "details": self.details,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at).isoformat(),
            "updated_at": self.updated_at,
            "updated_at_iso": datetime.fromtimestamp(self.updated_at).isoformat(),
        }
        
        if self.checkpoint:
            result["checkpoint"] = self.checkpoint.to_dict()
            
        if self.completed_at:
            result["completed_at"] = self.completed_at
            result["completed_at_iso"] = datetime.fromtimestamp(self.completed_at).isoformat()
            
        return result
    

class RecoveryManager:
    """
    Manager for handling recovery from failures in the FinFlow system.
    
    Provides:
    1. Checkpoint-based workflow recovery
    2. Partial result handling
    3. Recovery strategies for various failure scenarios
    4. Persistence of recovery state
    """
    
    # Singleton instance
    _instance: Optional['RecoveryManager'] = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, config: Dict[str, Any] = None) -> 'RecoveryManager':
        """Get the singleton instance of RecoveryManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or {})
            return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the recovery manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("finflow.recovery")
        
        # Configure recovery directory
        self.recovery_dir = config.get("recovery_dir") or os.path.join(
            os.getcwd(), "recovery_data"
        )
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Track active recovery operations
        self.active_recoveries: Dict[str, RecoveryPlan] = {}
        
        # Configure checkpoint retention
        self.checkpoint_retention_days = config.get("checkpoint_retention_days", 7)
        
        # Get metrics collector
        self.metrics = AppMetricsCollector.get_instance()
        
        # Initialize recovery components
        self._init_components()
        
        self.logger.info(f"Recovery manager initialized with recovery_dir={self.recovery_dir}")
    
    def _init_components(self) -> None:
        """Initialize recovery components."""
        # Load any existing recovery plans
        try:
            self._load_recovery_plans()
            self.logger.info("Recovery plans loaded")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Error initializing recovery components: {e}")
    
    def _load_recovery_plans(self) -> None:
        """Load existing recovery plans from disk."""
        recovery_files = Path(self.recovery_dir).glob("recovery_*.json")
        
        for file_path in recovery_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create recovery plan
                plan = RecoveryPlan(
                    entity_id=data["entity_id"],
                    entity_type=data["entity_type"],
                    strategy=RecoveryStrategy(data["strategy"]),
                    state=RecoveryState(data["state"]),
                    max_attempts=data["max_attempts"],
                    details=data["details"]
                )
                
                # Set other attributes
                plan.recovery_id = data["recovery_id"]
                plan.attempts = data["attempts"]
                plan.created_at = data["created_at"]
                plan.updated_at = data["updated_at"]
                
                if "completed_at" in data:
                    plan.completed_at = data["completed_at"]
                
                # Restore checkpoint if available
                if "checkpoint" in data:
                    plan.checkpoint = RecoveryCheckpoint.from_dict(data["checkpoint"])
                
                # Only add active recovery plans
                if plan.state in [RecoveryState.PENDING, RecoveryState.IN_PROGRESS]:
                    self.active_recoveries[plan.recovery_id] = plan
                    self.logger.info(f"Loaded active recovery plan: {plan.recovery_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading recovery plan from {file_path}: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints that have expired."""
        checkpoint_files = Path(self.recovery_dir).glob("checkpoint_*.json")
        cutoff_time = time.time() - (self.checkpoint_retention_days * 24 * 3600)
        
        deleted_count = 0
        for file_path in checkpoint_files:
            try:
                # Check file modification time
                mtime = file_path.stat().st_mtime
                if mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up checkpoint {file_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} expired checkpoints")
    
    @time_function("create_checkpoint")
    def create_checkpoint(
        self, 
        workflow_id: str, 
        state: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> RecoveryCheckpoint:
        """Create a recovery checkpoint for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            state: State to save
            metadata: Additional metadata
            
        Returns:
            RecoveryCheckpoint: The created checkpoint
        """
        # Create checkpoint object
        checkpoint = RecoveryCheckpoint(
            workflow_id=workflow_id,
            state=state,
            metadata=metadata or {}
        )
        
        # Save checkpoint to disk
        self._save_checkpoint(checkpoint)
        
        # Record metric
        self.metrics.counter("checkpoints_created").increment(
            labels={"workflow_id": workflow_id}
        )
        
        self.logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for workflow {workflow_id}")
        
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        """Save a checkpoint to disk.
        
        Args:
            checkpoint: The checkpoint to save
        """
        file_path = os.path.join(
            self.recovery_dir, 
            f"checkpoint_{checkpoint.workflow_id}_{checkpoint.checkpoint_id}.json"
        )
        
        with open(file_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f)
    
    def get_latest_checkpoint(self, workflow_id: str) -> Optional[RecoveryCheckpoint]:
        """Get the latest checkpoint for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Optional[RecoveryCheckpoint]: The latest checkpoint, or None if not found
        """
        # Find checkpoint files for this workflow
        checkpoint_files = list(Path(self.recovery_dir).glob(f"checkpoint_{workflow_id}_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Load the newest checkpoint
        try:
            with open(checkpoint_files[0], 'r') as f:
                data = json.load(f)
                
            return RecoveryCheckpoint.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    @time_function("create_recovery_plan")
    def create_recovery_plan(
        self,
        entity_id: str,
        entity_type: str,
        strategy: RecoveryStrategy,
        checkpoint: Optional[RecoveryCheckpoint] = None,
        details: Dict[str, Any] = None,
        max_attempts: int = 3
    ) -> RecoveryPlan:
        """Create a recovery plan for a failed entity.
        
        Args:
            entity_id: ID of the entity to recover
            entity_type: Type of entity
            strategy: Recovery strategy to use
            checkpoint: Optional checkpoint to use
            details: Additional details
            max_attempts: Maximum number of recovery attempts
            
        Returns:
            RecoveryPlan: The created recovery plan
        """
        # Create recovery plan
        plan = RecoveryPlan(
            entity_id=entity_id,
            entity_type=entity_type,
            strategy=strategy,
            checkpoint=checkpoint,
            details=details or {},
            max_attempts=max_attempts
        )
        
        # Save to active recoveries
        self.active_recoveries[plan.recovery_id] = plan
        
        # Save to disk
        self._save_recovery_plan(plan)
        
        # Record metric
        self.metrics.counter("recovery_plans_created").increment(
            labels={"entity_type": entity_type, "strategy": strategy}
        )
        
        self.logger.info(
            f"Created recovery plan {plan.recovery_id} for {entity_type} {entity_id} "
            f"using strategy {strategy}"
        )
        
        return plan
    
    def _save_recovery_plan(self, plan: RecoveryPlan) -> None:
        """Save a recovery plan to disk.
        
        Args:
            plan: The recovery plan to save
        """
        file_path = os.path.join(
            self.recovery_dir, 
            f"recovery_{plan.recovery_id}.json"
        )
        
        with open(file_path, 'w') as f:
            json.dump(plan.to_dict(), f)
    
    @time_function("execute_recovery")
    def execute_recovery(
        self, 
        recovery_id: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a recovery plan.
        
        Args:
            recovery_id: ID of the recovery plan
            context: Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        if recovery_id not in self.active_recoveries:
            raise ValueError(f"Recovery plan not found: {recovery_id}")
        
        plan = self.active_recoveries[recovery_id]
        
        # Check max attempts
        if plan.attempts >= plan.max_attempts:
            plan.state = RecoveryState.ABANDONED
            self._save_recovery_plan(plan)
            
            self.logger.warning(
                f"Recovery {recovery_id} abandoned: exceeded max attempts ({plan.max_attempts})"
            )
            
            return {
                "recovery_id": recovery_id,
                "status": "abandoned",
                "reason": "exceeded_max_attempts"
            }
        
        # Update state
        plan.state = RecoveryState.IN_PROGRESS
        plan.attempts += 1
        plan.updated_at = time.time()
        self._save_recovery_plan(plan)
        
        self.logger.info(f"Executing recovery {recovery_id} (attempt {plan.attempts}/{plan.max_attempts})")
        
        # Create timer
        timer = Timer(f"recovery_execution_{recovery_id}")
        timer.start()
        
        try:
            # Execute recovery based on strategy
            if plan.strategy == RecoveryStrategy.RETRY:
                result = self._execute_retry_strategy(plan, context or {})
                
            elif plan.strategy == RecoveryStrategy.CHECKPOINT:
                result = self._execute_checkpoint_strategy(plan, context or {})
                
            elif plan.strategy == RecoveryStrategy.ALTERNATE:
                result = self._execute_alternate_strategy(plan, context or {})
                
            elif plan.strategy == RecoveryStrategy.SKIP:
                result = self._execute_skip_strategy(plan, context or {})
                
            elif plan.strategy == RecoveryStrategy.PARTIAL:
                result = self._execute_partial_strategy(plan, context or {})
                
            elif plan.strategy == RecoveryStrategy.FALLBACK:
                result = self._execute_fallback_strategy(plan, context or {})
                
            else:
                raise ValueError(f"Unsupported recovery strategy: {plan.strategy}")
            
            # Update plan based on result
            if result.get("status") == "success":
                plan.state = RecoveryState.SUCCEEDED
                
            elif result.get("status") == "partial":
                # Partial success - may need further recovery
                if plan.attempts >= plan.max_attempts:
                    plan.state = RecoveryState.SUCCEEDED
                else:
                    # Keep in progress for further attempts
                    pass
                    
            else:
                # Failed recovery
                if plan.attempts >= plan.max_attempts:
                    plan.state = RecoveryState.FAILED
            
            # Update and save plan
            plan.updated_at = time.time()
            if plan.state in [RecoveryState.SUCCEEDED, RecoveryState.FAILED]:
                plan.completed_at = time.time()
                
            self._save_recovery_plan(plan)
            
            # Record metrics
            execution_time = timer.stop()
            self.metrics.histogram(
                "recovery_execution_time",
                execution_time,
                labels={
                    "entity_type": plan.entity_type,
                    "strategy": plan.strategy,
                    "status": result.get("status", "unknown")
                }
            )
            
            if result.get("status") == "success":
                self.metrics.counter("recovery_success").increment(
                    labels={"entity_type": plan.entity_type, "strategy": plan.strategy}
                )
            else:
                self.metrics.counter("recovery_failure").increment(
                    labels={"entity_type": plan.entity_type, "strategy": plan.strategy}
                )
            
            self.logger.info(
                f"Recovery {recovery_id} completed with status {result.get('status')} "
                f"in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            # Handle errors in recovery execution
            execution_time = timer.stop()
            
            self.logger.error(f"Error executing recovery {recovery_id}: {e}")
            
            # Update plan
            plan.updated_at = time.time()
            
            if plan.attempts >= plan.max_attempts:
                plan.state = RecoveryState.FAILED
                plan.completed_at = time.time()
            else:
                plan.state = RecoveryState.PENDING  # Ready for next attempt
                
            self._save_recovery_plan(plan)
            
            # Record error metric
            self.metrics.counter("recovery_errors").increment(
                labels={"entity_type": plan.entity_type, "error_type": type(e).__name__}
            )
            
            return {
                "recovery_id": recovery_id,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time
            }
    
    def _execute_retry_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a retry recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        # In a real implementation, this would retry the failed operation
        # using any configured retry mechanisms
        
        self.logger.info(
            f"Executing RETRY recovery for {plan.entity_type} {plan.entity_id}"
        )
        
        # For demonstration, we'll simulate success
        return {
            "recovery_id": plan.recovery_id,
            "status": "success",
            "details": {
                "retry_attempt": plan.attempts,
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def _execute_checkpoint_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a checkpoint-based recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        # Check if plan has a checkpoint
        if not plan.checkpoint:
            return {
                "recovery_id": plan.recovery_id,
                "status": "failed",
                "reason": "no_checkpoint_available"
            }
        
        self.logger.info(
            f"Executing CHECKPOINT recovery for {plan.entity_type} {plan.entity_id} "
            f"using checkpoint {plan.checkpoint.checkpoint_id}"
        )
        
        # For demonstration, we'll simulate success
        return {
            "recovery_id": plan.recovery_id,
            "status": "success",
            "details": {
                "checkpoint_id": plan.checkpoint.checkpoint_id,
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def _execute_alternate_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an alternate path recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        self.logger.info(
            f"Executing ALTERNATE recovery for {plan.entity_type} {plan.entity_id}"
        )
        
        # For demonstration, we'll simulate success
        return {
            "recovery_id": plan.recovery_id,
            "status": "success",
            "details": {
                "alternate_method": "simplified_processing",
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def _execute_skip_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a skip-failure recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        self.logger.info(
            f"Executing SKIP recovery for {plan.entity_type} {plan.entity_id}"
        )
        
        # For demonstration, we'll simulate success
        return {
            "recovery_id": plan.recovery_id,
            "status": "success",
            "details": {
                "skipped_step": plan.details.get("step_name", "unknown"),
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def _execute_partial_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a partial results recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        self.logger.info(
            f"Executing PARTIAL recovery for {plan.entity_type} {plan.entity_id}"
        )
        
        # For demonstration, we'll simulate partial success
        return {
            "recovery_id": plan.recovery_id,
            "status": "partial",
            "details": {
                "partial_fields": ["field1", "field2"],
                "missing_fields": ["field3"],
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def _execute_fallback_strategy(
        self, 
        plan: RecoveryPlan, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a fallback data recovery strategy.
        
        Args:
            plan: Recovery plan
            context: Additional context
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        self.logger.info(
            f"Executing FALLBACK recovery for {plan.entity_type} {plan.entity_id}"
        )
        
        # For demonstration, we'll simulate success
        return {
            "recovery_id": plan.recovery_id,
            "status": "success",
            "details": {
                "fallback_source": "cached_data",
                "fallback_quality": "medium",
                "entity_id": plan.entity_id,
                "entity_type": plan.entity_type
            }
        }
    
    def get_active_recoveries(self) -> List[Dict[str, Any]]:
        """Get list of all active recovery plans.
        
        Returns:
            List[Dict[str, Any]]: List of recovery plan dictionaries
        """
        return [plan.to_dict() for plan in self.active_recoveries.values()]
    
    def get_recovery_plan(self, recovery_id: str) -> Optional[RecoveryPlan]:
        """Get a recovery plan by ID.
        
        Args:
            recovery_id: ID of the recovery plan
            
        Returns:
            Optional[RecoveryPlan]: The recovery plan, or None if not found
        """
        return self.active_recoveries.get(recovery_id)
