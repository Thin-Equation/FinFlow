"""
Conditional branching for workflows.

This module provides conditional branching capabilities for workflows,
allowing dynamic selection of paths based on criteria.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TypeVar

from workflow.workflow_definitions import (
    WorkflowDefinition,
    WorkflowTask,
    WorkflowExecutionContext,
    TaskId
)

# Define type for condition functions
Condition = Callable[[WorkflowExecutionContext], bool]
T = TypeVar('T')

class BranchSelectionStrategy(str, Enum):
    """Strategy for branch selection when multiple branches match."""
    FIRST = "first"  # Take the first matching branch
    PRIORITY = "priority"  # Take the highest priority branch
    ALL = "all"  # Take all matching branches

class Branch:
    """A branch in a conditional workflow."""
    
    def __init__(
        self,
        name: str,
        condition: Condition,
        tasks: List[WorkflowTask],
        priority: int = 0
    ):
        """
        Initialize a branch.
        
        Args:
            name: The name of the branch
            condition: The condition function to evaluate
            tasks: The tasks to execute if the condition is true
            priority: Branch priority for conflict resolution (higher wins)
        """
        self.name = name
        self.condition = condition
        self.tasks = tasks
        self.priority = priority
    
    def evaluate(self, context: WorkflowExecutionContext) -> bool:
        """
        Evaluate the branch condition.
        
        Args:
            context: The workflow execution context
            
        Returns:
            bool: True if the condition is satisfied
        """
        try:
            return self.condition(context)
        except Exception as e:
            # Log the error and return False
            logging.getLogger(f"finflow.workflow.conditional.{self.name}").error(
                f"Error evaluating condition for branch '{self.name}': {str(e)}"
            )
            return False

class ConditionalBranching:
    """
    Utility class for adding conditional branching to workflows.
    Allows dynamic path selection based on runtime conditions.
    """
    
    @staticmethod
    def add_conditional_branch(
        workflow: WorkflowDefinition,
        branch: Branch,
        parent_task_id: Optional[TaskId] = None,
        exit_task_id: Optional[TaskId] = None
    ) -> List[TaskId]:
        """
        Add a conditional branch to a workflow.
        
        Args:
            workflow: The workflow to modify
            branch: The branch definition
            parent_task_id: The ID of the task that this branch depends on
            exit_task_id: The ID of the task that follows this branch
        
        Returns:
            List[TaskId]: The task IDs added to the workflow
        """
        branch_task_ids = []
        
        # Create a conditional router task that will decide whether to execute the branch
        router_task = WorkflowTask(
            id=f"{branch.name}_router",
            name=f"Branch Router: {branch.name}",
            description=f"Evaluates condition for branch '{branch.name}'",
            execute=lambda ctx: ConditionalBranching._evaluate_branch_condition(branch, ctx),
            dependencies=[parent_task_id] if parent_task_id else []
        )
        
        # Add router task to the workflow
        workflow.add_task(router_task)
        branch_task_ids.append(router_task.id)
        
        # Add tasks from the branch to the workflow
        previous_task_id = router_task.id
        for task in branch.tasks:
            # Make task dependent on the previous task in the branch
            if previous_task_id != router_task.id:
                task.dependencies.append(previous_task_id)
            else:
                # First task in branch depends on router
                task.dependencies.append(router_task.id)
            
            # Add task to workflow
            workflow.add_task(task)
            branch_task_ids.append(task.id)
            
            # Update previous task
            previous_task_id = task.id
        
        # Add connection to exit task if provided
        if exit_task_id and branch.tasks:
            exit_task = workflow.get_task(exit_task_id)
            if exit_task:
                # Make exit task dependent on the last task in the branch
                exit_task.dependencies.append(branch.tasks[-1].id)
        
        return branch_task_ids
    
    @staticmethod
    def _evaluate_branch_condition(branch: Branch, context: WorkflowExecutionContext) -> bool:
        """
        Evaluate a branch condition and store the result in the context.
        
        Args:
            branch: The branch to evaluate
            context: The workflow execution context
            
        Returns:
            bool: The result of the condition evaluation
        """
        result = branch.evaluate(context)
        
        # Store result in context for inspection later
        branch_results = context.get_state("branch_results", {})
        branch_results[branch.name] = result
        context.set_state("branch_results", branch_results)
        
        # If condition is false, mark the branch as skipped in the context
        if not result:
            skipped_branches = context.get_state("skipped_branches", set())
            skipped_branches.add(branch.name)
            context.set_state("skipped_branches", skipped_branches)
        
        return result
    
    @staticmethod
    def add_switch(
        workflow: WorkflowDefinition,
        switch_name: str,
        branches: List[Branch],
        parent_task_id: Optional[TaskId] = None,
        exit_task_id: Optional[TaskId] = None,
        strategy: BranchSelectionStrategy = BranchSelectionStrategy.FIRST,
        default_branch: Optional[Branch] = None
    ) -> Dict[str, List[TaskId]]:
        """
        Add a switch (multiple conditional branches) to a workflow.
        
        Args:
            workflow: The workflow to modify
            switch_name: The name of the switch
            branches: The branches to add
            parent_task_id: The ID of the task that this switch depends on
            exit_task_id: The ID of the task that follows this switch
            strategy: The strategy for selecting branches when multiple match
            default_branch: A default branch to take if no conditions match
            
        Returns:
            Dict[str, List[TaskId]]: Dictionary mapping branch names to their task IDs
        """
        branch_tasks = {}
        switch_branches = list(branches)
        
        # Sort branches by priority if using PRIORITY strategy
        if strategy == BranchSelectionStrategy.PRIORITY:
            switch_branches.sort(key=lambda b: b.priority, reverse=True)
        
        # Add the default branch if provided
        if default_branch:
            # Create a condition that returns true if no other branch matched
            orig_condition = default_branch.condition
            
            def default_condition(context: WorkflowExecutionContext) -> bool:
                branch_results = context.get_state("branch_results", {})
                # If any branch matched, don't take the default
                if any(branch_results.values()):
                    return False
                # Otherwise, evaluate the original condition if any
                return orig_condition(context) if callable(orig_condition) else True
            
            default_branch.condition = default_condition
            switch_branches.append(default_branch)
        
        # Create a switch router task
        switch_router = WorkflowTask(
            id=f"{switch_name}_router",
            name=f"Switch Router: {switch_name}",
            description=f"Routes execution through the '{switch_name}' switch",
            execute=lambda ctx: ConditionalBranching._init_switch_context(ctx, switch_name),
            dependencies=[parent_task_id] if parent_task_id else []
        )
        
        # Add the router task
        workflow.add_task(switch_router)
        
        # Add each branch
        for branch in switch_branches:
            # Make each branch depend on the switch router
            branch_task_ids = ConditionalBranching.add_conditional_branch(
                workflow=workflow,
                branch=branch,
                parent_task_id=switch_router.id,
                exit_task_id=exit_task_id
            )
            branch_tasks[branch.name] = branch_task_ids
        
        return branch_tasks
    
    @staticmethod
    def _init_switch_context(context: WorkflowExecutionContext, switch_name: str) -> Dict[str, Any]:
        """Initialize the context for a switch."""
        context.set_state(f"switch_{switch_name}_initialized", True)
        context.set_state("branch_results", {})
        context.set_state("skipped_branches", set())
        return {"status": "initialized"}
    
    @staticmethod
    def create_condition_from_predicate(
        predicate: Callable[[Dict[str, Any]], bool],
        state_key: Optional[str] = None,
        task_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Condition:
        """
        Create a condition function from a predicate function.
        
        Args:
            predicate: Function that takes a dict and returns a boolean
            state_key: Key in the context state to evaluate
            task_id: Task ID whose result should be evaluated
            parameters: Fixed parameters to pass to the predicate
            
        Returns:
            Condition: A condition function
        """
        def condition_func(context: WorkflowExecutionContext) -> bool:
            if state_key:
                value = context.get_state(state_key)
                return predicate(value)
            elif task_id:
                result = context.get_result(task_id)
                return predicate(result)
            elif parameters:
                return predicate(parameters)
            else:
                return predicate(context.parameters)
        
        return condition_func
    
    @staticmethod
    def eq(value: Any) -> Callable[[Any], bool]:
        """Create an equality predicate."""
        return lambda x: x == value
    
    @staticmethod
    def gt(value: float) -> Callable[[Any], bool]:
        """Create a greater than predicate."""
        return lambda x: isinstance(x, (int, float)) and x > value
    
    @staticmethod
    def lt(value: float) -> Callable[[Any], bool]:
        """Create a less than predicate."""
        return lambda x: isinstance(x, (int, float)) and x < value
    
    @staticmethod
    def contains(value: Any) -> Callable[[Any], bool]:
        """Create a contains predicate."""
        return lambda x: value in x if hasattr(x, '__contains__') else False
    
    @staticmethod
    def has_key(key: str) -> Callable[[Dict], bool]:
        """Create a has_key predicate for dictionaries."""
        return lambda x: isinstance(x, dict) and key in x
    
    @staticmethod
    def and_condition(*conditions: Condition) -> Condition:
        """Combine multiple conditions with AND logic."""
        def combined_condition(context: WorkflowExecutionContext) -> bool:
            return all(condition(context) for condition in conditions)
        return combined_condition
    
    @staticmethod
    def or_condition(*conditions: Condition) -> Condition:
        """Combine multiple conditions with OR logic."""
        def combined_condition(context: WorkflowExecutionContext) -> bool:
            return any(condition(context) for condition in conditions)
        return combined_condition
    
    @staticmethod
    def not_condition(condition: Condition) -> Condition:
        """Negate a condition."""
        def negated_condition(context: WorkflowExecutionContext) -> bool:
            return not condition(context)
        return negated_condition
