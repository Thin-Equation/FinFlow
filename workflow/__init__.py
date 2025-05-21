"""
Workflow module for FinFlow.

This module provides workflow definitions and execution engines for financial processes.
"""

from workflow.workflow_definitions import (
    Workflow,
    WorkflowTask,
    WorkflowDefinition,
    FinancialProcess,
    WorkflowResult,
    WorkflowExecutionContext
)

from workflow.sequential_agent import SequentialAgent
from workflow.parallel_agent import ParallelAgent
from workflow.conditional import ConditionalBranching

__all__ = [
    'Workflow',
    'WorkflowTask',
    'WorkflowDefinition',
    'FinancialProcess',
    'WorkflowResult',
    'WorkflowExecutionContext',
    'SequentialAgent',
    'ParallelAgent',
    'ConditionalBranching'
]
