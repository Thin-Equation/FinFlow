"""
Workflow runner for the FinFlow system.

This module provides a workflow execution framework for running defined workflows.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from workflow.workflow_definitions import WorkflowStatus, WorkflowExecutionContext

logger = logging.getLogger(__name__)


def get_workflow_definition(workflow_name: str) -> Dict[str, Any]:
    """
    Get a workflow definition by name.
    
    Args:
        workflow_name: Name of the workflow to get
        
    Returns:
        Dict[str, Any]: Workflow definition
    
    Raises:
        ValueError: If workflow not found
    """
    # In a production system, this would load from a database or registry
    # For now, we'll define some sample workflows
    
    workflows = {
        "standard": {
            "name": "Standard Document Processing",
            "description": "Standard document processing workflow",
            "steps": [
                {"name": "document_extraction", "agent": "document_processor", "action": "extract_document"},
                {"name": "validation", "agent": "validation_agent", "action": "validate_document"},
                {"name": "storage", "agent": "storage_agent", "action": "store_document"}
            ]
        },
        "invoice": {
            "name": "Invoice Processing",
            "description": "Invoice-specialized processing workflow",
            "steps": [
                {"name": "document_extraction", "agent": "document_processor", "action": "extract_invoice"},
                {"name": "rules_check", "agent": "rule_retrieval", "action": "get_rules"},
                {"name": "validation", "agent": "validation_agent", "action": "validate_invoice"},
                {"name": "storage", "agent": "storage_agent", "action": "store_document"},
                {"name": "analytics", "agent": "analytics_agent", "action": "analyze_invoice"}
            ]
        },
        "receipt": {
            "name": "Receipt Processing",
            "description": "Receipt-specialized processing workflow",
            "steps": [
                {"name": "document_extraction", "agent": "document_processor", "action": "extract_receipt"},
                {"name": "validation", "agent": "validation_agent", "action": "validate_receipt"},
                {"name": "storage", "agent": "storage_agent", "action": "store_document"},
                {"name": "analytics", "agent": "analytics_agent", "action": "analyze_receipt"}
            ]
        },
    }
    
    if workflow_name not in workflows:
        raise ValueError(f"Workflow not found: {workflow_name}")
    
    return workflows[workflow_name]


def run_workflow(
    workflow_name: str,
    agents: Dict[str, Any],
    config: Dict[str, Any],
    document_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a workflow by name.
    
    Args:
        workflow_name: Name of the workflow to run
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        document_path: Path to document to process
        context: Optional context dictionary to use (will override document_path)
        
    Returns:
        Dict[str, Any]: Workflow execution result
    """
    logger.info(f"Running workflow: {workflow_name}")
    
    # Get workflow definition
    workflow = get_workflow_definition(workflow_name)
    
    # Create workflow context
    workflow_context = context or {}
    workflow_id = f"workflow_{int(time.time())}_{workflow_name}"
    
    if document_path and not context:
        workflow_context = {
            "document_path": document_path,
            "workflow_type": workflow_name,
            "user_id": "workflow_runner",
            "session_id": f"workflow_{datetime.now().timestamp()}",
        }
    
    # Create execution context
    execution_context = WorkflowExecutionContext(
        workflow_id=workflow_id,
        parameters=workflow_context,
        state={},
        results={},
        start_time=datetime.now()
    )
    
    logger.info(f"Starting workflow execution: {workflow_id}")
    logger.info(f"Workflow: {workflow['name']}")
    
    # Track overall status
    status = WorkflowStatus.RUNNING
    results = {}
    errors = []
    
    # Execute workflow steps
    step_count = len(workflow["steps"])
    
    for i, step in enumerate(workflow["steps"]):
        step_name = step["name"]
        agent_name = step["agent"]
        action_name = step["action"]
        
        logger.info(f"Executing step {i+1}/{step_count}: {step_name} using {agent_name}.{action_name}")
        
        # Get the agent
        if agent_name not in agents:
            error_message = f"Agent not found: {agent_name}"
            logger.error(error_message)
            errors.append({"step": step_name, "error": error_message})
            status = WorkflowStatus.FAILED
            break
        
        agent = agents[agent_name]
        
        # Check if agent has the required action
        if not hasattr(agent, action_name):
            error_message = f"Action not found on agent {agent_name}: {action_name}"
            logger.error(error_message)
            errors.append({"step": step_name, "error": error_message})
            status = WorkflowStatus.FAILED
            break
        
        # Execute the action
        try:
            start_time = time.time()
            action = getattr(agent, action_name)
            
            # Update context with previous step results
            execution_context.state["current_step"] = step_name
            
            # Call the action with the execution context
            step_result = action(execution_context.parameters)
            
            # Record the step result
            execution_context.results[step_name] = step_result
            results[step_name] = step_result
            
            processing_time = time.time() - start_time
            logger.info(f"Step {step_name} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error executing step {step_name}: {e}")
            errors.append({"step": step_name, "error": str(e)})
            status = WorkflowStatus.FAILED
            break
    
    # Complete workflow
    execution_context.end_time = datetime.now()
    
    if status != WorkflowStatus.FAILED:
        status = WorkflowStatus.COMPLETED
    
    # Prepare final result
    final_result = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "status": status,
        "steps": len(workflow["steps"]),
        "steps_completed": len(results),
        "results": results,
        "errors": errors,
        "start_time": execution_context.start_time.isoformat(),
        "end_time": execution_context.end_time.isoformat(),
        "processing_time": (execution_context.end_time - execution_context.start_time).total_seconds(),
    }
    
    logger.info(f"Workflow {workflow_id} completed with status: {status}")
    
    return final_result
