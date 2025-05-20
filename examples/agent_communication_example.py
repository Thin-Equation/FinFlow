"""
Agent communication framework example for FinFlow.

This example demonstrates how to use the enhanced agent communication framework
for orchestrating workflow execution through intelligent, state-based delegation.
"""

import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent_communication_example")

# Import necessary components
from agents.base_agent import BaseAgent
from agents.document_processor import DocumentProcessorAgent
from agents.validation_agent import ValidationAgent
from agents.storage_agent import StorageAgent
from agents.analytics_agent import AnalyticsAgent
from agents.master_orchestrator import MasterOrchestratorAgent
from agents.agent_delegation import AgentDelegator, DelegatableAgent

from utils.agent_communication import (
    create_enhanced_agent_tool, delegate_task, register_agent_capabilities,
    create_workflow, transition_workflow, get_workflow_data,
    WorkflowState, DelegationStrategy,
    send_message, get_messages, create_response
)
from utils.agent_protocol import MessageType, PriorityLevel, StatusCode
from utils.session_state import SessionState, get_or_create_session_state


def create_example_context() -> Dict[str, Any]:
    """Create example context with session state."""
    # Create session state
    session_state = SessionState()
    
    # Initialize context
    context = {
        "session_id": session_state.session_id,
        "session_state": session_state.to_dict()
    }
    
    return context


def example_document_processing() -> None:
    """Example for document processing with enhanced agent communication."""
    logger.info("Starting document processing example with enhanced agent communication")
    
    # Create context
    context = create_example_context()
    
    # Create worker agents
    document_processor = DocumentProcessorAgent()
    validation_agent = ValidationAgent()
    storage_agent = StorageAgent()
    analytics_agent = AnalyticsAgent()
    
    # Create master orchestrator
    orchestrator = MasterOrchestratorAgent(
        document_processor=document_processor,
        validation_agent=validation_agent,
        storage_agent=storage_agent,
        analytics_agent=analytics_agent
    )
    
    # Register worker agents
    orchestrator.register_worker_agents()
    
    # Initialize delegation framework
    context = orchestrator.initialize_delegation_framework(context)
    
    # Process a sample document
    document_path = os.path.join("sample_data", "invoices", "sample_invoice_1.pdf")
    
    # Add document path to context
    context["document_path"] = document_path
    
    # Process document
    logger.info(f"Processing document: {document_path}")
    result = orchestrator.process_document(context)
    
    # Check result
    logger.info(f"Document processing completed with status: {result.get('status')}")
    logger.info(f"Document type: {result.get('document_type')}")
    logger.info(f"Is valid: {result.get('is_valid')}")
    logger.info(f"Steps completed: {result.get('steps_completed')}")
    
    # Check workflow
    workflow_id = result.get("workflow_id")
    if workflow_id:
        workflow_data = get_workflow_data(result, workflow_id)
        logger.info(f"Workflow state: {workflow_data.get('state')}")
        logger.info(f"Workflow history:")
        
        for i, transition in enumerate(workflow_data.get("history", [])):
            logger.info(f"  {i+1}. {transition.get('from_state')} -> {transition.get('to_state')} by {transition.get('performer_id')}")


def example_delegation_patterns() -> None:
    """Example for LLM-driven delegation patterns."""
    logger.info("Starting LLM-driven delegation patterns example")
    
    # Create context
    context = create_example_context()
    
    # Create delegator and worker agents
    delegator = AgentDelegator()
    
    # Create delegatable worker agents
    class ExampleWorker(DelegatableAgent):
        def process_task(self, task_description: str, context: Dict[str, Any], tool_context: Optional[Any] = None) -> Dict[str, Any]:
            logger.info(f"{self.name} processing task: {task_description}")
            return {
                "task_result": f"Task completed by {self.name}",
                "agent_name": self.name,
                "capabilities_used": self.capabilities,
            }
    
    workers = {
        "document_worker": ExampleWorker(
            name="document_worker",
            model="gemini-2.0-flash",
            description="Document processing worker agent",
            capabilities=["document_processing", "information_extraction"],
        ),
        "validation_worker": ExampleWorker(
            name="validation_worker",
            model="gemini-2.0-flash",
            description="Validation worker agent",
            capabilities=["validation", "rule_checking"],
        ),
        "storage_worker": ExampleWorker(
            name="storage_worker",
            model="gemini-2.0-flash",
            description="Storage worker agent",
            capabilities=["data_storage", "persistence"],
        ),
        "analytics_worker": ExampleWorker(
            name="analytics_worker", 
            model="gemini-2.0-flash",
            description="Analytics worker agent",
            capabilities=["data_analysis", "reporting"],
        ),
    }
    
    # Initialize delegator with worker agents
    delegator.worker_agents = workers
    delegator._register_worker_agents()
    context = delegator.initialize_context(context)
    
    # Create task for document processing
    task_description = "Extract information from invoice document"
    required_capabilities = ["document_processing", "information_extraction"]
    
    # Delegate task
    logger.info(f"Delegating task: {task_description}")
    result = delegator.delegate_agent_task(
        context,
        task_description=task_description,
        required_capabilities=required_capabilities,
        priority=PriorityLevel.HIGH,
        strategy=DelegationStrategy.CAPABILITY_BASED
    )
    
    # Check result
    logger.info(f"Delegation result: {result}")
    
    # Process messages for worker agents to handle delegated tasks
    for name, worker in workers.items():
        worker.process_incoming_messages(context)
    
    # Monitor task progress
    if result.get("success"):
        delegation_request_id = result.get("delegation_request_id")
        monitor_result = delegator.monitor_delegated_task(context, delegation_request_id)
        logger.info(f"Task monitoring result: {monitor_result}")
    
    # Process incoming messages for delegator to handle completions
    delegator.process_incoming_messages(context)
    
    # Show workflow state
    workflow_id = result.get("workflow_id")
    if workflow_id:
        workflow_data = get_workflow_data(context, workflow_id)
        logger.info(f"Workflow state: {workflow_data.get('state')}")


if __name__ == "__main__":
    logger.info("Running agent communication framework examples")
    
    # Run examples
    example_delegation_patterns()
    example_document_processing()
