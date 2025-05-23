#!/usr/bin/env python3
"""
Advanced Agent Communication Example

This example demonstrates how to use the enhanced agent communication framework
for robust agent-to-agent communication, delegation, and task execution.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from utils.agent_communication import (
    CommunicationProtocol,
    TaskExecutionFramework,
    create_enhanced_agent_tool,
    register_agent_capabilities,
    update_agent_status,
    delegate_task,
    DelegationStrategy,
    WorkflowState,
    create_workflow,
    transition_workflow,
    PriorityLevel
)
from utils.session_state import SessionState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_communication_example")

class DocumentProcessorAgent(BaseAgent):
    """Example document processor agent."""
    
    def __init__(self):
        super().__init__(
            name="DocumentProcessor",
            model="gemini-2.0-flash",
            description="Processes documents to extract information"
        )
        self.logger = logging.getLogger("finflow.agents.DocumentProcessor")
        
    def process(self, context: Dict[str, Any], tool_context=None) -> Dict[str, Any]:
        """Process a document and extract information."""
        self.logger.info("Processing document...")
        
        # Initialize communication protocol
        comms = CommunicationProtocol(context, self.name)
        
        # Simulate document processing
        time.sleep(1)
        
        # Create a response with extracted data
        document_path = context.get("document_path", "unknown")
        result = {
            "status": "success",
            "document_type": "invoice",
            "entities": {
                "invoice_number": "INV-12345",
                "date": "2025-05-15",
                "total_amount": 1000.0
            },
            "confidence": 0.95
        }
        
        # Acknowledge the processing request
        if "request_message_id" in context:
            comms.acknowledge_message(context["request_message_id"])
        
        self.logger.info(f"Successfully processed document: {document_path}")
        return result

class ValidationAgent(BaseAgent):
    """Example validation agent."""
    
    def __init__(self):
        super().__init__(
            name="ValidationAgent",
            model="gemini-2.0-flash",
            description="Validates document data against rules"
        )
        self.logger = logging.getLogger("finflow.agents.ValidationAgent")
        
    def process(self, context: Dict[str, Any], tool_context=None) -> Dict[str, Any]:
        """Validate document data against rules."""
        self.logger.info("Validating document data...")
        
        # Simulate validation
        time.sleep(0.5)
        
        # Get extraction result from context
        extraction_result = context.get("extraction_result", {})
        
        # Initialize task execution framework
        task_framework = TaskExecutionFramework(context, self.name)
        
        # Create the main validation task
        task_id = task_framework.create_task(
            task_description="Validate invoice data",
            task_type="validation",
            metadata={"document_type": extraction_result.get("document_type")}
        )
        
        # Define validation subtasks
        subtasks = [
            {
                "description": "Validate invoice number",
                "type": "field_validation",
                "metadata": {"field": "invoice_number"},
                "executor": lambda task: self._validate_invoice_number(
                    extraction_result.get("entities", {}).get("invoice_number")
                )
            },
            {
                "description": "Validate date",
                "type": "field_validation",
                "metadata": {"field": "date"},
                "executor": lambda task: self._validate_date(
                    extraction_result.get("entities", {}).get("date")
                )
            },
            {
                "description": "Validate total amount",
                "type": "field_validation",
                "metadata": {"field": "total_amount"},
                "executor": lambda task: self._validate_total_amount(
                    extraction_result.get("entities", {}).get("total_amount")
                )
            }
        ]
        
        # Execute all validation subtasks
        subtask_results = task_framework.create_and_execute_subtasks(
            parent_task_id=task_id,
            subtask_definitions=subtasks
        )
        
        # Check if all validations passed
        all_valid = all(
            result.get("valid", False) 
            for result in subtask_results.values()
        )
        
        validation_result = {
            "is_valid": all_valid,
            "validation_time": datetime.now().isoformat(),
            "field_results": subtask_results
        }
        
        self.logger.info(f"Validation completed. Document is {'valid' if all_valid else 'invalid'}")
        return validation_result
    
    def _validate_invoice_number(self, invoice_number: str) -> Dict[str, Any]:
        """Validate invoice number format."""
        # Simple validation - check if it starts with 'INV-'
        valid = invoice_number and invoice_number.startswith("INV-")
        return {
            "field": "invoice_number",
            "valid": valid,
            "reason": None if valid else "Invoice number must start with 'INV-'"
        }
    
    def _validate_date(self, date_str: str) -> Dict[str, Any]:
        """Validate date format."""
        # Simple validation - check if date is in YYYY-MM-DD format
        valid = date_str and len(date_str.split("-")) == 3
        return {
            "field": "date",
            "valid": valid,
            "reason": None if valid else "Date must be in YYYY-MM-DD format"
        }
    
    def _validate_total_amount(self, amount: float) -> Dict[str, Any]:
        """Validate total amount."""
        # Simple validation - check if amount is positive
        valid = amount is not None and amount > 0
        return {
            "field": "total_amount",
            "valid": valid,
            "reason": None if valid else "Total amount must be positive"
        }

class MasterOrchestrator:
    """
    Example master orchestrator that coordinates the entire workflow.
    """
    
    def __init__(self):
        self.name = "MasterOrchestrator"
        self.logger = logging.getLogger("finflow.agents.MasterOrchestrator")
        
        # Initialize worker agents
        self.document_processor = DocumentProcessorAgent()
        self.validation_agent = ValidationAgent()
        
        # Initialize agent registry and capabilities
        self.agent_registry = {
            "DocumentProcessor": {
                "agent": self.document_processor,
                "capabilities": ["document_processing", "information_extraction"],
                "status": "active",
                "availability": 1.0,
                "current_load": 0.0
            },
            "ValidationAgent": {
                "agent": self.validation_agent,
                "capabilities": ["validation", "rule_checking"],
                "status": "active",
                "availability": 1.0,
                "current_load": 0.0
            }
        }
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document using the communication framework.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Dict[str, Any]: Processing result
        """
        self.logger.info(f"Starting document processing for: {document_path}")
        
        # Initialize context with session state
        context = {
            "document_path": document_path,
            "agent_id": self.name,
            "start_time": datetime.now().isoformat(),
            "session_state": SessionState().to_dict()
        }
        
        # Register agent capabilities
        for agent_id, info in self.agent_registry.items():
            register_agent_capabilities(
                context=context,
                agent_id=agent_id,
                capabilities=info["capabilities"],
                metadata={
                    "description": getattr(info["agent"], "description", ""),
                    "agent_type": agent_id
                }
            )
            
            # Update agent status
            update_agent_status(
                context=context,
                agent_id=agent_id,
                status=info["status"],
                current_load=info["current_load"],
                availability=info["availability"]
            )
        
        # Create workflow for document processing
        workflow_id = create_workflow(
            context=context,
            workflow_type="document_processing",
            owner_id=self.name,
            initial_state=WorkflowState.INITIALIZED,
            metadata={
                "document_path": document_path,
                "start_time": context["start_time"],
                "description": f"Document processing workflow for {document_path}"
            }
        )
        
        context["workflow_id"] = workflow_id
        
        # Initialize communication protocol (will be used in future implementation)
        # comms = CommunicationProtocol(context, self.name)
        
        # Initialize task framework
        tasks = TaskExecutionFramework(context, self.name)
        
        # Create main processing task
        process_task_id = tasks.create_task(
            task_description=f"Process document {document_path}",
            task_type="document_processing_workflow",
            workflow_id=workflow_id
        )
        
        try:
            # Update workflow state
            transition_workflow(
                context=context,
                workflow_id=workflow_id,
                from_state=WorkflowState.INITIALIZED,
                to_state=WorkflowState.IN_PROGRESS,
                agent_id=self.name,
                reason="Starting document processing workflow"
            )
            
            # Step 1: Document Processing using delegation
            self.logger.info("Delegating document processing task...")
            
            success, delegation_result = delegate_task(
                context=context,
                task_description=f"Process document and extract information from {document_path}",
                required_capabilities=["document_processing", "information_extraction"],
                available_agents=self.get_available_agents(context),
                priority=PriorityLevel.HIGH,
                metadata={"step": "document_processing", "workflow_id": workflow_id},
                strategy=DelegationStrategy.CAPABILITY_BASED
            )
            
            if not success:
                raise ValueError(f"Failed to delegate document processing task: {delegation_result.get('reason')}")
                
            # Wait for task completion (in a real implementation, this would be async)
            extraction_result = self.wait_for_task_completion(context, delegation_result.get("request_id"))
            self.logger.info(f"Document processing completed: {extraction_result}")
            
            # Update context with extraction result
            context["extraction_result"] = extraction_result
            context["document_type"] = extraction_result.get("document_type", "unknown")
            
            # Update task progress
            tasks.update_task_status(
                process_task_id, 
                "in_progress", 
                progress=0.5
            )
            
            # Step 2: Validation using direct invocation
            self.logger.info("Invoking validation agent...")
            
            # Create tool for validation agent
            validation_tool = create_enhanced_agent_tool(self.validation_agent)
            
            # Execute validation with context
            validation_context = context.copy()
            validation_result = validation_tool.execute(validation_context)
            self.logger.info(f"Validation completed: {validation_result}")
            
            # Update context with validation result
            context["validation_result"] = validation_result
            context["is_valid"] = validation_result.get("is_valid", False)
            
            # Update task progress
            tasks.update_task_status(
                process_task_id, 
                "in_progress", 
                progress=0.9
            )
            
            # Complete workflow
            context["end_time"] = datetime.now().isoformat()
            context["status"] = "completed"
            
            # Update workflow state
            transition_workflow(
                context=context,
                workflow_id=workflow_id,
                from_state=WorkflowState.IN_PROGRESS,
                to_state=WorkflowState.COMPLETED,
                agent_id=self.name,
                reason="Document processing workflow completed successfully",
                metadata={
                    "document_type": context.get("document_type", "unknown"),
                    "is_valid": context.get("is_valid", False),
                    "end_time": context["end_time"]
                }
            )
            
            # Update task status
            tasks.update_task_status(
                process_task_id, 
                "completed", 
                progress=1.0,
                result={
                    "extraction_result": extraction_result,
                    "validation_result": validation_result
                }
            )
            
            self.logger.info(f"Document processing completed successfully for: {document_path}")
            
            # Compile final result
            result = {
                "status": "success",
                "document_path": document_path,
                "document_type": context["document_type"],
                "is_valid": context["is_valid"],
                "workflow_id": workflow_id,
                "extraction_result": extraction_result,
                "validation_result": validation_result,
                "processing_time": (
                    datetime.fromisoformat(context["end_time"]) - 
                    datetime.fromisoformat(context["start_time"])
                ).total_seconds()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            
            # Update workflow state to failed
            try:
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=None,  # Skip validation as we don't know current state
                    to_state=WorkflowState.FAILED,
                    agent_id=self.name,
                    reason=f"Document processing workflow failed: {str(e)}"
                )
                
                # Update task status
                tasks.update_task_status(
                    process_task_id, 
                    "failed", 
                    error=str(e)
                )
            except Exception as inner_e:
                self.logger.error(f"Error updating workflow state: {inner_e}")
            
            # Return error result
            return {
                "status": "error",
                "document_path": document_path,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    def get_available_agents(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get available agents from the registry."""
        from utils.session_state import get_or_create_session_state
        session_state = get_or_create_session_state(context)
        return session_state.get("agent_registry", {})
    
    def wait_for_task_completion(self, context: Dict[str, Any], delegation_request_id: str) -> Dict[str, Any]:
        """Wait for a delegated task to complete."""
        # Simulate a task completion in this example
        # In a real implementation, this would use proper async mechanisms
        
        # Get the delegation request
        from utils.session_state import get_or_create_session_state
        session_state = get_or_create_session_state(context)
        delegation_queue = session_state.get("delegation_queue", [])
        
        delegation_request = None
        for req in delegation_queue:
            if req.get("request_id") == delegation_request_id:
                delegation_request = req
                break
        
        if not delegation_request:
            return {"error": "Task not found"}
        
        # Get the delegated agent
        agent_id = delegation_request.get("delegatee_id")
        if not agent_id or agent_id not in self.agent_registry:
            return {"error": "Agent not found"}
            
        # Invoke the agent directly
        agent = self.agent_registry[agent_id]["agent"]
        
        # Create an execution context
        exec_context = context.copy()
        exec_context["request_message_id"] = delegation_request_id
        
        # Invoke the agent
        result = agent.process(exec_context)
        
        # Update the delegation result
        delegation_request["status"] = "completed"
        delegation_request["completion_time"] = datetime.now().isoformat()
        delegation_request["result"] = result
        
        # Save back to session state
        for i, req in enumerate(delegation_queue):
            if req.get("request_id") == delegation_request_id:
                delegation_queue[i] = delegation_request
                break
                
        session_state.set("delegation_queue", delegation_queue)
        
        return result

def main():
    """Run the example."""
    logger.info("Starting Agent Communication Example...")
    
    orchestrator = MasterOrchestrator()
    
    # Process a sample document
    result = orchestrator.process_document("/path/to/sample_invoice.pdf")
    
    logger.info("Processing completed:")
    logger.info(f"  Status: {result.get('status')}")
    logger.info(f"  Document type: {result.get('document_type')}")
    logger.info(f"  Valid: {result.get('is_valid')}")
    logger.info(f"  Processing time: {result.get('processing_time', 0):.2f} seconds")
    
    if result.get("status") != "success":
        logger.error(f"  Error: {result.get('error')}")

if __name__ == "__main__":
    main()
