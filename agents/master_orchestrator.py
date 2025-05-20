"""
Master orchestrator agent for the FinFlow system.

This agent orchestrates the entire document processing workflow by delegating tasks
to specialized worker agents, tracking workflow state, and managing communication
between agents.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from google.adk.tools import ToolContext # type: ignore

from agents.base_agent import BaseAgent

# Forward references for type hints
DocumentProcessorAgent = Any
ValidationAgent = Any
StorageAgent = Any
AnalyticsAgent = Any

from utils.prompt_templates import get_agent_prompt
from utils.logging_config import TraceContext, log_agent_call
from utils.session_state import get_or_create_session_state
from utils.agent_communication import (
    create_enhanced_agent_tool, AgentInvokeTool, 
    agent_task_decorator, delegate_task, register_agent_capabilities,
    create_workflow, transition_workflow, get_workflow_data, get_workflow_state,
    WorkflowState, DelegationStrategy,
    send_message, get_messages, create_response,
    update_agent_status, track_agent_metrics
)
from utils.agent_protocol import (
    MessageType, PriorityLevel, StatusCode,
    create_protocol_message, create_request, create_response, 
    create_error_response, create_notification
)

class MasterOrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that coordinates workflow execution and task delegation.
    
    This agent uses the enhanced agent communication framework to delegate tasks to specialized
    worker agents, track workflow state, and manage communication between agents.
    """
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessorAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
        storage_agent: Optional[StorageAgent] = None,
        analytics_agent: Optional[AnalyticsAgent] = None,
    ):
        """Initialize the master orchestrator agent.
        
        Args:
            document_processor: Document processor agent instance.
            validation_agent: Validation agent instance.
            storage_agent: Storage agent instance.
            analytics_agent: Analytics agent instance.
        """
        # Get the instruction prompt from template
        instruction = get_agent_prompt("master_orchestrator")
        
        super().__init__(
            name="FinFlow_MasterOrchestrator",
            model="gemini-2.0-pro",
            description="Coordinates workflow execution and delegates tasks to worker agents",
            instruction=instruction,
            temperature=0.2,
        )
        
        # Initialize worker agents dictionary with capabilities
        self.worker_agents = {
            "document_processor": document_processor,
            "validation_agent": validation_agent,
            "storage_agent": storage_agent,
            "analytics_agent": analytics_agent,
        }
        
        # Define agent capabilities
        self.agent_capabilities = {
            "document_processor": ["document_processing", "information_extraction", "document_classification"],
            "validation_agent": ["validation", "rule_checking", "compliance_verification"],
            "storage_agent": ["data_storage", "persistence", "retrieval"],
            "analytics_agent": ["data_analysis", "reporting", "visualization"],
        }
        
        # Set up logger
        self.logger = logging.getLogger(f"finflow.agents.{self.name}")
    
    def register_worker_agents(self) -> None:
        """
        Register available worker agents as tools.
        This should be called after all agent instances are created.
        """
        for name, agent in self.worker_agents.items():
            if agent is not None:
                self.logger.info(f"Registering worker agent as tool: {name}")
                
                # Register the agent using the enhanced agent tool
                self.add_tool(create_enhanced_agent_tool(
                    agent, 
                    name=f"{name}_invoke",
                    description=f"Invoke the {name} agent to perform {name.replace('_', ' ')} operations"
                ))
    
    def initialize_delegation_framework(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the delegation framework by registering agent capabilities.
        
        Args:
            context: Agent context dictionary
            
        Returns:
            Dict[str, Any]: Updated context with initialized delegation framework
        """
        # Get or create session state
        session_state = get_or_create_session_state(context)
        
        # Register agent capabilities in registry
        for name, agent in self.worker_agents.items():
            if agent is not None:
                capabilities = self.agent_capabilities.get(name, ["general_purpose"])
                
                # Register in the agent registry (will be stored in session state)
                register_agent_capabilities(
                    context=context,
                    agent_id=name,
                    capabilities=capabilities,
                    metadata={
                        "description": getattr(agent, "description", ""),
                        "model": getattr(agent, "model", "unknown"),
                        "agent_type": name
                    }
                )
                
                # Update agent status
                update_agent_status(
                    context=context,
                    agent_id=name,
                    status="active",
                    current_load=0.0,
                    availability=1.0
                )
        
        return context
    
    @agent_task_decorator(task_name="process_document", required_capabilities=["orchestration"], track_metrics=True)
    def process_document(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a document through the entire workflow using the enhanced agent communication framework.
        
        Args:
            context: Processing context with document information
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the processing results
        """
        # Extract document path from context
        document_path = context.get("document_path")
        if not document_path:
            return self.handle_error(ValueError("Document path not provided in context"), context)
        
        # Initialize the delegation framework
        context = self.initialize_delegation_framework(context)
        
        # Create a workflow for this document processing
        workflow_id = create_workflow(
            context=context,
            workflow_type="document_processing",
            owner_id=self.name,
            initial_state=WorkflowState.INITIALIZED,
            metadata={
                "document_path": document_path,
                "start_time": datetime.now().isoformat(),
                "description": f"Document processing workflow for {document_path}"
            }
        )
        
        # Store workflow ID in context
        context["workflow_id"] = workflow_id
        
        # Create trace context for this process
        with TraceContext() as trace:
            # Log the start of processing
            self.logger.info(f"Starting document processing for: {document_path}")
            log_agent_call(self.logger, self.name, context)
            
            # Track processing in context
            context["status"] = "started"
            context["trace_id"] = trace.trace_id
            context["start_time"] = datetime.now().isoformat()
            context["steps_completed"] = []
            context["current_step"] = "initialization"
            
            try:
                # Update workflow state to in progress
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=WorkflowState.INITIALIZED,
                    to_state=WorkflowState.IN_PROGRESS,
                    agent_id=self.name,
                    reason="Starting document processing workflow"
                )
                
                # Execute the document processing workflow with state tracking
                context = self.execute_workflow(context, tool_context)
                
                # Record completion
                context["end_time"] = datetime.now().isoformat()
                context["status"] = "completed"
                
                # Update workflow state to completed
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
                        "end_time": context["end_time"],
                        "steps_completed": context["steps_completed"]
                    }
                )
                
                # Log completion
                self.logger.info(f"Document processing completed successfully for: {document_path}")
                self.log_activity("document_processing_complete", {"document_path": document_path}, context)
                
            except Exception as e:
                # Handle any errors
                context = self.handle_error(e, context)
                context["end_time"] = datetime.now().isoformat()
                
                # Update workflow state to failed
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=None,  # Skip validation as we don't know current state
                    to_state=WorkflowState.FAILED,
                    agent_id=self.name,
                    reason=f"Document processing workflow failed: {str(e)}",
                    metadata={
                        "error": str(e),
                        "end_time": context["end_time"],
                        "steps_completed": context["steps_completed"],
                        "failed_step": context.get("current_step", "unknown")
                    }
                )
                
                # Log error
                self.logger.error(f"Document processing failed for: {document_path}")
                self.log_activity("document_processing_failed", {"document_path": document_path, "error": str(e)}, context)
        
        return context
    
    def execute_workflow(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Execute the document processing workflow steps using agent delegation.
        
        Args:
            context: Processing context
            tool_context: Tool context provided by ADK
            
        Returns:
            Updated context with workflow results
        """
        document_path = context["document_path"]
        workflow_id = context.get("workflow_id")
        
        # Step 1: Document Processing through delegation
        context["current_step"] = "document_processing"
        self.logger.info(f"Starting document extraction for: {document_path}")
        
        # Create task for document processing
        document_processing_task = {
            "task_description": f"Process document and extract information from {document_path}",
            "required_capabilities": ["document_processing", "information_extraction"],
            "priority": PriorityLevel.HIGH
        }
        
        # Delegate to document processor
        success, delegation_result = delegate_task(
            context=context,
            task_description=document_processing_task["task_description"],
            required_capabilities=document_processing_task["required_capabilities"],
            available_agents=get_or_create_session_state(context).get("agent_registry", {}),
            priority=document_processing_task["priority"],
            metadata={"step": "document_processing", "workflow_id": workflow_id},
            strategy=DelegationStrategy.CAPABILITY_BASED
        )
        
        if not success:
            raise ValueError(f"Failed to delegate document processing task: {delegation_result.get('reason')}")
        
        # Get task result from document processor
        extraction_result = self.wait_for_task_completion(context, delegation_result.get("request_id"))
        context["extraction_result"] = extraction_result
        context["document_type"] = extraction_result.get("document_type", "unknown")
        context["steps_completed"].append("document_processing")
        
        # Step 2: Rule Retrieval (local step)
        context["current_step"] = "rule_retrieval"
        self.logger.info(f"Retrieving rules for document type: {context['document_type']}")
        
        rules = self.retrieve_rules_step(context["document_type"], context, tool_context)
        context["applicable_rules"] = rules
        context["steps_completed"].append("rule_retrieval")
        
        # Step 3: Validation through delegation
        context["current_step"] = "validation"
        self.logger.info(f"Validating document against rules")
        
        # Create task for validation
        validation_task = {
            "task_description": f"Validate extracted document data against applicable rules",
            "required_capabilities": ["validation", "rule_checking"],
            "priority": PriorityLevel.HIGH
        }
        
        # Update context for validation task
        validation_context = context.copy()
        validation_context.update({
            "extraction_result": context["extraction_result"],
            "applicable_rules": context["applicable_rules"]
        })
        
        # Delegate to validation agent
        success, delegation_result = delegate_task(
            context=validation_context,
            task_description=validation_task["task_description"],
            required_capabilities=validation_task["required_capabilities"],
            available_agents=get_or_create_session_state(context).get("agent_registry", {}),
            priority=validation_task["priority"],
            metadata={"step": "validation", "workflow_id": workflow_id},
            strategy=DelegationStrategy.CAPABILITY_BASED
        )
        
        if not success:
            raise ValueError(f"Failed to delegate validation task: {delegation_result.get('reason')}")
        
        # Get validation result
        validation_result = self.wait_for_task_completion(context, delegation_result.get("request_id"))
        context["validation_result"] = validation_result
        context["is_valid"] = validation_result.get("is_valid", False)
        context["steps_completed"].append("validation")
        
        # Only proceed with storage and analytics if document is valid
        if context["is_valid"]:
            # Step 4: Storage through delegation
            context["current_step"] = "storage"
            self.logger.info(f"Storing validated document data")
            
            # Create task for storage
            storage_task = {
                "task_description": f"Store validated document data in the database",
                "required_capabilities": ["data_storage", "persistence"],
                "priority": PriorityLevel.NORMAL
            }
            
            # Update context for storage task
            storage_context = context.copy()
            storage_context.update({
                "document_data": context["extraction_result"],
                "validation_result": context["validation_result"]
            })
            
            # Delegate to storage agent
            success, delegation_result = delegate_task(
                context=storage_context,
                task_description=storage_task["task_description"],
                required_capabilities=storage_task["required_capabilities"],
                available_agents=get_or_create_session_state(context).get("agent_registry", {}),
                priority=storage_task["priority"],
                metadata={"step": "storage", "workflow_id": workflow_id},
                strategy=DelegationStrategy.CAPABILITY_BASED
            )
            
            if not success:
                raise ValueError(f"Failed to delegate storage task: {delegation_result.get('reason')}")
            
            # Get storage result
            storage_result = self.wait_for_task_completion(context, delegation_result.get("request_id"))
            context["storage_result"] = storage_result
            context["document_id"] = storage_result.get("document_id")
            context["steps_completed"].append("storage")
            
            # Step 5: Analytics through delegation
            context["current_step"] = "analytics"
            self.logger.info(f"Generating analytics for document")
            
            # Create task for analytics
            analytics_task = {
                "task_description": f"Generate analytics for processed document",
                "required_capabilities": ["data_analysis", "reporting"],
                "priority": PriorityLevel.LOW
            }
            
            # Update context for analytics task
            analytics_context = context.copy()
            analytics_context.update({
                "document_data": context["extraction_result"],
                "document_id": context["document_id"]
            })
            
            # Delegate to analytics agent
            success, delegation_result = delegate_task(
                context=analytics_context,
                task_description=analytics_task["task_description"],
                required_capabilities=analytics_task["required_capabilities"],
                available_agents=get_or_create_session_state(context).get("agent_registry", {}),
                priority=analytics_task["priority"],
                metadata={"step": "analytics", "workflow_id": workflow_id},
                strategy=DelegationStrategy.CAPABILITY_BASED
            )
            
            if not success:
                self.logger.warning(f"Failed to delegate analytics task: {delegation_result.get('reason')}")
                # Continue workflow even if analytics fails
            else:
                # Get analytics result
                analytics_result = self.wait_for_task_completion(context, delegation_result.get("request_id"))
                context["analytics_result"] = analytics_result
                context["steps_completed"].append("analytics")
        
        return context
        
    def wait_for_task_completion(self, context: Dict[str, Any], delegation_request_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for a delegated task to complete and return the result.
        
        Args:
            context: Agent context
            delegation_request_id: Delegation request ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict[str, Any]: Task result
        """
        # This is a simplified implementation - in a real system you'd implement
        # an asynchronous waiting mechanism with timeout handling
        
        # In this version, we'll just retrieve the task result directly from the session state
        # Normally, you'd poll for completion or use a notification mechanism
        
        session_state = get_or_create_session_state(context)
        delegation_queue = session_state.get("delegation_queue", [])
        
        # Find the delegation request
        for request in delegation_queue:
            if request.get("request_id") == delegation_request_id:
                if request.get("status") == "completed":
                    return request.get("result", {})
        
        # If we reach here, the task hasn't completed or couldn't be found
        # For this implementation, we'll return a placeholder result
        # In a real system, you'd implement proper timeout handling
        return {"error": "Task completion timeout or task not found"}
            
    def process_document_step(self, document_path: str, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Extract information from a document using Document Processor Agent."""
        # In a real implementation, this would invoke the Document Processor Agent
        self.logger.info(f"Processing document: {document_path}")
        
        # For now, return a mock result
        return {
            "document_type": "invoice",
            "confidence": 0.95,
            "entities": {
                "invoice_number": "INV-12345",
                "date": "2025-05-15",
                "total_amount": 1000.0,
                "vendor": "Acme Corp"
            },
            "status": "success"
        }
    
    def retrieve_rules_step(self, document_type: str, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Retrieve applicable rules for a document type."""
        # In a real implementation, this would invoke the Rule Retrieval Agent
        self.logger.info(f"Retrieving rules for document type: {document_type}")
        
        # For now, return mock rules
        return {
            "document_type": document_type,
            "rules": [
                {"id": "rule1", "description": "Invoice must have an invoice number", "severity": "critical"},
                {"id": "rule2", "description": "Invoice must have a date", "severity": "critical"},
                {"id": "rule3", "description": "Invoice must have a total amount", "severity": "critical"},
                {"id": "rule4", "description": "Invoice must have a vendor name", "severity": "warning"}
            ]
        }
    
    def validate_document_step(self, extraction_result: Dict[str, Any], rules: Dict[str, Any], context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Validate document against rules."""
        # In a real implementation, this would invoke the Validation Agent
        self.logger.info("Validating document")
        
        # Mock validation result
        is_valid = all([
            "invoice_number" in extraction_result["entities"],
            "date" in extraction_result["entities"],
            "total_amount" in extraction_result["entities"],
            "vendor" in extraction_result["entities"]
        ])
        
        return {
            "is_valid": is_valid,
            "validation_time": datetime.now().isoformat(),
            "issues": [] if is_valid else [{"rule_id": "rule3", "description": "Missing total amount"}]
        }
    
    def store_document_step(self, document_data: Dict[str, Any], context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Store document data."""
        # In a real implementation, this would invoke the Storage Agent
        self.logger.info("Storing document data")
        
        # Mock storage result
        document_id = f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "document_id": document_id,
            "storage_time": datetime.now().isoformat(),
            "status": "success"
        }
    
    def analyze_document_step(self, document_data: Dict[str, Any], context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Generate analytics for a document."""
        # In a real implementation, this would invoke the Analytics Agent
        self.logger.info("Analyzing document")
        
        # Mock analytics result
        return {
            "analysis_time": datetime.now().isoformat(),
            "insights": [
                {"type": "spend_trend", "description": "Spending with this vendor is 15% higher than last month"},
                {"type": "category_analysis", "description": "This invoice falls under 'Office Supplies' category"}
            ],
            "status": "success"
        }

    def log_activity(self, activity_type: str, activity_data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log agent activity for tracking and debugging.
        
        Args:
            activity_type: Type of activity
            activity_data: Activity data
            context: Agent context
        """
        # Create activity log entry
        activity = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.name,
            "activity_type": activity_type,
            "data": activity_data,
            "trace_id": context.get("trace_id", "unknown")
        }
        
        # Log to console
        self.logger.info(f"Activity: {activity_type} - {json.dumps(activity_data)}")
        
        # Store in session state
        session_state = get_or_create_session_state(context)
        activities = session_state.get("activities", [])
        activities.append(activity)
        session_state.set("activities", activities)
        
        # Update context
        context["session_state"] = session_state.to_dict()
