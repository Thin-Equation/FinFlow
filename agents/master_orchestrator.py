"""
Master orchestrator agent for the FinFlow system.

This agent orchestrates the entire document processing workflow by delegating tasks
to specialized worker agents, tracking workflow state, and managing communication
between agents.
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast, Tuple, Callable

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
    update_agent_status, track_agent_metrics,
    CommunicationProtocol, TaskExecutionFramework,
    apply_delegation_strategy
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
        
        # Initialize communication framework components
        self.comms_protocol = None  # Will be initialized during processing
        self.task_framework = None  # Will be initialized during processing
    
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
        
        # Initialize the enhanced delegation framework
        context = self.initialize_delegation_framework(context)
        
        # Initialize communication protocol
        self.comms_protocol = CommunicationProtocol(context, self.name)
        
        # Initialize task execution framework
        self.task_framework = TaskExecutionFramework(context, self.name)
        
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
        
        # Create main processing task
        main_task_id = self.task_framework.create_task(
            task_description=f"Process document {document_path}",
            task_type="document_processing_workflow",
            workflow_id=workflow_id,
            priority=PriorityLevel.HIGH
        )
        context["main_task_id"] = main_task_id
        
        # Create trace context for this process
        with TraceContext() as trace:
            # Log the start of processing
            self.logger.info(f"Starting document processing for: {document_path}")
            log_agent_call(self.logger, self.name, context)
            
            # Track processing in context
            context["status"] = "started"
            context["trace_id"] = trace.trace_id
            
            try:
                # Transition workflow to IN_PROGRESS state
                success = transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=WorkflowState.INITIALIZED,
                    to_state=WorkflowState.IN_PROGRESS,
                    agent_id=self.name,
                    reason="Starting document processing workflow"
                )
                
                if not success:
                    raise ValueError(f"Failed to transition workflow to IN_PROGRESS state")
                
                # Execute the document processing workflow using enhanced task framework
                result = self._execute_document_processing_workflow(context, main_task_id)
                
                # Update main task status
                self.task_framework.update_task_status(
                    task_id=main_task_id,
                    status="completed",
                    progress=1.0,
                    result=result
                )
                
                # Transition workflow to COMPLETED state
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=WorkflowState.IN_PROGRESS,
                    to_state=WorkflowState.COMPLETED,
                    agent_id=self.name,
                    reason="Document processing completed successfully",
                    metadata={
                        "document_type": result.get("document_type", "unknown"),
                        "is_valid": result.get("is_valid", False),
                        "completion_time": datetime.now().isoformat()
                    }
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                
                # Update task status to failed
                if self.task_framework and main_task_id:
                    self.task_framework.update_task_status(
                        task_id=main_task_id,
                        status="failed",
                        error=str(e)
                    )
                
                # Transition workflow to FAILED state
                if workflow_id:
                    transition_workflow(
                        context=context,
                        workflow_id=workflow_id,
                        from_state=None,  # Skip validation as we don't know current state
                        to_state=WorkflowState.FAILED,
                        agent_id=self.name,
                        reason=f"Document processing failed: {str(e)}"
                    )
                
                # Return error result
                return self.handle_error(e, context)
    
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
    
    def _execute_document_processing_workflow(
        self, 
        context: Dict[str, Any],
        main_task_id: str
    ) -> Dict[str, Any]:
        """
        Execute the document processing workflow using the enhanced task framework.
        
        Args:
            context: Processing context
            main_task_id: ID of the main processing task
            
        Returns:
            Dict containing the processing results
        """
        document_path = context.get("document_path")
        workflow_id = context.get("workflow_id")
        
        # Create subtasks for the main workflow phases
        subtask_definitions = [
            {
                "description": "Document extraction and classification",
                "type": "document_extraction",
                "metadata": {"document_path": document_path},
                "executor": lambda task: self._execute_document_extraction(context, task)
            },
            {
                "description": "Document validation",
                "type": "document_validation",
                "metadata": {"document_path": document_path},
                "executor": lambda task: self._execute_document_validation(context, task)
            },
            {
                "description": "Document storage",
                "type": "document_storage",
                "metadata": {"document_path": document_path},
                "executor": lambda task: self._execute_document_storage(context, task)
            },
            {
                "description": "Document analytics",
                "type": "document_analytics",
                "metadata": {"document_path": document_path},
                "executor": lambda task: self._execute_document_analytics(context, task)
            }
        ]
        
        # Execute all workflow subtasks
        subtask_results = self.task_framework.create_and_execute_subtasks(
            parent_task_id=main_task_id,
            subtask_definitions=subtask_definitions
        )
        
        # Aggregate results from all subtasks
        aggregated_result = {
            "document_path": document_path,
            "workflow_id": workflow_id,
            "processing_time": datetime.now().isoformat(),
            "status": "success"
        }
        
        # Add results from individual stages
        for task_id, result in subtask_results.items():
            task_info = self.task_framework.get_task(task_id)
            task_type = task_info.get("type") if task_info else "unknown"
            
            # Extract the stage name from the task type
            stage = task_type.replace("document_", "") if task_type.startswith("document_") else task_type
            aggregated_result[stage] = result
        
        # Extract key information from results
        if "extraction" in aggregated_result:
            extraction_data = aggregated_result["extraction"]
            if isinstance(extraction_data, dict):
                aggregated_result["document_type"] = extraction_data.get("document_type", "unknown")
                aggregated_result["entities"] = extraction_data.get("entities", {})
                aggregated_result["extraction_confidence"] = extraction_data.get("confidence", 0.0)
        
        if "validation" in aggregated_result:
            validation_data = aggregated_result["validation"]
            if isinstance(validation_data, dict):
                aggregated_result["is_valid"] = validation_data.get("is_valid", False)
                aggregated_result["validation_results"] = validation_data.get("field_results", {})
        
        self.logger.info(f"Document processing workflow completed for: {document_path}")
        
        return aggregated_result
    
    def _execute_document_extraction(self, context: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the document extraction subtask using the document processor agent.
        
        Args:
            context: Processing context
            task: Task information
            
        Returns:
            Dict containing the extraction results
        """
        document_path = task.get("metadata", {}).get("document_path")
        if not document_path:
            raise ValueError("Document path not provided in task metadata")
        
        self.logger.info(f"Starting document extraction for: {document_path}")
        
        # Check if document processor agent is available
        document_processor = self.worker_agents.get("document_processor")
        if not document_processor:
            raise ValueError("Document processor agent not available")
        
        # Create subtask for delegation vs direct invocation
        extraction_task_id = self.task_framework.create_task(
            task_description=f"Extract data from document {document_path}",
            task_type="extraction_delegation",
            parent_task_id=task["task_id"],
            metadata={"document_path": document_path}
        )
        
        # Update context with task information
        task_context = context.copy()
        task_context["extraction_task_id"] = extraction_task_id
        
        # Use enhanced delegation pattern first
        try:
            self.logger.info("Attempting extraction via delegation pattern")
            
            # Define required capabilities
            required_capabilities = ["document_processing", "information_extraction"]
            
            # Get available agents
            available_agents = self.get_available_agents(task_context)
            
            # Delegate the extraction task
            success, delegation_result = delegate_task(
                context=task_context,
                task_description=f"Extract information from document {document_path}",
                required_capabilities=required_capabilities,
                available_agents=available_agents,
                priority=PriorityLevel.HIGH,
                metadata={"document_path": document_path, "task_id": extraction_task_id},
                strategy=DelegationStrategy.ADAPTIVE
            )
            
            if success:
                delegatee_id = delegation_result.get("delegatee_id")
                self.logger.info(f"Successfully delegated extraction to agent: {delegatee_id}")
                
                # Wait for the result
                result = self._wait_for_delegated_task_completion(task_context, delegation_result)
                
                # Update extraction task status
                self.task_framework.update_task_status(
                    task_id=extraction_task_id,
                    status="completed",
                    progress=1.0,
                    result=result
                )
                
                return result
            else:
                self.logger.warning(f"Delegation failed: {delegation_result.get('reason')}")
                # Fall back to direct invocation
        except Exception as e:
            self.logger.warning(f"Error during delegation: {e}")
            # Fall back to direct invocation
        
        # Fall back to direct invocation if delegation fails
        self.logger.info("Falling back to direct invocation for extraction")
        
        # Create enhanced tool for document processor
        document_processor_tool = create_enhanced_agent_tool(document_processor)
        
        # Create invocation context
        invocation_context = {
            "document_path": document_path,
            "workflow_id": context.get("workflow_id"),
            "task_id": extraction_task_id,
            "session_state": context.get("session_state", {})
        }
        
        # Invoke the document processor agent
        start_time = datetime.now()
        result = document_processor_tool.execute(invocation_context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Track agent metrics
        track_agent_metrics(
            context=context,
            agent_id="document_processor",
            execution_time=execution_time,
            success=True,
            metadata={"task_type": "document_extraction"}
        )
        
        # Update extraction task status
        self.task_framework.update_task_status(
            task_id=extraction_task_id,
            status="completed",
            progress=1.0,
            result=result
        )
        
        return result
    
    def _execute_document_validation(self, context: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the document validation subtask using the validation agent.
        
        Args:
            context: Processing context
            task: Task information
            
        Returns:
            Dict containing the validation results
        """
        document_path = task.get("metadata", {}).get("document_path")
        if not document_path:
            raise ValueError("Document path not provided in task metadata")
        
        # Get extraction results from context
        extraction_results = context.get("extraction", {})
        if not extraction_results:
            self.logger.warning("No extraction results found in context for validation")
        
        self.logger.info(f"Starting document validation for: {document_path}")
        
        # Check if validation agent is available
        validation_agent = self.worker_agents.get("validation_agent")
        if not validation_agent:
            raise ValueError("Validation agent not available")
        
        # Create validation task
        validation_task_id = self.task_framework.create_task(
            task_description=f"Validate document data for {document_path}",
            task_type="validation",
            parent_task_id=task["task_id"],
            metadata={"document_path": document_path}
        )
        
        # Create enhanced tool for validation agent
        validation_tool = create_enhanced_agent_tool(validation_agent)
        
        # Create invocation context
        invocation_context = {
            "document_path": document_path,
            "workflow_id": context.get("workflow_id"),
            "task_id": validation_task_id,
            "extraction_result": extraction_results,
            "session_state": context.get("session_state", {})
        }
        
        # Invoke the validation agent
        start_time = datetime.now()
        result = validation_tool.execute(invocation_context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Track agent metrics
        track_agent_metrics(
            context=context,
            agent_id="validation_agent",
            execution_time=execution_time,
            success=True,
            metadata={"task_type": "document_validation"}
        )
        
        # Update validation task status
        self.task_framework.update_task_status(
            task_id=validation_task_id,
            status="completed",
            progress=1.0,
            result=result
        )
        
        # Store validation results in context for later stages
        context["validation"] = result
        
        return result
    
    def _execute_document_storage(self, context: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the document storage subtask using the storage agent.
        
        Args:
            context: Processing context
            task: Task information
            
        Returns:
            Dict containing the storage results
        """
        document_path = task.get("metadata", {}).get("document_path")
        if not document_path:
            raise ValueError("Document path not provided in task metadata")
        
        # Get extraction and validation results
        extraction_results = context.get("extraction", {})
        validation_results = context.get("validation", {})
        
        self.logger.info(f"Starting document storage for: {document_path}")
        
        # Check if storage agent is available
        storage_agent = self.worker_agents.get("storage_agent")
        if not storage_agent:
            raise ValueError("Storage agent not available")
        
        # Create storage task
        storage_task_id = self.task_framework.create_task(
            task_description=f"Store document data for {document_path}",
            task_type="storage",
            parent_task_id=task["task_id"],
            metadata={"document_path": document_path}
        )
        
        # Create enhanced tool for storage agent
        storage_tool = create_enhanced_agent_tool(storage_agent)
        
        # Create invocation context
        invocation_context = {
            "document_path": document_path,
            "workflow_id": context.get("workflow_id"),
            "task_id": storage_task_id,
            "extraction_result": extraction_results,
            "validation_result": validation_results,
            "is_valid": validation_results.get("is_valid", False),
            "session_state": context.get("session_state", {})
        }
        
        # Invoke the storage agent
        start_time = datetime.now()
        result = storage_tool.execute(invocation_context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Track agent metrics
        track_agent_metrics(
            context=context,
            agent_id="storage_agent",
            execution_time=execution_time,
            success=True,
            metadata={"task_type": "document_storage"}
        )
        
        # Update storage task status
        self.task_framework.update_task_status(
            task_id=storage_task_id,
            status="completed",
            progress=1.0,
            result=result
        )
        
        # Store storage results in context
        context["storage"] = result
        
        return result
    
    def _execute_document_analytics(self, context: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the document analytics subtask using the analytics agent.
        
        Args:
            context: Processing context
            task: Task information
            
        Returns:
            Dict containing the analytics results
        """
        document_path = task.get("metadata", {}).get("document_path")
        if not document_path:
            raise ValueError("Document path not provided in task metadata")
        
        # Get results from previous stages
        extraction_results = context.get("extraction", {})
        validation_results = context.get("validation", {})
        storage_results = context.get("storage", {})
        
        self.logger.info(f"Starting document analytics for: {document_path}")
        
        # Check if analytics agent is available
        analytics_agent = self.worker_agents.get("analytics_agent")
        if not analytics_agent:
            self.logger.warning("Analytics agent not available, skipping analytics")
            return {"status": "skipped", "reason": "Analytics agent not available"}
        
        # Create analytics task
        analytics_task_id = self.task_framework.create_task(
            task_description=f"Analyze document data for {document_path}",
            task_type="analytics",
            parent_task_id=task["task_id"],
            metadata={"document_path": document_path}
        )
        
        # Create enhanced tool for analytics agent
        analytics_tool = create_enhanced_agent_tool(analytics_agent)
        
        # Create invocation context
        invocation_context = {
            "document_path": document_path,
            "workflow_id": context.get("workflow_id"),
            "task_id": analytics_task_id,
            "extraction_result": extraction_results,
            "validation_result": validation_results,
            "storage_result": storage_results,
            "is_valid": validation_results.get("is_valid", False),
            "session_state": context.get("session_state", {})
        }
        
        # Invoke the analytics agent
        start_time = datetime.now()
        result = analytics_tool.execute(invocation_context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Track agent metrics
        track_agent_metrics(
            context=context,
            agent_id="analytics_agent",
            execution_time=execution_time,
            success=True,
            metadata={"task_type": "document_analytics"}
        )
        
        # Update analytics task status
        self.task_framework.update_task_status(
            task_id=analytics_task_id,
            status="completed",
            progress=1.0,
            result=result
        )
        
        return result
    
    def _wait_for_delegated_task_completion(
        self, 
        context: Dict[str, Any], 
        delegation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Wait for a delegated task to complete.
        
        Args:
            context: Processing context
            delegation_result: Result from the delegation process
            
        Returns:
            Dict containing the task execution result
        """
        # Get delegation request ID
        delegation_request_id = delegation_result.get("request_id")
        if not delegation_request_id:
            raise ValueError("No delegation request ID provided")
        
        # Get the delegation request
        session_state = get_or_create_session_state(context)
        delegation_queue = session_state.get("delegation_queue", [])
        
        # Initialize message receiver
        comms = self.comms_protocol or CommunicationProtocol(context, self.name)
        
        # Wait for response message with a timeout (in a real implementation, this would be async)
        max_wait_time = 30  # seconds
        wait_interval = 0.5  # seconds
        total_waited = 0
        
        while total_waited < max_wait_time:
            # Check for unread messages
            unread_messages = comms.get_unread_messages()
            
            # Look for response messages related to our delegation
            for message in unread_messages:
                content = message.get("content", {})
                if (content.get("action") == "delegated_task_completed" and 
                    content.get("data", {}).get("delegation_request_id") == delegation_request_id):
                    
                    # Mark message as read
                    comms.mark_as_read(message.get("message_id", ""))
                    
                    # Return the result
                    return content.get("data", {}).get("result", {})
            
            # Check delegation queue directly for completion
            for req in delegation_queue:
                if req.get("request_id") == delegation_request_id and req.get("status") == "completed":
                    return req.get("result", {})
            
            # Wait before checking again
            time.sleep(wait_interval)
            total_waited += wait_interval
        
        # If we get here, we timed out
        raise TimeoutError(f"Timed out waiting for delegated task completion: {delegation_request_id}")
        
    def get_available_agents(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get available agents from the registry.
        
        Args:
            context: Processing context
            
        Returns:
            Dict of available agents with their capabilities
        """
        session_state = get_or_create_session_state(context)
        return session_state.get("agent_registry", {})
