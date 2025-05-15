"""
Master orchestrator agent for the FinFlow system.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext # type: ignore

from agents.base_agent import BaseAgent

# Forward references for type hints
DocumentProcessorAgent = Any
RuleRetrievalAgent = Any
ValidationAgent = Any
StorageAgent = Any
AnalyticsAgent = Any

from utils.prompt_templates import get_agent_prompt
from utils.logging_config import TraceContext, log_agent_call

class MasterOrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that coordinates workflow execution and task delegation.
    """
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessorAgent] = None,
        rule_retrieval: Optional[RuleRetrievalAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
        storage_agent: Optional[StorageAgent] = None,
        analytics_agent: Optional[AnalyticsAgent] = None,
    ):
        """Initialize the master orchestrator agent.
        
        Args:
            document_processor: Document processor agent instance.
            rule_retrieval: Rule retrieval agent instance.
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
        
        # Initialize worker agents dictionary
        self.worker_agents = {
            "document_processor": document_processor,
            "rule_retrieval": rule_retrieval,
            "validation_agent": validation_agent,
            "storage_agent": storage_agent,
            "analytics_agent": analytics_agent,
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
                # Register the agent as a tool using our utility function
                from utils.agent_communication import create_agent_tool
                self.add_tool(create_agent_tool(agent))
    
    def process_document(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a document through the entire workflow.
        
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
                # Execute the document processing workflow
                context = self.execute_workflow(context, tool_context)
                
                # Record completion
                context["end_time"] = datetime.now().isoformat()
                context["status"] = "completed"
                
                # Log completion
                self.logger.info(f"Document processing completed successfully for: {document_path}")
                self.log_activity("document_processing_complete", {"document_path": document_path}, context)
                
            except Exception as e:
                # Handle any errors
                context = self.handle_error(e, context)
                context["end_time"] = datetime.now().isoformat()
                
                # Log error
                self.logger.error(f"Document processing failed for: {document_path}")
                self.log_activity("document_processing_failed", {"document_path": document_path, "error": str(e)}, context)
        
        return context
    
    def execute_workflow(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Execute the document processing workflow steps.
        
        Args:
            context: Processing context
            tool_context: Tool context provided by ADK
            
        Returns:
            Updated context with workflow results
        """
        document_path = context["document_path"]
        
        # Step 1: Document Processing
        context["current_step"] = "document_processing"
        self.logger.info(f"Starting document extraction for: {document_path}")
        
        extraction_result = self.process_document_step(document_path, context, tool_context)
        context["extraction_result"] = extraction_result
        context["document_type"] = extraction_result.get("document_type", "unknown")
        context["steps_completed"].append("document_processing")
        
        # Step 2: Rule Retrieval
        context["current_step"] = "rule_retrieval"
        self.logger.info(f"Retrieving rules for document type: {context['document_type']}")
        
        rules = self.retrieve_rules_step(context["document_type"], context, tool_context)
        context["applicable_rules"] = rules
        context["steps_completed"].append("rule_retrieval")
        
        # Step 3: Validation
        context["current_step"] = "validation"
        self.logger.info(f"Validating document against rules")
        
        validation_result = self.validate_document_step(context["extraction_result"], context["applicable_rules"], context, tool_context)
        context["validation_result"] = validation_result
        context["is_valid"] = validation_result.get("is_valid", False)
        context["steps_completed"].append("validation")
        
        # Only proceed with storage and analytics if document is valid
        if context["is_valid"]:
            # Step 4: Storage
            context["current_step"] = "storage"
            self.logger.info(f"Storing validated document data")
            
            storage_result = self.store_document_step(context["extraction_result"], context, tool_context)
            context["storage_result"] = storage_result
            context["document_id"] = storage_result.get("document_id")
            context["steps_completed"].append("storage")
            
            # Step 5: Analytics
            context["current_step"] = "analytics"
            self.logger.info(f"Generating analytics for document")
            
            analytics_result = self.analyze_document_step(context["extraction_result"], context, tool_context)
            context["analytics_result"] = analytics_result
            context["steps_completed"].append("analytics")
        
        return context
            
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
