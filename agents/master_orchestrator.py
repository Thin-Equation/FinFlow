"""
Master orchestrator agent for the FinFlow system.
"""

import logging
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from agents.base_agent import BaseAgent
from agents.document_processor import DocumentProcessorAgent
from agents.rule_retrieval import RuleRetrievalAgent
from agents.validation_agent import ValidationAgent
from agents.storage_agent import StorageAgent
from agents.analytics_agent import AnalyticsAgent

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
        super().__init__(
            name="FinFlow_Master_Orchestrator",
            model="gemini-2.0-pro",
            description="Coordinates workflow execution and delegates tasks to worker agents",
            instruction="""
            You are the master orchestrator for the FinFlow financial document processing system.
            Your job is to:
            
            1. Receive document processing requests
            2. Determine the appropriate workflow based on document type
            3. Delegate tasks to specialized worker agents
            4. Track and report progress
            5. Handle errors and coordinate retries
            6. Ensure end-to-end processing completes successfully
            
            You should maintain context throughout the conversation, including the state of
            document processing and any errors encountered.
            """
        )
        
        # Register worker agents
        self.worker_agents = {
            "document_processor": document_processor,
            "rule_retrieval": rule_retrieval,
            "validation_agent": validation_agent,
            "storage_agent": storage_agent,
            "analytics_agent": analytics_agent,
        }
        
        # Register active worker agents as tools
        self._register_worker_agents()
    
    def _register_worker_agents(self):
        """Register available worker agents as tools."""
        for name, agent in self.worker_agents.items():
            if agent:
                self.logger.info(f"Registering worker agent: {name}")
                self.add_tool(AgentTool(agent))
    
    async def process_document(self, document_path: str, document_type: str = None) -> Dict[str, Any]:
        """
        Process a document through the entire workflow.
        
        Args:
            document_path: Path to the document.
            document_type: Optional document type hint.
            
        Returns:
            Dict containing the processing results.
        """
        self.logger.info(f"Starting document processing for: {document_path}")
        
        # Create context for the document processing workflow
        context = {
            "document_path": document_path,
            "document_type": document_type,
            "status": "started",
            "steps_completed": [],
            "current_step": "document_processing",
        }
        
        try:
            # Step 1: Document processing
            if self.worker_agents["document_processor"]:
                self.logger.info("Delegating to document processor agent")
                extraction_result = await self.worker_agents["document_processor"].process_document(document_path)
                context["extraction_result"] = extraction_result
                context["steps_completed"].append("document_processing")
                context["current_step"] = "rule_retrieval"
            else:
                raise ValueError("Document processor agent not registered")
            
            # Additional steps will be implemented here
            
            # Mark as completed
            context["status"] = "completed"
            self.logger.info(f"Document processing completed for: {document_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}", exc_info=True)
            context["status"] = "error"
            context["error"] = str(e)
        
        return context
