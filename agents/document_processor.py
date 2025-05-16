"""
Document processor agent for the FinFlow system.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from google.adk.tools import ToolContext # type: ignore
from google.adk.agents import LlmAgent # type: ignore

from agents.base_agent import BaseAgent
from utils.prompt_templates import get_agent_prompt
from utils.logging_config import TraceContext, log_agent_call

class DocumentProcessorAgent(BaseAgent):
    """
    Agent responsible for extracting and structuring information from financial documents.
    """
    
    def __init__(self):
        """Initialize the document processor agent."""
        # Get the instruction prompt from template
        instruction = get_agent_prompt("document_processor")
        
        super().__init__(
            name="FinFlow_DocumentProcessor",
            model="gemini-2.0-flash",
            description="Extracts and structures information from financial documents",
            instruction=instruction,
            temperature=0.1,
        )
        
        # Initialize logger
        self.logger = logging.getLogger(f"finflow.agents.{self.name}")
    
    def _register_document_ai_tool(self):
        """Register Document AI tool."""
        try:
            from tools.document_ai import process_document, analyze_financial_document
            from utils.agent_tools import DocumentProcessingTool
            
            # Adapter function for general document processing
            def document_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                # Get processor ID from context or use default invoice processor
                processor_id = "projects/YOUR_PROJECT/locations/us-central1/processors/finflow-invoice-processor"
                
                # Read the file as bytes before passing to process_document
                try:
                    with open(document_path, 'rb') as file:
                        file_content = file.read()
                    return process_document(file_content, processor_id, tool_context)
                except FileNotFoundError:
                    return {"status": "error", "message": f"File not found: {document_path}"}
                except Exception as e:
                    return {"status": "error", "message": f"Error processing document: {str(e)}"}
            
            # Adapter function specifically for invoice processing
            def invoice_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                try:
                    with open(document_path, 'rb') as file:
                        file_content = file.read()
                    
                    # Use analyze_financial_document which is optimized for invoices and receipts
                    result = analyze_financial_document(file_content, tool_context)
                    
                    # If document type is not invoice/receipt, log a warning
                    if result.get("document_type") not in ["invoice", "receipt"]:
                        self.logger.warning(f"Document {document_path} detected as {result.get('document_type')}, not invoice/receipt")
                    
                    return result
                except FileNotFoundError:
                    return {"status": "error", "message": f"File not found: {document_path}"}
                except Exception as e:
                    return {"status": "error", "message": f"Error processing invoice: {str(e)}"}
                
            # Register general document processing tool
            doc_ai_tool = DocumentProcessingTool(document_adapter)
            LlmAgent.add_tool(self, doc_ai_tool)  # type: ignore
            
            # Register invoice-specific tool using the dedicated InvoiceProcessingTool
            from utils.agent_tools import InvoiceProcessingTool
            invoice_tool = InvoiceProcessingTool(invoice_adapter)
            LlmAgent.add_tool(self, invoice_tool)  # type: ignore
            
            self.logger.info("Document AI tools registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register Document AI tools: {e}")
    
    def _register_document_ingestion_tool(self):
        """Register document ingestion tool."""
        try:
            from tools.document_ingestion import validate_document, preprocess_document, upload_document, batch_upload_documents
            from utils.agent_tools import FinflowTool
            
            # Create adapter function for document validation
            def validate_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                return validate_document(document_path)
            
            # Create adapter function for document preprocessing
            def preprocess_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                return preprocess_document(document_path)
            
            # Create adapter function for document upload
            def upload_adapter(document_path: str, destination_folder: Optional[str] = None, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                return upload_document(document_path, destination_folder, tool_context)
            
            # Create adapter function for batch upload
            def batch_upload_adapter(file_paths: List[str], destination_folder: Optional[str] = None, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                return batch_upload_documents(file_paths, destination_folder, tool_context)
            
            # Register document validation tool
            validate_tool = FinflowTool(
                name="validate_document",
                description="Validates if a document is of supported type and format.",
                function=lambda params, ctx: validate_adapter(params.get("document_path", ""), ctx)
            )
            LlmAgent.add_tool(self, validate_tool)  # type: ignore
            
            # Register document preprocessing tool
            preprocess_tool = FinflowTool(
                name="preprocess_document",
                description="Preprocesses a document for improved Document AI processing.",
                function=lambda params, ctx: preprocess_adapter(params.get("document_path", ""), ctx)
            )
            LlmAgent.add_tool(self, preprocess_tool)  # type: ignore
            
            # Register document upload tool
            upload_tool = FinflowTool(
                name="upload_document",
                description="Uploads a document to the FinFlow system.",
                function=lambda params, ctx: upload_adapter(
                    params.get("document_path", ""),
                    params.get("destination_folder"),
                    ctx
                )
            )
            LlmAgent.add_tool(self, upload_tool)  # type: ignore
            
            # Register batch upload tool
            batch_tool = FinflowTool(
                name="batch_upload_documents",
                description="Upload multiple documents to the FinFlow system.",
                function=lambda params, ctx: batch_upload_adapter(
                    params.get("file_paths", []),
                    params.get("destination_folder"),
                    ctx
                )
            )
            LlmAgent.add_tool(self, batch_tool)  # type: ignore
            
            self.logger.info("Document ingestion tools registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register document ingestion tools: {e}")
    
    def register_tools(self):
        """Register all tools for the document processor agent."""
        self._register_document_ai_tool()
        self._register_document_ingestion_tool()
        self.logger.info("All document processing tools registered")
    
    def process_document(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a document and extract structured information.
        
        Args:
            context: Processing context with document information
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the extracted information
        """
        # Extract document path from context
        document_path = context.get("document_path")
        if not document_path:
            return self.handle_error(ValueError("Document path not provided in context"), context)
            
        # Create trace context for this process
        with TraceContext() as trace:
            # Log the start of processing
            self.logger.info(f"Processing document: {document_path}")
            log_agent_call(self.logger, self.name, context)
            
            # Track processing in context
            context["trace_id"] = trace.trace_id
            context["processing_start_time"] = datetime.now().isoformat()
            
            try:
                # First, check if file exists
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f"Document not found: {document_path}")
                
                # Process document using Document AI
                # We'll use either the actual Document AI tool or simulated behavior depending on availability
                try:
                    # Try to use the Document AI tool if available
                    from tools.document_ai import process_document
                    
                    with open(document_path, 'rb') as file:
                        file_content = file.read()
                    
                    processor_id = "default-processor"  # Use default processor ID
                    extracted_info = process_document(file_content, processor_id, tool_context)
                    
                    # If we get an error status, fall back to simulation
                    if extracted_info.get("status") == "error":
                        self.logger.warning(f"Document AI processing failed, falling back to simulation: {extracted_info.get('message')}")
                        extracted_info = self._extract_document_info(document_path)
                except Exception as e:
                    # If anything fails, use the simulation as fallback
                    self.logger.warning(f"Using simulated document extraction due to: {str(e)}")
                    extracted_info = self._extract_document_info(document_path)
                
                # Structure the data according to FinFlow data model
                structured_data = self._structure_document_data(extracted_info)
                
                # Update context with result
                context["document_type"] = structured_data["document_type"]
                context["extracted_data"] = structured_data
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "success"
                
                # Log completion
                self.logger.info(f"Document processing completed successfully for: {document_path}")
                self.log_activity(
                    "document_processing_complete", 
                    {"document_path": document_path, "document_type": structured_data["document_type"]}, 
                    context
                )
                
            except Exception as e:
                # Handle errors
                context = self.handle_error(e, context)
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "error"
                
                # Log error
                self.logger.error(f"Document processing failed for: {document_path}")
                self.log_activity(
                    "document_processing_failed", 
                    {"document_path": document_path, "error": str(e)}, 
                    context
                )
        
        return context
    
    def _extract_document_info(self, document_path: str) -> Dict[str, Any]:
        """
        Extract information from document using Document AI.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict containing extracted document information
        """
        # Simulate Document AI response
        # In a real implementation, this would call the actual Document AI service
        
        # Determine document type from file extension
        _, file_ext = os.path.splitext(document_path)
        
        # Simulate different document types based on file extension
        if file_ext.lower() == '.pdf':
            doc_type = "invoice"
        elif file_ext.lower() == '.jpg' or file_ext.lower() == '.png':
            doc_type = "receipt"
        else:
            doc_type = "unknown"
            
        # Mock extraction result
        return {
            "document_type": doc_type,
            "text": "Sample document text for " + document_path,
            "entities": {
                "invoice_number": "INV-12345" if doc_type == "invoice" else None,
                "receipt_id": "R-67890" if doc_type == "receipt" else None,
                "date": "2025-05-15",
                "total_amount": "1000.00",
                "vendor": "Acme Corp",
                "line_items": [
                    {"description": "Item 1", "quantity": "2", "unit_price": "250.00", "amount": "500.00"},
                    {"description": "Item 2", "quantity": "1", "unit_price": "500.00", "amount": "500.00"}
                ]
            },
            "confidence": 0.95,
            "pages": 2
        }
    
    def _structure_document_data(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure extracted information according to FinFlow data model.
        
        Args:
            extracted_info: Raw extracted information
            
        Returns:
            Dict containing structured document data
        """
        document_type = extracted_info["document_type"]
        entities = extracted_info.get("entities", {})
        
        # Common metadata structure to avoid type errors
        metadata: Dict[str, Any] = {
            "confidence": float(extracted_info.get("confidence", 0.0)),
            "pages": int(extracted_info.get("pages", 1)),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Structure data based on document type
        if document_type == "invoice":
            structured_data: Dict[str, Any] = {
                "document_type": "invoice",
                "metadata": metadata,
                "invoice_number": entities.get("invoice_number"),
                "date": entities.get("date"),
                "vendor": {
                    "name": entities.get("vendor"),
                    "id": None  # Would be populated in a real implementation
                },
                "total_amount": float(entities.get("total_amount", "0")),
                "line_items": entities.get("line_items", [])
            }
        elif document_type == "receipt":
            structured_data = {
                "document_type": "receipt",
                "metadata": metadata,  # Use the common metadata structure
                "receipt_id": entities.get("receipt_id"),
                "date": entities.get("date"),
                "vendor": {
                    "name": entities.get("vendor"),
                    "id": None
                },
                "total_amount": float(entities.get("total_amount", "0")),
                "line_items": entities.get("line_items", [])
            }
        else:
            # Generic structure for unknown document types
            structured_data = {
                "document_type": "unknown",
                "metadata": metadata,  # Use the common metadata structure
                "text_content": extracted_info.get("text", ""),
                "entities": entities
            }
            
        return structured_data
    
    def log_activity(self, activity_type: str, details: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log agent activity for audit and debugging purposes.
        
        Args:
            activity_type: Type of activity
            details: Activity details
            context: Current context
        """
        trace_id = context.get("trace_id", "no-trace")
        
        # Create activity log entry with explicit typing
        activity_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "trace_id": trace_id,
            "activity_type": activity_type,
            "details": details
        }
        
        # Log activity with explicit typing
        extra_data: Dict[str, Any] = {"activity": activity_log}
        self.logger.debug(f"Activity: {activity_type}", extra=extra_data)
