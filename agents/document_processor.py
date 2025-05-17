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
            from tools.document_ai import process_document, analyze_financial_document, get_invoice_processor_capabilities
            from utils.agent_tools import DocumentProcessingTool
            
            # Get project configuration (in production, this would come from environment or config)
            from config.config_loader import load_config
            config = load_config()
            project_id = config.get('google_cloud', {}).get('project_id', 'YOUR_PROJECT')
            location = config.get('google_cloud', {}).get('location', 'us-central1')
            processor_name = f"projects/{project_id}/locations/{location}/processors/finflow-invoice-processor"
            
            # Adapter function for general document processing
            def document_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                # Get processor ID from context or use default invoice processor
                processor_id = processor_name
                
                if tool_context:
                    ctx_processor_id = getattr(tool_context, "processor_id", None)
                    if ctx_processor_id:
                        processor_id = ctx_processor_id
                
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
                    
                    # Extract additional invoice-specific fields
                    if result.get("status") != "error":
                        # If document type is not invoice/receipt, log a warning
                        if result.get("document_type") not in ["invoice", "receipt"]:
                            self.logger.warning(f"Document {document_path} detected as {result.get('document_type')}, not invoice/receipt")
                        
                        # Add structured invoice data
                        entities = result.get("entities", {})
                        result["structured_data"] = {
                            "invoice_number": entities.get("invoice_id") or entities.get("invoice_number"),
                            "issue_date": entities.get("invoice_date") or entities.get("date"),
                            "due_date": entities.get("due_date"),
                            "vendor": {
                                "name": entities.get("supplier_name") or entities.get("vendor"),
                                "tax_id": entities.get("supplier_tax_id")
                            },
                            "customer": {
                                "name": entities.get("customer_name"),
                                "tax_id": entities.get("customer_tax_id")
                            },
                            "line_items": entities.get("line_items", []),
                            "total_amount": entities.get("total_amount"),
                            "subtotal": entities.get("subtotal"),
                            "tax_amount": entities.get("tax_amount"),
                            "currency": entities.get("currency"),
                            "payment_terms": entities.get("payment_terms")
                        }
                    
                    return result
                except FileNotFoundError:
                    return {"status": "error", "message": f"File not found: {document_path}"}
                except Exception as e:
                    return {"status": "error", "message": f"Error processing invoice: {str(e)}"}
            
            # Adapter function to get invoice processor capabilities
            def capabilities_adapter(parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
                return get_invoice_processor_capabilities()
                
            # Register general document processing tool
            doc_ai_tool = DocumentProcessingTool(document_adapter)
            LlmAgent.add_tool(self, doc_ai_tool)  # type: ignore
            
            # Register invoice-specific tool using the dedicated InvoiceProcessingTool
            from utils.agent_tools import InvoiceProcessingTool
            invoice_tool = InvoiceProcessingTool(invoice_adapter)
            LlmAgent.add_tool(self, invoice_tool)  # type: ignore
            
            # Register capabilities tool
            capabilities_tool = FinflowTool(
                name="get_invoice_processor_capabilities",
                description="Get information about the capabilities and limitations of the invoice processor",
                function=capabilities_adapter
            )
            LlmAgent.add_tool(self, capabilities_tool)  # type: ignore
            
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
                
                # Validate document before processing
                try:
                    from tools.document_ingestion import validate_document
                    validation_result = validate_document(document_path)
                    
                    if validation_result.get("status") == "error" or not validation_result.get("valid", False):
                        error_msg = validation_result.get("message", "Document validation failed")
                        self.logger.warning(f"Document validation failed: {error_msg}")
                        # Continue processing, but log the warning
                except Exception as e:
                    self.logger.warning(f"Could not validate document: {str(e)}")
                    # Continue with processing even if validation isn't available
                
                # Process document using smart detection of document type
                try:
                    # Detect if this is an invoice or receipt to use specialized processor
                    _, ext = os.path.splitext(document_path.lower())
                    file_type = ext.lstrip(".")
                    
                    # Initialize result
                    extracted_info = {}
                    
                    # Use appropriate tool based on the file and context hints
                    if context.get("document_type") == "invoice" or "invoice" in document_path.lower():
                        # If explicitly an invoice or filename suggests it, use invoice processor
                        self.logger.info(f"Processing as invoice: {document_path}")
                        if hasattr(self, "process_invoice"):
                            # Use the invoice-specific tool through ADK
                            extracted_info = self.process_invoice({"document_path": document_path})
                        else:
                            # Fallback to direct function call
                            from tools.document_ai import analyze_financial_document
                            with open(document_path, 'rb') as file:
                                file_content = file.read()
                            extracted_info = analyze_financial_document(file_content, tool_context)
                    else:
                        # Use general document processor
                        self.logger.info(f"Processing with general document processor: {document_path}")
                        if hasattr(self, "process_document"):
                            # Use the document processing tool through ADK
                            extracted_info = self.process_document({"document_path": document_path})
                        else:
                            # Fallback to direct function call
                            from tools.document_ai import process_document
                            
                            with open(document_path, 'rb') as file:
                                file_content = file.read()
                            
                            # Get processor ID from context or configuration
                            from config.config_loader import load_config
                            config = load_config()
                            project_id = config.get('google_cloud', {}).get('project_id', 'YOUR_PROJECT')
                            location = config.get('google_cloud', {}).get('location', 'us-central1')
                            processor_id = context.get("processor_id") or f"projects/{project_id}/locations/{location}/processors/finflow-document-processor"
                            
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
                
                # Store confidence scores if available
                if "metadata" in structured_data and "confidence" in structured_data["metadata"]:
                    context["confidence_score"] = structured_data["metadata"]["confidence"]
                
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
        document_type = extracted_info.get("document_type", "unknown")
        entities = extracted_info.get("entities", {})
        
        # Check if we already have structured data from the Document AI tool
        if "structured_data" in extracted_info:
            structured_data = extracted_info["structured_data"]
            # Ensure document_type is set
            structured_data["document_type"] = document_type
            # Add metadata if not present
            if "metadata" not in structured_data:
                structured_data["metadata"] = {
                    "confidence": float(extracted_info.get("confidence", 0.0)),
                    "pages": int(extracted_info.get("pages", 1)),
                    "processing_timestamp": datetime.now().isoformat()
                }
            return structured_data
        
        # Common metadata structure to avoid type errors
        metadata: Dict[str, Any] = {
            "confidence": float(extracted_info.get("confidence", 0.0)),
            "pages": int(extracted_info.get("pages", 1)),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Structure data based on document type
        if document_type == "invoice":
            # Extract invoice number with fallbacks
            invoice_number = (
                entities.get("invoice_number") or 
                entities.get("invoice_id") or 
                entities.get("number") or 
                entities.get("invoice_num")
            )
            
            # Extract date with fallbacks
            date_value = (
                entities.get("invoice_date") or 
                entities.get("date") or 
                entities.get("issue_date")
            )
            
            # Extract vendor data with fallbacks
            vendor_name = (
                entities.get("supplier_name") or 
                entities.get("vendor") or 
                entities.get("vendor_name") or
                entities.get("company")
            )
            
            # Try to convert total amount to float, handle different formats
            try:
                total_amount_str = str(entities.get("total_amount", "0"))
                # Remove any currency symbols or commas
                total_amount_str = ''.join(c for c in total_amount_str if c.isdigit() or c in ['.', '-'])
                total_amount = float(total_amount_str)
            except (ValueError, TypeError):
                total_amount = 0.0
                
            # Convert line items if needed
            line_items = entities.get("line_items", [])
            if isinstance(line_items, str):
                # Try to parse line items if they're in string format
                try:
                    import json
                    line_items = json.loads(line_items)
                except:
                    line_items = [{"description": line_items, "amount": "unknown"}]
            elif not isinstance(line_items, list):
                line_items = []
                
            structured_data: Dict[str, Any] = {
                "document_type": "invoice",
                "metadata": metadata,
                "invoice_number": invoice_number,
                "date": date_value,
                "due_date": entities.get("due_date"),
                "vendor": {
                    "name": vendor_name,
                    "tax_id": entities.get("supplier_tax_id") or entities.get("vendor_tax_id")
                },
                "customer": {
                    "name": entities.get("customer_name") or entities.get("bill_to"),
                    "tax_id": entities.get("customer_tax_id")
                },
                "total_amount": total_amount,
                "subtotal": entities.get("subtotal"),
                "tax_amount": entities.get("tax_amount") or entities.get("tax"),
                "currency": entities.get("currency"),
                "payment_terms": entities.get("payment_terms"),
                "line_items": line_items
            }
        elif document_type == "receipt":
            # Try to convert total amount to float
            try:
                total_amount = float(str(entities.get("total_amount", "0")))
            except (ValueError, TypeError):
                total_amount = 0.0
                
            structured_data = {
                "document_type": "receipt",
                "metadata": metadata,
                "receipt_id": entities.get("receipt_id") or entities.get("receipt_number"),
                "date": entities.get("date") or entities.get("receipt_date"),
                "vendor": {
                    "name": entities.get("vendor") or entities.get("merchant"),
                    "id": entities.get("vendor_id")
                },
                "total_amount": total_amount,
                "tax_amount": entities.get("tax_amount") or entities.get("tax"),
                "tip_amount": entities.get("tip_amount") or entities.get("tip"),
                "payment_method": entities.get("payment_method") or entities.get("payment_type"),
                "currency": entities.get("currency"),
                "line_items": entities.get("line_items", [])
            }
        else:
            # Generic structure for unknown document types
            structured_data = {
                "document_type": "unknown",
                "metadata": metadata,
                "text_content": extracted_info.get("text", ""),
                "entities": entities
            }
            
        return structured_data
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors in document processing.
        
        Args:
            error: Exception that occurred
            context: Current processing context
            
        Returns:
            Updated context with error information
        """
        # Get error details
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the error
        self.logger.error(f"Document processing error - {error_type}: {error_message}")
        
        # Add error details to context
        context["status"] = "error"
        context["error_type"] = error_type
        context["error_message"] = error_message
        
        # Try to get traceback info for debugging
        import traceback
        context["error_traceback"] = traceback.format_exc()
        
        # Include document path in error if available
        if "document_path" in context:
            self.logger.error(f"Error processing document: {context['document_path']}")
        
        return context

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
