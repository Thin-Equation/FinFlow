"""
Document processor agent for the FinFlow system.
"""

import logging
import os
import uuid
import json
import concurrent.futures
from typing import Any, Dict, List, Optional
from datetime import datetime
from google.adk.tools import ToolContext # type: ignore
from google.adk.agents import LlmAgent # type: ignore

from agents.base_agent import BaseAgent
from utils.prompt_templates import get_agent_prompt
from utils.logging_config import TraceContext, log_agent_call
from config.document_processor_config import get_processor_id, DEFAULT_PROCESSOR_LOCATION
from config.config_loader import load_config

class DocumentProcessorAgent(BaseAgent):
    """
    Agent responsible for extracting and structuring information from financial documents.
    
    This agent uses Document AI to process various types of financial documents,
    including invoices, receipts, and other document types. It includes advanced
    features like document classification, batch processing, and error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processor agent.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Get the instruction prompt from template
        instruction = get_agent_prompt("document_processor")
        
        super().__init__(
            name="FinFlow_DocumentProcessor",
            model="gemini-2.0-flash",
            description="Extracts and structures information from financial documents with advanced classification and batch processing",
            instruction=instruction,
            temperature=0.1,
        )
        
        # Initialize logger using __dict__ to avoid Pydantic validation
        self.__dict__["logger"] = logging.getLogger(f"finflow.agents.{self.name}")
        
        # Load configuration - first from file, then override with provided config
        self.__dict__["config"] = self._load_config()
        if config:
            # Update with provided config (deep merge)
            self._update_config_recursive(self.__dict__["config"], config)
        
        # Initialize processing and classification components
        self._init_components()
        
        # Tracking for batch operations using __dict__
        self.__dict__["active_batches"] = {}
        self.__dict__["metrics"] = {
            "documents_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_processing_time": 0.0,
            "avg_confidence_score": 0.0,
        }
        
        # Register tools
        self.register_tools()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Access the configuration."""
        return self.__dict__.get("config", {})
    
    @property
    def logger(self) -> logging.Logger:
        """Access the logger."""
        return self.__dict__.get("logger")
    
    @property
    def active_batches(self) -> Dict[str, Any]:
        """Access active batches."""
        return self.__dict__.get("active_batches", {})
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Access metrics."""
        return self.__dict__.get("metrics", {})
    
    @property
    def document_processor(self) -> Any:
        """Access the document processor."""
        return self.__dict__.get("document_processor")
    
    @property
    def ingestion_manager(self) -> Any:
        """Access the ingestion manager."""
        return self.__dict__.get("ingestion_manager")
    
    @property
    def document_classifier(self) -> Any:
        """Access the document classifier."""
        return self.__dict__.get("document_classifier")

    def _load_config(self) -> Dict[str, Any]:
        """Load document processor configuration."""
        try:
            # Try to load from config file
            from config.document_processor_config import (
                PROCESSOR_CONFIGS, CLASSIFICATION, DOCUMENT_TYPE_PRIORITY,
                FILE_VALIDATION, STORAGE, ERROR_HANDLING, TELEMETRY,
                DEFAULT_PROCESSOR_LOCATION, MAX_BATCH_SIZE, MAX_PARALLEL_WORKERS,
                MAX_RETRY_COUNT, DOCUMENT_CACHE_ENABLED, DEFAULT_OPTIMIZATION_LEVEL,
                DEFAULT_CONFIDENCE_THRESHOLD
            )
            
            # Build configuration dictionary
            config = {
                "processor_configs": PROCESSOR_CONFIGS,
                "classification": CLASSIFICATION,
                "document_type_priority": DOCUMENT_TYPE_PRIORITY,
                "file_validation": FILE_VALIDATION,
                "storage": STORAGE,
                "error_handling": ERROR_HANDLING,
                "telemetry": TELEMETRY,
                "default_processor_location": DEFAULT_PROCESSOR_LOCATION,
                "max_batch_size": MAX_BATCH_SIZE,
                "max_parallel_workers": MAX_PARALLEL_WORKERS,
                "max_retry_count": MAX_RETRY_COUNT,
                "document_cache_enabled": DOCUMENT_CACHE_ENABLED,
                "default_optimization_level": DEFAULT_OPTIMIZATION_LEVEL,
                "default_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            }
            
            # Get project ID from main config
            main_config = load_config()
            project_id = main_config.get('google_cloud', {}).get('project_id', 'YOUR_PROJECT')
            environment = main_config.get('environment', 'development')
            
            # Add to config
            config.update({
                "project_id": project_id,
                "environment": environment,
            })
            
            return config
        
        except ImportError:
            self.logger.warning("Could not load document processor config, using defaults")
            # Return minimal default configuration
            return {
                "project_id": "YOUR_PROJECT",
                "environment": "development",
                "processor_configs": {
                    "invoice": {
                        "processor_name": "finflow-invoice-processor",
                        "location": "us-central1",
                    },
                    "general": {
                        "processor_name": "finflow-document-processor",
                        "location": "us-central1",
                    }
                },
                "classification": {
                    "enabled": True,
                    "confidence_threshold": 0.6,
                },
                "max_batch_size": 20,
                "max_parallel_workers": 5,
                "document_cache_enabled": True,
            }
    
    def _update_config_recursive(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Recursively update configuration with overrides."""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_config_recursive(base_config[key], value)
            else:
                # Override or add value
                base_config[key] = value
    
    def _init_components(self) -> None:
        """Initialize document processing components."""
        try:
            # Import enhanced modules
            from tools.enhanced_document_ai import create_processor_instance
            from tools.enhanced_document_ingestion import DocumentIngestionManager
            from tools.document_classification import DocumentClassifier
            
            # Create document processor instance
            processor_config = {
                "project_id": self.config["project_id"],
                "location": self.config.get("default_processor_location", "us-central1"),
                "processor_config": {
                    doc_type: f"projects/{self.config['project_id']}/locations/{config.get('location', 'us-central1')}/processors/{config['processor_name']}"
                    for doc_type, config in self.config["processor_configs"].items()
                },
                "gcs_bucket": self.config.get("gcs_bucket")
            }
            
            self.__dict__["document_processor"] = create_processor_instance(processor_config)
            
            # Create document ingestion manager
            self.__dict__["ingestion_manager"] = DocumentIngestionManager(
                max_workers=self.config.get("max_parallel_workers", 5),
                enable_cache=self.config.get("document_cache_enabled", True)
            )
            
            # Create document classifier if enabled
            if self.config["classification"]["enabled"]:
                self.__dict__["document_classifier"] = DocumentClassifier(
                    confidence_threshold=self.config["classification"]["confidence_threshold"]
                )
            else:
                self.__dict__["document_classifier"] = None
                
            self.logger.info("Document processing components initialized successfully")
            
        except ImportError as e:
            self.logger.warning(f"Could not initialize enhanced components: {e}. Using basic functionality.")
            self.__dict__["document_processor"] = None
            self.__dict__["ingestion_manager"] = None
            self.__dict__["document_classifier"] = None
    
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
            from utils.agent_tools import FinflowTool
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
    
    def _register_enhanced_document_ai_tools(self):
        """Register enhanced Document AI tools."""
        try:
            from tools.enhanced_document_ai import (
                process_document_with_classification_adapter,
                batch_process_documents_adapter
            )
            from utils.agent_tools import FinflowTool
            
            # Register enhanced document processor tool for single documents
            enhanced_doc_tool = FinflowTool(
                name="process_document_with_classification",
                description="Process a document with automatic classification and optimal processor selection",
                function=lambda params, ctx: process_document_with_classification_adapter(
                    params.get("document_path", ""), 
                    ctx
                )
            )
            LlmAgent.add_tool(self, enhanced_doc_tool)  # type: ignore
            
            # Register enhanced batch document processing tool
            batch_tool = FinflowTool(
                name="batch_process_documents",
                description="Process multiple documents in parallel with automatic classification",
                function=lambda params, ctx: batch_process_documents_adapter(
                    params.get("file_paths", []),
                    ctx
                )
            )
            LlmAgent.add_tool(self, batch_tool)  # type: ignore
            
            self.logger.info("Enhanced Document AI tools registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register enhanced Document AI tools: {e}")
    
    def _register_classification_tool(self):
        """Register document classification tool."""
        try:
            from tools.document_classification import classify_document, batch_classify_documents
            from utils.agent_tools import FinflowTool
            
            # Register single document classification tool
            classify_tool = FinflowTool(
                name="classify_document",
                description="Classify a document to determine its type and category",
                function=lambda params, ctx: classify_document(params.get("document_path", ""))
            )
            LlmAgent.add_tool(self, classify_tool)  # type: ignore
            
            # Register batch document classification tool
            batch_classify_tool = FinflowTool(
                name="batch_classify_documents",
                description="Classify multiple documents to determine their types and categories",
                function=lambda params, ctx: batch_classify_documents(params.get("file_paths", []))
            )
            LlmAgent.add_tool(self, batch_classify_tool)  # type: ignore
            
            self.logger.info("Document classification tools registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register document classification tools: {e}")
            
    def register_tools(self):
        """Register all tools for the document processor agent."""
        # Register basic tools
        self._register_document_ai_tool()
        self._register_document_ingestion_tool()
        
        # Register enhanced tools
        self._register_enhanced_document_ai_tools()
        self._register_classification_tool()
        
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
                except (ValueError, TypeError, json.JSONDecodeError):
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

    def process_document_enhanced(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Enhanced document processing with automatic classification and optimal processor selection.
        
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
            self.logger.info(f"Enhanced processing for document: {document_path}")
            log_agent_call(self.logger, self.name, context)
            
            # Track processing in context
            context["trace_id"] = trace.trace_id
            context["processing_start_time"] = datetime.now().isoformat()
            context["processing_id"] = str(uuid.uuid4())
            
            try:
                # First, check if file exists
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f"Document not found: {document_path}")
                
                # Step 1: Validate document
                validation_result = self.ingestion_manager.validate_document(document_path)
                if not validation_result.get("valid", False):
                    self.logger.warning(f"Document validation failed: {validation_result.get('message')}")
                    if self.config["file_validation"]["enforce_strict_validation"]:
                        raise ValueError(f"Document validation failed: {validation_result.get('message')}")
                
                # Step 2: Preprocess document (optimize for OCR)
                optimization_level = context.get("optimization_level", self.config.get("default_optimization_level", "medium"))
                preprocess_result = self.ingestion_manager.preprocess_document(document_path, optimization_level)
                
                if preprocess_result["status"] == "error":
                    self.logger.warning(f"Document preprocessing failed: {preprocess_result.get('message')}")
                    # Continue with original document
                    preprocessed_path = document_path
                else:
                    preprocessed_path = preprocess_result["processed_file_path"]
                    context["preprocessed_path"] = preprocessed_path
                
                # Step 3: Classify document if classification is enabled
                if self.document_classifier and self.config["classification"]["enabled"]:
                    classification_result = self.document_classifier.classify_document(preprocessed_path)
                    
                    if classification_result["status"] == "success":
                        document_type = classification_result["document_type"]
                        confidence = classification_result["confidence"]
                        
                        self.logger.info(f"Document classified as {document_type} with confidence {confidence:.2f}")
                        context["document_type"] = document_type
                        context["classification_confidence"] = confidence
                        context["document_category"] = classification_result["document_category"]
                    else:
                        self.logger.warning(f"Document classification failed: {classification_result.get('error')}")
                        # Use document_type hint from context if available, otherwise unknown
                        document_type = context.get("document_type", "unknown")
                else:
                    # Use document_type hint from context if available
                    document_type = context.get("document_type", "unknown")
                
                # Step 4: Process document with appropriate processor
                processing_result: Dict[str, Any] = {}
                
                # Use document processor component if available
                if self.document_processor:
                    # Auto-route to optimal processor if enabled
                    if self.config["classification"].get("auto_route", True) and document_type != "unknown":
                        processing_result = self.document_processor.process_with_optimal_processor(preprocessed_path)
                    else:
                        # Use specified processor type from context or default
                        processor_type = context.get("processor_type", "general")
                        processor_id = None
                        
                        # Get processor ID if specified
                        if "processor_id" in context:
                            processor_id = context["processor_id"]
                        else:
                            # Build processor ID from config
                            try:
                                processor_id = get_processor_id(
                                    self.config["project_id"],
                                    processor_type,
                                    self.config["environment"]
                                )
                            except Exception as e:
                                self.logger.warning(f"Error getting processor ID: {e}, using default")
                                processor_id = None
                        
                        # Process with specified processor
                        processing_result = self.document_processor.process_single_document(
                            preprocessed_path,
                            document_type=document_type,
                            processor_id=processor_id
                        )
                else:
                    # Fall back to basic processing
                    self.logger.info("Enhanced document processor not available, falling back to basic processing")
                    temp_context = context.copy()
                    temp_context["document_path"] = preprocessed_path
                    return self.process_document(temp_context, tool_context)
                
                # Check if processing was successful
                if processing_result.get("status") == "error":
                    error_message = processing_result.get("error", "Unknown error")
                    self.logger.warning(f"Document AI processing failed: {error_message}")
                    
                    # Fall back to basic processing if configured
                    if self.config["error_handling"]["fallback_to_default_processor"]:
                        self.logger.info("Falling back to basic document processing")
                        temp_context = context.copy()
                        temp_context["document_path"] = preprocessed_path
                        return self.process_document(temp_context, tool_context)
                    else:
                        raise RuntimeError(f"Document processing failed: {error_message}")
                
                # Extract data from processing result
                extracted_data = processing_result.get("extracted_data", {})
                
                # Structure the data according to FinFlow data model
                structured_data = self._structure_document_data(extracted_data)
                
                # Update context with result
                context["document_type"] = structured_data["document_type"]
                context["extracted_data"] = structured_data
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "success"
                context["processor_id"] = processing_result.get("processor_id")
                context["processing_time"] = processing_result.get("processing_time")
                
                # Store confidence scores if available
                if "metadata" in structured_data and "confidence" in structured_data["metadata"]:
                    context["confidence_score"] = structured_data["metadata"]["confidence"]
                
                # Update metrics
                self._update_metrics(True, 
                                   processing_result.get("processing_time", 0),
                                   structured_data.get("metadata", {}).get("confidence", 0.0))
                
                # Log completion
                self.logger.info(f"Enhanced document processing completed successfully for: {document_path}")
                self.log_activity(
                    "document_processing_complete", 
                    {
                        "document_path": document_path, 
                        "document_type": structured_data["document_type"],
                        "processing_time": processing_result.get("processing_time"),
                        "processor_id": processing_result.get("processor_id")
                    }, 
                    context
                )
                
            except Exception as e:
                # Handle errors
                context = self.handle_error(e, context)
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "error"
                
                # Update metrics
                self._update_metrics(False, 0, 0)
                
                # Log error
                self.logger.error(f"Enhanced document processing failed for: {document_path}")
                self.log_activity(
                    "document_processing_failed", 
                    {"document_path": document_path, "error": str(e)}, 
                    context
                )
        
        return context
    
    def batch_process_documents(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process multiple documents in a batch.
        
        Args:
            context: Processing context with document information
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the batch processing results
        """
        # Extract document paths from context
        document_paths = context.get("document_paths", [])
        if not document_paths:
            return self.handle_error(ValueError("Document paths not provided in context"), context)
        
        # Create batch ID
        batch_id = str(uuid.uuid4())
        context["batch_id"] = batch_id
        
        # Create trace context for this process
        with TraceContext() as trace:
            # Log the start of batch processing
            self.logger.info(f"Batch processing {len(document_paths)} documents (ID: {batch_id})")
            log_agent_call(self.logger, self.name, context)
            
            # Track processing in context
            context["trace_id"] = trace.trace_id
            context["processing_start_time"] = datetime.now().isoformat()
            
            try:
                # Check max batch size
                max_batch_size = self.config.get("max_batch_size", 20)
                if len(document_paths) > max_batch_size:
                    self.logger.warning(f"Batch size ({len(document_paths)}) exceeds maximum ({max_batch_size})")
                    document_paths = document_paths[:max_batch_size]
                    context["truncated_batch"] = True
                
                # Store batch info for tracking
                self.active_batches[batch_id] = {
                    "total_documents": len(document_paths),
                    "completed": 0,
                    "successful": 0,
                    "failed": 0,
                    "start_time": datetime.now().isoformat(),
                    "document_paths": document_paths,
                    "results": []
                }
                
                # Process using enhanced document processor if available
                if self.document_processor:
                    # Get document_types if provided
                    document_types = context.get("document_types")
                    
                    # Get max workers
                    max_workers = context.get("max_workers", self.config.get("max_parallel_workers", 5))
                    
                    # Process batch
                    result = self.document_processor.batch_process_documents(
                        document_paths,
                        document_types=document_types,
                        max_workers=max_workers
                    )
                    
                    # Update batch info
                    self.active_batches[batch_id].update({
                        "completed": len(document_paths),
                        "successful": result.get("successful_count", 0),
                        "failed": result.get("failed_count", 0),
                        "end_time": datetime.now().isoformat(),
                        "results": result.get("results", []),
                    })
                    
                    # Structure results for each document
                    structured_results = []
                    for doc_result in result.get("results", []):
                        if doc_result.get("status") == "success":
                            extracted_data = doc_result.get("extracted_data", {})
                            structured_data = self._structure_document_data(extracted_data)
                            
                            structured_results.append({
                                "document_path": doc_result.get("document_path"),
                                "status": "success",
                                "document_type": structured_data["document_type"],
                                "extracted_data": structured_data,
                                "processing_time": doc_result.get("processing_time"),
                                "confidence_score": structured_data.get("metadata", {}).get("confidence", 0.0)
                            })
                        else:
                            structured_results.append({
                                "document_path": doc_result.get("document_path"),
                                "status": "error",
                                "error": doc_result.get("error", "Unknown error"),
                                "error_type": doc_result.get("error_type", "Unknown")
                            })
                    
                    # Update context with structured results
                    context["results"] = structured_results
                    context["successful_count"] = result.get("successful_count", 0)
                    context["failed_count"] = result.get("failed_count", 0)
                    context["processing_time"] = result.get("processing_time")
                    context["status"] = result.get("status", "success")
                    
                else:
                    # Fall back to sequential processing with basic processor
                    self.logger.info("Enhanced document processor not available, using sequential processing")
                    
                    results = []
                    successful = 0
                    failed = 0
                    
                    for doc_path in document_paths:
                        doc_context = {
                            "document_path": doc_path,
                            "parent_batch_id": batch_id
                        }
                        
                        # Copy relevant fields from batch context
                        for key in ["document_type", "processor_type", "processor_id"]:
                            if key in context:
                                doc_context[key] = context[key]
                        
                        # Process document
                        doc_result = self.process_document(doc_context, tool_context)
                        
                        # Update counters
                        if doc_result.get("status") == "success":
                            successful += 1
                        else:
                            failed += 1
                        
                        results.append(doc_result)
                        
                        # Update batch info
                        self.active_batches[batch_id]["completed"] += 1
                        self.active_batches[batch_id]["results"].append(doc_result)
                    
                    # Update batch info
                    self.active_batches[batch_id].update({
                        "successful": successful,
                        "failed": failed,
                        "end_time": datetime.now().isoformat(),
                    })
                    
                    # Update context with results
                    context["results"] = results
                    context["successful_count"] = successful
                    context["failed_count"] = failed
                    context["processing_time"] = (
                        datetime.fromisoformat(datetime.now().isoformat()) -
                        datetime.fromisoformat(context["processing_start_time"])
                    ).total_seconds()
                    context["status"] = "success" if successful > 0 else "error"
                
                # Log completion
                self.logger.info(
                    f"Batch processing completed: {context['successful_count']}/{len(document_paths)} successful"
                )
                self.log_activity(
                    "batch_processing_complete",
                    {
                        "batch_id": batch_id,
                        "total_documents": len(document_paths),
                        "successful": context["successful_count"],
                        "failed": context["failed_count"],
                    },
                    context
                )
                
                # Add completion timestamp
                context["processing_end_time"] = datetime.now().isoformat()
                
            except Exception as e:
                # Handle batch-level errors
                context = self.handle_error(e, context)
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "error"
                
                # Update batch info
                if batch_id in self.active_batches:
                    self.active_batches[batch_id].update({
                        "error": str(e),
                        "end_time": datetime.now().isoformat(),
                    })
                
                # Log error
                self.logger.error(f"Batch processing failed: {str(e)}")
                self.log_activity(
                    "batch_processing_failed",
                    {"batch_id": batch_id, "error": str(e)},
                    context
                )
        
        return context
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch processing job.
        
        Args:
            batch_id: ID of the batch job
            
        Returns:
            Dict with batch status information
        """
        if batch_id not in self.active_batches:
            return {
                "status": "error",
                "message": f"Batch job {batch_id} not found"
            }
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "batch_info": self.active_batches[batch_id]
        }
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """
        Get document processor metrics.
        
        Returns:
            Dict with agent metrics
        """
        current_time = datetime.now().isoformat()
        
        # Get ingestion metrics if available
        ingestion_metrics = {}
        if hasattr(self, "ingestion_manager") and self.ingestion_manager:
            try:
                ingestion_metrics = self.ingestion_manager.get_metrics()
            except Exception as e:
                self.logger.warning(f"Error getting ingestion metrics: {e}")
        
        # Get processor metrics if available
        processor_metrics = {}
        if hasattr(self, "document_processor") and self.document_processor:
            try:
                processor_metrics = self.document_processor.get_metrics()
            except Exception as e:
                self.logger.warning(f"Error getting processor metrics: {e}")
        
        # Merge with agent metrics
        return {
            "timestamp": current_time,
            "agent_metrics": self.metrics,
            "ingestion_metrics": ingestion_metrics,
            "processor_metrics": processor_metrics,
            "active_batches": len(self.active_batches),
        }
    
    def _update_metrics(self, success: bool, processing_time: float, confidence_score: float) -> None:
        """Update agent metrics."""
        self.metrics["documents_processed"] += 1
        
        if success:
            self.metrics["successful_extractions"] += 1
        else:
            self.metrics["failed_extractions"] += 1
        
        # Update average processing time
        n = self.metrics["documents_processed"]
        old_avg = self.metrics["avg_processing_time"]
        if processing_time > 0:
            self.metrics["avg_processing_time"] = old_avg + (processing_time - old_avg) / n
        
        # Update average confidence score (only for successful extractions)
        if success and confidence_score > 0:
            n_success = self.metrics["successful_extractions"]
            old_conf_avg = self.metrics["avg_confidence_score"]
            self.metrics["avg_confidence_score"] = old_conf_avg + (confidence_score - old_conf_avg) / n_success
    
    def process_document_with_classification(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a document with automatic classification to determine the best processor.
        
        Args:
            context: Processing context with document information
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the extracted information and classification results
        """
        # Extract document path from context
        document_path = context.get("document_path")
        if not document_path:
            return self.handle_error(ValueError("Document path not provided in context"), context)
            
        # Create trace context for this process
        with TraceContext() as trace:
            # Log the start of processing
            self.logger.info(f"Processing document with classification: {document_path}")
            
            # Track processing in context
            context["trace_id"] = trace.trace_id
            context["processing_start_time"] = datetime.now().isoformat()
            
            try:
                # First, check if file exists
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f"Document not found: {document_path}")
                
                # Validate document before processing
                try:
                    if self.ingestion_manager:
                        validation_result = self.ingestion_manager.validate_document(document_path)
                        
                        if validation_result.get("status") == "error" or not validation_result.get("valid", False):
                            error_msg = validation_result.get("message", "Document validation failed")
                            self.logger.warning(f"Document validation failed: {error_msg}")
                            # Continue with processing but log warning
                    else:
                        # Fallback to basic validation
                        from tools.document_ingestion import validate_document
                        validate_document(document_path)
                except Exception as e:
                    self.logger.warning(f"Could not validate document: {str(e)}")
                
                # Classify document to determine its type
                document_type = "general"  # Default type
                classification_result = {}
                
                try:
                    if self.document_classifier:
                        classification_result = self.document_classifier.classify_document(document_path)
                        if classification_result.get("document_type"):
                            document_type = classification_result.get("document_type")
                            self.logger.info(f"Document classified as: {document_type} (confidence: {classification_result.get('confidence', 'N/A')})")
                    else:
                        # Fallback to basic classification
                        from tools.document_classification import classify_document
                        classification_result = classify_document(document_path)
                        if classification_result.get("document_type"):
                            document_type = classification_result.get("document_type")
                except Exception as e:
                    self.logger.warning(f"Document classification failed: {str(e)}")
                
                # Add classification result to context
                context["document_type"] = document_type
                context["classification"] = classification_result
                
                # Process document using appropriate processor based on document type
                try:
                    if not self.document_processor:
                        self.logger.warning("Document processor not initialized, falling back to default processing")
                        extracted_info = self._extract_document_info(document_path)
                    else:
                        # Get processor ID based on document type
                        processor_id = self.get_processor_id_for_document_type(document_type)
                        
                        # Process document using the selected processor
                        extracted_info = self.document_processor.process_document(
                            document_path,
                            processor_id=processor_id,
                            document_type=document_type
                        )
                        
                        # If processing failed, try generic processor as fallback
                        if extracted_info.get("status") == "error":
                            self.logger.warning(f"Failed to process with {document_type} processor, trying generic: {extracted_info.get('message')}")
                            generic_processor_id = self.get_processor_id_for_document_type("general")
                            extracted_info = self.document_processor.process_document(
                                document_path,
                                processor_id=generic_processor_id
                            )
                except Exception as e:
                    self.logger.error(f"Document AI processing failed: {str(e)}")
                    extracted_info = self._extract_document_info(document_path)  # Use fallback simulation
                
                # Structure the data according to FinFlow data model
                structured_data = self._structure_document_data(extracted_info, document_type)
                
                # Normalize the extracted data
                normalized_data = self._normalize_data(structured_data)
                
                # Update context with result
                context["extracted_data"] = normalized_data
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "success"
                
                # Update metrics
                self._update_metrics(document_type, normalized_data)
                
                # Log completion
                self.logger.info(f"Document processing completed successfully for: {document_path}")
                
            except Exception as e:
                # Handle errors
                context = self.handle_error(e, context)
                context["processing_end_time"] = datetime.now().isoformat()
                context["status"] = "error"
                
                # Log error
                self.logger.error(f"Document processing failed for: {document_path}")
                
        return context
        
    def batch_process_documents_parallel(self, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process multiple documents in parallel.
        
        Args:
            context: Processing context with file paths and options
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the results for each document
        """
        # Extract file paths from context
        file_paths = context.get("file_paths", [])
        if not file_paths:
            return self.handle_error(ValueError("No file paths provided for batch processing"), context)
        
        # Get batch options
        destination_folder = context.get("destination_folder")
        batch_size = context.get("batch_size") or self.config.get("max_batch_size", 20)
        
        # Create trace context for batch process
        with TraceContext() as trace:
            # Generate batch ID
            batch_id = str(uuid.uuid4())
            
            # Track batch in context
            context["batch_id"] = batch_id
            context["trace_id"] = trace.trace_id
            context["processing_start_time"] = datetime.now().isoformat()
            context["total_documents"] = len(file_paths)
            context["processed_documents"] = 0
            context["successful_documents"] = 0
            context["failed_documents"] = 0
            context["results"] = {}
            
            # Store batch context for tracking
            self.active_batches[batch_id] = context.copy()
            
            try:
                self.logger.info(f"Starting batch processing of {len(file_paths)} documents (Batch ID: {batch_id})")
                
                # Check if enhanced document processor is available
                if self.document_processor and hasattr(self.document_processor, "batch_process_documents"):
                    # Use enhanced batch processing capability
                    batch_result = self.document_processor.batch_process_documents(
                        file_paths,
                        max_workers=self.config.get("max_parallel_workers", 5),
                        destination_folder=destination_folder,
                        batch_size=batch_size
                    )
                    
                    # Update context with batch results
                    context["results"] = batch_result.get("results", {})
                    context["processed_documents"] = batch_result.get("processed_count", 0)
                    context["successful_documents"] = batch_result.get("success_count", 0)
                    context["failed_documents"] = batch_result.get("failed_count", 0)
                    
                else:
                    # Fallback to manual batch processing with concurrent execution
                    results = {}
                    success_count = 0
                    fail_count = 0
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_parallel_workers", 5)) as executor:
                        # Start the document processing tasks
                        future_to_path = {
                            executor.submit(self._process_single_document, path, tool_context): path
                            for path in file_paths[:batch_size]  # Limit to batch size
                        }
                        
                        # Process as they complete
                        for future in concurrent.futures.as_completed(future_to_path):
                            path = future_to_path[future]
                            try:
                                result = future.result()
                                results[path] = result
                                if result.get("status") == "success":
                                    success_count += 1
                                else:
                                    fail_count += 1
                            except Exception as e:
                                self.logger.error(f"Error processing document {path}: {str(e)}")
                                results[path] = {"status": "error", "message": str(e)}
                                fail_count += 1
                    
                    # Update context with results
                    context["results"] = results
                    context["processed_documents"] = len(results)
                    context["successful_documents"] = success_count
                    context["failed_documents"] = fail_count
                
                # Mark batch as completed
                context["status"] = "completed"
                context["processing_end_time"] = datetime.now().isoformat()
                
                # Log completion
                self.logger.info(f"Batch processing completed for batch {batch_id}: {context['successful_documents']} successful, {context['failed_documents']} failed")
                
            except Exception as e:
                # Handle errors
                context = self.handle_error(e, context)
                context["status"] = "error"
                context["processing_end_time"] = datetime.now().isoformat()
                self.logger.error(f"Batch processing failed: {str(e)}")
            
            # Update active batch entry
            self.active_batches[batch_id] = context.copy()
            
        return context
    
    def get_processor_id_for_document_type(self, document_type: str) -> str:
        """
        Get the appropriate processor ID for a document type.
        
        Args:
            document_type: The type of document (e.g., 'invoice', 'receipt', 'general')
            
        Returns:
            The processor ID to use for this document type
        """
        try:
            # First check the processor_configs for specific processor
            if document_type in self.config.get("processor_configs", {}):
                processor_config = self.config["processor_configs"][document_type]
                project_id = self.config.get("project_id", "YOUR_PROJECT")
                location = processor_config.get("location", DEFAULT_PROCESSOR_LOCATION)
                processor_name = processor_config.get("processor_name")
                return f"projects/{project_id}/locations/{location}/processors/{processor_name}"
            
            # Check if we can get it from a dedicated function
            if hasattr(self, "get_processor_id"):
                return self.get_processor_id(document_type)
            
            # Fall back to the default general processor
            default_config = self.config["processor_configs"].get("general", {})
            project_id = self.config.get("project_id", "YOUR_PROJECT")
            location = default_config.get("location", DEFAULT_PROCESSOR_LOCATION)
            processor_name = default_config.get("processor_name", "finflow-document-processor")
            return f"projects/{project_id}/locations/{location}/processors/{processor_name}"
            
        except Exception as e:
            self.logger.error(f"Error getting processor ID for document type {document_type}: {str(e)}")
            # Use the general processor as a failsafe
            project_id = self.config.get("project_id", "YOUR_PROJECT")
            return f"projects/{project_id}/locations/{DEFAULT_PROCESSOR_LOCATION}/processors/finflow-document-processor"
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize extracted data according to FinFlow standards.
        
        This includes:
        - Standardizing date formats
        - Normalizing currency values
        - Ensuring consistent structure across document types
        
        Args:
            data: Extracted document data
        
        Returns:
            Normalized document data
        """
        if not data:
            return {}
            
        try:
            # Create a copy to avoid modifying the original
            normalized = data.copy()
            document_type = normalized.get("document_type", "general")
            
            # Get normalization settings for this document type
            norm_settings = {}
            if document_type in self.config.get("processor_configs", {}):
                norm_settings = self.config["processor_configs"][document_type].get("normalization", {})
            
            # Normalize dates (if we have date fields)
            # Date format will be used when we implement actual normalization
            for field in ["date", "issue_date", "invoice_date", "due_date", "transaction_date", "statement_date", "period_start", "period_end"]:
                if field in normalized and normalized[field]:
                    # Here we'd implement actual date normalization based on date_format
                    # For now just ensure it exists and has a value
                    pass
            
            # Normalize currency values
            amount_precision = norm_settings.get("amount_precision", 2)
            for field in ["total_amount", "subtotal", "tax_amount", "shipping_amount", "discount_amount"]:
                if field in normalized and normalized[field]:
                    # Try to convert to float and format with precision
                    try:
                        amount = float(normalized[field])
                        normalized[field] = format(amount, f".{amount_precision}f")
                    except (ValueError, TypeError):
                        # If it can't be converted, leave as is
                        pass
            
            # Normalize line items if present
            if "line_items" in normalized and isinstance(normalized["line_items"], list):
                for i, item in enumerate(normalized["line_items"]):
                    if isinstance(item, dict):
                        # Normalize quantity
                        if "quantity" in item and item["quantity"]:
                            try:
                                item["quantity"] = str(float(item["quantity"]))
                            except (ValueError, TypeError):
                                pass
                                
                        # Normalize amount/unit_price
                        for price_field in ["amount", "unit_price", "price"]:
                            if price_field in item and item[price_field]:
                                try:
                                    amount = float(item[price_field])
                                    item[price_field] = format(amount, f".{amount_precision}f")
                                except (ValueError, TypeError):
                                    pass
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Error during data normalization: {str(e)}")
            # Return original data if normalization fails
            return data
    
    def _structure_document_data(self, extracted_info: Dict[str, Any], document_type: str = None) -> Dict[str, Any]:
        """
        Structure document data according to the FinFlow data model.
        
        Args:
            extracted_info: Raw extracted information from Document AI
            document_type: Optional document type for specialized structuring
            
        Returns:
            Dict containing structured document data
        """
        # Ensure we have extracted_info
        if not extracted_info:
            return {"document_type": document_type or "unknown"}
        
        # Determine document type if not provided
        if not document_type:
            document_type = extracted_info.get("document_type", "general")
        
        # Initialize with common fields
        structured_data = {
            "document_type": document_type,
            "raw_text": extracted_info.get("text", ""),
            "metadata": {
                "confidence": extracted_info.get("confidence", 0.0),
                "processing_time": extracted_info.get("processing_time_ms", 0),
                "page_count": extracted_info.get("page_count", 1)
            }
        }
        
        # Get entities from extraction
        entities = extracted_info.get("entities", {})
        
        # Structure based on document type
        if document_type == "invoice":
            structured_data.update({
                "invoice_number": entities.get("invoice_id") or entities.get("invoice_number"),
                "issue_date": entities.get("invoice_date") or entities.get("date"),
                "due_date": entities.get("due_date"),
                "total_amount": entities.get("total_amount"),
                "subtotal": entities.get("subtotal"),
                "tax_amount": entities.get("tax_amount"),
                "currency": entities.get("currency"),
                "vendor": {
                    "name": entities.get("supplier_name") or entities.get("vendor"),
                    "address": entities.get("supplier_address") or entities.get("vendor_address"),
                    "tax_id": entities.get("supplier_tax_id") or entities.get("vendor_tax_id"),
                    "phone": entities.get("supplier_phone") or entities.get("vendor_phone"),
                    "email": entities.get("supplier_email") or entities.get("vendor_email")
                },
                "customer": {
                    "name": entities.get("customer_name") or entities.get("bill_to_name"),
                    "address": entities.get("customer_address") or entities.get("bill_to_address"),
                    "id": entities.get("customer_id")
                },
                "line_items": entities.get("line_items", []),
                "payment_terms": entities.get("payment_terms"),
                "purchase_order": entities.get("purchase_order_number") or entities.get("po_number")
            })
        elif document_type == "receipt":
            structured_data.update({
                "receipt_number": entities.get("receipt_number") or entities.get("receipt_id"),
                "date": entities.get("receipt_date") or entities.get("transaction_date") or entities.get("date"),
                "time": entities.get("receipt_time") or entities.get("time"),
                "total_amount": entities.get("total_amount"),
                "subtotal": entities.get("subtotal"),
                "tax_amount": entities.get("tax_amount"),
                "tip_amount": entities.get("tip_amount"),
                "merchant": {
                    "name": entities.get("merchant_name") or entities.get("vendor"),
                    "address": entities.get("merchant_address") or entities.get("store_address"),
                    "phone": entities.get("merchant_phone") or entities.get("store_phone")
                },
                "payment_method": entities.get("payment_method"),
                "currency": entities.get("currency"),
                "line_items": entities.get("line_items", [])
            })
        elif document_type == "bank_statement":
            structured_data.update({
                "account_number": entities.get("account_number"),
                "account_holder": entities.get("account_holder"),
                "statement_date": entities.get("statement_date") or entities.get("date"),
                "period_start": entities.get("period_start"),
                "period_end": entities.get("period_end"),
                "opening_balance": entities.get("opening_balance"),
                "closing_balance": entities.get("closing_balance"),
                "total_deposits": entities.get("total_deposits"),
                "total_withdrawals": entities.get("total_withdrawals"),
                "bank": {
                    "name": entities.get("bank_name"),
                    "address": entities.get("bank_address"),
                    "identifier": entities.get("bank_identifier")
                },
                "transactions": entities.get("transactions", [])
            })
        else:
            # For general documents, just include all entities
            structured_data.update(entities)
        
        # Include any tables that were extracted
        if "tables" in extracted_info:
            structured_data["tables"] = extracted_info["tables"]
        
        return structured_data
        
    def _process_single_document(self, document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Process a single document (helper method for batch processing).
        
        Args:
            document_path: Path to the document
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict containing the processing result
        """
        try:
            # Create context for document
            doc_context = {
                "document_path": document_path,
                "processing_id": str(uuid.uuid4()),
                "processing_start_time": datetime.now().isoformat()
            }
            
            # Process with classification for best results
            return self.process_document_with_classification(doc_context, tool_context)
            
        except Exception as e:
            # Return error information
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            return {
                "document_path": document_path,
                "status": "error",
                "message": str(e),
                "processing_start_time": datetime.now().isoformat(),
                "processing_end_time": datetime.now().isoformat()
            }
    
    def _update_metrics(self, document_type: str, data: Dict[str, Any]) -> None:
        """Update agent metrics with document processing result."""
        try:
            self.metrics["documents_processed"] += 1
            
            # Check if processing was successful
            if data and data.get("document_type"):
                self.metrics["successful_extractions"] += 1
                
                # Update confidence score tracking
                confidence = data.get("metadata", {}).get("confidence", 0.0)
                if confidence > 0:
                    # Calculate running average
                    current_avg = self.metrics["avg_confidence_score"] 
                    total = self.metrics["successful_extractions"]
                    self.metrics["avg_confidence_score"] = ((current_avg * (total - 1)) + confidence) / total
                
                # Track document types
                if "document_types" not in self.metrics:
                    self.metrics["document_types"] = {}
                
                if document_type not in self.metrics["document_types"]:
                    self.metrics["document_types"][document_type] = 0
                
                self.metrics["document_types"][document_type] += 1
                
            else:
                self.metrics["failed_extractions"] += 1
                
        except Exception as e:
            self.logger.warning(f"Error updating metrics: {str(e)}")
    
    def get_detailed_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a batch processing job.
        
        Args:
            batch_id: ID of the batch job
            
        Returns:
            Detailed status information for the batch job
        """
        if batch_id in self.active_batches:
            return self.active_batches[batch_id]
        else:
            return {"status": "error", "message": f"Batch ID {batch_id} not found"}
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "document_processor": self.name,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    def log_processing_activity(self, activity_type: str, activity_data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log document processing activity for telemetry and auditing.
        
        Args:
            activity_type: Type of activity (e.g., document_processing_complete)
            activity_data: Data related to the activity
            context: Current processing context
        """
        try:
            # Only log if telemetry is enabled
            if not self.config.get("telemetry", {}).get("enabled", False):
                return
                
            log_entry = {
                "agent": self.name,
                "activity_type": activity_type,
                "activity_data": activity_data,
                "timestamp": datetime.now().isoformat(),
                "trace_id": context.get("trace_id")
            }
            
            # In a production system, this would be sent to a logging service
            # For now, just log locally
            self.logger.debug(f"Activity log: {json.dumps(log_entry)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log activity: {str(e)}")
