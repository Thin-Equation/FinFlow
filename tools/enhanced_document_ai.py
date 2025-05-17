"""
Enhanced Document AI processor for the FinFlow system.

This module provides advanced document processing capabilities using Google Document AI,
including batch processing, error handling, and performance optimization.
"""

import os
import logging
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import asyncio
from pathlib import Path
from datetime import datetime
import json
import uuid

# Google Cloud imports
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.api_core.exceptions import RetryError, ResourceExhausted
from google.adk.tools import ToolContext  # type: ignore

# Local imports
from tools.document_ai import process_document, analyze_financial_document
from tools.document_ingestion import validate_document

# Configure logging
logger = logging.getLogger("finflow.tools.enhanced_document_ai")

class DocumentProcessor:
    """Enhanced Document AI processor with production-level capabilities."""
    
    def __init__(self, 
                 project_id: str, 
                 location: str = "us-central1", 
                 processor_config: Optional[Dict[str, Any]] = None,
                 gcs_bucket: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location for Document AI
            processor_config: Dictionary mapping document types to processor IDs
            gcs_bucket: Optional GCS bucket for batch processing
        """
        self.project_id = project_id
        self.location = location
        self.client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client(project=project_id) if gcs_bucket else None
        self.gcs_bucket_name = gcs_bucket
        
        # Default processor configurations
        self.default_processor_id = f"projects/{project_id}/locations/{location}/processors/finflow-document-processor"
        self.processor_configs = processor_config or {
            "invoice": f"projects/{project_id}/locations/{location}/processors/finflow-invoice-processor",
            "receipt": f"projects/{project_id}/locations/{location}/processors/finflow-receipt-processor",
            "default": self.default_processor_id
        }
        
        # For tracking and telemetry
        self.process_metrics = {
            "total_documents": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0.0,
        }
        
        # Error recovery settings
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        self.timeout = 300.0  # 5 minutes
        
        logger.info(f"Document processor initialized for project {project_id} in {location}")

    def process_single_document(self, 
                               content: Union[bytes, str], 
                               document_type: Optional[str] = None,
                               mime_type: Optional[str] = None,
                               processor_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single document with retry logic and error handling.
        
        Args:
            content: Document bytes or file path
            document_type: Type of document (invoice, receipt, etc.)
            mime_type: MIME type of the document
            processor_id: Override default processor ID
            
        Returns:
            Dict with processing results or error information
        """
        start_time = time.time()
        processing_id = str(uuid.uuid4())
        
        try:
            # Convert file path to content if needed
            if isinstance(content, str) and os.path.exists(content):
                with open(content, 'rb') as f:
                    content = f.read()
                # Auto-detect MIME type if not provided
                if not mime_type:
                    import magic
                    mime_type = magic.from_buffer(content, mime=True)
            
            # Ensure content is bytes
            if not isinstance(content, bytes):
                raise ValueError("Document content must be bytes or a valid file path")

            # Select processor based on document type
            if not processor_id:
                if document_type and document_type in self.processor_configs:
                    processor_id = self.processor_configs[document_type]
                else:
                    # Use default processor
                    processor_id = self.processor_configs.get("default", self.default_processor_id)
            
            logger.debug(f"Processing document {processing_id} with processor {processor_id}")
            
            # Initialize retry counter
            retry_count = 0
            
            while True:
                try:
                    # Process document
                    document = {"content": content, "mime_type": mime_type or "application/pdf"}
                    request = {"name": processor_id, "document": document}
                    
                    # Process the document
                    result = self.client.process_document(request=request)
                    
                    # Parse result
                    if document_type == "invoice":
                        # Use invoice-specific parsing
                        extracted_data = self._parse_invoice_result(result.document)
                    else:
                        # Use general parsing
                        extracted_data = self._parse_general_document(result.document)
                    
                    # Update metrics
                    self._update_metrics(True, time.time() - start_time)
                    
                    return {
                        "status": "success",
                        "processing_id": processing_id,
                        "processor_id": processor_id,
                        "document_type": document_type or extracted_data.get("document_type", "unknown"),
                        "processing_time": time.time() - start_time,
                        "extracted_data": extracted_data
                    }
                    
                except (RetryError, ResourceExhausted) as e:
                    # Retry logic for retriable errors
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.warning(f"Retry {retry_count}/{self.max_retries} for document {processing_id}: {str(e)}")
                        time.sleep(self.retry_delay * retry_count)  # Exponential backoff
                        continue
                    else:
                        raise
                except Exception as e:
                    # Non-retriable error
                    raise
                
                # Exit retry loop if successful
                break
                
        except Exception as e:
            logger.error(f"Error processing document {processing_id}: {str(e)}")
            self._update_metrics(False, time.time() - start_time)
            
            return {
                "status": "error",
                "processing_id": processing_id,
                "document_type": document_type or "unknown",
                "processing_time": time.time() - start_time,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def batch_process_documents(self, 
                               document_paths: List[str], 
                               document_types: Optional[List[str]] = None,
                               max_workers: int = 5) -> Dict[str, Any]:
        """
        Process multiple documents in parallel.
        
        Args:
            document_paths: List of paths to documents
            document_types: Optional list of document types (same length as document_paths)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dict with batch processing results
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        results = []
        
        # Normalize document_types to match document_paths length
        if document_types is None:
            document_types = [None] * len(document_paths)
        elif len(document_types) != len(document_paths):
            raise ValueError("document_types must have the same length as document_paths")
        
        try:
            # Process documents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(
                        self.process_single_document, 
                        path, 
                        doc_type
                    ): (path, doc_type) for path, doc_type in zip(document_paths, document_types)
                }
                
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_path, doc_type = future_to_doc[future]
                    try:
                        result = future.result()
                        result["document_path"] = doc_path
                        results.append(result)
                    except Exception as exc:
                        # Handle unexpected errors
                        logger.error(f"Exception processing {doc_path}: {exc}")
                        results.append({
                            "status": "error",
                            "document_path": doc_path,
                            "document_type": doc_type or "unknown",
                            "error": str(exc),
                            "error_type": type(exc).__name__
                        })
            
            # Summarize results
            successful = sum(1 for r in results if r["status"] == "success")
            
            return {
                "status": "success" if successful == len(document_paths) else "partial_success" if successful > 0 else "error",
                "batch_id": batch_id,
                "total_documents": len(document_paths),
                "successful_count": successful,
                "failed_count": len(document_paths) - successful,
                "processing_time": time.time() - start_time,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing {batch_id}: {str(e)}")
            return {
                "status": "error",
                "batch_id": batch_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time
            }

    async def batch_process_documents_async(self, 
                                          document_paths: List[str], 
                                          document_types: Optional[List[str]] = None,
                                          max_concurrency: int = 5) -> Dict[str, Any]:
        """
        Process multiple documents asynchronously.
        
        Args:
            document_paths: List of paths to documents
            document_types: Optional list of document types
            max_concurrency: Maximum number of concurrent tasks
            
        Returns:
            Dict with batch processing results
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        results = []
        
        # Normalize document_types
        if document_types is None:
            document_types = [None] * len(document_paths)
        elif len(document_types) != len(document_paths):
            raise ValueError("document_types must have the same length as document_paths")
        
        # Define async wrapper for process_single_document
        async def process_async(path: str, doc_type: Optional[str]) -> Dict[str, Any]:
            # Use executor to run CPU-bound processing in a thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self.process_single_document(path, doc_type)
            )
            result["document_path"] = path
            return result
        
        try:
            # Process in batches to control concurrency
            sem = asyncio.Semaphore(max_concurrency)
            
            async def process_with_semaphore(path: str, doc_type: Optional[str]) -> Dict[str, Any]:
                async with sem:
                    return await process_async(path, doc_type)
            
            # Create tasks
            tasks = []
            for path, doc_type in zip(document_paths, document_types):
                task = asyncio.create_task(process_with_semaphore(path, doc_type))
                tasks.append(task)
                
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exceptions
                    processed_results.append({
                        "status": "error",
                        "document_path": document_paths[i],
                        "document_type": document_types[i] or "unknown",
                        "error": str(result),
                        "error_type": type(result).__name__
                    })
                else:
                    processed_results.append(result)
            
            # Summarize results
            successful = sum(1 for r in processed_results if r["status"] == "success")
            
            return {
                "status": "success" if successful == len(document_paths) else "partial_success" if successful > 0 else "error",
                "batch_id": batch_id,
                "total_documents": len(document_paths),
                "successful_count": successful,
                "failed_count": len(document_paths) - successful,
                "processing_time": time.time() - start_time,
                "results": processed_results
            }
            
        except Exception as e:
            logger.error(f"Error in async batch processing {batch_id}: {str(e)}")
            return {
                "status": "error",
                "batch_id": batch_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time
            }

    def classify_document(self, content: Union[bytes, str]) -> Dict[str, Any]:
        """
        Classify a document to determine its type.
        
        Args:
            content: Document bytes or file path
            
        Returns:
            Dict with document classification
        """
        try:
            # Convert file path to content if needed
            if isinstance(content, str) and os.path.exists(content):
                with open(content, 'rb') as f:
                    content = f.read()
            
            # Use a general processor to extract basic information
            document = {"content": content, "mime_type": "application/pdf"}
            request = {"name": self.default_processor_id, "document": document}
            
            # Process the document
            result = self.client.process_document(request=request)
            
            # Extract text for classification
            text = result.document.text if hasattr(result.document, 'text') else ""
            
            # Classify based on content
            classification = self._classify_document_content(text, result.document)
            
            return {
                "status": "success",
                "document_type": classification["document_type"],
                "confidence": classification["confidence"],
                "features": classification["features"]
            }
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                "status": "error",
                "document_type": "unknown",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def process_with_optimal_processor(self, content: Union[bytes, str]) -> Dict[str, Any]:
        """
        Process a document by first classifying it and then using the optimal processor.
        
        Args:
            content: Document bytes or file path
            
        Returns:
            Dict with processing results
        """
        try:
            # First, classify the document
            classification = self.classify_document(content)
            
            if classification["status"] == "error":
                return classification
            
            document_type = classification["document_type"]
            
            # Get the appropriate processor ID
            processor_id = self.processor_configs.get(document_type, self.default_processor_id)
            
            # Process with the optimal processor
            return self.process_single_document(
                content, 
                document_type=document_type,
                processor_id=processor_id
            )
            
        except Exception as e:
            logger.error(f"Error in optimal processing: {str(e)}")
            return {
                "status": "error",
                "document_type": "unknown",
                "error": str(e),
                "error_type": type(e).__name__
            }

    # Private helper methods
    
    def _parse_invoice_result(self, document: Any) -> Dict[str, Any]:
        """Parse invoice-specific document results."""
        entities = {}
        line_items = []
        
        try:
            # Extract entities
            if hasattr(document, 'entities'):
                for entity in document.entities:
                    if hasattr(entity, 'type_') and hasattr(entity, 'mention_text'):
                        entity_type = str(entity.type_)
                        entity_text = str(entity.mention_text)
                        
                        # Handle line items separately
                        if entity_type == 'line_item':
                            if hasattr(entity, 'properties'):
                                line_item = {}
                                for prop in entity.properties:
                                    if hasattr(prop, 'type_') and hasattr(prop, 'mention_text'):
                                        prop_type = str(prop.type_)
                                        prop_text = str(prop.mention_text)
                                        line_item[prop_type] = prop_text
                                if line_item:
                                    line_items.append(line_item)
                        else:
                            entities[entity_type] = entity_text
            
            # Add line items to entities
            entities["line_items"] = line_items
            
            # Structure the data for invoices
            structured_data = {
                "document_type": "invoice",
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
                "line_items": line_items,
                "total_amount": entities.get("total_amount"),
                "subtotal": entities.get("subtotal"),
                "tax_amount": entities.get("tax_amount"),
                "currency": entities.get("currency"),
                "payment_terms": entities.get("payment_terms"),
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error parsing invoice document: {str(e)}")
            return {
                "document_type": "invoice",
                "error": str(e),
                "entities": entities
            }

    def _parse_general_document(self, document: Any) -> Dict[str, Any]:
        """Parse general document results."""
        entities = {}
        
        try:
            # Extract text
            text = document.text if hasattr(document, 'text') else ""
            
            # Extract entities
            if hasattr(document, 'entities'):
                for entity in document.entities:
                    if hasattr(entity, 'type_') and hasattr(entity, 'mention_text'):
                        entity_type = str(entity.type_)
                        entity_text = str(entity.mention_text)
                        entities[entity_type] = entity_text
            
            # Determine document type
            document_type = self._detect_document_type(text, entities)
            
            return {
                "document_type": document_type,
                "text": text[:1000] + "..." if len(text) > 1000 else text,  # Truncate for efficiency
                "entities": entities,
                "pages": len(document.pages) if hasattr(document, 'pages') else 1
            }
            
        except Exception as e:
            logger.error(f"Error parsing general document: {str(e)}")
            return {
                "document_type": "unknown",
                "error": str(e),
                "entities": entities
            }

    def _classify_document_content(self, text: str, document: Any) -> Dict[str, Any]:
        """
        Classify document based on its content.
        
        Args:
            text: Document text content
            document: Document AI document object
            
        Returns:
            Dict with classification information
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract key features for classification
        features = {
            "has_invoice_number": any(kw in text_lower for kw in ["invoice #", "invoice number", "invoice no"]),
            "has_receipt": any(kw in text_lower for kw in ["receipt", "proof of purchase"]),
            "has_bank_statement": any(kw in text_lower for kw in ["account statement", "bank statement", "balance"]),
            "has_tax_form": any(kw in text_lower for kw in ["tax form", "tax return", "form 1040", "w-2"]),
            "has_contract": any(kw in text_lower for kw in ["agreement", "contract", "terms and conditions"]),
        }
        
        # Classification logic
        doc_type = "unknown"
        confidence = 0.5  # Default confidence
        
        if features["has_invoice_number"] or "invoice" in text_lower:
            doc_type = "invoice"
            confidence = 0.8 if features["has_invoice_number"] else 0.6
        elif features["has_receipt"]:
            doc_type = "receipt"
            confidence = 0.7
        elif features["has_bank_statement"]:
            doc_type = "bank_statement"
            confidence = 0.8
        elif features["has_tax_form"]:
            doc_type = "tax_document"
            confidence = 0.9
        elif features["has_contract"]:
            doc_type = "contract"
            confidence = 0.7
        
        return {
            "document_type": doc_type,
            "confidence": confidence,
            "features": features
        }

    def _detect_document_type(self, text: str, entities: Dict[str, Any]) -> str:
        """Detect document type based on text content and entities."""
        text_lower = text.lower()
        
        # Check for invoice indicators
        if ("invoice" in text_lower or "bill" in text_lower or 
            any(k in entities for k in ["invoice_id", "invoice_number", "total_amount"])):
            return "invoice"
        
        # Check for receipt indicators
        elif ("receipt" in text_lower or 
              any(k in entities for k in ["receipt_id", "receipt_number"])):
            return "receipt"
        
        # Check for bank statement indicators
        elif ("statement" in text_lower and ("account" in text_lower or "bank" in text_lower) or
              any(k in entities for k in ["account_number", "statement_date", "closing_balance"])):
            return "bank_statement"
        
        # Check for tax document indicators
        elif ("tax" in text_lower and ("return" in text_lower or "form" in text_lower) or
              any(k in entities for k in ["tax_year", "tax_id", "taxable_income"])):
            return "tax_document"
        
        # Default case
        else:
            return "unknown"

    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Update processing metrics."""
        self.process_metrics["total_documents"] += 1
        
        if success:
            self.process_metrics["successful"] += 1
        else:
            self.process_metrics["failed"] += 1
        
        # Update average processing time using running average
        n = self.process_metrics["total_documents"]
        old_avg = self.process_metrics["avg_processing_time"]
        self.process_metrics["avg_processing_time"] = old_avg + (processing_time - old_avg) / n

    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            **self.process_metrics,
            "success_rate": (self.process_metrics["successful"] / self.process_metrics["total_documents"] 
                            if self.process_metrics["total_documents"] > 0 else 0.0)
        }


# Helper functions for the DocumentProcessor

def create_processor_instance(config: Dict[str, Any]) -> DocumentProcessor:
    """
    Create a DocumentProcessor instance from configuration.
    
    Args:
        config: Configuration dictionary with project_id, location, etc.
        
    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(
        project_id=config.get("project_id"),
        location=config.get("location", "us-central1"),
        processor_config=config.get("processor_config"),
        gcs_bucket=config.get("gcs_bucket")
    )

def process_document_with_classification(file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document with automatic classification.
    
    Args:
        file_path: Path to the document file
        config: Configuration for the processor
        
    Returns:
        Dict with processing results
    """
    # Validate the document first
    validation = validate_document(file_path)
    if not validation.get("valid", False):
        return {
            "status": "error",
            "message": validation.get("message", "Document validation failed"),
            "document_path": file_path
        }
    
    # Create processor instance
    processor = create_processor_instance(config)
    
    # Process with optimal processor
    return processor.process_with_optimal_processor(file_path)

def batch_process_folder(folder_path: str, config: Dict[str, Any], pattern: str = "*") -> Dict[str, Any]:
    """
    Process all documents in a folder.
    
    Args:
        folder_path: Path to the folder containing documents
        config: Configuration for the processor
        pattern: File pattern to match (default: "*" for all files)
        
    Returns:
        Dict with batch processing results
    """
    import glob
    
    # Find documents matching pattern
    file_pattern = os.path.join(folder_path, pattern)
    document_paths = glob.glob(file_pattern)
    
    if not document_paths:
        return {
            "status": "error",
            "message": f"No documents found matching pattern {pattern} in {folder_path}"
        }
    
    # Create processor instance
    processor = create_processor_instance(config)
    
    # Process documents in batch
    return processor.batch_process_documents(document_paths)

# Adapter functions for ADK tools

def process_document_with_classification_adapter(document_path: str, 
                                               tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Adapter function for processing a document with classification for ADK tools.
    
    Args:
        document_path: Path to the document file
        tool_context: Tool context from ADK
        
    Returns:
        Dict with processing results
    """
    # Get configuration from tool context or default
    config = {"project_id": "YOUR_PROJECT", "location": "us-central1"}
    
    if tool_context:
        if hasattr(tool_context, "get"):
            project_id = tool_context.get("project_id")  # type: ignore
            if project_id:
                config["project_id"] = project_id
                
            location = tool_context.get("location")  # type: ignore
            if location:
                config["location"] = location
    
    # Process the document with classification
    return process_document_with_classification(document_path, config)

def batch_process_documents_adapter(file_paths: List[str], 
                                  tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Adapter function for batch processing documents for ADK tools.
    
    Args:
        file_paths: List of paths to document files
        tool_context: Tool context from ADK
        
    Returns:
        Dict with batch processing results
    """
    # Get configuration from tool context or default
    config = {"project_id": "YOUR_PROJECT", "location": "us-central1"}
    
    if tool_context:
        if hasattr(tool_context, "get"):
            project_id = tool_context.get("project_id")  # type: ignore
            if project_id:
                config["project_id"] = project_id
                
            location = tool_context.get("location")  # type: ignore
            if location:
                config["location"] = location
    
    # Create processor instance
    processor = create_processor_instance(config)
    
    # Process documents in batch
    return processor.batch_process_documents(file_paths)