"""
Enhanced document processor agent for the FinFlow system.

This version includes improved error handling, performance optimizations,
metrics tracking, and recovery mechanisms for document processing.
"""

import logging
import os
import uuid
import time
import random
import concurrent.futures
from typing import Any, Dict, List, Optional
from datetime import datetime
import threading

# Import base agent class and utilities
from agents.base_agent import BaseAgent
from utils.prompt_templates import get_agent_prompt

# Import enhanced utilities for metrics and error handling
from utils.metrics import (
    AppMetricsCollector, time_function, count_invocations, track_errors,
    Timer, MetricType, Counter, Histogram
)
from utils.error_handling import (
    AgentError, ErrorSeverity, ErrorManager, 
    retry, circuit_protected, DocumentProcessingError, ConfigurationError
)

# Cache mechanism to avoid reprocessing documents
class DocumentCache:
    """Simple cache for document processing results to improve performance."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize the document cache.
        
        Args:
            max_size: Maximum number of documents to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("finflow.cache.document")
    
    def get(self, document_path: str, document_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a document from the cache if it exists and is not expired.
        
        Args:
            document_path: Path to the document
            document_type: Optional document type for more specific caching
            
        Returns:
            Cached document data if found, None otherwise
        """
        with self._lock:
            key = self._make_key(document_path, document_type)
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            timestamp = self._access_times.get(key, 0)
            if time.time() - timestamp > self._ttl_seconds:
                # Entry has expired
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return entry
    
    def set(self, document_path: str, data: Dict[str, Any], document_type: Optional[str] = None) -> None:
        """Add a document to the cache.
        
        Args:
            document_path: Path to the document
            data: Document data to cache
            document_type: Optional document type for more specific caching
        """
        with self._lock:
            key = self._make_key(document_path, document_type)
            
            # Make room if cache is full
            if len(self._cache) >= self._max_size:
                # Remove least recently used
                oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            # Store document
            self._cache[key] = data
            self._access_times[key] = time.time()
            self.logger.debug(f"Cached document: {key}")
    
    def _make_key(self, document_path: str, document_type: Optional[str] = None) -> str:
        """Create a cache key from document path and type.
        
        Args:
            document_path: Path to the document
            document_type: Optional document type
            
        Returns:
            Cache key string
        """
        if document_type:
            return f"{document_path}:{document_type}"
        return document_path
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.logger.debug("Document cache cleared")
    
    def invalidate(self, document_path: str, document_type: Optional[str] = None) -> None:
        """Remove a specific document from the cache.
        
        Args:
            document_path: Path to the document
            document_type: Optional document type
        """
        with self._lock:
            key = self._make_key(document_path, document_type)
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                self.logger.debug(f"Invalidated cache entry: {key}")


class EnhancedDocumentProcessorAgent(BaseAgent):
    """
    Enhanced agent responsible for extracting and structuring information from financial documents.
    
    This agent uses Document AI to process various types of financial documents,
    and includes performance optimizations, robust error handling, and metric tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced document processor agent.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Get the instruction prompt from template
        instruction = get_agent_prompt("document_processor")
        
        super().__init__(
            name="FinFlow_EnhancedDocumentProcessor",
            model="gemini-2.0-flash",
            description="Extracts and structures information from financial documents with optimized performance and error handling",
            instruction=instruction,
            temperature=0.1,
            retry_attempts=3,
            circuit_breaker_enabled=True,
        )
        
        # Initialize logger
        self.logger = logging.getLogger(f"finflow.agents.{self.name}")
        
        # Initialize metrics collector
        self.metrics_collector = AppMetricsCollector.get_instance()
        
        # Create performance metrics
        self.processing_timer = Timer("document_processing_time")
        self.extraction_counter = Counter("documents_extracted")
        self.processing_histogram = Histogram("document_processing_time_ms")
        
        # Load configuration - first from file, then override with provided config
        self.config = self._load_config()
        if config:
            # Update with provided config (deep merge)
            self._update_config_recursive(self.config, config)
        
        # Initialize processing and classification components
        self._init_components()
        
        # Initialize document cache if enabled
        cache_enabled = self.config.get("document_cache_enabled", True)
        cache_size = self.config.get("cache_size", 100)
        cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        
        if cache_enabled:
            self.document_cache = DocumentCache(max_size=cache_size, ttl_seconds=cache_ttl)
            self.logger.info(f"Document cache enabled with size {cache_size}, TTL {cache_ttl}s")
        else:
            self.document_cache = None
            self.logger.info("Document cache disabled")
        
        # Tracking for batch operations
        self.active_batches = {}
        self.performance_metrics = {
            "documents_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_processing_time": 0.0,
            "avg_confidence_score": 0.0,
        }
        
        # Register tools
        self.register_tools()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load document processor configuration with error handling."""
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
            
            return config
            
        except ImportError:
            # Fall back to default config if module not found
            self.logger.warning("Configuration module not found, using defaults")
            
            return {
                "processor_configs": {},
                "classification": {
                    "enabled": True,
                    "confidence_threshold": 0.7
                },
                "document_type_priority": ["invoice", "receipt", "contract", "statement"],
                "file_validation": {
                    "max_file_size_mb": 10,
                    "allowed_extensions": [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
                },
                "storage": {
                    "result_storage_path": "./document_results",
                    "original_storage_path": "./document_originals"
                },
                "error_handling": {
                    "retry_count": 3,
                    "retry_delay": 2.0
                },
                "telemetry": {
                    "enabled": True,
                    "log_level": "INFO"
                },
                "default_processor_location": "us-central1",
                "max_batch_size": 20,
                "max_parallel_workers": 4,
                "max_retry_count": 3,
                "document_cache_enabled": True,
                "default_optimization_level": "balanced",
                "default_confidence_threshold": 0.7
            }
            
        except Exception as e:
            # Report configuration error
            error_manager = ErrorManager.get_instance()
            error = ConfigurationError(
                f"Error loading document processor configuration: {str(e)}",
                severity=ErrorSeverity.HIGH,
                details={"error": str(e)},
                cause=e
            )
            error_manager.report_error(error)
            self.logger.error(f"Configuration error: {error}")
            
            # Return minimal config to allow operation
            return {
                "document_cache_enabled": True,
                "max_retry_count": 3,
                "max_parallel_workers": 2,
                "file_validation": {
                    "max_file_size_mb": 10
                }
            }
    
    def _init_components(self) -> None:
        """Initialize document processing components with performance optimizations."""
        try:
            # Initialize any document processing components
            # This is a placeholder in this example
            self.logger.info("Initializing document processing components")
            
            # Initialize performance monitoring
            self.metrics_collector.gauge("document_processor_ready").set(1.0)
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            # Report error but allow agent to continue with limited functionality
            error_manager = ErrorManager.get_instance()
            error_manager.report_error(
                AgentError(
                    f"Component initialization error in {self.name}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    details={"agent": self.name},
                    cause=e
                )
            )
    
    def _update_config_recursive(self, base_config: Dict, update_config: Dict) -> None:
        """
        Update base config with values from update_config, handling nested dictionaries.
        
        Args:
            base_config: Base configuration to update
            update_config: New configuration values
        """
        for key, value in update_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                # Recursively update nested dictionaries
                self._update_config_recursive(base_config[key], value)
            else:
                # Update value
                base_config[key] = value
    
    def register_tools(self) -> None:
        """Register tools for the agent."""
        # Register document processor tools
        # This is a placeholder in this example
        self.logger.debug("Registering document processor tools")
    
    @time_function(name="validate_document", metric_type=MetricType.TIMER)
    def validate_document(self, document_path: str) -> Dict[str, Any]:
        """
        Validate that a document can be processed.
        
        Args:
            document_path: Path to document to validate
            
        Returns:
            Dict[str, Any]: Validation result
            
        Raises:
            DocumentProcessingError: If document is invalid
        """
        try:
            # Track validation metrics
            self.metrics_collector.counter("document_validations").increment()
            
            # Check if document exists
            if not os.path.exists(document_path):
                raise DocumentProcessingError(
                    f"Document does not exist: {document_path}",
                    error_code="DOC_NOT_FOUND"
                )
            
            # Check file size
            max_size_mb = self.config.get("file_validation", {}).get("max_file_size_mb", 10)
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                raise DocumentProcessingError(
                    f"Document exceeds maximum size: {file_size_mb:.2f}MB > {max_size_mb}MB",
                    error_code="DOC_TOO_LARGE"
                )
            
            # Check file extension
            allowed_extensions = self.config.get("file_validation", {}).get(
                "allowed_extensions", [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
            )
            
            file_extension = os.path.splitext(document_path)[1].lower()
            if file_extension not in allowed_extensions:
                raise DocumentProcessingError(
                    f"Unsupported document format: {file_extension}",
                    error_code="DOC_UNSUPPORTED_FORMAT"
                )
            
            # Document is valid
            self.logger.debug(f"Document validated: {document_path}")
            return {
                "valid": True,
                "path": document_path,
                "size_mb": file_size_mb,
                "format": file_extension
            }
            
        except DocumentProcessingError as e:
            # Re-raise document processing errors
            self.metrics_collector.counter("document_validation_failures").increment(
                labels={"error_code": e.error_code}
            )
            self.logger.warning(f"Document validation failed: {e}")
            raise
            
        except Exception as e:
            # Convert other exceptions to DocumentProcessingError
            error = DocumentProcessingError(
                f"Error validating document: {str(e)}",
                error_code="DOC_VALIDATION_ERROR",
                cause=e
            )
            
            self.metrics_collector.counter("document_validation_failures").increment(
                labels={"error_code": error.error_code}
            )
            
            self.logger.error(f"Document validation error: {e}")
            raise error
    
    @time_function(name="extract_document", metric_type=MetricType.TIMER)
    @count_invocations(name="document_extraction_calls")
    @track_errors(name="document_extraction_errors")
    @circuit_protected(failure_threshold=5, reset_timeout=60)
    def extract_document(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from a document.
        
        Args:
            context: Processing context with document path
            
        Returns:
            Dict[str, Any]: Extraction result
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        document_path = context.get("document_path")
        if not document_path:
            raise ValueError("Missing document_path in context")
        
        # Start processing timer
        self.processing_timer.start()
        
        try:
            # Check cache if enabled
            if self.document_cache:
                cached_result = self.document_cache.get(document_path)
                if cached_result:
                    self.logger.info(f"Using cached result for {document_path}")
                    self.metrics_collector.counter("cache_hits").increment()
                    return cached_result
                else:
                    self.metrics_collector.counter("cache_misses").increment()
            
            # Validate document (validation occurs but result not used yet)
            self.validate_document(document_path)
            
            # Determine document type through classification
            document_type = self._classify_document(document_path)
            
            # Extract using appropriate processor
            result = self._process_document(document_path, document_type)
            
            # Cache the result if caching is enabled
            if self.document_cache:
                self.document_cache.set(document_path, result)
            
            # Update performance metrics
            self.performance_metrics["documents_processed"] += 1
            self.performance_metrics["successful_extractions"] += 1
            
            processing_time = self.processing_timer.stop()
            
            # Update running average for processing time
            n = self.performance_metrics["documents_processed"]
            current_avg = self.performance_metrics["avg_processing_time"]
            self.performance_metrics["avg_processing_time"] = (
                (current_avg * (n - 1) + processing_time) / n
            )
            
            # Record histogram metric
            self.processing_histogram.observe(
                processing_time * 1000,  # Convert to milliseconds
                labels={"document_type": document_type}
            )
            
            # Increment extraction counter
            self.extraction_counter.increment(labels={"document_type": document_type})
            
            self.logger.info(
                f"Document extracted: {document_path} ({document_type}) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            # Handle errors
            processing_time = self.processing_timer.stop() if self.processing_timer.is_running else 0
            
            # Update error metrics
            self.performance_metrics["failed_extractions"] += 1
            self.metrics_collector.counter("document_extraction_failures").increment(
                labels={"error_type": type(e).__name__}
            )
            
            # Log error
            self.logger.error(f"Error extracting document {document_path}: {e}")
            
            # Re-raise as DocumentProcessingError
            if isinstance(e, DocumentProcessingError):
                raise
            else:
                raise DocumentProcessingError(
                    f"Document extraction failed: {str(e)}",
                    error_code="EXTRACTION_FAILED",
                    cause=e
                )
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
    def _classify_document(self, document_path: str) -> str:
        """
        Classify the document type.
        
        Args:
            document_path: Path to the document
            
        Returns:
            str: Document type
        """
        # This would normally use a document classification model
        # For this example, we'll just use the file extension
        
        # Start timer for classification
        timer = Timer("document_classification")
        timer.start()
        
        try:
            # Use filename to guess document type
            filename = os.path.basename(document_path).lower()
            
            if "invoice" in filename:
                doc_type = "invoice"
            elif "receipt" in filename:
                doc_type = "receipt"
            elif "statement" in filename:
                doc_type = "statement"
            elif "contract" in filename:
                doc_type = "contract"
            else:
                doc_type = "unknown"
            
            # Record classification time
            classification_time = timer.stop()
            self.metrics_collector.histogram(
                "document_classification_time_ms",
                classification_time * 1000
            )
            
            return doc_type
            
        except Exception as e:
            # Record classification failure
            timer.stop()
            self.metrics_collector.counter("classification_errors").increment()
            self.logger.error(f"Classification error: {e}")
            
            # Default to unknown document type
            return "unknown"
    
    @time_function(name="document_processing", metric_type=MetricType.TIMER)
    def _process_document(self, document_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process a document using the appropriate processor.
        
        Args:
            document_path: Path to the document
            document_type: Type of document
            
        Returns:
            Dict[str, Any]: Processing result
        """
        # This would normally use Document AI or another processing system
        # For this example, we'll simulate document processing
        
        self.logger.info(f"Processing document: {document_path} as {document_type}")
        
        # Simulate processing time based on document size
        file_size = os.path.getsize(document_path)
        processing_time = min(0.5 + (file_size / 1000000) * 0.1, 2.0)  # Scale with size, max 2 seconds
        time.sleep(processing_time)
        
        # Create a basic result based on document type
        if document_type == "invoice":
            return {
                "document_path": document_path,
                "document_type": document_type,
                "extracted_data": {
                    "invoice_number": f"INV-{uuid.uuid4().hex[:8]}",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "total_amount": round(random.uniform(100, 1000), 2),
                    "vendor": "Example Vendor Inc.",
                },
                "confidence_score": 0.92,
                "processing_time": processing_time
            }
            
        elif document_type == "receipt":
            return {
                "document_path": document_path,
                "document_type": document_type,
                "extracted_data": {
                    "receipt_number": f"RCPT-{uuid.uuid4().hex[:8]}",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "total_amount": round(random.uniform(10, 200), 2),
                    "merchant": "Example Store",
                },
                "confidence_score": 0.89,
                "processing_time": processing_time
            }
            
        else:
            return {
                "document_path": document_path,
                "document_type": document_type,
                "extracted_data": {
                    "id": f"DOC-{uuid.uuid4().hex[:8]}",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                },
                "confidence_score": 0.75,
                "processing_time": processing_time
            }
    
    @time_function(name="extract_invoice", metric_type=MetricType.TIMER)
    def extract_invoice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from an invoice document.
        
        Args:
            context: Processing context with document path
            
        Returns:
            Dict[str, Any]: Extraction result
        """
        # Use the general extraction method and add invoice-specific processing
        result = self.extract_document(context)
        
        # Add invoice-specific enhancements (would normally include specialized extraction)
        result["document_type"] = "invoice"
        result["invoice_specific_data"] = {
            "line_items": [
                {"item": "Service 1", "quantity": 1, "price": 100.00, "total": 100.00},
                {"item": "Service 2", "quantity": 2, "price": 50.00, "total": 100.00},
            ],
            "subtotal": 200.00,
            "tax": 20.00,
            "total": 220.00
        }
        
        return result
    
    @time_function(name="extract_receipt", metric_type=MetricType.TIMER)
    def extract_receipt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from a receipt document.
        
        Args:
            context: Processing context with document path
            
        Returns:
            Dict[str, Any]: Extraction result
        """
        # Use the general extraction method and add receipt-specific processing
        result = self.extract_document(context)
        
        # Add receipt-specific enhancements
        result["document_type"] = "receipt"
        result["receipt_specific_data"] = {
            "items": [
                {"item": "Product 1", "price": 10.99},
                {"item": "Product 2", "price": 5.99},
            ],
            "payment_method": "Credit Card",
            "card_last_four": "1234"
        }
        
        return result
    
    @time_function(name="batch_process", metric_type=MetricType.TIMER)
    def batch_process(
        self, 
        document_paths: List[str], 
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of documents in parallel.
        
        Args:
            document_paths: List of paths to documents to process
            batch_id: Optional batch ID
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        if not document_paths:
            return {"error": "No documents provided", "results": []}
        
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"Starting batch processing: {batch_id} with {len(document_paths)} documents")
        
        # Create batch context
        batch_context = {
            "batch_id": batch_id,
            "document_count": len(document_paths),
            "start_time": datetime.now().isoformat(),
        }
        
        # Track batch
        self.active_batches[batch_id] = batch_context
        
        # Process documents in parallel
        results = []
        successful = 0
        failed = 0
        
        # Get configured max parallel workers
        max_workers = min(
            self.config.get("max_parallel_workers", 4),
            len(document_paths)
        )
        
        # Start batch timer
        batch_timer = Timer(f"batch_processing_{batch_id}")
        batch_timer.start()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing tasks
            future_to_path = {
                executor.submit(
                    self.extract_document, {"document_path": path}
                ): path for path in document_paths
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                document_path = future_to_path[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    successful += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing document {document_path}: {e}")
                    failed += 1
                    results.append({
                        "document_path": document_path,
                        "error": str(e),
                        "status": "failed"
                    })
        
        # Complete batch processing
        batch_time = batch_timer.stop()
        
        # Update batch context
        batch_context.update({
            "end_time": datetime.now().isoformat(),
            "processing_time": batch_time,
            "successful": successful,
            "failed": failed,
            "completed": True
        })
        
        # Log completion
        self.logger.info(
            f"Batch {batch_id} completed in {batch_time:.2f}s: "
            f"{successful} successful, {failed} failed"
        )
        
        # Record batch metrics
        self.metrics_collector.histogram(
            "batch_processing_time",
            batch_time,
            labels={"batch_size": len(document_paths)}
        )
        self.metrics_collector.counter("batch_documents_processed").increment(value=len(document_paths))
        self.metrics_collector.counter("batch_documents_successful").increment(value=successful)
        self.metrics_collector.counter("batch_documents_failed").increment(value=failed)
        
        return {
            "batch_id": batch_id,
            "total": len(document_paths),
            "successful": successful,
            "failed": failed,
            "processing_time": batch_time,
            "results": results
        }


