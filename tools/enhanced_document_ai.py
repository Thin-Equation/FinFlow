"""
Enhanced Document AI processor for the FinFlow system.

This module provides advanced document processing capabilities using Google Document AI,
including batch processing, error handling, and performance optimization.
"""

import os
import logging
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime
import json
import uuid
import random
import hashlib

# Google Cloud imports
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.api_core.exceptions import RetryError, ResourceExhausted
from google.adk.tools import ToolContext  # type: ignore

# Image and document processing
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF

# Local imports
from tools.document_ingestion import validate_document
from config.document_processor_config import get_processor_id

# Configure logging
logger = logging.getLogger("finflow.tools.enhanced_document_ai")

class DocumentProcessor:
    """Enhanced Document AI processor with production-level capabilities."""
    
    def __init__(self, 
                 project_id: str, 
                 environment: str = "development",
                 location: str = "us-central1", 
                 processor_config: Optional[Dict[str, Any]] = None,
                 gcs_bucket: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            project_id: Google Cloud project ID
            environment: Environment (development, staging, production)
            location: Google Cloud location for Document AI
            processor_config: Dictionary mapping document types to processor IDs
            gcs_bucket: Optional GCS bucket for batch processing
        """
        self.project_id = project_id
        self.location = location
        self.environment = environment
        self.client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client(project=project_id) if gcs_bucket else None
        self.gcs_bucket_name = gcs_bucket
        
        # Import config-related functions
        from config.document_processor_config import (
            get_processor_id, get_processor_config, get_validation_settings,
            ERROR_HANDLING, TELEMETRY, PERFORMANCE
        )
        
        # Load configuration based on environment
        self.get_processor_id = lambda doc_type: get_processor_id(doc_type, project_id, environment)
        self.get_processor_config = lambda doc_type: get_processor_config(doc_type, environment)
        self.validation_settings = get_validation_settings(environment)
        
        # Use config settings for error handling, telemetry, and performance
        self.error_config = ERROR_HANDLING
        self.telemetry_config = TELEMETRY
        self.performance_config = PERFORMANCE
        
        # Default processor ID
        self.default_processor_id = self.get_processor_id("general")
        
        # Custom processor configurations if provided
        self.processor_configs = processor_config or {}
        
        # For tracking and telemetry
        self.process_metrics = {
            "total_documents": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0.0,
            "confidence_scores": {},
            "processing_times": [],
            "error_counts": {},
            "throughput": {
                "last_minute": 0,
                "last_hour": 0,
                "last_day": 0
            }
        }
        
        # Initialize telemetry if enabled
        if self.telemetry_config["collect_metrics"]:
            self._setup_telemetry()
        
        # Error recovery settings from config
        self.max_retries = self.error_config.get("max_retries", 3)
        self.retry_delay = self.error_config.get("retry_delay_seconds", 2.0)
        self.timeout = 300.0  # 5 minutes
        self.exponential_backoff = self.error_config.get("exponential_backoff", True)
        
        # Initialize circuit breaker state
        self.circuit_breaker = {
            "enabled": self.error_config.get("circuit_breaker", {}).get("enabled", False),
            "failure_count": 0,
            "failure_threshold": self.error_config.get("circuit_breaker", {}).get("failure_threshold", 5),
            "open": False,
            "last_failure_time": None,
            "reset_timeout": self.error_config.get("circuit_breaker", {}).get("reset_timeout_seconds", 300)
        }
        
        # Cache setup
        self.cache = {}
        self.cache_enabled = self.performance_config.get("cache_enabled", True)
        self.cache_ttl_hours = self.performance_config.get("cache_ttl_hours", 24)
        
        logger.info(f"Document processor initialized for project {project_id} in {location} ({environment} environment)")
    
    def _setup_telemetry(self) -> None:
        """Set up telemetry collection."""
        # Configure logging based on telemetry settings
        log_level = getattr(logging, self.telemetry_config.get("log_level", "INFO"))
        logger.setLevel(log_level)
        
        # Set up periodic reporting if enabled
        if self.telemetry_config.get("periodic_reporting", {}).get("enabled", False):
            import threading
            interval = self.telemetry_config.get("periodic_reporting", {}).get("interval_minutes", 60)
            
            def report_metrics():
                self._report_metrics()
                # Schedule next report
                threading.Timer(interval * 60, report_metrics).start()
            
            # Start the first report timer
            threading.Timer(interval * 60, report_metrics).start()
    
    def _report_metrics(self) -> None:
        """Report collected metrics."""
        logger.info("==== Document Processor Metrics ====")
        logger.info(f"Total documents processed: {self.process_metrics['total_documents']}")
        logger.info(f"Successful: {self.process_metrics['successful']} - " +
                   f"Failed: {self.process_metrics['failed']}")
        
        if self.process_metrics['total_documents'] > 0:
            success_rate = (self.process_metrics['successful'] / 
                           self.process_metrics['total_documents']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
        
        if self.process_metrics['processing_times']:
            avg_time = sum(self.process_metrics['processing_times']) / len(self.process_metrics['processing_times'])
            logger.info(f"Average processing time: {avg_time:.2f} seconds")
        
        # Log top errors if any
        if self.process_metrics['error_counts']:
            logger.info("Top errors:")
            for error, count in sorted(self.process_metrics['error_counts'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                logger.info(f"  - {error}: {count} occurrences")

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open.
        
        Returns:
            True if requests should be allowed, False if circuit breaker is open
        """
        if not self.circuit_breaker["enabled"] or not self.circuit_breaker["open"]:
            return True
        
        # Check if reset timeout has passed
        if self.circuit_breaker["last_failure_time"] is not None:
            elapsed = time.time() - self.circuit_breaker["last_failure_time"]
            if elapsed > self.circuit_breaker["reset_timeout"]:
                # Reset circuit breaker
                self.circuit_breaker["open"] = False
                self.circuit_breaker["failure_count"] = 0
                logger.info("Circuit breaker reset after timeout period")
                return True
        
        # Circuit breaker is open and timeout hasn't passed
        return False
    
    def _update_circuit_breaker(self, success: bool) -> None:
        """Update circuit breaker state based on request success/failure."""
        if not self.circuit_breaker["enabled"]:
            return
        
        if success:
            # On success, decrement failure count but not below zero
            self.circuit_breaker["failure_count"] = max(0, self.circuit_breaker["failure_count"] - 1)
        else:
            # On failure, increment failure count and update last failure time
            self.circuit_breaker["failure_count"] += 1
            self.circuit_breaker["last_failure_time"] = time.time()
            
            # Check if threshold reached
            if self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
                self.circuit_breaker["open"] = True
                logger.warning(
                    f"Circuit breaker opened after {self.circuit_breaker['failure_count']} " +
                    f"consecutive failures. Will reset in {self.circuit_breaker['reset_timeout']} seconds."
                )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay time before retry based on attempt number.
        
        Args:
            attempt: The attempt number (0-based)
            
        Returns:
            Delay time in seconds
        """
        if self.exponential_backoff:
            # Exponential backoff with jitter: base * 2^attempt + random jitter
            base_delay = self.retry_delay * (2 ** attempt)
            jitter = base_delay * 0.1 * (2 * (0.5 - random.random()))  # 10% jitter
            return base_delay + jitter
        else:
            # Fixed delay with small jitter
            jitter = self.retry_delay * 0.1 * (2 * (0.5 - random.random()))
            return self.retry_delay + jitter
    
    def _document_fingerprint(self, content: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique fingerprint for a document to enable caching.
        
        Args:
            content: Document content bytes
            metadata: Optional metadata to include in fingerprint
            
        Returns:
            Document fingerprint as string
        """
        # Generate SHA-256 hash of content
        hasher = hashlib.sha256()
        hasher.update(content)
        
        # Add metadata to hash if provided
        if metadata:
            metadata_str = json.dumps(metadata, sort_keys=True)
            hasher.update(metadata_str.encode())
        
        return hasher.hexdigest()
    
    def _check_cache(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Check if document is in cache.
        
        Args:
            fingerprint: Document fingerprint
            
        Returns:
            Cached result or None if not found
        """
        if not self.cache_enabled:
            return None
        
        cache_entry = self.cache.get(fingerprint)
        if not cache_entry:
            return None
        
        # Check if entry is expired
        timestamp, result = cache_entry
        cache_ttl_seconds = self.cache_ttl_hours * 3600
        if time.time() - timestamp > cache_ttl_seconds:
            # Entry expired, remove from cache
            del self.cache[fingerprint]
            return None
        
        logger.info(f"Cache hit for document {fingerprint[:8]}")
        return result
    
    def _update_cache(self, fingerprint: str, result: Dict[str, Any]) -> None:
        """
        Update cache with document processing result.
        
        Args:
            fingerprint: Document fingerprint
            result: Processing result
        """
        if not self.cache_enabled:
            return
        
        self.cache[fingerprint] = (time.time(), result)
        
        # Prune cache if it gets too large (over 1000 entries)
        if len(self.cache) > 1000:
            # Remove oldest 20% of entries
            entries = sorted(self.cache.items(), key=lambda x: x[1][0])
            for key, _ in entries[:200]:
                del self.cache[key]
    
    def process_single_document(self, 
                               content: Union[bytes, str], 
                               document_type: Optional[str] = None,
                               mime_type: Optional[str] = None,
                               processor_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single document with retry logic and error handling.
        
        Args:
            content: Document bytes or file path
            document_type: Type of document (invoice, receipt, etc.)
            mime_type: MIME type of the document
            processor_id: Override default processor ID
            metadata: Optional metadata to include with the document
            
        Returns:
            Dict with processing results or error information
        """
        start_time = time.time()
        processing_id = str(uuid.uuid4())
        doc_type = document_type or "general"
        
        # Check circuit breaker first
        if not self._check_circuit_breaker():
            error_msg = "Circuit breaker open, request rejected"
            logger.warning(f"{error_msg} (document type: {doc_type})")
            
            if self.error_config.get("log_detailed_errors", True):
                logger.info(f"Circuit breaker state: {self.circuit_breaker}")
            
            return {
                "status": "error",
                "error_type": "circuit_breaker_open",
                "message": error_msg,
                "processing_id": processing_id,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Convert file path to content if needed
            if isinstance(content, str) and os.path.exists(content):
                with open(content, 'rb') as f:
                    content = f.read()
                    
            # Generate fingerprint for caching
            if isinstance(content, bytes):
                fingerprint = self._document_fingerprint(content, metadata)
                
                # Check cache
                cached_result = self._check_cache(fingerprint)
                if cached_result:
                    # Update metrics for cached results
                    self.process_metrics["total_documents"] += 1
                    self.process_metrics["successful"] += 1
                    
                    # Add cache metadata
                    cached_result["from_cache"] = True
                    cached_result["processing_id"] = processing_id
                    return cached_result
            else:
                # Content is neither bytes nor a valid file path
                raise ValueError("Content must be either bytes or a valid file path")
                
            # Determine mime type if not provided
            if not mime_type:
                import magic
                mime_type = magic.from_buffer(content, mime=True)
                
            # Choose processor based on document type or use provided processor_id
            if not processor_id:
                if document_type:
                    processor_id = self.get_processor_id(document_type)
                else:
                    processor_id = self.default_processor_id
                
            # Get processor configuration for timeout settings
            processor_timeout = 120  # default timeout in seconds
            if document_type:
                config = self.get_processor_config(document_type)
                processor_timeout = config.get("processor_timeout_seconds", processor_timeout)
            
            # Initialize result structure
            result = {
                "status": "processing",
                "document_type": document_type,
                "processing_id": processing_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Process with retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    # Create Document AI document object
                    document = {"content": content, "mime_type": mime_type}
                    
                    # Prepare request with optional metadata
                    request = {"name": processor_id, "document": document}
                    if metadata:
                        request["document_metadata"] = metadata
                    
                    # Process the document with timeout
                    logger.info(f"Processing document {processing_id} (attempt {attempt+1}/{self.max_retries+1})")
                    
                    # Set timeout for this request
                    from google.api_core import retry_async
                    from google.api_core import gapic_v1
                    
                    # Create custom retry strategy
                    retry = retry_async.AsyncRetry(
                        predicate=retry_async.if_exception_type(
                            ConnectionError,
                            TimeoutError
                        ),
                        maximum=3,
                    )
                    
                    # Create timeout object
                    timeout = gapic_v1.method.DEFAULT.with_timeout(processor_timeout)
                    
                    # Process document with retry and timeout settings
                    api_result = self.client.process_document(
                        request=request,
                        retry=retry,
                        timeout=timeout
                    )
                    
                    # Extract and format results
                    processed_result = self._extract_document_data(api_result.document, doc_type)
                    
                    # Add metadata
                    processed_result["processing_id"] = processing_id
                    processed_result["status"] = "success"
                    processed_result["processing_time"] = time.time() - start_time
                    processed_result["attempts"] = attempt + 1
                    processed_result["timestamp"] = datetime.now().isoformat()
                    
                    # Update metrics
                    self.process_metrics["total_documents"] += 1
                    self.process_metrics["successful"] += 1
                    self.process_metrics["avg_processing_time"] = (
                        (self.process_metrics["avg_processing_time"] * (self.process_metrics["successful"] - 1) +
                         processed_result["processing_time"]) / self.process_metrics["successful"]
                    )
                    self.process_metrics["processing_times"].append(processed_result["processing_time"])
                    
                    # Update circuit breaker
                    self._update_circuit_breaker(True)
                    
                    # Cache the result
                    if isinstance(content, bytes):
                        self._update_cache(fingerprint, processed_result)
                    
                    return processed_result
                
                except (RetryError, ResourceExhausted, ConnectionError, TimeoutError) as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    if attempt < self.max_retries:
                        # Calculate retry delay with backoff
                        delay = self._calculate_retry_delay(attempt)
                        
                        logger.warning(
                            f"Retryable error ({error_type}) processing document {processing_id}: "
                            f"{error_msg}. Retrying in {delay:.2f}s. Attempt {attempt+1}/{self.max_retries}"
                        )
                        
                        # Wait before retrying
                        time.sleep(delay)
                    else:
                        # All retries exhausted
                        logger.error(
                            f"Failed to process document {processing_id} after {self.max_retries + 1} attempts: "
                            f"{error_type}: {error_msg}"
                        )
                        
                        # Update metrics
                        self.process_metrics["total_documents"] += 1
                        self.process_metrics["failed"] += 1
                        
                        # Track error type
                        error_key = error_type
                        if error_key not in self.process_metrics["error_counts"]:
                            self.process_metrics["error_counts"][error_key] = 0
                        self.process_metrics["error_counts"][error_key] += 1
                        
                        # Update circuit breaker
                        self._update_circuit_breaker(False)
                        
                        # Return error result
                        return {
                            "status": "error",
                            "error_type": error_type,
                            "message": error_msg,
                            "processing_time": time.time() - start_time,
                            "attempts": self.max_retries + 1,
                            "processing_id": processing_id,
                            "document_type": document_type,
                            "timestamp": datetime.now().isoformat()
                        }
                
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    # Log the error
                    logger.error(
                        f"Unexpected error processing document {processing_id}: "
                        f"{error_type}: {error_msg}"
                    )
                    
                    # Update metrics
                    self.process_metrics["total_documents"] += 1
                    self.process_metrics["failed"] += 1
                    
                    # Track error type
                    error_key = error_type
                    if error_key not in self.process_metrics["error_counts"]:
                        self.process_metrics["error_counts"][error_key] = 0
                    self.process_metrics["error_counts"][error_key] += 1
                    
                    # Update circuit breaker
                    self._update_circuit_breaker(False)
                    
                    # Return error result
                    return {
                        "status": "error",
                        "error_type": error_type,
                        "message": error_msg,
                        "processing_time": time.time() - start_time,
                        "processing_id": processing_id,
                        "document_type": document_type,
                        "timestamp": datetime.now().isoformat()
                    }
        
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Log the error
            logger.error(
                f"Error preparing document {processing_id} for processing: "
                f"{error_type}: {error_msg}"
            )
            
            # Update metrics
            self.process_metrics["total_documents"] += 1
            self.process_metrics["failed"] += 1
            
            # Return error result
            return {
                "status": "error",
                "error_type": error_type,
                "message": error_msg,
                "processing_time": time.time() - start_time,
                "processing_id": processing_id,
                "timestamp": datetime.now().isoformat()
            }
            
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
                except Exception:
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

    def batch_process_documents(self, document_paths: List[str], max_workers: int = 5, 
                           batch_size: int = 20, destination_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple documents in parallel.
        
        Args:
            document_paths: List of paths to the documents to process
            max_workers: Maximum number of parallel workers
            batch_size: Size of batches to process
            destination_folder: Optional folder to store processed results
            
        Returns:
            Dictionary with batch processing results
        """
        # Validate inputs
        if not document_paths:
            return {"status": "error", "message": "No documents provided"}
        
        # Create processing context
        processing_id = str(uuid.uuid4())
        logger.info(f"Starting batch processing {processing_id} with {len(document_paths)} documents")
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            self._update_circuit_breaker(success=False)
            return {
                "status": "error",
                "message": "Circuit breaker is open due to previous failures",
                "processing_id": processing_id
            }
        
        # Limit batch size to avoid overloading
        if len(document_paths) > batch_size:
            document_paths = document_paths[:batch_size]
            logger.warning(f"Limited batch to {batch_size} documents")
        
        # Results tracking
        results = {}
        success_count = 0
        failed_count = 0
        total_time = 0.0
        
        # Track start time
        start_time = time.time()
        
        # Process documents in parallel
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create tasks dictionary to keep track of futures
                future_to_path = {
                    executor.submit(self._process_document_with_retry, path, destination_folder): path
                    for path in document_paths
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                        
                        # Update metrics
                        if result.get('status') == 'success':
                            success_count += 1
                            # Collect processing time for metrics
                            if 'processing_time' in result:
                                total_time += result['processing_time']
                        else:
                            failed_count += 1
                            # Track error type for metrics
                            error_type = result.get('error_type', 'unknown')
                            if 'error_counts' not in self.process_metrics:
                                self.process_metrics['error_counts'] = {}
                            if error_type not in self.process_metrics['error_counts']:
                                self.process_metrics['error_counts'][error_type] = 0
                            self.process_metrics['error_counts'][error_type] += 1
                    
                    except Exception as e:
                        # Handle exceptions from the future
                        logger.error(f"Error processing {path}: {str(e)}")
                        results[path] = {
                            'status': 'error',
                            'message': f"Exception during processing: {str(e)}",
                            'path': path
                        }
                        failed_count += 1
        
        except Exception as e:
            # Handle exceptions from the executor itself
            logger.error(f"Batch processing error: {str(e)}")
            self._update_circuit_breaker(success=False)
            return {
                "status": "error",
                "message": f"Batch processing failed: {str(e)}",
                "processing_id": processing_id,
                "results": results,
                "success_count": success_count, 
                "failed_count": failed_count,
                "elapsed_time": time.time() - start_time
            }
        
        # Calculate batch metrics
        total_processed = success_count + failed_count
        success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
        avg_time_per_doc = (total_time / success_count) if success_count > 0 else 0
        
        # Update processor metrics
        self.process_metrics["total_documents"] += total_processed
        self.process_metrics["successful"] += success_count
        self.process_metrics["failed"] += failed_count
        
        if success_count > 0:
            # Update average processing time
            if not self.process_metrics["processing_times"]:
                self.process_metrics["processing_times"] = []
            self.process_metrics["processing_times"].append(avg_time_per_doc)
            
            # Calculate new running average
            avg_processing_time = sum(self.process_metrics["processing_times"]) / len(self.process_metrics["processing_times"])
            self.process_metrics["avg_processing_time"] = avg_processing_time
        
        # Calculate overall elapsed time
        elapsed_time = time.time() - start_time
        
        # Log completion
        logger.info(f"Batch {processing_id} completed in {elapsed_time:.2f}s: {success_count} success, " +
                   f"{failed_count} failed ({success_rate:.1f}% success rate)")
        
        # Return detailed results
        return {
            "status": "completed",
            "processing_id": processing_id,
            "results": results,
            "processed_count": total_processed,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": success_rate,
            "avg_processing_time": avg_time_per_doc,
            "elapsed_time": elapsed_time
        }
        
    def _process_document_with_retry(self, document_path: str, 
                                   destination_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Process document with automatic retry and error handling.
        
        Args:
            document_path: Path to document to process
            destination_folder: Optional destination folder for results
            
        Returns:
            Document processing result
        """
        # Check if document exists
        if not os.path.exists(document_path):
            return {
                "status": "error", 
                "message": f"Document not found: {document_path}",
                "error_type": "file_not_found"
            }
            
        # Generate processing ID for tracking
        processing_id = str(uuid.uuid4())
        
        # Initial retry delay
        retry_delay = self.retry_delay
        
        # Attempt processing with retry
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{self.max_retries} for {document_path} (ID: {processing_id})")
                    
                # Process document (this calls the method we defined earlier)
                result = self.process_document(
                    document_path=document_path,
                    destination_folder=destination_folder
                )
                
                # If successful, update circuit breaker and return result
                if result.get("status") != "error":
                    self._update_circuit_breaker(success=True)
                    return result
                    
                # If we reach here, processing failed but didn't throw an exception
                logger.warning(f"Processing failed (attempt {attempt+1}/{self.max_retries+1}): {result.get('message')}")
                
                # Special handling for rate limiting
                if "rate limit" in result.get("message", "").lower():
                    # Use a longer delay for rate limits
                    time.sleep(min(retry_delay * 2, 10))
                elif attempt < self.max_retries:  # Don't sleep on the last attempt
                    # Use exponential backoff if configured
                    current_delay = retry_delay * (2 ** attempt) if self.exponential_backoff else retry_delay
                    time.sleep(current_delay)
                    
            except Exception as e:
                logger.error(f"Processing error (attempt {attempt+1}/{self.max_retries+1}): {str(e)}")
                
                if attempt < self.max_retries:  # Don't sleep on the last attempt
                    # Use exponential backoff if configured
                    current_delay = retry_delay * (2 ** attempt) if self.exponential_backoff else retry_delay
                    time.sleep(current_delay)
        
        # If we reached here, all attempts failed
        self._update_circuit_breaker(success=False)
        
        # Check if circuit breaker threshold exceeded
        if self.circuit_breaker["enabled"] and self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
            self.circuit_breaker["open"] = True
            self.circuit_breaker["last_failure_time"] = time.time()
            logger.warning("Circuit breaker opened due to consecutive failures")
        
        # Return error
        return {
            "status": "error",
            "message": f"Document processing failed after {self.max_retries + 1} attempts",
            "path": document_path,
            "processing_id": processing_id,
            "error_type": "max_retries_exceeded"
        }

    def optimize_document_for_processing(self, document_path: str) -> str:
        """
        Optimize a document for better processing results.
        
        Args:
            document_path: Path to the document to optimize
            
        Returns:
            Path to optimized document (may be the same as input if optimization not needed)
        """
        try:
            # Check if optimization is needed
            _, ext = os.path.splitext(document_path.lower())
            
            # PDF optimization
            if ext == '.pdf':
                # Start with basic checks
                is_optimized = self._check_if_optimized(document_path)
                
                if is_optimized:
                    logger.debug(f"Document {document_path} is already optimized")
                    return document_path
                
                # Create optimized version
                optimized_path = self._get_optimized_path(document_path)
                
                # Use PyMuPDF to optimize
                doc = fitz.open(document_path)
                
                # Check and process each page
                for page_num in range(len(doc)):
                    # Document page would be accessed as: doc[page_num]
                    # Perform optimization actions here
                    # In a production system, you would implement:
                    # - OCR if needed
                    # - Image compression
                    # - Text extraction and enhancement
                    pass
                    
                # Save optimized document
                doc.save(optimized_path)
                doc.close()
                
                logger.info(f"Optimized document saved to {optimized_path}")
                return optimized_path
                
            # Image optimization
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                # Create optimized version
                optimized_path = self._get_optimized_path(document_path)
                
                # Open and optimize image
                with Image.open(document_path) as img:
                    # Resize if too large
                    max_dimension = 3000  # Max dimension for Document AI
                    if img.width > max_dimension or img.height > max_dimension:
                        # Resize while maintaining aspect ratio
                        ratio = min(max_dimension / img.width, max_dimension / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)  # Increase contrast by 20%
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save optimized image
                    img.save(optimized_path, optimize=True, quality=85)
                
                logger.info(f"Optimized image saved to {optimized_path}")
                return optimized_path
                
            # Other file types
            else:
                # No optimization needed/available
                return document_path
                
        except Exception as e:
            logger.error(f"Error optimizing document {document_path}: {str(e)}")
            # Return original document path on error
            return document_path
            
    def _check_if_optimized(self, document_path: str) -> bool:
        """Check if a document is already optimized."""
        return 'optimized' in document_path.lower()
        
    def _get_optimized_path(self, document_path: str) -> str:
        """Get path for optimized version of document."""
        # Get directory, filename, and extension
        directory = os.path.dirname(document_path)
        filename = os.path.basename(document_path)
        name, ext = os.path.splitext(filename)
        
        # Create optimized filename
        optimized_filename = f"{name}_optimized{ext}"
        
        # Return optimized path
        return os.path.join(directory, optimized_filename)

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
    Create and initialize a DocumentProcessor instance.
    
    Args:
        config: Configuration dictionary for the processor
        
    Returns:
        Initialized DocumentProcessor instance
    """
    try:
        # Extract configuration parameters
        project_id = config.get("project_id")
        if not project_id:
            raise ValueError("Project ID must be provided")
            
        location = config.get("location", "us-central1")
        environment = config.get("environment", "development")
        processor_config = config.get("processor_config", {})
        gcs_bucket = config.get("gcs_bucket")
        
        # Create the document processor
        processor = DocumentProcessor(
            project_id=project_id,
            environment=environment,
            location=location,
            processor_config=processor_config,
            gcs_bucket=gcs_bucket
        )
        
        logger.info(f"Document processor initialized for project {project_id} in {location}")
        return processor
        
    except Exception as e:
        logger.error(f"Failed to create document processor: {str(e)}")
        raise

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

def process_document_with_classification_adapter(document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Process a document with automatic classification and optimal processor selection.
    
    This is an adapter function that can be called via ADK tools framework.
    
    Args:
        document_path: Path to the document to process
        tool_context: ADK tool context
        
    Returns:
        Document processing result with classification
    """
    try:
        # Validate input
        if not document_path or not os.path.exists(document_path):
            return {"status": "error", "message": f"Invalid or missing document: {document_path}"}
            
        # Get context parameters if available
        project_id = None
        if tool_context:
            project_id = getattr(tool_context, "project_id", None)
            
        # If no project ID, try to get it from config
        if not project_id:
            from config.config_loader import load_config
            config = load_config()
            project_id = config.get('google_cloud', {}).get('project_id')
        
        # First, classify the document
        from tools.document_classification import classify_document
        classification_result = classify_document(document_path)
        
        document_type = classification_result.get("document_type", "general")
        confidence = classification_result.get("confidence", 0.0)
        
        logger.info(f"Document classified as {document_type} with confidence {confidence}")
        
        # Get appropriate processor ID for the document type
        from config.document_processor_config import get_processor_id
        processor_id = get_processor_id(document_type, project_id)
        
        # Initialize processor for this document
        processor_config = {
            "project_id": project_id,
            "processor_config": {document_type: processor_id}
        }
        
        processor = create_processor_instance(processor_config)
        
        # Process the document
        result = processor.process_document(document_path, processor_id=processor_id, document_type=document_type)
        
        # Add classification info to result
        result["document_type"] = document_type
        result["classification"] = classification_result
        
        return result
        
    except Exception as e:
        logger.error(f"Error in document processing with classification: {str(e)}")
        return {
            "status": "error",
            "message": f"Document processing failed: {str(e)}",
            "document_path": document_path
        }

def batch_process_documents_adapter(file_paths: List[str], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Process multiple documents in parallel with automatic classification.
    
    This is an adapter function that can be called via ADK tools framework.
    
    Args:
        file_paths: List of paths to documents to process
        tool_context: ADK tool context
        
    Returns:
        Batch processing results
    """
    try:
        # Validate input
        if not file_paths:
            return {"status": "error", "message": "No file paths provided"}
            
        # Get context parameters if available
        project_id = None
        max_workers = 5  # Default
        
        if tool_context:
            project_id = getattr(tool_context, "project_id", None)
            max_workers_param = getattr(tool_context, "max_workers", None)
            if max_workers_param:
                try:
                    max_workers = int(max_workers_param)
                except ValueError:
                    pass
            
        # If no project ID, try to get it from config
        if not project_id:
            from config.config_loader import load_config
            config = load_config()
            project_id = config.get('google_cloud', {}).get('project_id')
            
        # Get processor configuration
        from config.document_processor_config import PROCESSOR_CONFIGS
        
        # Initialize processor
        processor_config = {
            "project_id": project_id,
            "processor_config": {
                doc_type: get_processor_id(doc_type, project_id)
                for doc_type in PROCESSOR_CONFIGS.keys()
            }
        }
        
        processor = create_processor_instance(processor_config)
        
        # Process documents in batch
        max_batch_size = min(50, len(file_paths))  # Document AI has limits
        
        return processor.batch_process_documents(
            file_paths,
            max_workers=max_workers,
            batch_size=max_batch_size
        )
        
    except Exception as e:
        logger.error(f"Error in batch document processing: {str(e)}")
        return {
            "status": "error",
            "message": f"Batch document processing failed: {str(e)}",
            "file_count": len(file_paths),
            "processed_count": 0
        }