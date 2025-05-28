"""
Optimized batch document processing module for FinFlow.

This module provides high-performance batch processing capabilities with:
- Advanced parallel processing with worker management
- Chunking and prioritization of documents
- Comprehensive error handling and recovery
- Detailed progress tracking and metrics collection
- Resource-aware processing to prevent system overload
"""

import os
import sys
import logging
import time
import json
import uuid
import psutil
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import signal

# Import FinFlow utilities
from utils.metrics import (
    AppMetricsCollector, time_function, Timer, Counter, Histogram, MetricPoint, MetricType
)
from utils.error_handling import (
    retry
)
from utils.recovery import (
    RecoveryManager
)

# Import from base batch processor
from batch.batch_processor import SUPPORTED_EXTENSIONS

# Create module logger
logger = logging.getLogger(__name__)


class DocumentBatch:
    """
    Represents a batch of documents to be processed together.
    Manages document queuing, chunking, and status tracking.
    """
    
    def __init__(self, batch_id: Optional[str] = None, max_chunk_size: int = 10):
        """
        Initialize a document batch.
        
        Args:
            batch_id: Optional ID for the batch (auto-generated if not provided)
            max_chunk_size: Maximum number of documents in a processing chunk
        """
        self.batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        self.max_chunk_size = max_chunk_size
        self.documents: List[Dict[str, Any]] = []
        self.document_paths: Set[str] = set()
        self.chunks: List[List[Dict[str, Any]]] = []
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.failed: Dict[str, Dict[str, Any]] = {}
        self.in_progress: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.is_complete = False
        self.lock = threading.RLock()
        
    def add_document(self, document_path: str, priority: int = 1, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a document to the batch.
        
        Args:
            document_path: Path to the document
            priority: Processing priority (higher numbers = higher priority)
            metadata: Optional document metadata
            
        Returns:
            Boolean indicating if document was added
        """
        with self.lock:
            if document_path in self.document_paths:
                return False
                
            self.document_paths.add(document_path)
            
            doc_entry = {
                "document_path": document_path,
                "document_id": f"doc_{uuid.uuid4().hex[:8]}",
                "priority": priority,
                "metadata": metadata or {},
                "status": "queued",
                "queued_at": time.time()
            }
            
            self.documents.append(doc_entry)
            return True
            
    def create_chunks(self) -> None:
        """Split documents into processing chunks based on priority."""
        with self.lock:
            # Sort by priority (descending)
            sorted_docs = sorted(
                self.documents, 
                key=lambda x: (x["priority"], x["queued_at"]),
                reverse=True
            )
            
            # Create chunks of documents
            self.chunks = [
                sorted_docs[i:i + self.max_chunk_size]
                for i in range(0, len(sorted_docs), self.max_chunk_size)
            ]
            
            logger.info(f"Created {len(self.chunks)} document chunks for batch {self.batch_id}")
    
    def mark_document_in_progress(self, document_path: str) -> None:
        """Mark a document as being processed."""
        with self.lock:
            for doc in self.documents:
                if doc["document_path"] == document_path:
                    doc["status"] = "in_progress"
                    doc["started_at"] = time.time()
                    self.in_progress[document_path] = doc
                    break
    
    def mark_document_complete(self, document_path: str, result: Dict[str, Any]) -> None:
        """Mark a document as successfully processed."""
        with self.lock:
            self.processed[document_path] = result
            if document_path in self.in_progress:
                del self.in_progress[document_path]
                
            for doc in self.documents:
                if doc["document_path"] == document_path:
                    doc["status"] = "completed"
                    doc["completed_at"] = time.time()
                    doc["processing_time"] = doc["completed_at"] - doc.get("started_at", doc["queued_at"])
                    break
    
    def mark_document_failed(self, document_path: str, error: str) -> None:
        """Mark a document as failed during processing."""
        with self.lock:
            failure_info = {
                "document_path": document_path,
                "error": error,
                "failed_at": time.time()
            }
            
            self.failed[document_path] = failure_info
            if document_path in self.in_progress:
                del self.in_progress[document_path]
                
            for doc in self.documents:
                if doc["document_path"] == document_path:
                    doc["status"] = "failed"
                    doc["failed_at"] = time.time()
                    doc["error"] = error
                    break
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current batch processing progress."""
        with self.lock:
            total = len(self.documents)
            completed = len(self.processed)
            failed = len(self.failed)
            in_progress = len(self.in_progress)
            queued = total - completed - failed - in_progress
            
            progress_pct = (completed + failed) / total * 100 if total > 0 else 0
            
            elapsed = time.time() - self.start_time
            estimated_total = None
            
            # Estimate remaining time if we have progress
            if completed > 0:
                estimated_total = elapsed * total / completed
                
            return {
                "batch_id": self.batch_id,
                "total_documents": total,
                "completed": completed,
                "failed": failed,
                "in_progress": in_progress,
                "queued": queued,
                "progress_percent": progress_pct,
                "elapsed_seconds": elapsed,
                "estimated_total_seconds": estimated_total,
                "estimated_remaining_seconds": estimated_total - elapsed if estimated_total else None
            }
    
    def mark_complete(self) -> None:
        """Mark the batch as complete."""
        with self.lock:
            self.is_complete = True
            self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary representation."""
        with self.lock:
            total_time = (self.end_time or time.time()) - self.start_time
            
            return {
                "batch_id": self.batch_id,
                "total_documents": len(self.documents),
                "completed": len(self.processed),
                "failed": len(self.failed),
                "is_complete": self.is_complete,
                "total_time_seconds": total_time,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            }


class ResourceMonitor:
    """
    Monitors system resources and provides guidance on optimal parallel processing.
    """
    
    def __init__(self, target_cpu_percent: float = 80.0, check_interval: int = 5):
        """
        Initialize the resource monitor.
        
        Args:
            target_cpu_percent: Target CPU utilization percentage
            check_interval: Seconds between resource checks
        """
        self.target_cpu_percent = target_cpu_percent
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._cpu_percent = 0.0
        self._memory_percent = 0.0
        self._lock = threading.RLock()
        self.metrics = AppMetricsCollector.get_instance()
        self.logger = logging.getLogger("finflow.batch.resources")
        
    def start(self) -> None:
        """Start the resource monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ResourceMonitorThread"
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitor started")
    
    def stop(self) -> None:
        """Stop the resource monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        self.logger.info("Resource monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_resources()
                self._stop_event.wait(self._calculate_interval())
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")
                self._stop_event.wait(self.check_interval)
    
    def _check_resources(self) -> None:
        """Check current system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            with self._lock:
                self._cpu_percent = cpu_percent
                self._memory_percent = memory_percent
            
            # Record resource metrics
            self.metrics.record(MetricPoint(
                name="system_cpu_percent", 
                value=cpu_percent,
                metric_type=MetricType.GAUGE
            ))
            self.metrics.record(MetricPoint(
                name="system_memory_percent", 
                value=memory_percent,
                metric_type=MetricType.GAUGE
            ))
            
            self.logger.debug(f"Resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to check resources: {e}")
    
    def _calculate_interval(self) -> float:
        """Calculate next check interval based on system load."""
        with self._lock:
            # Check more frequently when system is under heavy load
            if self._cpu_percent > 90:
                return 1.0
            elif self._cpu_percent > 80:
                return 2.0
            else:
                return self.check_interval
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        with self._lock:
            return {
                "cpu_percent": self._cpu_percent,
                "memory_percent": self._memory_percent
            }
    
    def get_optimal_workers(self, min_workers: int = 2, max_workers: int = 16) -> int:
        """
        Calculate optimal worker count based on current resource usage.
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads
            
        Returns:
            Recommended number of worker threads
        """
        with self._lock:
            cpu_count = os.cpu_count() or 4
            
            if self._cpu_percent > 85:
                # Under heavy load, reduce workers
                optimal = max(min_workers, cpu_count // 2)
            elif self._cpu_percent > 70:
                # Moderate load, use default workers
                optimal = max(min_workers, cpu_count - 1)
            else:
                # Light load, can use more workers
                optimal = max(min_workers, cpu_count + 1)
            
            # Ensure we stay within bounds
            return min(max(optimal, min_workers), max_workers)


class OptimizedBatchProcessor:
    """
    High-performance batch processor for document processing with parallel execution,
    recovery mechanisms, and adaptive resource management.
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 8,
        checkpoint_interval: int = 10,
        recovery_dir: Optional[str] = None
    ):
        """
        Initialize the optimized batch processor.
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads
            checkpoint_interval: Number of documents to process before creating checkpoint
            recovery_dir: Directory for recovery files (defaults to .finflow/recovery)
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        
        # Set up recovery directory
        if recovery_dir:
            self.recovery_dir = Path(recovery_dir)
        else:
            self.recovery_dir = Path.home() / ".finflow" / "recovery"
        
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Initialize components
        self.logger = logging.getLogger("finflow.batch.optimized")
        self.metrics = AppMetricsCollector.get_instance()
        
        # Create performance metrics
        self.batch_timer = Timer("batch_processing_time")
        self.document_counter = Counter("batch_documents_processed")
        self.document_histogram = Histogram("batch_document_processing_time")
        
        # Set up resource monitor
        self.resource_monitor = ResourceMonitor(target_cpu_percent=75.0)
        
        # Set up recovery manager
        self.recovery_manager = RecoveryManager()
        
        # Active batch tracking
        self.active_batches: Dict[str, DocumentBatch] = {}
        self.batch_lock = threading.RLock()
        
        # Processing flag
        self._shutdown_requested = False
        
    def _create_checkpoint(self, batch: DocumentBatch) -> None:
        """Create a recovery checkpoint for a batch."""
        try:
            checkpoint_path = self.recovery_dir / f"batch_{batch.batch_id}.json"
            
            # Create checkpoint data
            checkpoint_data = {
                "batch_id": batch.batch_id,
                "timestamp": time.time(),
                "processed": list(batch.processed.keys()),
                "failed": list(batch.failed.keys()),
                "documents": batch.documents
            }
            
            # Write checkpoint to file
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)
                
            self.logger.debug(f"Created checkpoint for batch {batch.batch_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for batch {batch.batch_id}: {e}")
            
    def _load_checkpoint(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load a batch checkpoint if it exists."""
        checkpoint_path = self.recovery_dir / f"batch_{batch_id}.json"
        
        if not checkpoint_path.exists():
            return None
            
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for batch {batch_id}: {e}")
            return None
    
    @time_function("prepare_documents")
    def _prepare_documents(self, batch_dir: str) -> List[str]:
        """
        Find and prepare documents for processing.
        
        Args:
            batch_dir: Directory containing documents to process
            
        Returns:
            List of document paths
        """
        documents = []
        
        for root, _, files in os.walk(batch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in SUPPORTED_EXTENSIONS:
                    documents.append(file_path)
        
        # Sort documents by size (smallest first for quick wins)
        try:
            documents.sort(key=lambda x: os.path.getsize(x))
        except (OSError, FileNotFoundError) as e:
            # If sorting fails, just use the original order
            self.logger.warning(f"Failed to sort documents by size: {str(e)}")
            pass
        
        self.logger.info(f"Found {len(documents)} documents to process in {batch_dir}")
        return documents
    
    @retry(max_attempts=3, delay=1)
    def _process_document(
        self,
        agents: Dict[str, Any],
        config: Dict[str, Any],
        document_path: str,
        workflow_type: str,
        batch_id: str
    ) -> Dict[str, Any]:
        """
        Process a single document with error handling and metrics.
        
        Args:
            agents: Dictionary of initialized agents
            config: Configuration dictionary
            document_path: Path to the document
            workflow_type: Type of workflow to use
            batch_id: ID of the batch this document belongs to
            
        Returns:
            Dict[str, Any]: Processing result with metadata
        """
        start_time = time.time()
        doc_id = os.path.basename(document_path)
        
        # Update batch tracking
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.mark_document_in_progress(document_path)
        
        # Record metrics
        self.document_counter.increment(labels={"doc_type": workflow_type})
        
        try:
            # Create processing context
            context = {
                "document_path": document_path,
                "workflow_type": workflow_type,
                "user_id": "batch_processor",
                "session_id": f"batch_{batch_id}_{uuid.uuid4().hex[:8]}",
                "batch_id": batch_id
            }
            
            # Get the orchestrator agent
            orchestrator = agents.get("master_orchestrator") or agents.get("enhanced_document_processor")
            
            if not orchestrator:
                raise ValueError("No document processing agent available in agent dictionary")
                
            # Process the document
            result = orchestrator.process_document(context)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            self.document_histogram.update(processing_time)
            
            result.update({
                "document_path": document_path,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "batch_id": batch_id
            })
            
            # Update batch tracking
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                batch.mark_document_complete(document_path, result)
                
            self.logger.info(f"Processed document {doc_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {doc_id}: {e}")
            
            # Record error in metrics
            self.metrics.record(MetricPoint(
                name="document_processing_errors",
                value=1,
                labels={"batch_id": batch_id, "error_type": type(e).__name__},
                metric_type=MetricType.COUNTER
            ))
            
            # Update batch tracking
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                batch.mark_document_failed(document_path, str(e))
            
            # Return error information
            return {
                "document_path": document_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "batch_id": batch_id
            }
    
    @time_function("process_document_batch")
    def process_batch(
        self,
        agents: Dict[str, Any],
        config: Dict[str, Any],
        batch_dir: str,
        workflow_type: str = "optimized",
        adaptive_workers: bool = True,
        output_dir: Optional[str] = None,
        batch_id: Optional[str] = None,
        resume_batch: bool = False
    ) -> Dict[str, Any]:
        """
        Process a batch of documents with optimized parallel execution,
        automatic resource management, and recovery mechanisms.
        
        Args:
            agents: Dictionary of initialized agents
            config: Configuration dictionary
            batch_dir: Directory containing documents to process
            workflow_type: Type of workflow to use
            adaptive_workers: Whether to adapt worker count based on system resources
            output_dir: Directory for output results (defaults to batch_dir/results)
            batch_id: Optional ID for the batch (auto-generated if not provided)
            resume_batch: Whether to attempt resuming a previous batch
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        # Validate batch directory
        if not os.path.isdir(batch_dir):
            raise ValueError(f"Batch directory not found: {batch_dir}")
        
        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(batch_dir, "results")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate batch ID if not provided
        batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        
        # Initialize resource monitor
        self.resource_monitor.start()
        
        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._create_signal_handler(batch_id))
        
        # Start timer for batch processing
        with self.batch_timer:
            try:
                # Load or create batch
                if resume_batch:
                    loaded_data = self._load_checkpoint(batch_id)
                    if loaded_data:
                        self.logger.info(f"Resuming batch {batch_id}")
                        # TODO: Resume from checkpoint
                
                # Create new batch
                documents = self._prepare_documents(batch_dir)
                
                if not documents:
                    self.logger.warning(f"No supported documents found in {batch_dir}")
                    return {
                        "batch_id": batch_id,
                        "total": 0,
                        "success": 0,
                        "failed": 0,
                        "documents": []
                    }
                
                # Create batch object
                batch = DocumentBatch(batch_id=batch_id)
                
                # Register batch
                with self.batch_lock:
                    self.active_batches[batch_id] = batch
                
                # Add documents to batch
                for doc_path in documents:
                    batch.add_document(doc_path)
                
                # Create processing chunks
                batch.create_chunks()
                
                # Process documents with thread pool
                success_count = 0
                failed_count = 0
                
                # Set initial number of workers
                current_workers = (
                    self.resource_monitor.get_optimal_workers(self.min_workers, self.max_workers)
                    if adaptive_workers else self.max_workers
                )
                
                self.logger.info(f"Starting batch processing with {current_workers} workers")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
                    # Create document processing tasks
                    future_to_document = {}
                    
                    for doc_path in documents:
                        future = executor.submit(
                            self._process_document, 
                            agents, config, doc_path, workflow_type, batch_id
                        )
                        future_to_document[future] = doc_path
                    
                    # Process results as they complete
                    completed = 0
                    checkpoint_counter = 0
                    
                    for future in concurrent.futures.as_completed(future_to_document):
                        document_path = future_to_document[future]
                        
                        try:
                            result = future.result()
                            
                            # Save individual result to output directory
                            document_filename = os.path.basename(document_path)
                            result_filename = f"{os.path.splitext(document_filename)[0]}_result.json"
                            result_path = os.path.join(output_dir, result_filename)
                            
                            with open(result_path, 'w') as f:
                                json.dump(result, f, indent=2)
                            
                            if result.get("status") == "success":
                                success_count += 1
                            else:
                                failed_count += 1
                                
                            # Update checkpoint counter
                            completed += 1
                            checkpoint_counter += 1
                            
                            # Create checkpoint periodically
                            if checkpoint_counter >= self.checkpoint_interval:
                                self._create_checkpoint(batch)
                                checkpoint_counter = 0
                                
                            # Adjust worker count if needed and if we're not almost done
                            if (adaptive_workers and 
                                completed < len(documents) * 0.8 and 
                                completed % 5 == 0):
                                
                                optimal_workers = self.resource_monitor.get_optimal_workers(
                                    self.min_workers, self.max_workers
                                )
                                
                                if optimal_workers != current_workers:
                                    self.logger.info(f"Adjusting worker count: {current_workers} -> {optimal_workers}")
                                    current_workers = optimal_workers
                                
                            # Show progress periodically
                            if completed % 10 == 0 or completed == len(documents):
                                progress = batch.get_progress()
                                self.logger.info(
                                    f"Batch progress: {progress['progress_percent']:.1f}% complete, "
                                    f"{progress['completed']} succeeded, {progress['failed']} failed"
                                )
                                
                        except Exception as e:
                            self.logger.error(f"Error getting result for {document_path}: {e}")
                            failed_count += 1
                
                # Save final checkpoint
                self._create_checkpoint(batch)
                
                # Mark batch as complete
                batch.mark_complete()
                
                # Save batch summary
                summary = {
                    "batch_id": batch_id,
                    "total": len(documents),
                    "success": success_count,
                    "failed": failed_count,
                    "timestamp": datetime.now().isoformat(),
                    "batch_dir": batch_dir,
                    "workflow_type": workflow_type,
                    "processing_stats": batch.to_dict()
                }
                
                summary_path = os.path.join(output_dir, f"batch_summary_{batch_id}.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.logger.info(f"Batch processing complete. Processed {len(documents)} documents, "
                            f"{success_count} successful, {failed_count} failed.")
                
                return summary
                
            finally:
                # Stop resource monitor
                self.resource_monitor.stop()
                
                # Remove batch from active batches
                with self.batch_lock:
                    if batch_id in self.active_batches:
                        del self.active_batches[batch_id]
                
                # Restore original signal handler
                signal.signal(signal.SIGINT, original_sigint_handler)
    
    def _create_signal_handler(self, batch_id: str) -> Callable:
        """Create a signal handler for graceful shutdown."""
        def signal_handler(sig, frame):
            self.logger.info(f"Shutdown requested, creating checkpoint for batch {batch_id}...")
            self._shutdown_requested = True
            
            # Create checkpoint
            if batch_id in self.active_batches:
                self._create_checkpoint(self.active_batches[batch_id])
                
            self.logger.info("Checkpoint created, shutting down...")
            sys.exit(0)
            
        return signal_handler
        
    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for an active batch."""
        with self.batch_lock:
            if batch_id in self.active_batches:
                return self.active_batches[batch_id].get_progress()
            return None
    
    def list_active_batches(self) -> List[str]:
        """Get a list of active batch IDs."""
        with self.batch_lock:
            return list(self.active_batches.keys())
    
    def list_recoverable_batches(self) -> List[Dict[str, Any]]:
        """Get a list of batches that can be recovered from checkpoints."""
        try:
            checkpoint_files = list(self.recovery_dir.glob("batch_*.json"))
            recoverable = []
            
            for cp_file in checkpoint_files:
                try:
                    with open(cp_file, "r") as f:
                        data = json.load(f)
                        recoverable.append({
                            "batch_id": data.get("batch_id", cp_file.stem),
                            "timestamp": data.get("timestamp"),
                            "timestamp_iso": datetime.fromtimestamp(data.get("timestamp", 0)).isoformat(),
                            "processed_count": len(data.get("processed", [])),
                            "failed_count": len(data.get("failed", [])),
                            "total_count": len(data.get("documents", [])),
                            "checkpoint_file": str(cp_file)
                        })
                except Exception as e:
                    self.logger.warning(f"Could not parse checkpoint file {cp_file}: {e}")
                    
            return recoverable
            
        except Exception as e:
            self.logger.error(f"Error listing recoverable batches: {e}")
            return []


# Make batch processor available at module level
def create_batch_processor(**kwargs) -> OptimizedBatchProcessor:
    """Create and return an optimized batch processor with the given config."""
    return OptimizedBatchProcessor(**kwargs)
