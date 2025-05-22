"""
FinFlow API Server

This module provides a FastAPI server implementation for the FinFlow platform
with enhanced metrics collection, health monitoring, and batch processing endpoints.
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Body, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import enhanced utilities
from utils.health_check import HealthCheckManager, HealthStatus
from utils.metrics import AppMetricsCollector, MetricPoint, MetricType
from utils.recovery_manager import RecoveryManager

# Import the optimized batch processor
from batch.optimized_batch import OptimizedBatchProcessor

logger = logging.getLogger(__name__)


class ProcessRequest(BaseModel):
    """Request model for document processing."""
    workflow_type: str = Field(default="standard", description="Type of workflow to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class BatchProcessRequest(BaseModel):
    """Request model for batch document processing."""
    directory_path: str = Field(..., description="Directory containing documents to process")
    workflow_type: str = Field(default="optimized", description="Type of workflow to use")
    parallel: bool = Field(default=True, description="Enable parallel processing")
    adaptive_workers: bool = Field(default=True, description="Adjust workers based on system load")
    max_workers: int = Field(default=8, description="Maximum worker threads")
    output_directory: Optional[str] = Field(default=None, description="Output directory (defaults to input_dir/results)")
    batch_id: Optional[str] = Field(default=None, description="Optional batch ID")


class DocumentResponse(BaseModel):
    """Response model for document processing."""
    document_id: str
    status: str
    result: Dict[str, Any]
    processing_time: float
    timestamp: str


class BatchStatusResponse(BaseModel):
    """Response model for batch processing status."""
    batch_id: str
    status: str
    total: int
    processed: int
    failed: int
    progress_percent: float
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float]
    timestamp: str


def create_app(agents: Dict[str, Any], config: Dict[str, Any]) -> FastAPI:
    """
    Create a FastAPI application for the FinFlow platform.
    
    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="FinFlow API",
        description="Financial document processing and analysis API",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize core components
    metrics_collector = AppMetricsCollector.get_instance()
    health_manager = HealthCheckManager.get_instance(config.get("health_check", {}))
    recovery_manager = RecoveryManager()
    batch_processor = OptimizedBatchProcessor()
    
    # Start health check system
    health_manager.start()
    
    # Static files for API docs
    # app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next) -> Response:
        """Middleware to track request metrics."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        # Track request
        metrics_collector.record(MetricPoint(
            name="http_requests_total",
            value=1,
            labels={"method": request.method, "path": request.url.path},
            metric_type=MetricType.COUNTER
        ))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Track response
            metrics_collector.record(MetricPoint(
                name="http_response_time_ms",
                value=duration_ms,
                labels={"method": request.method, "path": request.url.path, "status": response.status_code},
                metric_type=MetricType.HISTOGRAM
            ))
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            # Log error
            logger.error(f"Request error: {e}")
            
            # Track error
            metrics_collector.record(MetricPoint(
                name="http_errors_total",
                value=1, 
                labels={"method": request.method, "path": request.url.path, "error": str(e)},
                metric_type=MetricType.COUNTER
            ))
            
            # Pass through the exception
            raise
    
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {"message": "FinFlow API", "version": "1.0.0"}
    
    @app.get("/status")
    async def status():
        """System status endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "environment": config.get("environment", "unknown"),
            "version": "1.0.0"
        }
    
    @app.post("/process", response_model=DocumentResponse)
    async def process_document(
        request: ProcessRequest = Body(...),
        background_tasks: BackgroundTasks = None,
        file: UploadFile = File(...)
    ):
        """
        Process a document using the agent system.
        
        Args:
            request: Processing request with workflow and options
            background_tasks: FastAPI background tasks
            file: Uploaded document file
            
        Returns:
            DocumentResponse: Processing result
        """
        start_time = datetime.now()
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        try:
            # Save uploaded file to temporary location
            file_path = f"/tmp/finflow_{start_time.timestamp()}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Document saved to temporary location: {file_path}")
            
            # Create processing context
            context = {
                "document_path": file_path,
                "document_id": document_id,
                "workflow_type": request.workflow_type,
                "options": request.options,
                "user_id": "api_user",  # Would be from authentication in production
                "session_id": f"api_{start_time.timestamp()}",
            }
            
            # Track document submission
            metrics_collector.record(MetricPoint(
                name="documents_submitted",
                value=1,
                labels={"workflow_type": request.workflow_type, "api_endpoint": "process"},
                metric_type=MetricType.COUNTER
            ))
            
            # Get the appropriate orchestrator based on workflow type
            if request.workflow_type == "optimized" and "enhanced_document_processor" in agents:
                orchestrator = agents["enhanced_document_processor"]
            else:
                orchestrator = agents["master_orchestrator"]
            
            # Process the document
            result = orchestrator.process_document(context)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Track processing time
            metrics_collector.record(MetricPoint(
                name="document_processing_time",
                value=processing_time,
                labels={"workflow_type": request.workflow_type},
                metric_type=MetricType.HISTOGRAM
            ))
            
            # If processing was successful, create background task to clean up
            if background_tasks:
                background_tasks.add_task(lambda: os.remove(file_path))
            
            # Create response
            response = {
                "document_id": result.get("document_id", document_id),
                "status": result.get("status", "unknown"),
                "result": result,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            
            # Track processing error
            metrics_collector.record(MetricPoint(
                name="document_processing_errors",
                value=1,
                labels={"workflow_type": request.workflow_type, "error_type": type(e).__name__},
                metric_type=MetricType.COUNTER
            ))
            
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    @app.post("/batch/process", response_model=Dict[str, Any])
    async def process_batch(request: BatchProcessRequest = Body(...)):
        """
        Start batch processing of documents in a directory.
        
        Args:
            request: Batch processing request with options
            
        Returns:
            Dict: Batch processing info with batch ID
        """
        try:
            # Validate directory exists
            if not os.path.exists(request.directory_path):
                raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
            
            # Start batch processing in background thread
            batch_id = request.batch_id or f"batch_{uuid.uuid4().hex[:8]}"
            
            def run_batch_processing():
                try:
                    batch_processor.process_batch(
                        agents=agents,
                        config=config,
                        batch_dir=request.directory_path,
                        workflow_type=request.workflow_type,
                        adaptive_workers=request.adaptive_workers,
                        output_dir=request.output_directory,
                        batch_id=batch_id
                    )
                except Exception as e:
                    logger.error(f"Error in batch processing thread: {e}")
            
            # Start background thread
            import threading
            batch_thread = threading.Thread(
                target=run_batch_processing,
                name=f"BatchProcessor-{batch_id}",
                daemon=True
            )
            batch_thread.start()
            
            # Return batch information
            return {
                "batch_id": batch_id,
                "status": "started",
                "timestamp": datetime.now().isoformat(),
                "message": f"Batch processing started with ID {batch_id}"
            }
            
        except HTTPException:
            # Pass through HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error starting batch process: {e}")
            raise HTTPException(status_code=500, detail=f"Error starting batch process: {str(e)}")
    
    @app.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
    async def get_batch_status(batch_id: str):
        """
        Get the status of a batch processing job.
        
        Args:
            batch_id: ID of the batch processing job
            
        Returns:
            BatchStatusResponse: Current batch status
        """
        try:
            # Get progress from batch processor
            progress = batch_processor.get_batch_progress(batch_id)
            
            if not progress:
                raise HTTPException(status_code=404, detail=f"Batch with ID {batch_id} not found or completed")
            
            # Transform to response model
            return {
                "batch_id": batch_id,
                "status": "in_progress",
                "total": progress["total_documents"],
                "processed": progress["completed"],
                "failed": progress["failed"],
                "progress_percent": progress["progress_percent"],
                "elapsed_seconds": progress["elapsed_seconds"],
                "estimated_remaining_seconds": progress["estimated_remaining_seconds"],
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            # Pass through HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error getting batch status: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting batch status: {str(e)}")
    
    @app.get("/batch/list/active")
    async def list_active_batches():
        """
        List all active batch processing jobs.
        
        Returns:
            Dict: List of active batch IDs
        """
        try:
            active_batches = batch_processor.list_active_batches()
            return {
                "active_batches": active_batches,
                "count": len(active_batches)
            }
        except Exception as e:
            logger.error(f"Error listing active batches: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing active batches: {str(e)}")
    
    @app.get("/batch/list/recoverable")
    async def list_recoverable_batches():
        """
        List all recoverable batch processing jobs.
        
        Returns:
            Dict: List of batches that can be recovered
        """
        try:
            recoverable = batch_processor.list_recoverable_batches()
            return {
                "recoverable_batches": recoverable,
                "count": len(recoverable)
            }
        except Exception as e:
            logger.error(f"Error listing recoverable batches: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing recoverable batches: {str(e)}")
    
    @app.post("/batch/{batch_id}/resume")
    async def resume_batch(batch_id: str):
        """
        Resume a previously interrupted batch processing job.
        
        Args:
            batch_id: ID of the batch to resume
            
        Returns:
            Dict: Result of resume operation
        """
        try:
            # Check if batch is recoverable
            recoverable = batch_processor.list_recoverable_batches()
            batch_exists = any(b["batch_id"] == batch_id for b in recoverable)
            
            if not batch_exists:
                raise HTTPException(status_code=404, detail=f"Recoverable batch with ID {batch_id} not found")
            
            # Start background thread to resume batch
            def resume_batch_processing():
                try:
                    # The resume process needs to determine the batch directory from checkpoints
                    # This is a simplified implementation
                    for batch in recoverable:
                        if batch["batch_id"] == batch_id:
                            # In a real implementation, this information should come from the checkpoint
                            batch_dir = "/path/from/checkpoint"  # Placeholder
                            batch_processor.process_batch(
                                agents=agents,
                                config=config,
                                batch_dir=batch_dir,
                                batch_id=batch_id,
                                resume_batch=True
                            )
                            break
                except Exception as e:
                    logger.error(f"Error in batch resume thread: {e}")
            
            # Start background thread
            import threading
            resume_thread = threading.Thread(
                target=resume_batch_processing,
                name=f"BatchResume-{batch_id}",
                daemon=True
            )
            resume_thread.start()
            
            return {
                "batch_id": batch_id,
                "status": "resuming",
                "timestamp": datetime.now().isoformat(),
                "message": f"Batch processing resumed for batch ID {batch_id}"
            }
            
        except HTTPException:
            # Pass through HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error resuming batch: {e}")
            raise HTTPException(status_code=500, detail=f"Error resuming batch: {str(e)}")
    
    @app.get("/workflows")
    async def list_workflows():
        """List available workflows."""
        try:
            # This would pull from a workflow registry in production
            workflows = [
                {"id": "standard", "name": "Standard Processing", "description": "Standard document processing workflow"},
                {"id": "optimized", "name": "Optimized Processing", "description": "Performance-optimized processing workflow"},
                {"id": "invoice", "name": "Invoice Processing", "description": "Invoice-specialized processing workflow"},
                {"id": "receipt", "name": "Receipt Processing", "description": "Receipt-specialized processing workflow"},
            ]
            return {"workflows": workflows}
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing workflows: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """
        Enhanced health check endpoint with detailed system status.
        
        Returns:
            Dict: Detailed health status
        """
        # Get full health report
        health_report = health_manager.get_full_report()
        
        # Determine overall status
        overall_status = health_report.get("status", HealthStatus.UNKNOWN)
        
        # Set response status code based on health
        status_code = 200
        if overall_status == HealthStatus.DEGRADED:
            status_code = 200  # Still functioning but with issues
        elif overall_status == HealthStatus.UNHEALTHY:
            status_code = 503  # Service unavailable
            
        return JSONResponse(
            content=health_report,
            status_code=status_code
        )
    
    @app.get("/metrics")
    async def get_metrics(format: str = Query("json", description="Output format (json or prometheus)")):
        """
        Get system metrics in requested format.
        
        Args:
            format: Output format (json or prometheus)
            
        Returns:
            Response with metrics data
        """
        try:
            if format.lower() == "prometheus":
                metrics_data = metrics_collector.export_prometheus()
                return Response(content=metrics_data, media_type="text/plain")
            else:
                metrics_data = metrics_collector.export_json()
                return JSONResponse(content=json.loads(metrics_data))
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting metrics: {str(e)}")
    
    @app.get("/diagnostics")
    async def get_diagnostics():
        """
        Get system diagnostics information.
        
        Returns:
            Dict: System diagnostic information
        """
        try:
            import psutil
            
            # Collect system info
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get process info
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads": len(process.threads()),
                "uptime_seconds": time.time() - process.create_time()
            }
            
            # Get active batch info
            active_batches = batch_processor.list_active_batches()
            
            # Combine diagnostics
            diagnostics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_total_mb": memory.total / (1024 * 1024),
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "memory_percent": memory.percent,
                    "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                    "disk_percent": disk.percent
                },
                "process": process_info,
                "application": {
                    "active_batches": len(active_batches)
                }
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting diagnostics: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting diagnostics: {str(e)}")
        
    return app
