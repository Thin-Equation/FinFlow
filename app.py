#!/usr/bin/env python3
"""
FinFlow: Financial document processing and analysis platform.
Main application entry point.

This version includes:
- Comprehensive error handling
- Recovery mechanisms
- Performance monitoring
- System health checks
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Dict, Any, cast, List

# Import system initialization
from initialize_agents import initialize_system
from utils.logging_config import configure_logging, TraceContext
from utils.system_init import initialize_robustness_systems
from utils.metrics import (
    AppMetricsCollector,
    time_function, count_invocations
)
from utils.recovery import (
    RecoveryManager, create_workflow_checkpointer
)
from utils.error_handling import (
    ErrorManager, FinFlowError, ErrorSeverity, DocumentProcessingError, 
    ErrorBoundary
)

# Import configuration
from config.config_loader import load_config

# Import for type checking
from agents.master_orchestrator import MasterOrchestratorAgent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FinFlow: Financial document processing platform")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--env", help="Environment (development, staging, production)", default="development")
    parser.add_argument("--log-level", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--document", help="Path to document to process")
    parser.add_argument("--recovery", help="Enable automatic recovery", action="store_true")
    parser.add_argument("--monitoring", help="Enable performance monitoring", action="store_true", default=True)
    parser.add_argument("--profiling", help="Enable CPU and memory profiling", action="store_true")
    parser.add_argument("--batch", help="Process multiple documents in batch mode", action="store_true")
    parser.add_argument("--batch-dir", help="Directory containing documents for batch processing")
    parser.add_argument("--health-port", help="Port for health API", type=int, default=8888)
    return parser.parse_args()


@time_function("document_processing_time")
@count_invocations("document_processing_count")
def process_document(document_path: str, agents: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document using the agent system with comprehensive error handling and metrics.
    
    Args:
        document_path: Path to the document to process
        agents: Dictionary of initialized agents
        config: Application configuration
        
    Returns:
        Processing results
    """
    # Generate a unique workflow ID for tracking
    import uuid
    workflow_id = f"doc-{uuid.uuid4().hex[:8]}"
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing document: {document_path} (workflow: {workflow_id})")
    
    # Get metrics collector
    metrics = AppMetricsCollector.get_instance()
    
    # Get error manager
    error_manager = ErrorManager.get_instance()
    
    # Create processing context
    context = {
        "document_path": document_path,
        "user_id": "system",  # In a real app, this would be the actual user
        "session_id": "test_session",  # In a real app, this would be a real session ID
        "workflow_type": "document_processing",
        "workflow_id": workflow_id,
        "start_time": time.time()
    }
    
    # Create a workflow checkpointer for recovery
    checkpointer = create_workflow_checkpointer("document_processing", context)
    
    # Create trace context for distributed tracing
    with TraceContext(workflow_id) as trace:
        try:
            # Start metrics timer
            doc_timer = metrics.track_document("general")
            
            # Checkpoint the start
            checkpointer.checkpoint("start", context)
            
            # Get document extension for metrics
            _, ext = os.path.splitext(document_path)
            doc_type = ext[1:] if ext else "unknown"
            
            # Use the master orchestrator to process the document
            master_orchestrator = cast(MasterOrchestratorAgent, agents["master_orchestrator"])
            
            # Create error boundary
            boundary = ErrorBoundary(
                f"document_processing_{workflow_id}", 
                fallback_value={"status": "error", "error": "Processing failed"},
                retries=2,
                error_manager=error_manager
            )
            
            # Process within error boundary
            with doc_timer, boundary:
                # Start the actual processing
                result = master_orchestrator.process_document(context)
                
                # Add timing information
                end_time = time.time()
                result["processing_time"] = end_time - context["start_time"]
                result["workflow_id"] = workflow_id
                
                # Checkpoint completion
                checkpointer.checkpoint("complete", {"result": result})
                
                logger.info(
                    f"Document processing completed in {result['processing_time']:.2f}s "
                    f"with status: {result.get('status', 'unknown')}"
                )
                
                # Complete the workflow
                checkpointer.complete(
                    success=result.get("status") == "success",
                    final_state={"result": result}
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            
            # Convert to FinFlowError if needed
            if not isinstance(e, FinFlowError):
                e = DocumentProcessingError(
                    f"Document processing failed: {str(e)}",
                    document_id=os.path.basename(document_path),
                    severity=ErrorSeverity.HIGH,
                    cause=e
                )
            
            # Handle through error manager
            error_manager.handle_error(e)
            
            # Log to metrics
            metrics.track_error("document_processing", e.__class__.__name__)
            
            # Complete the workflow with failure
            checkpointer.complete(
                success=False,
                final_state={
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }
            )
            
            # Return error result
            return {
                "status": "error",
                "error": str(e),
                "error_type": e.__class__.__name__,
                "workflow_id": workflow_id
            }


def batch_process_documents(directory: str, agents: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process multiple documents in batch mode.
    
    Args:
        directory: Directory containing documents to process
        agents: Dictionary of initialized agents
        config: Application configuration
        
    Returns:
        List of processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing from directory: {directory}")
    
    if not os.path.isdir(directory):
        logger.error(f"Batch directory not found: {directory}")
        return [{"status": "error", "error": f"Batch directory not found: {directory}"}]
    
    # Get list of documents
    valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.doc', '.docx']
    documents = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions:
                documents.append(os.path.join(root, file))
    
    if not documents:
        logger.warning(f"No valid documents found in {directory}")
        return [{"status": "warning", "message": f"No valid documents found in {directory}"}]
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Process each document
    results = []
    for doc in documents:
        try:
            result = process_document(doc, agents, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch processing for {doc}: {e}")
            results.append({
                "document": doc,
                "status": "error",
                "error": str(e)
            })
    
    # Generate batch summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(
        f"Batch processing complete: {len(results)} documents processed, "
        f"{success_count} succeeded, {error_count} failed"
    )
    
    return results


def start_health_api(port: int, robustness_systems: Dict[str, Any]) -> None:
    """
    Start a simple health API on a background thread.
    
    Args:
        port: Port number to use
        robustness_systems: Dictionary of initialized robustness systems
    """
    try:
        import http.server
        import socketserver
        import json
        
        health_system = robustness_systems["metrics"]["health_system"]
        metrics_registry = robustness_systems["metrics"]["registry"]
        
        logger = logging.getLogger("finflow.health_api")
        logger.info(f"Starting health API on port {port}")
        
        class HealthHandler(http.server.BaseHTTPRequestHandler):
            """Handler for health check requests."""
            
            def _set_headers(self, status_code=200, content_type="application/json"):
                """Set response headers."""
                self.send_response(status_code)
                self.send_header("Content-type", content_type)
                self.end_headers()
                
            def do_GET(self):
                """Handle GET requests."""
                if self.path == "/health" or self.path == "/":
                    # Get health status
                    health_report = health_system.get_health_report()
                    status_code = 200
                    if health_report["status"] == "unhealthy":
                        status_code = 500
                    elif health_report["status"] == "degraded":
                        status_code = 429
                        
                    self._set_headers(status_code)
                    self.wfile.write(json.dumps(health_report).encode())
                    
                elif self.path == "/metrics":
                    # Get metrics
                    metrics_data = metrics_registry.get_metrics_data(
                        since=time.time() - 300  # Last 5 minutes
                    )
                    self._set_headers()
                    self.wfile.write(json.dumps(metrics_data).encode())
                    
                else:
                    self._set_headers(404)
                    self.wfile.write(json.dumps({"error": "Not found"}).encode())
            
            def log_message(self, format, *args):
                """Override default logging."""
                logger.debug(f"Health API request: {format % args}")
        
        def run_server():
            """Run the health API server."""
            with socketserver.TCPServer(("0.0.0.0", port), HealthHandler) as httpd:
                logger.info(f"Health API serving at port {port}")
                httpd.serve_forever()
        
        # Start server on a thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
    except ImportError:
        logger = logging.getLogger("finflow.health_api")
        logger.warning("Could not start health API due to missing dependencies")
    except Exception as e:
        logger = logging.getLogger("finflow.health_api")
        logger.error(f"Failed to start health API: {e}")


def enable_profiling() -> None:
    """
    Enable CPU and memory profiling.
    
    This creates a profile of the application's CPU and memory usage
    that can be analyzed to find performance bottlenecks.
    """
    try:
        import yappi
        import psutil
        import threading
        
        logger = logging.getLogger("finflow.profiling")
        logger.info("Starting CPU and memory profiling")
        
        # Start CPU profiling
        yappi.set_clock_type("cpu")  # Use CPU time
        yappi.start()
        
        # Set up memory tracking
        process = psutil.Process()
        
        def memory_tracker():
            """Track memory usage over time."""
            memory_log = []
            while True:
                try:
                    mem_info = process.memory_info()
                    memory_log.append({
                        "timestamp": time.time(),
                        "rss_mb": mem_info.rss / (1024 * 1024),
                        "vms_mb": mem_info.vms / (1024 * 1024)
                    })
                    
                    # Log every 5 minutes
                    if len(memory_log) % 30 == 0:
                        recent_mem = memory_log[-1]["rss_mb"]
                        logger.info(f"Current memory usage: {recent_mem:.1f} MB")
                        
                        # Write to file periodically
                        with open("memory_profile.json", "w") as f:
                            import json
                            json.dump(memory_log, f)
                except Exception as e:
                    logger.error(f"Error in memory tracker: {e}")
                
                time.sleep(10)  # Sample every 10 seconds
        
        # Start memory tracking thread
        mem_thread = threading.Thread(target=memory_tracker, daemon=True)
        mem_thread.start()
        
        # Register exit handler to save profiling data
        import atexit
        
        def save_profiling_data():
            """Save profiling data on exit."""
            logger.info("Saving profiling information")
            try:
                # Save CPU profiling info
                yappi.stop()
                
                # Retrieve statistics
                stats = yappi.get_func_stats()
                stats.save("cpu_profile.prof", type="pstat")
                
                # Also save in readable format
                with open("cpu_profile.txt", "w") as f:
                    stats.print_all(out=f)
                    
                logger.info("Profiling data saved to cpu_profile.prof and cpu_profile.txt")
            except Exception as e:
                logger.error(f"Error saving profiling data: {e}")
                
        atexit.register(save_profiling_data)
        
    except ImportError:
        logger = logging.getLogger("finflow.profiling")
        logger.warning("Could not enable profiling due to missing dependencies")
        logger.warning("Install with: pip install yappi psutil")
    except Exception as e:
        logger = logging.getLogger("finflow.profiling")
        logger.error(f"Failed to enable profiling: {e}")


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set environment variable for config
    os.environ.setdefault('FINFLOW_ENV', args.env)
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Catch uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        # Ignore KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    # Load configuration
    config = load_config()
    logger.info(f"Starting FinFlow in {config['environment']} mode")
    
    try:
        # Initialize robustness systems first
        logger.info("Initializing robustness systems")
        robustness_systems = initialize_robustness_systems(config)
        logger.info("Robustness systems initialized")
        
        # Start health API if enabled
        if args.monitoring and args.health_port:
            start_health_api(args.health_port, robustness_systems)
        
        # Enable profiling if requested
        if args.profiling:
            enable_profiling()
        
        # Start automatic recovery if requested
        if args.recovery:
            logger.info("Enabling automatic recovery")
            recovery_manager = RecoveryManager.get_instance()
            recovery_manager.automatic_recovery_loop(interval=120.0)  # Check every 2 minutes
        
        # Initialize agent system
        logger.info("Initializing agent system")
        agents = initialize_system(config)
        logger.info("Agent system initialized successfully")
        
        # Process document or batch
        if args.batch and args.batch_dir:
            # Batch mode
            results = batch_process_documents(args.batch_dir, agents, config)
            logger.info(f"Batch processing completed with {len(results)} results")
        elif args.document:
            # Single document mode
            result = process_document(args.document, agents, config)
            logging.info(f"Document processing result: {result}")
        else:
            logger.info("No document or batch directory specified for processing")
            logger.info("Run with --document path/to/document to process a document")
            logger.info("Run with --batch --batch-dir path/to/dir to process multiple documents")
        
        logger.info("FinFlow initialization complete")
        
        # Keep main thread alive if we have background threads
        if args.monitoring or args.recovery:
            logger.info("Background services running, press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Exiting by user request")
                
    except Exception as e:
        logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
