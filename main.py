#!/usr/bin/env python3
"""
FinFlow: Financial document processing and analysis platform.
Main application entry point with support for various run modes.

Usage:
    python main.py [options]
    
Run modes:
    - server: Run as a web server (default)
    - cli: Run as a command line tool
    - batch: Run in batch processing mode
    - workflow: Run a specific workflow
    - monitor: Run in monitoring mode (health checks and metrics)
    - optimized: Run with enhanced document processor
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
import importlib.util
from datetime import datetime
import time
import signal

# Import configuration
from config.config_loader import load_config

# Configure logging
from utils.logging_config import configure_logging

# Import system initialization
from initialize_agents import initialize_system

# Import metrics and health monitoring
from utils.metrics import AppMetricsCollector
from utils.health_check import HealthCheckManager
from utils.error_handling import ErrorManager

# Version
__version__ = "1.1.0"

# Run modes
RUN_MODE_SERVER = "server"
RUN_MODE_CLI = "cli"
RUN_MODE_BATCH = "batch"
RUN_MODE_WORKFLOW = "workflow"
RUN_MODE_MONITOR = "monitor"
RUN_MODE_OPTIMIZED = "optimized"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FinFlow: Financial document processing platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--env", help="Environment (development, staging, production)", default="development")
    parser.add_argument("--log-level", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--document", help="Path to document to process")
    parser.add_argument("--mode", help="Run mode", 
                        choices=[RUN_MODE_SERVER, RUN_MODE_CLI, RUN_MODE_BATCH, RUN_MODE_WORKFLOW, RUN_MODE_MONITOR, RUN_MODE_OPTIMIZED], 
                        default=RUN_MODE_SERVER)
    parser.add_argument("--workflow", help="Workflow name to run (for workflow mode)")
    parser.add_argument("--batch-dir", help="Directory with documents to process in batch mode")
    parser.add_argument("--port", help="Port for server mode", type=int, default=8000)
    parser.add_argument("--host", help="Host for server mode", default="0.0.0.0")
    parser.add_argument("--parallel", help="Enable parallel execution", action="store_true")
    parser.add_argument("--max-workers", help="Maximum number of parallel workers", type=int, default=4)
    parser.add_argument("--with-recovery", help="Enable recovery mechanisms", action="store_true")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    return parser.parse_args()


def run_server_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace, 
                    monitoring: Dict[str, Any]) -> None:
    """Run in server mode."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting FinFlow server on {args.host}:{args.port}")
    
    try:
        # Check if FastAPI is available
        if importlib.util.find_spec("fastapi") is None:
            logger.error("FastAPI is required for server mode. Install with: pip install fastapi uvicorn")
            sys.exit(1)
            
        # Import here to avoid dependency issues
        from server.app import create_app
        import uvicorn
        
        # Create FastAPI application with monitoring
        app = create_app(
            agents=agents, 
            config=config, 
            metrics=monitoring["metrics"],
            health_manager=monitoring["health_manager"]
        )
        
        # Run server
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
        
    except ImportError as e:
        logger.error(f"Failed to start server: {e}")
        logger.error("Make sure server dependencies are installed: pip install -r requirements-server.txt")
        monitoring["error_manager"].report_error(e)
        sys.exit(1)


def run_cli_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace,
                monitoring: Dict[str, Any]) -> None:
    """Run in command line interface mode."""
    from cli.cli_app import run_cli
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FinFlow CLI")
    
    try:
        run_cli(agents=agents, config=config)
    except Exception as e:
        logger.error(f"Error in CLI mode: {e}")
        monitoring["error_manager"].report_error(e)
        sys.exit(1)


def run_batch_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace,
                  monitoring: Dict[str, Any]) -> None:
    """Run in batch processing mode."""
    logger = logging.getLogger(__name__)
    
    if not args.batch_dir:
        logger.error("Batch directory (--batch-dir) is required for batch mode")
        sys.exit(1)
        
    if not os.path.isdir(args.batch_dir):
        logger.error(f"Batch directory not found: {args.batch_dir}")
        sys.exit(1)
    
    try:
        # Use optimized batch processor if parallel flag is set
        if args.parallel:
            from batch.optimized_batch import process_batch_parallel
            
            logger.info(f"Starting optimized parallel batch processing from directory: {args.batch_dir}")
            results = process_batch_parallel(
                agents=agents, 
                config=config, 
                batch_dir=args.batch_dir,
                max_workers=args.max_workers
            )
        else:
            from batch.batch_processor import process_batch
            
            logger.info(f"Starting batch processing from directory: {args.batch_dir}")
            results = process_batch(
                agents=agents, 
                config=config, 
                batch_dir=args.batch_dir
            )
        
        logger.info(
            f"Batch processing completed. Processed {results['total']} documents, "
            f"{results['success']} success, {results['failed']} failed."
        )
        
        # Record batch metrics
        monitoring["metrics"].counter("documents_processed_total").increment(value=results["total"])
        monitoring["metrics"].counter("documents_processed_success").increment(value=results["success"])
        monitoring["metrics"].counter("documents_processed_failed").increment(value=results["failed"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        monitoring["error_manager"].report_error(e)
        sys.exit(1)
    

def run_workflow_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace,
                     monitoring: Dict[str, Any]) -> None:
    """Run a specific workflow."""
    logger = logging.getLogger(__name__)
    
    if not args.workflow:
        logger.error("Workflow name (--workflow) is required for workflow mode")
        sys.exit(1)
    
    try:
        # Use optimized workflow runner with parallelism and recovery
        if args.parallel or args.with_recovery:
            from workflow.optimized_runner import run_workflow, run_parallel_workflow, run_recoverable_workflow
            
            if args.with_recovery:
                logger.info(f"Running recoverable workflow: {args.workflow}")
                result = run_recoverable_workflow(
                    workflow_name=args.workflow,
                    agents=agents,
                    config=config,
                    document_path=args.document
                )
            elif args.parallel:
                logger.info(f"Running parallel workflow: {args.workflow}")
                result = run_parallel_workflow(
                    workflow_name=args.workflow,
                    agents=agents,
                    config=config,
                    document_path=args.document
                )
            else:
                logger.info(f"Running optimized workflow: {args.workflow}")
                result = run_workflow(
                    workflow_name=args.workflow,
                    agents=agents,
                    config=config,
                    document_path=args.document,
                    max_parallel=args.max_workers
                )
        else:
            # Use standard workflow runner
            from workflow.workflow_runner import run_workflow
            
            logger.info(f"Running workflow: {args.workflow}")
            result = run_workflow(
                workflow_name=args.workflow,
                agents=agents,
                config=config,
                document_path=args.document
            )
        
        logger.info(f"Workflow completed with status: {result.get('status', 'unknown')}")
        
        # Record workflow metrics
        monitoring["metrics"].counter("workflows_completed").increment(
            labels={"workflow": args.workflow, "status": result.get("status", "unknown")}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        monitoring["error_manager"].report_error(e)
        sys.exit(1)


def run_monitor_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace,
                    monitoring: Dict[str, Any]) -> None:
    """Run in monitoring mode."""
    logger = logging.getLogger(__name__)
    logger.info("Starting FinFlow in monitoring mode")
    
    health_manager = monitoring["health_manager"]
    metrics = monitoring["metrics"]
    
    # Register agent checks
    from utils.health_check import register_agent_checks
    register_agent_checks(health_manager, list(agents.keys()))
    
    # Run health check system indefinitely
    try:
        logger.info("Monitoring mode active. Press Ctrl+C to exit.")
        
        # Update health status periodically
        while True:
            # Get and log health status
            status = health_manager.get_health_status()
            logger.info(f"Health status: {status['status']}")
            
            # Log metrics
            metric_counts = metrics.get_snapshot()
            logger.info(f"Active metrics: {len(metric_counts)} metrics collected")
            
            # Wait for next check
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        health_manager.stop()


def run_optimized_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace,
                      monitoring: Dict[str, Any]) -> None:
    """Run in optimized mode with enhanced document processor."""
    logger = logging.getLogger(__name__)
    
    if not args.document:
        logger.error("Document path (--document) is required for optimized mode")
        sys.exit(1)
        
    if not os.path.exists(args.document):
        logger.error(f"Document not found: {args.document}")
        sys.exit(1)
    
    try:
        # Use enhanced document processor
        from agents.enhanced_document_processor import EnhancedDocumentProcessorAgent
        
        # Create enhanced processor
        processor = EnhancedDocumentProcessorAgent()
        
        # Process document
        logger.info(f"Processing document using enhanced processor: {args.document}")
        context = {
            "document_path": args.document,
            "workflow_type": "standard",
            "user_id": "optimized_mode",
            "session_id": f"opt_{datetime.now().timestamp()}",
        }
        
        start_time = time.time()
        result = processor.extract_document(context)
        processing_time = time.time() - start_time
        
        logger.info(f"Document processed in {processing_time:.2f}s with confidence {result.get('confidence_score', 'N/A')}")
        
        # Record metrics
        monitoring["metrics"].histogram(
            "optimized_processing_time",
            processing_time,
            labels={"document_type": result.get("document_type", "unknown")}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in optimized processing: {e}")
        monitoring["error_manager"].report_error(e)
        sys.exit(1)


def initialize_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize monitoring and observability systems.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Monitoring components
    """
    logger = logging.getLogger(__name__)
    
    # Initialize metrics collector
    metrics_config = config.get("metrics", {})
    metrics = AppMetricsCollector.get_instance(metrics_config)
    logger.info("Metrics collector initialized")
    
    # Start with basic app metrics
    metrics.gauge("app_info").set(1.0, labels={"version": __version__, "env": config["environment"]})
    metrics.counter("app_starts").increment()
    
    # Initialize error manager
    error_config = config.get("error_handling", {})
    error_manager = ErrorManager.get_instance(error_config)
    logger.info("Error manager initialized")
    
    # Initialize health check manager
    health_config = config.get("health_checks", {})
    health_manager = HealthCheckManager.get_instance(health_config)
    logger.info("Health check manager initialized")
    
    # Start health checks
    health_manager.start()
    
    return {
        "metrics": metrics,
        "error_manager": error_manager,
        "health_manager": health_manager,
    }


def setup_signal_handlers(monitoring: Dict[str, Any]) -> None:
    """Set up signal handlers for graceful shutdown.
    
    Args:
        monitoring: Monitoring components
    """
    def handle_signal(sig, frame):
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {sig}, shutting down...")
        
        # Stop health checks
        if "health_manager" in monitoring:
            monitoring["health_manager"].stop()
        
        # Record shutdown metric
        if "metrics" in monitoring:
            monitoring["metrics"].counter("app_shutdowns").increment()
        
        # Exit with success
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Show version and exit if requested
    if args.version:
        print(f"FinFlow version {__version__}")
        sys.exit(0)
    
    # Set environment variable for config
    os.environ.setdefault('FINFLOW_ENV', args.env)
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Banner
    logger.info("=" * 60)
    logger.info(f"FinFlow v{__version__} - Financial document processing platform")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    logger.info(f"Starting FinFlow in {config['environment']} mode")
    
    # Initialize agent system
    logger.info("Initializing agent system")
    agents = initialize_system(config)
    logger.info("Agent system initialized successfully")
    
    # Initialize monitoring systems
    logger.info("Initializing monitoring systems")
    monitoring_components = initialize_monitoring(config)
    logger.info("Monitoring systems initialized successfully")
    
    # Set up signal handlers
    setup_signal_handlers(monitoring_components)
    
    # Run in specified mode
    if args.mode == RUN_MODE_SERVER:
        run_server_mode(agents, config, args, monitoring_components)
    elif args.mode == RUN_MODE_CLI:
        run_cli_mode(agents, config, args, monitoring_components)
    elif args.mode == RUN_MODE_BATCH:
        run_batch_mode(agents, config, args, monitoring_components)
    elif args.mode == RUN_MODE_WORKFLOW:
        run_workflow_mode(agents, config, args, monitoring_components)
    elif args.mode == RUN_MODE_MONITOR:
        run_monitor_mode(agents, config, args, monitoring_components)
    elif args.mode == RUN_MODE_OPTIMIZED:
        run_optimized_mode(agents, config, args, monitoring_components)
    else:
        # Should never happen due to argparse choices
        logger.error(f"Unknown run mode: {args.mode}")
        sys.exit(1)
    
    logger.info("FinFlow shutdown complete")


if __name__ == "__main__":
    main()
