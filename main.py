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
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional
import importlib.util
from datetime import datetime

# Import configuration
from config.config_loader import load_config, ConfigLoader

# Configure logging
from utils.logging_config import configure_logging

# Import system initialization
from initialize_agents import initialize_system

# Version
__version__ = "1.0.0"

# Run modes
RUN_MODE_SERVER = "server"
RUN_MODE_CLI = "cli"
RUN_MODE_BATCH = "batch"
RUN_MODE_WORKFLOW = "workflow"


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
    parser.add_argument("--mode", help="Run mode", choices=[RUN_MODE_SERVER, RUN_MODE_CLI, RUN_MODE_BATCH, RUN_MODE_WORKFLOW], default=RUN_MODE_SERVER)
    parser.add_argument("--workflow", help="Workflow name to run (for workflow mode)")
    parser.add_argument("--batch-dir", help="Directory with documents to process in batch mode")
    parser.add_argument("--port", help="Port for server mode", type=int, default=8000)
    parser.add_argument("--host", help="Host for server mode", default="0.0.0.0")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    return parser.parse_args()


def run_server_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> None:
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
        
        # Create FastAPI application
        app = create_app(agents=agents, config=config)
        
        # Run server
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
        
    except ImportError as e:
        logger.error(f"Failed to start server: {e}")
        logger.error("Make sure server dependencies are installed: pip install -r requirements-server.txt")
        sys.exit(1)


def run_cli_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run in command line interface mode."""
    from cli.cli_app import run_cli
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FinFlow CLI")
    
    run_cli(agents=agents, config=config)


def run_batch_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run in batch processing mode."""
    logger = logging.getLogger(__name__)
    
    if not args.batch_dir:
        logger.error("Batch directory (--batch-dir) is required for batch mode")
        sys.exit(1)
        
    if not os.path.isdir(args.batch_dir):
        logger.error(f"Batch directory not found: {args.batch_dir}")
        sys.exit(1)
    
    from batch.batch_processor import process_batch
    
    logger.info(f"Starting batch processing from directory: {args.batch_dir}")
    results = process_batch(agents=agents, config=config, batch_dir=args.batch_dir)
    
    logger.info(f"Batch processing completed. Processed {results['total']} documents, {results['success']} success, {results['failed']} failed.")
    

def run_workflow_mode(agents: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run a specific workflow."""
    logger = logging.getLogger(__name__)
    
    if not args.workflow:
        logger.error("Workflow name (--workflow) is required for workflow mode")
        sys.exit(1)
    
    try:
        from workflow.workflow_runner import run_workflow
        
        logger.info(f"Running workflow: {args.workflow}")
        result = run_workflow(
            workflow_name=args.workflow,
            agents=agents,
            config=config,
            document_path=args.document
        )
        
        logger.info(f"Workflow completed with status: {result.get('status', 'unknown')}")
        return result
        
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        sys.exit(1)


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
    
    # Run in specified mode
    if args.mode == RUN_MODE_SERVER:
        run_server_mode(agents, config, args)
    elif args.mode == RUN_MODE_CLI:
        run_cli_mode(agents, config, args)
    elif args.mode == RUN_MODE_BATCH:
        run_batch_mode(agents, config, args)
    elif args.mode == RUN_MODE_WORKFLOW:
        run_workflow_mode(agents, config, args)
    else:
        # Should never happen due to argparse choices
        logger.error(f"Unknown run mode: {args.mode}")
        sys.exit(1)
    
    logger.info("FinFlow shutdown complete")


if __name__ == "__main__":
    main()
