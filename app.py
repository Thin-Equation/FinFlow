#!/usr/bin/env python3
"""
FinFlow: Financial document processing and analysis platform.
Main application entry point.
"""

import os
import logging
import argparse
from typing import Dict, Any, cast

# Import system initialization
from initialize_agents import initialize_system
from utils.logging_config import configure_logging

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
    return parser.parse_args()

def process_document(document_path: str, agents: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document using the agent system.
    
    Args:
        document_path: Path to the document to process
        agents: Dictionary of initialized agents
        
    Returns:
        Processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing document: {document_path}")
    
    # Create processing context
    context = {
        "document_path": document_path,
        "user_id": "system",  # In a real app, this would be the actual user
        "session_id": "test_session",  # In a real app, this would be a real session ID
        "workflow_type": "standard"  # Could be different workflow types
    }
    
    # Use the master orchestrator to process the document
    # Cast to specific type for type safety
    master_orchestrator = cast(MasterOrchestratorAgent, agents["master_orchestrator"])
    result = master_orchestrator.process_document(context)
    
    logger.info(f"Document processing completed with status: {result.get('status', 'unknown')}")
    return result

def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set environment variable for config
    os.environ.setdefault('FINFLOW_ENV', args.env)
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    logger.info(f"Starting FinFlow in {config['environment']} mode")
    
    # Initialize agent system
    logger.info("Initializing agent system")
    agents = initialize_system(config)
    logger.info("Agent system initialized successfully")
    
    # Process document if specified
    if args.document:
        result = process_document(args.document, agents)
        logging.info(f"Document processing result: {result}")
    else:
        logger.info("No document specified for processing")
        logger.info("Run with --document path/to/document to process a document")
    
    logger.info("FinFlow initialization complete")

if __name__ == "__main__":
    main()
