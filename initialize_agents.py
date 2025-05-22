"""
Initialization script for FinFlow agents.

This script initializes and configures all agents in the FinFlow system.
"""

import logging
import os
from typing import Dict, Any

# Set up logging
from utils.logging_config import configure_logging

# Import configuration
from config.config_loader import load_config

# Import agents
from agents.master_orchestrator import MasterOrchestratorAgent
from agents.document_processor import DocumentProcessorAgent
from agents.validation_agent import ValidationAgent
from agents.storage_agent import StorageAgent
from agents.analytics_agent import AnalyticsAgent

def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the FinFlow agent system.
    
    Args:
        config: System configuration
    
    Returns:
        Dict containing initialized agents
    """
    logging.info("Initializing FinFlow agent system")
    
    # Initialize agents
    document_processor = DocumentProcessorAgent()

    validation_agent = ValidationAgent()
    storage_agent = StorageAgent()
    analytics_agent = AnalyticsAgent()
    
    # Initialize master orchestrator with worker agents
    master_orchestrator = MasterOrchestratorAgent(
        document_processor=document_processor,
        validation_agent=validation_agent,
        storage_agent=storage_agent,
        analytics_agent=analytics_agent
    )
    
    # Register worker agents as tools
    master_orchestrator.register_worker_agents()
    
    # Register Document AI tool for document processor
    document_processor.register_tools()
    
    logging.info("FinFlow agent system initialization complete")
    
    return {
        "master_orchestrator": master_orchestrator,
        "document_processor": document_processor,
        "validation_agent": validation_agent,
        "storage_agent": storage_agent,
        "analytics_agent": analytics_agent
    }

def main():
    """Main entry point."""
    # Set environment variable for config
    os.environ.setdefault('FINFLOW_ENV', 'development')
    
    # Configure logging
    configure_logging(log_level='DEBUG')
    
    # Load configuration
    config = load_config()
    
    # Initialize system
    agents = initialize_system(config)
    
    logging.info("FinFlow system initialized successfully")
    
    # Return the initialized agents for use in other scripts
    return agents

if __name__ == "__main__":
    main()
