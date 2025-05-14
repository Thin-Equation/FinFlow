#!/usr/bin/env python3
"""
FinFlow: Financial document processing and analysis platform.
Main application entry point.
"""

import logging
from config import config_loader

def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = config_loader.load_config()
    logger.info(f"Starting FinFlow in {config['environment']} mode")
    
    # Application initialization logic goes here
    
    logger.info("FinFlow initialization complete")

if __name__ == "__main__":
    main()
