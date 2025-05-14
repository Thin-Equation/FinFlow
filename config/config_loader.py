"""
Configuration loader for FinFlow.
Loads the appropriate configuration based on environment.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config():
    """
    Load configuration based on environment.
    Returns:
        dict: Configuration dictionary
    """
    # Determine environment - default to development
    env = os.environ.get("FINFLOW_ENV", "development")
    
    # Load base configuration
    config_path = os.path.join(os.path.dirname(__file__), f"{env}.yaml")
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            config['environment'] = env
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
