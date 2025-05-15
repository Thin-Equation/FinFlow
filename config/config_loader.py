"""
Configuration loader for FinFlow.
Loads the appropriate configuration based on environment.
"""

import os
import logging
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Load the appropriate configuration based on environment.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Determine environment
    env = os.environ.get('FINFLOW_ENV', 'development')
    
    # Base path for configuration files
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load base config
    config_path = os.path.join(config_dir, f"{env}.yaml")
    
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Set environment in config
    config['environment'] = env
    
    # Check for local overrides
    local_config_path = os.path.join(config_dir, f"{env}.local.yaml")
    if os.path.exists(local_config_path):
        logger.info(f"Loading local override configuration from {local_config_path}")
        with open(local_config_path, 'r') as local_config_file:
            local_config = yaml.safe_load(local_config_file)
            if local_config:
                _deep_merge(config, local_config)
    
    return config

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
    
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Add type ignore comment to suppress the unknown argument type error
            _deep_merge(base[key], value)  # type: ignore
        else:
            base[key] = value
    
    return base
