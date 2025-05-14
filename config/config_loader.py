"""
Configuration loader for FinFlow.
Loads the appropriate configuration based on environment.
"""

import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Check for local overrides
    local_config_path = os.path.join(config_dir, f"{env}.local.yaml")
    if os.path.exists(local_config_path):
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
            _deep_merge(base[key], value)
        else:
            base[key] = value
    
    return base
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
