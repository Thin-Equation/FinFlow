"""
Configuration loader for FinFlow.
Loads the appropriate configuration based on environment.
"""

import os
import logging
import importlib
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for FinFlow."""
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.env = os.environ.get('FINFLOW_ENV', 'development')
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load the appropriate configuration based on environment.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Return cached config if available
        if "main" in self.config_cache:
            return self.config_cache["main"]
        
        # Load base config
        config_path = os.path.join(self.config_dir, f"{self.env}.yaml")
        
        logger.info(f"Loading configuration from {config_path}")
        
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        # Set environment in config
        config['environment'] = self.env
        
        # Check for local overrides
        local_config_path = os.path.join(self.config_dir, f"{self.env}.local.yaml")
        if os.path.exists(local_config_path):
            logger.info(f"Loading local override configuration from {local_config_path}")
            with open(local_config_path, 'r') as local_config_file:
                local_config = yaml.safe_load(local_config_file)
                if local_config:
                    _deep_merge(config, local_config)
        
        # Load specialized configurations
        self._load_specialized_configs(config)
        
        # Cache the config
        self.config_cache["main"] = config
        
        return config
    
    def _load_specialized_configs(self, config: Dict[str, Any]) -> None:
        """
        Load specialized configurations from modules and merge them.
        
        Args:
            config: Base configuration to merge into
        """
        # Load storage configuration
        try:
            from config.storage_config import get_storage_config
            storage_config = get_storage_config()
            if "storage" not in config:
                config["storage"] = {}
            _deep_merge(config, {"storage": storage_config})
            logger.info("Storage configuration loaded and merged")
        except ImportError:
            logger.info("No specialized storage configuration found")
        
        # Additional specialized configs can be added here following the same pattern
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get the storage configuration.
        
        Returns:
            Dict[str, Any]: Storage configuration
        """
        # Load full config first to ensure specialized configs are merged
        config = self.load_config()
        
        # Return storage section of config
        return config.get("storage", {})


# For backwards compatibility with existing code
def load_config() -> Dict[str, Any]:
    """
    Load the appropriate configuration based on environment.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return ConfigLoader().load_config()

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
