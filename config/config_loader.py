# filepath: /Users/dhairyagundechia/Downloads/finflow/config/config_loader.py
"""
Configuration loader for FinFlow.
Loads the appropriate configuration based on environment with support for
overrides, secrets management, and component-specific configurations.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for FinFlow with robust environment and override support.
    
    Features:
    - Environment-based configuration (dev/staging/prod)
    - Local override support
    - Secrets management
    - Component-specific configuration
    - Configuration validation
    """
    
    def __init__(self, env: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            env: Optional environment override (development, staging, production)
        """
        self.env = env or os.environ.get('FINFLOW_ENV', 'development')
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Validate environment
        if self.env not in ('development', 'staging', 'production'):
            logger.warning(f"Invalid environment: {self.env}, defaulting to development")
            self.env = 'development'
        
        logger.info(f"Initialized ConfigLoader for environment: {self.env}")
    
    def load_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load the appropriate configuration based on environment.
        
        Args:
            reload: Force reload configuration from disk
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Return cached config if available and not reloading
        if "main" in self.config_cache and not reload:
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
        
        # Validate configuration
        if not self.validate_config(config):
            logger.warning("Configuration validation failed")
        
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
            _deep_merge(config["storage"], storage_config)
            logger.info("Storage configuration loaded and merged")
        except ImportError:
            logger.info("No specialized storage configuration found")
        
        # Load document processor configuration
        try:
            from config.document_processor_config import get_document_processor_config
            doc_config = get_document_processor_config()
            if "document_processor" not in config:
                config["document_processor"] = {}
            _deep_merge(config["document_processor"], doc_config)
            logger.info("Document processor configuration loaded and merged")
        except ImportError:
            logger.info("No specialized document processor configuration found")
        
        # Additional specialized configs can be added here
    
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
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the loaded configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check required configurations
        required_keys = ["google_cloud"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        # Check for Google Cloud credentials
        if "google_cloud" in config:
            if "project_id" not in config["google_cloud"]:
                logger.error("Missing required Google Cloud project_id in configuration")
                return False
                
        return True


# Helper function for deep merging dictionaries
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
    
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# For backwards compatibility with existing code
def load_config(reload: bool = False) -> Dict[str, Any]:
    """
    Load the appropriate configuration based on environment.
    
    Args:
        reload: Force reload configuration from disk
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_config(reload=reload)
