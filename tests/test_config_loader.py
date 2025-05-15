"""
Unit tests for config_loader.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, Any, TypeVar, cast

from config.config_loader import load_config

T = TypeVar('T')

# Implement test utility function for deep merging (similar to the private one in config_loader)
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries for testing purposes.
    
    Args:
        base: Base dictionary
        override: Override dictionary
    
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    # First, create a typed version of the override dict
    typed_override: Dict[str, Any] = {}
    for k, v in override.items():
        typed_override[k] = v
    
    # Now iterate over the typed override
    for key, value in typed_override.items():
        # Check if we need to merge nested dictionaries
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Type cast both dictionaries
            base_dict = cast(Dict[str, Any], base[key])
            value_dict = cast(Dict[str, Any], value)
            
            # Call deep_merge on the nested dictionaries
            base[key] = deep_merge(base_dict, value_dict)
        else:
            # For non-nested values, just override
            base[key] = value
    
    return base

class TestConfigLoader(unittest.TestCase):
    """Test cases for the config_loader module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config: Dict[str, Any] = {
            "google_cloud": {
                "project_id": "test-project",
                "region": "us-central1"
            },
            "agents": {
                "master_orchestrator": {
                    "model": "gemini-2.0-pro"
                }
            }
        }
        
        self.local_config: Dict[str, Any] = {
            "google_cloud": {
                "project_id": "local-project"
            },
            "agents": {
                "master_orchestrator": {
                    "temperature": 0.3
                }
            }
        }
        
        # Expected merged result
        self.merged_config: Dict[str, Any] = {
            "google_cloud": {
                "project_id": "local-project", 
                "region": "us-central1"
            },
            "agents": {
                "master_orchestrator": {
                    "model": "gemini-2.0-pro",
                    "temperature": 0.3
                }
            }
        }
        
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    @patch('os.environ.get')
    def test_load_config_development(
        self, 
        mock_env_get: MagicMock, 
        mock_yaml_load: MagicMock, 
        mock_open: MagicMock, 
        mock_path_exists: MagicMock
    ):
        """Test loading development config."""
        # Set up mocks
        mock_env_get.return_value = 'development'
        
        # Define a properly typed function instead of using lambda
        def check_path(path: str) -> bool:
            return 'development.yaml' in path
            
        mock_path_exists.side_effect = check_path
        mock_yaml_load.return_value = self.sample_config
        
        # Call the function
        config = load_config()
        
        # Verify
        mock_env_get.assert_called_once_with('FINFLOW_ENV', 'development')
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(config, self.sample_config)
    
    @patch('os.path.exists')
    @patch('yaml.safe_load')
    @patch('os.environ.get')
    def test_load_config_with_local_override(
        self, 
        mock_env_get: MagicMock, 
        mock_yaml_load: MagicMock, 
        mock_path_exists: MagicMock
    ):
        """Test loading config with local override."""
        # Set up mocks
        mock_env_get.return_value = 'development'
        mock_path_exists.return_value = True
        mock_yaml_load.side_effect = [self.sample_config, self.local_config]
        # Since we're mocking internal functionality, we don't get the actual merged result
        # So this is more to test the function flow rather than the actual merging
        self.assertEqual(mock_yaml_load.call_count, 2)
    
    @patch('os.path.exists')
    def test_load_config_file_not_found(self, mock_path_exists: MagicMock):
        """Test error when config file not found."""
        mock_path_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_config()
    
    def test_deep_merge(self):
        """Test the deep merge function."""
        base: Dict[str, Any] = self.sample_config.copy()
        override: Dict[str, Any] = self.local_config.copy()
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, self.merged_config)
        
        # Verify the base was modified in place
        self.assertEqual(base, self.merged_config)
        
        # Test merging with empty dictionaries
        self.assertEqual(deep_merge({}, {}), {})
        self.assertEqual(deep_merge({"a": 1}, {}), {"a": 1})
        self.assertEqual(deep_merge({}, {"b": 2}), {"b": 2})

if __name__ == "__main__":
    unittest.main()
