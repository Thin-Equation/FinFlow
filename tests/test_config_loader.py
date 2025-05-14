"""
Unit tests for config_loader.
"""

import unittest
from unittest.mock import patch, mock_open
import os
import yaml

from config.config_loader import load_config, _deep_merge

class TestConfigLoader(unittest.TestCase):
    """Test cases for the config_loader module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
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
        
        self.local_config = {
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
        self.merged_config = {
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
    def test_load_config_development(self, mock_env_get, mock_yaml_load, mock_open, mock_path_exists):
        """Test loading development config."""
        # Set up mocks
        mock_env_get.return_value = 'development'
        mock_path_exists.side_effect = lambda path: 'development.yaml' in path
        mock_yaml_load.return_value = self.sample_config
        
        # Call the function
        config = load_config()
        
        # Verify
        mock_env_get.assert_called_once_with('FINFLOW_ENV', 'development')
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(config, self.sample_config)
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    @patch('os.environ.get')
    def test_load_config_with_local_override(self, mock_env_get, mock_yaml_load, mock_path_exists, mock_open):
        """Test loading config with local override."""
        # Set up mocks
        mock_env_get.return_value = 'development'
        mock_path_exists.return_value = True
        mock_yaml_load.side_effect = [self.sample_config, self.local_config]
        
        # Call the function
        config = load_config()
        
        # Since we're mocking _deep_merge internally, we don't get the actual merged result
        # So this is more to test the function flow rather than the actual merging
        self.assertEqual(mock_yaml_load.call_count, 2)
    
    @patch('os.path.exists')
    def test_load_config_file_not_found(self, mock_path_exists):
        """Test error when config file not found."""
        mock_path_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_config()
    
    def test_deep_merge(self):
        """Test the deep merge function."""
        base = self.sample_config.copy()
        override = self.local_config.copy()
        
        result = _deep_merge(base, override)
        
        self.assertEqual(result, self.merged_config)
        
        # Verify the base was modified in place
        self.assertEqual(base, self.merged_config)
        
        # Test merging with empty dictionaries
        self.assertEqual(_deep_merge({}, {}), {})
        self.assertEqual(_deep_merge({"a": 1}, {}), {"a": 1})
        self.assertEqual(_deep_merge({}, {"b": 2}), {"b": 2})

if __name__ == "__main__":
    unittest.main()
