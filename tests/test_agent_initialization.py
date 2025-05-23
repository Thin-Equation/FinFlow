"""
Tests for agent initialization and configuration.

This module tests the agent initialization process and configuration loading.
"""

import unittest
import sys
from unittest.mock import MagicMock
import os
import tempfile
import yaml
from typing import Any, Dict
from agents.base_agent import BaseAgent

# Set up mocks for the ADK imports
sys.modules['google.adk.agents'] = MagicMock()
sys.modules['google.adk.tools'] = MagicMock()

class TestAgentInitialization(unittest.TestCase):
    """Test cases for agent initialization and configuration."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary config file for testing
        self.config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        
        # Sample configuration for testing
        self.test_config: Dict[str, Any] = {
            "system": {
                "name": "FinFlow Test",
                "environment": "test"
            },
            "logging": {
                "level": "INFO",
                "file": None
            },
            "agents": {
                "document_processor": {
                    "model": "gemini-2.0-flash",
                    "temperature": 0.2
                },
                "master_orchestrator": {
                    "model": "gemini-2.0-flash",
                    "temperature": 0.1
                }
            }
        }
        
        # Write config to the temporary file
        with open(self.config_file.name, 'w') as f:
            yaml.dump(self.test_config, f)
            
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        os.unlink(self.config_file.name)
        
    def test_basic_agent_initialization(self) -> None:
        """Test basic initialization of an agent."""
        agent = BaseAgent(
            name="TestAgent",
            model="gemini-2.0-flash",
            description="Test agent description",
            instruction="Test instruction"
        )
        
        # Check basic properties
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.description, "Test agent description")
        
    # def test_agent_with_tools(self) -> None:
    #     """Test creating an agent with tools."""
    #     # Create a test tool
    #     test_tool = MockBaseTool(name="TestTool", description="A test tool")
        
    #     # Create an agent with the tool
    #     agent = BaseAgent(
    #         name="ToolAgent",
    #         model="gemini-2.0-flash",
    #         description="An agent with tools",
    #         instruction="Test with tools",
    #         tools=[test_tool]
    #     )
        
    #     # Verify the agent was created with the correct name
    #     self.assertEqual(agent.name, "ToolAgent")
    #     self.assertEqual(agent.description, "An agent with tools")
        
    def test_config_based_agent(self) -> None:
        """Test creating an agent with configuration parameters."""
        # Extract configuration for a specific agent
        agent_config = self.test_config["agents"]["document_processor"]
        
        # Create an agent with the configuration
        agent = BaseAgent(
            name="ConfigAgent",
            model=agent_config["model"],
            description="Agent from config",
            instruction="Test from config",
            temperature=agent_config["temperature"]
        )
        
        # Verify agent creation
        self.assertEqual(agent.name, "ConfigAgent")
        self.assertEqual(agent.description, "Agent from config")
        
    def test_session_state_across_agents(self) -> None:
        """Test that session state can be shared between agents."""
        # Create two agents
        agent1 = BaseAgent(name="Agent1", model="gemini-2.0-flash")
        agent2 = BaseAgent(name="Agent2", model="gemini-2.0-flash")
        
        # Create context
        context: Dict[str, Any] = {}
        
        # First agent updates session state
        context = agent1.update_session_state("test_key", "test_value", context)
        
        # Second agent reads the state
        value = agent2.get_session_state("test_key", context)
        
        # Verify state is shared
        self.assertEqual(value, "test_value")
        
if __name__ == "__main__":
    unittest.main()
