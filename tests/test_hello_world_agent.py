"""
Hello World test case for basic agent functionality.

This test case verifies that an agent can be created, initialized, 
and can respond to a simple query.
"""

import unittest
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

# Use mock objects for testing
from tests.mocks import MockBaseTool, MockLlmAgent

# Import the agent module directly, using mock objects instead of BaseAgent
sys.modules['google.adk.agents'] = MagicMock()
sys.modules['google.adk.tools'] = MagicMock()
sys.modules['google.adk.tools'].BaseTool = MockBaseTool
sys.modules['google.adk.agents'].LlmAgent = MockLlmAgent

# Now we can import BaseAgent
from agents.base_agent import BaseAgent

class TestHelloWorldAgent(unittest.TestCase):
    """Hello World test case for basic agent functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent = BaseAgent(
            name="HelloWorldAgent",
            model="gemini-2.0-flash",
            description="A simple test agent that says hello",
            instruction="When asked to say hello, respond with 'Hello, World!'"
        )
    
    def test_agent_initialization(self) -> None:
        """Test that the agent initializes correctly with proper attributes."""
        self.assertEqual(self.agent.name, "HelloWorldAgent")
        self.assertEqual(self.agent.description, "A simple test agent that says hello")

    def test_session_state_management(self) -> None:
        """Test the agent's session state management functions."""
        # Create a test context
        context: Dict[str, Any] = {}
        
        # Test updating session state
        context = self.agent.update_session_state("test_key", "test_value", context)
        
        # Verify the session state was updated correctly
        self.assertIn("session_state", context)
        self.assertIn("test_key", context["session_state"])
        self.assertEqual(context["session_state"]["test_key"], "test_value")
        
        # Test retrieving session state
        value = self.agent.get_session_state("test_key", context)
        self.assertEqual(value, "test_value")
        
        # Test retrieving non-existent key with default
        default_value = "default"
        value = self.agent.get_session_state("non_existent_key", context, default_value)
        self.assertEqual(value, default_value)
        
        # Test updating an existing key
        context = self.agent.update_session_state("test_key", "updated_value", context)
        value = self.agent.get_session_state("test_key", context)
        self.assertEqual(value, "updated_value")

    def test_error_handling(self) -> None:
        """Test the agent's error handling functionality."""
        # Create a test context
        context: Dict[str, Any] = {}
        
        # Create a test error
        test_error = ValueError("Test error message")
        
        # Handle the error
        updated_context = self.agent.handle_error(test_error, context)
        
        # Verify error information was added to the context
        self.assertIn("error", updated_context)
        self.assertEqual(updated_context["error"]["type"], "ValueError")
        self.assertEqual(updated_context["error"]["message"], "Test error message")
        self.assertEqual(updated_context["error"]["agent"], "HelloWorldAgent")

    def test_agent_to_agent_transfer(self) -> None:
        """Test agent-to-agent transfer functionality."""
        # Create a test context
        context: Dict[str, Any] = {}
        
        # Test transfer to another agent
        target_agent = "TargetAgent"
        result = self.agent.transfer_to_agent(target_agent, context)
        
        # Verify transfer information
        self.assertEqual(result["status"], "transferred")
        self.assertEqual(result["from_agent"], "HelloWorldAgent")
        self.assertEqual(result["to_agent"], target_agent)
        
        # Verify agent flow was tracked
        transferred_context = result["context"]
        self.assertIn("agent_flow", transferred_context)
        self.assertEqual(len(transferred_context["agent_flow"]), 1)
        self.assertEqual(transferred_context["agent_flow"][0]["from"], "HelloWorldAgent")
        self.assertEqual(transferred_context["agent_flow"][0]["to"], target_agent)
        
    def test_hello_world_with_tools(self) -> None:
        """Test that an agent can be created with tools."""
        # Create a simple test tool
        test_tool = MockBaseTool(name="TestTool", description="A test tool")
        
        # Create a hello world agent with the test tool
        hello_agent = BaseAgent(
            name="HelloAgent",
            model="gemini-2.0-flash",
            description="An agent that says hello",
            instruction="Say hello when asked",
            tools=[test_tool]
        )
        
        # Verify the agent has the correct name and description
        self.assertEqual(hello_agent.name, "HelloAgent")
        self.assertEqual(hello_agent.description, "An agent that says hello")

if __name__ == "__main__":
    unittest.main()
