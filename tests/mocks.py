"""
Mock implementations of agent classes for testing.

This module provides mock versions of agent classes that can be used in tests
without requiring the actual ADK dependencies.
"""

from typing import Any, Dict, Optional

class MockBaseTool:
    """A mock implementation of BaseTool for testing."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class MockLlmAgent:
    """A mock implementation of LlmAgent for testing."""
    
    def __init__(
        self, 
        name: str,
        model: str,
        description: str = "",
        instruction: str = ""
    ):
        self.name = name
        self.model = model
        self.description = description
        self.instruction = instruction
        self.tools = []
        
    def add_tool(self, tool: MockBaseTool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        
    def generate_content(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Mock method for generating content."""
        response = MockResponse("This is a mock response.")
        return response

class MockResponse:
    """A mock response from an agent."""
    
    def __init__(self, text: str):
        self.text = text
