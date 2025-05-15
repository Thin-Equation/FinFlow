"""
Hello World agent for testing ADK functionality.

This simple agent is designed for testing the ADK CLI and verifying
that the basic infrastructure is working correctly.
"""

from google.adk.agents import LlmAgent  # type: ignore
from agents.base_agent import BaseAgent

class HelloWorldAgent(BaseAgent):
    """A simple Hello World agent for testing ADK functionality."""
    
    def __init__(self):
        """Initialize the Hello World agent."""
        super().__init__(
            name="HelloWorldAgent",
            model="gemini-2.0-flash",
            description="A simple agent that responds with a greeting",
            instruction="""
            You are a simple test agent designed to verify that ADK is working correctly.
            Your primary task is to respond to greetings with a friendly message.
            
            - When the user says 'hello', respond with 'Hello, World! I'm the FinFlow Hello World Agent.'
            - If asked about your purpose, explain that you're a test agent for the FinFlow financial system.
            - For any other questions, politely explain that you're just a test agent with limited functionality.
            
            Be concise and friendly in your responses.
            """
        )

# Create an instance of the agent that ADK can discover
hello_world_agent = HelloWorldAgent()

if __name__ == "__main__":
    # This allows the agent to be run directly for testing
    print("Hello World Agent initialized. Use ADK CLI to interact with it.")
