"""
Base agent class for all FinFlow agents.
"""

import logging
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool

class BaseAgent(LlmAgent):
    """Base agent class with common functionality for all FinFlow agents."""
    
    def __init__(
        self,
        name: str,
        model: str = "gemini-2.0-flash", # or gemini-pro
        description: str = "",
        instruction: str = "",
        tools: Optional[List[BaseTool]] = None,
        temperature: float = 0.2,
    ):
        """Initialize the base agent.
        
        Args:
            name: The name of the agent.
            model: The model to use for the agent.
            description: A short description of the agent.
            instruction: The instruction prompt for the agent.
            tools: List of tools to add to the agent.
            temperature: The temperature for the agent's model.
        """
        super().__init__(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            temperature=temperature,
        )
        
        self.logger = logging.getLogger(f"finflow.agents.{name}")
        
        # Add tools if provided
        if tools:
            for tool in tools:
                self.add_tool(tool)
    
    def log_context(self, context: Dict[str, Any]) -> None:
        """Log the current context for debugging purposes."""
        self.logger.debug(f"Agent context: {context}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors during agent execution.
        
        Args:
            error: The exception that was raised
            context: The current context
            
        Returns:
            Dict[str, Any]: Updated context with error information
        """
        self.logger.error(f"Error in {self.name}: {str(error)}", exc_info=True)
        
        # Add error to context
        context["error"] = {
            "message": str(error),
            "type": error.__class__.__name__
        }
        
        return context
    
    def update_session_state(self, state_key: str, state_value: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the session state with the provided key and value.
        
        Args:
            state_key: The key to update
            state_value: The value to update
            context: The current context
            
        Returns:
            Dict[str, Any]: Updated context with new session state
        """
        if "session_state" not in context:
            context["session_state"] = {}
            
        context["session_state"][state_key] = state_value
        return context
    
    def get_session_state(self, state_key: str, context: Dict[str, Any], default: Any = None) -> Any:
        """
        Get a value from the session state.
        
        Args:
            state_key: The key to retrieve
            context: The current context
            default: Default value if the key doesn't exist
            
        Returns:
            Any: The value associated with the key, or the default
        """
        if "session_state" not in context:
            return default
            
        return context["session_state"].get(state_key, default)
    
    def transfer_to_agent(self, agent_name: str, context: Dict[str, Any]) -> None:
        """
        Transfer control to another agent.
        
        Args:
            agent_name: The name of the agent to transfer control to
            context: The current context
        """
        # In ADK, this is handled by the AgentTool mechanism
        self.logger.info(f"Transferring control from {self.name} to {agent_name}")
        
        # We'll use a placeholder here that will be replaced by actual AgentTool
        # when the full system is implemented
        pass
        """Handle errors that occur during agent execution.
        
        Args:
            error: The exception that occurred.
            context: The current context.
            
        Returns:
            Updated context with error information.
        """
        self.logger.error(f"Agent error: {error}", exc_info=True)
        
        # Add error to context
        context["error"] = {
            "message": str(error),
            "type": type(error).__name__
        }
        
        return context
