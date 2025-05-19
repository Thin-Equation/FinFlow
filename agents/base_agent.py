"""
Base agent class for all FinFlow agents.
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TypeVar

# Google ADK imports with type ignores for dependencies
from google.adk.agents import LlmAgent  # type: ignore
from google.adk.tools import BaseTool, ToolContext  # type: ignore
from utils.logging_config import TraceContext

# Define type variable for generic function returns
T = TypeVar('T')

class BaseAgent(LlmAgent):
    """Base agent class with common functionality for all FinFlow agents."""
    
    # Pre-define logger field to make it compatible with Pydantic
    logger: logging.Logger = None
    
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
        # Create logger before initializing parent to avoid Pydantic validation issues
        self.__dict__["logger"] = logging.getLogger(f"finflow.agents.{name}")
        
        # Call parent constructor with necessary parameters
        # We need to use type: ignore because of complex typing in ADK
        super().__init__(  # type: ignore
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            # Temperature is not directly supported in the constructor
        )
        
        # Explicitly add a type-annotated add_tool method to handle typing issues
        # The original is inherited from LlmAgent but needs explicit typing for TypeScript
        self.add_tool: Callable[[BaseTool], None]
        
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
        error_id = datetime.now().strftime("%Y%m%d%H%M%S")
        error_stack = traceback.format_exc()
        
        self.logger.error(
            f"Error in {self.name} [ID: {error_id}]: {str(error)}",
            exc_info=True
        )
        
        # Add detailed error info to context
        context["error"] = {
            "id": error_id,
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "message": str(error),
            "type": error.__class__.__name__,
            "stack_trace": error_stack
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
        self.logger.debug(f"Updated session state: {state_key} = {state_value}")
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
            
        value = context["session_state"].get(state_key, default)
        self.logger.debug(f"Retrieved session state: {state_key} = {value}")
        return value
    
    def transfer_to_agent(self, agent_name: str, context: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Transfer control to another agent.
        
        Args:
            agent_name: The name of the agent to transfer control to
            context: The current context
            tool_context: Tool context provided by ADK
            
        Returns:
            Dict[str, Any]: Response from the target agent
        """
        self.logger.info(f"Transferring control from {self.name} to {agent_name}")
        
        # Record the agent flow for tracing
        if "agent_flow" not in context:
            context["agent_flow"] = []
            
        # Create a flow entry
        flow_entry = {
            "from": self.name,
            "to": agent_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use type: ignore to bypass the type checking error
        context["agent_flow"].append(flow_entry)  # type: ignore
        
        # In a real implementation, this would use the actual AgentTool
        # For now, just return the context with transfer information
        return {
            "status": "transferred",
            "from_agent": self.name,
            "to_agent": agent_name,
            "context": context
        }
    
    def execute_with_tracing(self, func: Any, context: Dict[str, Any], **kwargs: Any) -> Any:
        """
        Execute a function with tracing.
        
        Args:
            func: Function to execute that takes context as first parameter
            context: Context to pass to the function
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            Any: Result of the function
        """
        # Get or create trace ID
        trace_id = context.get("trace_id", None)
        
        # Create trace context
        with TraceContext(trace_id=trace_id) as trace:
            # Update context with trace ID
            context["trace_id"] = trace.trace_id
            
            # Log function call
            self.logger.debug(f"Executing {func.__name__} with trace ID {trace.trace_id}")
            
            # Execute function with any parameter
            start_time = datetime.now()
            result = func(context=context, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log completion
            self.logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            
            return result
    
    def log_activity(self, activity_type: str, details: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log agent activity for audit and debugging purposes.
        
        Args:
            activity_type: Type of activity (e.g., 'document_processing', 'validation')
            details: Activity details
            context: Current context
        """
        timestamp = datetime.now().isoformat()
        trace_id = context.get("trace_id", "no-trace")
        
        # Create activity log with explicit typing to fix the type issue
        activity_log: Dict[str, Any] = {
            "timestamp": timestamp,
            "agent": self.name,
            "activity_type": activity_type,
            "trace_id": trace_id,
            "details": details
        }
        
        # Log as structured data
        self.logger.info(f"Activity: {json.dumps(activity_log)}")
