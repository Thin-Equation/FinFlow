"""
Base agent class for all FinFlow agents.
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Callable, Union

# Google ADK imports with type ignores for dependencies
from google.adk.agents import LlmAgent  # type: ignore
from google.adk.tools import BaseTool, ToolContext  # type: ignore
from utils.logging_config import TraceContext
from utils.error_handling import (
    FinFlowError, AgentError, ErrorSeverity, ErrorManager, 
    retry, circuit_protected, ErrorBoundary, capture_exceptions
)
from utils.metrics import (
    AppMetricsCollector, time_function, count_invocations, track_errors
)

# Define type variable for generic function returns
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

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
        retry_attempts: int = 3,
        circuit_breaker_enabled: bool = True,
        error_manager: Optional[ErrorManager] = None,
    ):
        """Initialize the base agent.
        
        Args:
            name: The name of the agent.
            model: The model to use for the agent.
            description: A short description of the agent.
            instruction: The instruction prompt for the agent.
            tools: List of tools to add to the agent.
            temperature: The temperature for the agent's model.
            retry_attempts: Number of retry attempts for agent operations.
            circuit_breaker_enabled: Whether to use circuit breaker pattern.
            error_manager: Optional custom error manager.
        """
        # Create a proper logger and configure it
        logger = logging.getLogger(f"finflow.agents.{name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        
        # Store logger in __dict__ to bypass Pydantic validation
        self.__dict__["logger"] = logger
        
        # Store agent configuration
        self.__dict__["agent_name"] = name
        self.__dict__["retry_attempts"] = retry_attempts
        self.__dict__["circuit_breaker_enabled"] = circuit_breaker_enabled
        self.__dict__["error_manager"] = error_manager or ErrorManager.get_instance()
        
        # Initialize metrics collector
        self.__dict__["metrics"] = AppMetricsCollector.get_instance()
        
        # Call parent constructor with necessary parameters
        # We need to use type: ignore because of complex typing in ADK
        super().__init__(  # type: ignore
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            # Temperature is not directly supported in the constructor
        )
        
        # Use the logger that was already created and stored
        # Don't overwrite it by getting from __dict__
        
        # Define a direct implementation of add_tool since parent doesn't have it
        def add_tool_impl(tool: BaseTool) -> None:
            """Implementation of add_tool that stores tools in the agent."""
            if logger:
                logger.debug(f"Adding tool: {tool.name}")
            
            # Store the tool in the agent's tools collection
            if "tools" not in self.__dict__:
                self.__dict__["tools"] = []
                
            self.__dict__["tools"].append(tool)
            
            # In a real implementation, this would register the tool with the LlmAgent
            # but since that's not available, we'll just store it
        
        # Store directly in __dict__ to avoid Pydantic validation issues
        self.__dict__["add_tool"] = add_tool_impl
        
        # Also add it as a method to the class for easier access from child classes
        setattr(self.__class__, "add_tool", add_tool_impl)
        
        # Add tools if provided
        if tools:
            for tool in tools:
                self.__dict__["add_tool"](tool)
                
        logger.info(f"Agent {name} initialized with model {model}")
    
    def log_context(self, context: Dict[str, Any]) -> None:
        """Log the current context for debugging purposes."""
        self.logger.debug(f"Agent context: {context}")
    
    @count_invocations("agent_operation_count")
    @time_function("agent_operation_time")
    @track_errors("agent_error_count")
    @capture_exceptions(AgentError)
    def execute_with_monitoring(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute an agent operation with monitoring and error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            AgentError: If the operation fails
        """
        agent_name = self.__dict__.get("agent_name", "unknown")
        operation = func.__name__
        
        # Start metrics timer
        metrics = self.__dict__.get("metrics")
        timer_context = metrics.track_agent_call(agent_name, operation)
        
        try:
            with timer_context:
                return func(*args, **kwargs)
        except Exception as e:
            # Track error in metrics
            metrics.track_agent_error(agent_name, e.__class__.__name__)
            
            # Wrap in AgentError for consistent handling
            if not isinstance(e, AgentError):
                raise AgentError(
                    f"Agent operation {operation} failed: {str(e)}",
                    agent_name=agent_name,
                    severity=ErrorSeverity.HIGH,
                    cause=e
                ) from e
            
            # Re-raise agent errors
            raise
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors during agent execution with robust error recovery.
        
        Args:
            error: The error that occurred
            context: The execution context
            
        Returns:
            Updated context with error information
        """
        agent_name = self.__dict__.get("agent_name", "unknown")
        error_manager = self.__dict__.get("error_manager")
        
        # Log the error with full traceback
        self.logger.error(
            f"Error in agent {agent_name}: {error}", 
            exc_info=True
        )
        
        # Convert to FinFlowError if it's not already
        if not isinstance(error, FinFlowError):
            error = AgentError(
                str(error),
                agent_name=agent_name,
                severity=ErrorSeverity.HIGH,
                cause=error
            )
        
        # Add error details to context
        error_details = {
            "error": error.to_dict() if isinstance(error, FinFlowError) else str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
        }
        
        context["error"] = error_details
        context["success"] = False
        
        # Send to error manager for centralized handling
        if error_manager:
            error_manager.handle_error(error)
        
        # Track in metrics
        metrics = self.__dict__.get("metrics")
        if metrics:
            metrics.track_agent_error(agent_name, error.__class__.__name__)
        
        return context
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def execute_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with automatic retry for transient failures.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            AgentError: If all retries fail
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Wrap in AgentError for consistent handling
            agent_name = self.__dict__.get("agent_name", "unknown")
            raise AgentError(
                f"Function execution failed after retries: {str(e)}",
                agent_name=agent_name,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
    
    def safe_execute(self, func: Callable[..., T], *args: Any, fallback: Any = None, **kwargs: Any) -> Union[T, Any]:
        """
        Execute a function with error boundary protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            fallback: Fallback value if execution fails
            **kwargs: Keyword arguments
            
        Returns:
            Function result or fallback value if execution fails
        """
        agent_name = self.__dict__.get("agent_name", "unknown")
        boundary = ErrorBoundary(f"agent_{agent_name}", fallback_value=fallback)
        return boundary.execute(func, *args, **kwargs)
    
    def with_circuit_breaker(self, circuit_name: str) -> Callable[[F], F]:
        """
        Create a decorator for circuit breaker protection.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            Decorator function that applies circuit breaker protection
        """
        agent_name = self.__dict__.get("agent_name", "unknown")
        full_circuit_name = f"{agent_name}_{circuit_name}"
        
        # Only apply circuit breaker if enabled
        if self.__dict__.get("circuit_breaker_enabled", True):
            return circuit_protected(
                circuit_name=full_circuit_name,
                failure_threshold=5,
                recovery_timeout=60.0
            )
        else:
            # Return a pass-through decorator
            def passthrough_decorator(func: F) -> F:
                return func
            return passthrough_decorator
    
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
