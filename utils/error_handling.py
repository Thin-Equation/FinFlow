"""
Comprehensive error handling framework for FinFlow.

This module provides:
1. Custom exception hierarchy
2. Error reporting system
3. Retry mechanisms with exponential backoff
4. Circuit breaker pattern implementation
5. Graceful degradation utilities
"""

import logging
import time
import random
import functools
import json
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from utils.logging_config import get_current_trace_id

# Type variable for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# -----------------------------------------------------------------------------
# Exception Hierarchy
# -----------------------------------------------------------------------------

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Non-critical, can be ignored in most cases
    MEDIUM = "medium"     # Important but not critical, should be logged
    HIGH = "high"         # Critical, requires attention
    FATAL = "fatal"       # System-stopping, requires immediate intervention


class FinFlowError(Exception):
    """Base exception for all FinFlow errors."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            severity: Error severity level
            error_code: Application-specific error code
            details: Additional error details
            cause: The exception that caused this one
        """
        self.message = message
        self.severity = severity
        self.error_code = error_code or "ERR_UNDEFINED"
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        self.trace_id = get_current_trace_id()
        
        # Format the message with details
        formatted_message = f"{self.error_code}: {message}"
        super().__init__(formatted_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary."""
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }
        
        if self.details:
            result["details"] = self.details
            
        if self.cause:
            result["cause"] = str(self.cause)
            
        return result
    
    def to_json(self) -> str:
        """Convert the exception to a JSON string."""
        return json.dumps(self.to_dict())


class ConfigurationError(FinFlowError):
    """Error related to system configuration."""
    
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_CONFIG")
        super().__init__(message, **kwargs)


class AgentError(FinFlowError):
    """Error related to agent operations."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_AGENT")
        details = kwargs.get("details", {})
        if agent_name:
            details["agent_name"] = agent_name
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class DocumentProcessingError(FinFlowError):
    """Error related to document processing."""
    
    def __init__(self, message: str, document_id: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_DOC_PROC")
        details = kwargs.get("details", {})
        if document_id:
            details["document_id"] = document_id
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ValidationError(FinFlowError):
    """Error related to validation operations."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_VALIDATION")
        details = kwargs.get("details", {})
        if validation_type:
            details["validation_type"] = validation_type
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class StorageError(FinFlowError):
    """Error related to storage operations."""
    
    def __init__(self, message: str, storage_type: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_STORAGE")
        details = kwargs.get("details", {})
        if storage_type:
            details["storage_type"] = storage_type
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ExternalServiceError(FinFlowError):
    """Error related to external service integrations."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_EXT_SERVICE")
        details = kwargs.get("details", {})
        if service_name:
            details["service_name"] = service_name
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class TimeoutError(FinFlowError):
    """Error related to operation timeouts."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_TIMEOUT")
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AuthenticationError(FinFlowError):
    """Error related to authentication."""
    
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_AUTH")
        super().__init__(message, **kwargs)


class AuthorizationError(FinFlowError):
    """Error related to authorization."""
    
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "ERR_AUTHZ")
        super().__init__(message, **kwargs)


# -----------------------------------------------------------------------------
# Error Reporting System
# -----------------------------------------------------------------------------

class ErrorManager:
    """Central error management system."""
    
    _instance: Optional['ErrorManager'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'ErrorManager':
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the error manager."""
        if self.__class__._instance is not None:
            raise RuntimeError("This class is a singleton. Use get_instance() instead.")
        
        self.logger = logging.getLogger("finflow.errors")
        self.error_handlers: Dict[str, List[Callable[[FinFlowError], None]]] = {}
        self.error_counts: Dict[str, int] = {}
        self._subscribers: Set[Callable[[FinFlowError], None]] = set()
    
    def register_handler(self, error_code: str, handler: Callable[[FinFlowError], None]) -> None:
        """Register a handler for a specific error code."""
        if error_code not in self.error_handlers:
            self.error_handlers[error_code] = []
        self.error_handlers[error_code].append(handler)
    
    def subscribe(self, handler: Callable[[FinFlowError], None]) -> Callable[[], None]:
        """
        Subscribe to all errors.
        
        Returns:
            A function that can be called to unsubscribe.
        """
        self._subscribers.add(handler)
        
        def unsubscribe() -> None:
            self._subscribers.discard(handler)
        
        return unsubscribe
    
    def handle_error(self, error: Union[FinFlowError, Exception]) -> None:
        """Handle an error through the error management system."""
        # Convert standard exceptions to FinFlowError
        if not isinstance(error, FinFlowError):
            error = FinFlowError(
                str(error), 
                severity=ErrorSeverity.HIGH, 
                error_code="ERR_UNEXPECTED",
                cause=error
            )
        
        # Log the error
        if error.severity == ErrorSeverity.FATAL:
            self.logger.critical(error.message, exc_info=True)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(error.message, exc_info=True)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error.message)
        else:
            self.logger.info(error.message)
        
        # Increment error count
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        
        # Call specific handlers
        if error.error_code in self.error_handlers:
            for handler in self.error_handlers[error.error_code]:
                try:
                    handler(error)
                except Exception as e:
                    self.logger.error(f"Error in error handler: {e}")
        
        # Call subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(error)
            except Exception as e:
                self.logger.error(f"Error in error subscriber: {e}")


# -----------------------------------------------------------------------------
# Retry Mechanism
# -----------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier applied to delay between retries
        jitter: Random factor to add to delay (0.1 = 10% random jitter)
        exceptions: Exception(s) to catch and retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if isinstance(exceptions, list):
                exception_tuple = tuple(exceptions)
            else:
                exception_tuple = (exceptions,)
            
            last_exception = None
            curr_delay = delay
            
            logger = logging.getLogger(func.__module__)
            
            # Try up to max_attempts times
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception_tuple as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    
                    # Add jitter to delay
                    jitter_amount = random.uniform(-jitter, jitter) * curr_delay
                    wait = curr_delay + jitter_amount
                    
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"in {wait:.2f}s after error: {str(e)}"
                    )
                    
                    time.sleep(wait)
                    curr_delay *= backoff
            
            # Should never reach here due to the raise in the loop
            raise last_exception if last_exception else RuntimeError("Unexpected retry failure")
        
        return cast(F, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Circuit Breaker Pattern
# -----------------------------------------------------------------------------

class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"       # Normal operation, requests allowed
    OPEN = "open"           # Failure detected, requests blocked
    HALF_OPEN = "half_open" # Testing if service is back, limited requests


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    Implements the circuit breaker pattern to automatically detect failures and
    prevent repeated failures by temporarily blocking requests that are likely to fail.
    """
    
    _instances: Dict[str, 'CircuitBreaker'] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, name: str, **kwargs: Any) -> 'CircuitBreaker':
        """Get a named circuit breaker instance."""
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    cls._instances[name] = cls(name=name, **kwargs)
        return cls._instances[name]
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        test_attempts: int = 1
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before testing recovery
            test_attempts: Number of successful test attempts to close circuit
        """
        self.name = name
        self.logger = logging.getLogger(f"finflow.circuit_breaker.{name}")
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.test_attempts = test_attempts
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.successful_test_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_failure: Optional[Exception] = None
        
        self._lock = threading.Lock()
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by func
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time is not None and \
                   time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.logger.info(
                        f"Circuit {self.name} state: OPEN -> HALF_OPEN, "
                        f"testing service after {self.recovery_timeout}s"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.successful_test_count = 0
                else:
                    error = ExternalServiceError(
                        f"Circuit {self.name} is OPEN, service is unavailable",
                        service_name=self.name,
                        error_code="ERR_CIRCUIT_OPEN",
                        details={"last_failure": str(self.last_failure)} if self.last_failure else {}
                    )
                    self.logger.warning(str(error))
                    raise error
        
        try:
            result = func(*args, **kwargs)
            
            # Update state on success
            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.successful_test_count += 1
                    if self.successful_test_count >= self.test_attempts:
                        self.logger.info(f"Circuit {self.name} state: HALF_OPEN -> CLOSED, service recovered")
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitState.CLOSED:
                    # Reset failure count on successful requests
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            # Update state on failure
            with self._lock:
                self.last_failure = e
                self.last_failure_time = time.time()
                
                if self.state == CircuitState.HALF_OPEN:
                    self.logger.warning(
                        f"Circuit {self.name} state: HALF_OPEN -> OPEN, "
                        f"service still failing: {str(e)}"
                    )
                    self.state = CircuitState.OPEN
                elif self.state == CircuitState.CLOSED:
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.logger.warning(
                            f"Circuit {self.name} state: CLOSED -> OPEN, "
                            f"failure threshold reached: {self.failure_count} failures"
                        )
                        self.state = CircuitState.OPEN
            
            # Re-raise the exception
            raise


def circuit_protected(
    circuit_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    fallback: Optional[Callable[..., Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before testing recovery
        fallback: Optional fallback function if circuit is open
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            circuit = CircuitBreaker.get_instance(
                circuit_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
            try:
                return circuit.execute(func, *args, **kwargs)
            except ExternalServiceError as e:
                if e.error_code == "ERR_CIRCUIT_OPEN" and fallback is not None:
                    return fallback(*args, **kwargs)
                raise
                
        return cast(F, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Graceful Degradation
# -----------------------------------------------------------------------------

class FeatureFlag:
    """Feature flag management for graceful degradation."""
    
    _instance: Optional['FeatureFlag'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'FeatureFlag':
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the feature flag manager."""
        self.flags: Dict[str, bool] = {}
        self.flag_defaults: Dict[str, bool] = {}
        self.logger = logging.getLogger("finflow.feature_flags")
        self._lock = threading.Lock()
    
    def register_flag(self, flag_name: str, default_value: bool = True) -> None:
        """Register a new feature flag."""
        with self._lock:
            self.flag_defaults[flag_name] = default_value
            if flag_name not in self.flags:
                self.flags[flag_name] = default_value
    
    def enable(self, flag_name: str) -> None:
        """Enable a feature flag."""
        with self._lock:
            self.flags[flag_name] = True
            self.logger.info(f"Feature flag {flag_name} enabled")
    
    def disable(self, flag_name: str) -> None:
        """Disable a feature flag."""
        with self._lock:
            self.flags[flag_name] = False
            self.logger.info(f"Feature flag {flag_name} disabled")
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        with self._lock:
            if flag_name not in self.flags and flag_name in self.flag_defaults:
                self.flags[flag_name] = self.flag_defaults[flag_name]
            return self.flags.get(flag_name, False)
    
    def reset(self, flag_name: str) -> None:
        """Reset a flag to its default value."""
        with self._lock:
            if flag_name in self.flag_defaults:
                self.flags[flag_name] = self.flag_defaults[flag_name]


def with_feature_flag(flag_name: str, fallback: Optional[Callable[..., Any]] = None) -> Callable[[F], F]:
    """
    Decorator to conditionally execute a function based on a feature flag.
    
    Args:
        flag_name: Name of the feature flag
        fallback: Optional fallback function if feature is disabled
        
    Returns:
        Decorated function that checks feature flag status
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            feature_flags = FeatureFlag.get_instance()
            
            if feature_flags.is_enabled(flag_name):
                return func(*args, **kwargs)
            elif fallback is not None:
                return fallback(*args, **kwargs)
            else:
                logger = logging.getLogger(func.__module__)
                logger.warning(f"Feature {flag_name} is disabled, skipping {func.__name__}")
                return None
                
        return cast(F, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def safe_execute(func: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
    """
    Execute a function and catch any exceptions.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        The function result or None if an exception occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger("finflow.safe_execute")
        logger.error(f"Error executing {func.__name__}: {e}")
        return None


def capture_exceptions(
    wrapper_class: Type[FinFlowError] = FinFlowError,
    **wrapper_kwargs: Any
) -> Callable[[F], F]:
    """
    Decorator to capture and convert exceptions to FinFlowError types.
    
    Args:
        wrapper_class: Type of FinFlowError to create
        **wrapper_kwargs: Additional arguments for the wrapper class
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except FinFlowError:
                # Don't wrap FinFlowError instances
                raise
            except Exception as e:
                # Create a new instance of the wrapper class with the original exception
                raise wrapper_class(
                    str(e),
                    cause=e,
                    **wrapper_kwargs
                ) from e
                
        return cast(F, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Error Boundary for Async Code
# -----------------------------------------------------------------------------

class ErrorBoundary:
    """
    Context manager for creating error boundaries in code.
    
    An error boundary catches exceptions and provides various recovery options.
    """
    
    def __init__(
        self,
        boundary_name: str,
        fallback_value: Any = None,
        retries: int = 0,
        error_manager: Optional[ErrorManager] = None
    ):
        """
        Initialize the error boundary.
        
        Args:
            boundary_name: Name for this boundary (for logging)
            fallback_value: Value to return if an error occurs
            retries: Number of retry attempts (0 = no retry)
            error_manager: Optional error manager to use
        """
        self.boundary_name = boundary_name
        self.fallback_value = fallback_value
        self.retries = retries
        self.error_manager = error_manager or ErrorManager.get_instance()
        self.logger = logging.getLogger(f"finflow.error_boundary.{boundary_name}")
    
    def __enter__(self) -> 'ErrorBoundary':
        """Enter the error boundary."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Exit the error boundary and handle any exceptions.
        
        Returns:
            True if the exception was handled, False to re-raise
        """
        if exc_type is None:
            # No exception occurred
            return False
        
        # Log the exception
        self.logger.error(
            f"Error in boundary {self.boundary_name}: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb)
        )
        
        # Pass to error manager
        if isinstance(exc_val, Exception):
            self.error_manager.handle_error(exc_val)
        
        # Suppress the exception
        return True
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Union[T, Any]:
        """
        Execute a function within this error boundary.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The function's return value or the fallback value
        """
        attempts = 1 + self.retries
        
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < attempts:
                    self.logger.warning(
                        f"Retry {attempt}/{attempts} for {func.__name__} after error: {str(e)}"
                    )
                    continue
                else:
                    # Final attempt failed
                    self.logger.error(
                        f"Error in boundary {self.boundary_name}: {str(e)}",
                        exc_info=True
                    )
                    
                    # Pass to error manager
                    self.error_manager.handle_error(e)
                    
                    # Return fallback
                    return self.fallback_value