# filepath: /Users/dhairyagundechia/Downloads/finflow/utils/logging_config.py
"""
Logging configuration for the FinFlow system.
"""

import logging
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Custom log format with trace IDs
class TraceIDLogFormatter(logging.Formatter):
    """Custom formatter that includes trace IDs in log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with trace ID."""
        # Check if the record has a trace_id attribute, add default if not
        if not hasattr(record, 'trace_id'):
            record.trace_id = 'no-trace'
            
        # Apply the standard formatting
        return super().format(record)

def get_current_trace_id() -> str:
    """Get the current trace ID from thread local storage."""
    import threading
    if (hasattr(threading.local(), 'trace_context') and 
            'current_trace_id' in threading.local().trace_context):
        return threading.local().trace_context['current_trace_id']
    return 'no-trace'

class TraceContext:
    """Context manager for trace logging."""
    
    def __init__(self, trace_id: Optional[str] = None, parent_id: Optional[str] = None):
        """
        Initialize trace context.
        
        Args:
            trace_id: Trace ID (will generate if None)
            parent_id: Parent trace ID
        """
        from uuid import uuid4
        self.trace_id = trace_id if trace_id is not None else f"trace-{uuid4().hex[:8]}"
        self.parent_id = parent_id
        self.start_time = datetime.now()
        self.logger = logging.getLogger('finflow.trace')
        
        # Add to thread local storage for access across functions
        import threading
        if not hasattr(threading.local(), 'trace_context'):
            threading.local().trace_context = {}
        threading.local().trace_context['current_trace_id'] = self.trace_id

    def __enter__(self) -> 'TraceContext':
        """Enter the trace context."""
        # Store the original logger class
        self._old_factory = logging.getLogRecordFactory()
        
        # Create a new factory that adds trace_id to LogRecord instances
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self._old_factory(*args, **kwargs)
            record.trace_id = self.trace_id
            return record
        
        # Set new factory
        logging.setLogRecordFactory(record_factory)
        
        # Log the trace start
        parent_info = f" (parent: {self.parent_id})" if self.parent_id else ""
        self.logger.info(f"Trace started: {self.trace_id}{parent_info}")
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the trace context."""
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        
        # Log any exception
        if exc_type:
            self.logger.error(
                f"Trace {self.trace_id} ended with error after {duration:.3f}s: {exc_val}"
            )
        else:
            self.logger.info(f"Trace {self.trace_id} completed in {duration:.3f}s")
        
        # Restore original factory
        logging.setLogRecordFactory(self._old_factory)
        
        # Clean up thread local storage
        import threading
        if hasattr(threading.local(), 'trace_context'):
            threading.local().trace_context.pop('current_trace_id', None)
            
        # Don't suppress exceptions
        return False

def log_agent_call(
    logger: logging.Logger, 
    agent_name: str, 
    context: Dict[str, Any],
    level: int = logging.INFO
) -> None:
    """
    Log an agent call with relevant context.
    
    Args:
        logger: Logger to use
        agent_name: Name of the agent being called
        context: Context being passed to the agent
        level: Logging level
    """
    trace_id = context.get('trace_id', get_current_trace_id())
    
    # Create a filtered context that doesn't include sensitive information
    filtered_context = {
        k: v for k, v in context.items() 
        if k not in ('api_keys', 'credentials', 'secrets')
    }
    
    # Create a simplified context for logging (exclude large fields)
    log_context = {k: v for k, v in filtered_context.items() 
                  if not isinstance(v, (bytes, bytearray)) and 
                  (not isinstance(v, str) or len(v) < 1000)}
    
    # Log the call
    logger.log(
        level,
        f"[{trace_id}] Calling agent {agent_name}",
        extra={
            'trace_id': trace_id,
            'agent_name': agent_name,
            'event_type': 'agent_call',
            'context': json.dumps(log_context, default=str)
        }
    )

def log_agent_response(
    logger: logging.Logger,
    agent_name: str,
    response: Dict[str, Any],
    level: int = logging.INFO,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an agent response.
    
    Args:
        logger: Logger to use
        agent_name: Name of the agent that responded
        response: Response from the agent
        level: Logging level
        context: Optional context for trace_id
    """
    trace_id = 'no-trace'
    if context is not None:
        trace_id = context.get('trace_id', get_current_trace_id())
    else:
        trace_id = get_current_trace_id()
    
    # Create a simplified response for logging (exclude large fields)
    log_response = {k: v for k, v in response.items() 
                   if not isinstance(v, (bytes, bytearray)) and
                   (not isinstance(v, str) or len(v) < 1000)}
    
    # Log the response
    logger.log(
        level,
        f"[{trace_id}] Response from agent {agent_name}",
        extra={
            'trace_id': trace_id,
            'agent_name': agent_name,
            'event_type': 'agent_response',
            'response': json.dumps(log_response, default=str)
        }
    )

def configure_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the FinFlow system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
    """
    # Get log level from environment or use INFO as default
    if log_level is None:
        log_level = os.environ.get('FINFLOW_LOG_LEVEL', 'INFO')
        
    # Convert to numeric level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory if it doesn't exist
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = TraceIDLogFormatter(
        '%(asctime)s - %(trace_id)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_logger('finflow', numeric_level)
    configure_logger('finflow.agents', numeric_level)
    configure_logger('finflow.tools', numeric_level)
    configure_logger('finflow.models', numeric_level)
    
    # Log configuration complete
    logging.info("Logging configured with level %s", log_level)

def configure_logger(name: str, level: int) -> None:
    """Configure a specific logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
