"""
Define the agent communication protocol for the FinFlow system.

This module contains the protocol definitions, message schemas, and protocol-specific
utilities for agent-to-agent communication.
"""

from typing import Any, Dict, List, Optional, Union, TypedDict, Literal
from datetime import datetime
import json
import uuid
from enum import Enum, auto

# Message types enum
class MessageType(str, Enum):
    """Enum for message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    INFO = "info"

# Priority levels enum
class PriorityLevel(str, Enum):
    """Enum for priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Status codes
class StatusCode(str, Enum):
    """Enum for status codes in responses."""
    OK = "ok"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"
    WAITING = "waiting"

# TypedDict for better type checking
class MessageContent(TypedDict, total=False):
    """Base type for message content."""
    action: str
    data: Dict[str, Any]
    status: StatusCode
    error: Optional[str]

class Message(TypedDict):
    """Type definition for a message in the protocol."""
    message_id: str
    timestamp: str
    sender_id: str
    recipient_id: str
    message_type: str
    priority: str
    content: MessageContent
    reference_id: Optional[str]

# Protocol version
PROTOCOL_VERSION = "1.0.0"

def create_protocol_message(
    sender_id: str,
    recipient_id: str,
    message_type: Union[MessageType, str],
    content: Dict[str, Any],
    message_id: Optional[str] = None,
    priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
    reference_id: Optional[str] = None
) -> Message:
    """
    Create a message according to the protocol specifications.
    
    Args:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        message_type: Type of message
        content: Message content dictionary
        message_id: Optional message ID (generated if not provided)
        priority: Message priority level
        reference_id: Optional reference to another message
        
    Returns:
        Message: Protocol-compliant message
    """
    # Convert enum values to strings if necessary
    if isinstance(message_type, MessageType):
        message_type = message_type.value
        
    if isinstance(priority, PriorityLevel):
        priority = priority.value
    
    # Generate message ID if not provided
    if not message_id:
        message_id = str(uuid.uuid4())
    
    # Create the message
    message: Message = {
        "message_id": message_id,
        "timestamp": datetime.now().isoformat(),
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "message_type": message_type,
        "priority": priority,
        "content": content,
        "reference_id": reference_id
    }
    
    return message

def create_request(
    sender_id: str,
    recipient_id: str,
    action: str,
    data: Dict[str, Any],
    priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL
) -> Message:
    """
    Create a request message.
    
    Args:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        action: The action being requested
        data: Data for the request
        priority: Priority level
        
    Returns:
        Message: Request message
    """
    content: MessageContent = {
        "action": action,
        "data": data
    }
    
    return create_protocol_message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MessageType.REQUEST,
        content=content,
        priority=priority
    )

def create_response(
    request_message: Message,
    data: Dict[str, Any],
    status: Union[StatusCode, str] = StatusCode.OK,
    error: Optional[str] = None
) -> Message:
    """
    Create a response message to a request.
    
    Args:
        request_message: The original request message
        data: Response data
        status: Status code
        error: Optional error message
        
    Returns:
        Message: Response message
    """
    # Convert enum value to string if necessary
    if isinstance(status, StatusCode):
        status = status.value
    
    content: MessageContent = {
        "data": data,
        "status": status
    }
    
    if error:
        content["error"] = error
    
    return create_protocol_message(
        sender_id=request_message["recipient_id"],
        recipient_id=request_message["sender_id"],
        message_type=MessageType.RESPONSE,
        content=content,
        priority=request_message["priority"],
        reference_id=request_message["message_id"]
    )

def create_error_response(
    request_message: Message,
    error_message: str,
    data: Optional[Dict[str, Any]] = None
) -> Message:
    """
    Create an error response.
    
    Args:
        request_message: The original request message
        error_message: Error description
        data: Optional additional data
        
    Returns:
        Message: Error response message
    """
    return create_response(
        request_message=request_message,
        data=data or {},
        status=StatusCode.ERROR,
        error=error_message
    )

def create_notification(
    sender_id: str,
    recipient_id: str,
    subject: str,
    data: Dict[str, Any],
    priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL
) -> Message:
    """
    Create a notification message.
    
    Args:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        subject: Notification subject
        data: Notification data
        priority: Priority level
        
    Returns:
        Message: Notification message
    """
    content: MessageContent = {
        "action": "notification",
        "data": {
            "subject": subject,
            **data
        }
    }
    
    return create_protocol_message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MessageType.NOTIFICATION,
        content=content,
        priority=priority
    )

def message_to_json(message: Message) -> str:
    """
    Convert a message to JSON string.
    
    Args:
        message: Message object
        
    Returns:
        str: JSON string
    """
    return json.dumps(message)

def json_to_message(json_str: str) -> Message:
    """
    Convert a JSON string to a message.
    
    Args:
        json_str: JSON string
        
    Returns:
        Message: Message object
    """
    message_data = json.loads(json_str)
    return message_data
