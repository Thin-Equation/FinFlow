"""
Utilities for agent-to-agent communication in the FinFlow system.
"""

from typing import Any, Dict, List, Optional, TypeVar, cast
from datetime import datetime
from google.adk.tools.agent_tool import AgentTool # type: ignore

from utils.session_state import get_or_create_session_state

# For improved typing
T = TypeVar('T')

# Message types for agent communication
MESSAGE_TYPE_REQUEST = "request"
MESSAGE_TYPE_RESPONSE = "response"
MESSAGE_TYPE_NOTIFICATION = "notification"
MESSAGE_TYPE_ERROR = "error"

# Priority levels for messages
PRIORITY_LOW = "low"
PRIORITY_NORMAL = "normal"
PRIORITY_HIGH = "high"
PRIORITY_URGENT = "urgent"

def create_agent_tool(agent: Any) -> AgentTool:
    """
    Create an AgentTool for invoking another agent.
    
    Args:
        agent: The agent instance to create a tool for
        
    Returns:
        AgentTool: Tool for invoking the specified agent
    """
    return AgentTool(agent)

def transfer_context(
    source_context: Dict[str, Any], 
    target_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transfer session state and other relevant information between contexts.
    
    Args:
        source_context: Source context dictionary
        target_context: Target context dictionary
        
    Returns:
        Dict[str, Any]: Updated target context
    """
    # Transfer session state if it exists
    if "session_state" in source_context:
        if "session_state" not in target_context:
            target_context["session_state"] = {}
        
        for key, value in source_context["session_state"].items():
            target_context["session_state"][key] = value
    
    # Transfer document information if it exists
    if "document" in source_context:
        target_context["document"] = source_context["document"]
    
    return target_context

def create_agent_tools(agents: List[Any]) -> List[AgentTool]:
    """
    Create a list of AgentTools based on provided agent instances.
    
    Args:
        agents: List of agent instances
        
    Returns:
        List[AgentTool]: List of agent tools
    """
    return [create_agent_tool(agent) for agent in agents]

def create_message(
    sender_id: str,
    recipient_id: str,
    message_type: str,
    content: Dict[str, Any],
    message_id: Optional[str] = None,
    priority: str = PRIORITY_NORMAL,
    reference_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a structured message for agent-to-agent communication.
    
    Args:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        message_type: Type of message (request, response, notification, error)
        content: Message content dictionary
        message_id: Optional unique message identifier (generated if not provided)
        priority: Message priority level
        reference_id: Optional reference to another message ID
        
    Returns:
        Dict[str, Any]: Message dictionary
    """
    timestamp = datetime.now().isoformat()
    
    # Generate a simple message_id if not provided
    if not message_id:
        message_id = f"{sender_id}-{recipient_id}-{timestamp}"
    
    # Create message structure with explicit typing
    message: Dict[str, Any] = {
        "message_id": message_id,
        "timestamp": timestamp,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "message_type": message_type,
        "priority": priority,
        "content": content
    }
    
    # Add reference id if provided
    if reference_id:
        message["reference_id"] = reference_id
        
    return message

def send_message(
    context: Dict[str, Any],
    sender_id: str,
    recipient_id: str,
    message_type: str,
    content: Dict[str, Any],
    priority: str = PRIORITY_NORMAL,
    reference_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a message to another agent via the session state.
    
    Args:
        context: Agent context dictionary
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        message_type: Type of message
        content: Message content dictionary
        priority: Message priority level
        reference_id: Optional reference to another message ID
        
    Returns:
        Dict[str, Any]: The sent message
    """
    # Get the session state
    session_state = get_or_create_session_state(context)
    
    # Create the message
    message = create_message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        content=content,
        priority=priority,
        reference_id=reference_id
    )
    
    # Store messages in session state
    if "messages" not in session_state.data:
        session_state.set("messages", [])
        
    messages = session_state.get("messages", [])
    messages.append(message)
    session_state.set("messages", messages)
    
    # Also store message in recipient's inbox
    inbox_key = f"{recipient_id}_inbox"
    inbox = session_state.get(inbox_key, [])
    inbox.append(message)
    session_state.set(inbox_key, inbox)
    
    # Update the session state in the context
    context["session_state"] = session_state.to_dict()
    
    return message

def get_messages(
    context: Dict[str, Any],
    agent_id: str,
    filter_sender: Optional[str] = None,
    filter_type: Optional[str] = None,
    only_unread: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retrieve messages for an agent from the session state.
    
    Args:
        context: Agent context dictionary
        agent_id: ID of the agent to get messages for
        filter_sender: Optional sender ID to filter by
        filter_type: Optional message type to filter by
        only_unread: Whether to only return unread messages
        
    Returns:
        List[Dict[str, Any]]: List of messages
    """
    # Get the session state
    session_state = get_or_create_session_state(context)
    
    # Get the agent's inbox
    inbox_key = f"{agent_id}_inbox"
    inbox = cast(List[Dict[str, Any]], session_state.get(inbox_key, []))
    
    # Get the agent's read messages
    read_key = f"{agent_id}_read_messages"
    read_message_ids = cast(List[str], session_state.get(read_key, []))
    
    # Apply filters with type hints
    filtered_messages: List[Dict[str, Any]] = []
    for message in inbox:
        # Check if we only want unread messages
        message_id = cast(str, message.get("message_id", ""))
        if only_unread and message_id in read_message_ids:
            continue
            
        # Filter by sender if provided
        sender_id = cast(str, message.get("sender_id", ""))
        if filter_sender and sender_id != filter_sender:
            continue
            
        # Filter by message type if provided
        message_type = cast(str, message.get("message_type", ""))
        if filter_type and message_type != filter_type:
            continue
            
        filtered_messages.append(message)
        
    return filtered_messages

def mark_message_read(
    context: Dict[str, Any],
    agent_id: str,
    message_id: str
) -> None:
    """
    Mark a message as read.
    
    Args:
        context: Agent context dictionary
        agent_id: ID of the agent
        message_id: ID of the message to mark as read
    """
    # Get the session state
    session_state = get_or_create_session_state(context)
    
    # Get the agent's read messages
    read_key = f"{agent_id}_read_messages"
    read_message_ids = session_state.get(read_key, [])
    
    # Add the message ID if not already present
    if message_id not in read_message_ids:
        read_message_ids.append(message_id)
        session_state.set(read_key, read_message_ids)
        
        # Update the session state in the context
        context["session_state"] = session_state.to_dict()

def create_response(
    context: Dict[str, Any],
    original_message: Dict[str, Any],
    content: Dict[str, Any],
    priority: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a response to a previously received message.
    
    Args:
        context: Agent context dictionary
        original_message: The original message to respond to
        content: Response content dictionary
        priority: Message priority level (defaults to original message priority)
        
    Returns:
        Dict[str, Any]: The sent response message
    """
    # Use the original sender as the recipient, with type safety
    recipient_id = cast(str, original_message.get("sender_id", "unknown"))
    
    # Use the original recipient as the sender, with type safety
    sender_id = cast(str, original_message.get("recipient_id", "unknown"))
    
    # Use original priority if not specified, with type safety
    if not priority:
        priority = cast(str, original_message.get("priority", PRIORITY_NORMAL))
    
    # Send the response message
    return send_message(
        context=context,
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MESSAGE_TYPE_RESPONSE,
        content=content,
        priority=priority,
        reference_id=cast(Optional[str], original_message.get("message_id"))
    )
