"""
Utilities for agent-to-agent communication in the FinFlow system.

This module provides a comprehensive framework for agent-to-agent communication,
including message passing, agent tool creation, state management, and delegation patterns.
"""

from typing import Any, Dict, List, Optional, TypeVar, cast, Callable, Union, Tuple
from datetime import datetime
import logging
import uuid
import functools
from enum import Enum

from google.adk.tools import BaseTool, ToolContext # type: ignore
from google.adk.tools.agent_tool import AgentTool # type: ignore

from utils.session_state import get_or_create_session_state
from utils.agent_protocol import (
    MessageType, StatusCode, PriorityLevel,
    create_protocol_message, create_response
)

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

class AgentInvokeTool(BaseTool):
    """
    Tool for direct agent-to-agent communication and invocation.
    Provides a unified interface for agent invocation with built-in state management
    and communication protocol handling.
    """
    
    def __init__(self, target_agent: Any, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize an agent invocation tool.
        
        Args:
            target_agent: The agent instance this tool will invoke
            name: Optional custom name for the tool (default: target agent name + "_invoke")
            description: Optional description of what the agent does
        """
        self.target_agent = target_agent
        
        # Get agent name and generate tool name if not provided
        self.agent_id = getattr(target_agent, "name", str(id(target_agent)))
        tool_name = name or f"{self.agent_id}_invoke"
        
        # Generate description from agent description if available
        agent_desc = getattr(target_agent, "description", "")
        tool_desc = description or f"Invoke the {self.agent_id} agent. {agent_desc}"
        
        # Call the BaseTool constructor with the appropriate name and description
        super().__init__(name=tool_name, description=tool_desc)
        
        self.logger = logging.getLogger(f"finflow.tools.{tool_name}")
    
    def execute(self, parameters: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """
        Execute the tool by invoking the target agent.
        
        Args:
            parameters: Parameters for the agent invocation
            tool_context: Tool context containing caller information
            
        Returns:
            Dict[str, Any]: Results from the agent invocation
        """
        # Get caller information from tool context if available
        caller_id = "unknown"
        if tool_context:
            caller = getattr(tool_context, "caller", None)
            if caller:
                caller_id = getattr(caller, "name", str(id(caller)))
        
        # Create execution context with session state
        context = parameters.copy()
        if tool_context and hasattr(tool_context, "context"):
            # Merge with existing context if available
            if isinstance(tool_context.context, dict):
                for key, value in tool_context.context.items():
                    if key not in context:
                        context[key] = value
        
        # Ensure session state exists in context
        session_state = get_or_create_session_state(context)
        context["session_state"] = session_state.to_dict()
        
        # Record invocation in context
        invocation_id = str(uuid.uuid4())
        context["invocation_id"] = invocation_id
        context["caller_id"] = caller_id
        context["timestamp"] = datetime.now().isoformat()
        
        # Log the invocation
        self.logger.debug(f"Invoking agent {self.agent_id} from {caller_id} with context: {context}")
        
        # Invoke the target agent
        try:
            # Try different agent interfaces (handle various agent implementations)
            if hasattr(self.target_agent, "process") and callable(getattr(self.target_agent, "process")):
                result = self.target_agent.process(context, tool_context)
            elif hasattr(self.target_agent, "execute") and callable(getattr(self.target_agent, "execute")):
                result = self.target_agent.execute(context)
            elif hasattr(self.target_agent, "run") and callable(getattr(self.target_agent, "run")):
                result = self.target_agent.run(context)
            elif callable(self.target_agent):
                result = self.target_agent(context)
            else:
                raise ValueError(f"Target agent {self.agent_id} has no valid execution interface")
                
            # Ensure we return a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
                
            # Add invocation tracking
            result["invocation_id"] = invocation_id
            result["agent_id"] = self.agent_id
            
            # Return the result
            return result
            
        except Exception as e:
            # Log the exception
            self.logger.error(f"Error invoking agent {self.agent_id}: {e}")
            
            # Return error information
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "invocation_id": invocation_id,
                "status": "error"
            }

def create_agent_tool(agent: Any) -> AgentTool:
    """
    Create an AgentTool for invoking another agent via the Google ADK.
    
    Args:
        agent: The agent instance to create a tool for
        
    Returns:
        AgentTool: Tool for invoking the specified agent
    """
    return AgentTool(agent)

def create_enhanced_agent_tool(agent: Any, name: Optional[str] = None, description: Optional[str] = None) -> AgentInvokeTool:
    """
    Create an enhanced agent tool with built-in state management and protocol handling.
    
    Args:
        agent: The agent instance to create a tool for
        name: Optional custom name for the tool
        description: Optional description of what the agent does
        
    Returns:
        AgentInvokeTool: Enhanced tool for invoking the specified agent
    """
    return AgentInvokeTool(agent, name, description)

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

# State transitions and management
class WorkflowState(str, Enum):
    """Enum for workflow state management."""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"
    SUSPENDED = "suspended"

def get_workflow_state(context: Dict[str, Any], workflow_id: str) -> str:
    """
    Get current workflow state.
    
    Args:
        context: Agent context dictionary
        workflow_id: Workflow identifier
        
    Returns:
        str: Current workflow state
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    workflow = workflows.get(workflow_id, {})
    return workflow.get("state", WorkflowState.INITIALIZED)

def update_workflow_state(
    context: Dict[str, Any], 
    workflow_id: str, 
    new_state: Union[WorkflowState, str],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update the state of a workflow.
    
    Args:
        context: Agent context dictionary
        workflow_id: Workflow identifier
        new_state: New workflow state
        metadata: Optional metadata to store with state update
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    
    # Ensure the workflow exists
    if workflow_id not in workflows:
        workflows[workflow_id] = {}
    
    # Update the state
    if isinstance(new_state, WorkflowState):
        new_state = new_state.value
        
    workflows[workflow_id]["state"] = new_state
    workflows[workflow_id]["last_updated"] = datetime.now().isoformat()
    
    # Add metadata if provided
    if metadata:
        if "metadata" not in workflows[workflow_id]:
            workflows[workflow_id]["metadata"] = {}
            
        for key, value in metadata.items():
            workflows[workflow_id]["metadata"][key] = value
    
    # Save back to session state
    session_state.set("workflows", workflows)
    
    # Update context
    context["session_state"] = session_state.to_dict()

def create_workflow(
    context: Dict[str, Any],
    workflow_id: Optional[str] = None,
    workflow_type: str = "default",
    owner_id: Optional[str] = None,
    initial_state: Union[WorkflowState, str] = WorkflowState.INITIALIZED,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new workflow and initialize its state.
    
    Args:
        context: Agent context dictionary
        workflow_id: Optional workflow identifier (generated if not provided)
        workflow_type: Type of workflow
        owner_id: ID of the agent that owns this workflow
        initial_state: Initial workflow state
        metadata: Optional metadata for the workflow
        
    Returns:
        str: Workflow identifier
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    
    # Generate workflow_id if not provided
    if not workflow_id:
        workflow_id = str(uuid.uuid4())
    
    # Check for existing workflow
    if workflow_id in workflows:
        raise ValueError(f"Workflow {workflow_id} already exists")
    
    # Convert enum to value if needed
    if isinstance(initial_state, WorkflowState):
        initial_state = initial_state.value
    
    # Create the workflow
    workflows[workflow_id] = {
        "id": workflow_id,
        "type": workflow_type,
        "owner_id": owner_id or context.get("agent_id", "unknown"),
        "state": initial_state,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "metadata": metadata or {},
        "history": []
    }
    
    # Save back to session state
    session_state.set("workflows", workflows)
    
    # Update context
    context["session_state"] = session_state.to_dict()
    context["current_workflow_id"] = workflow_id
    
    return workflow_id

def transition_workflow(
    context: Dict[str, Any],
    workflow_id: str,
    from_state: Union[WorkflowState, str, None],
    to_state: Union[WorkflowState, str],
    agent_id: Optional[str] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Transition a workflow from one state to another, with validation and history tracking.
    
    Args:
        context: Agent context dictionary
        workflow_id: Workflow identifier
        from_state: Expected current state (transition fails if current state doesn't match)
        to_state: Target state for the transition
        agent_id: ID of the agent performing the transition
        reason: Optional reason for the state transition
        metadata: Optional metadata for the transition
        
    Returns:
        bool: Whether the transition was successful
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    
    # Check if workflow exists
    if workflow_id not in workflows:
        return False
    
    # Convert enum values to strings if needed
    current_state = workflows[workflow_id]["state"]
    if isinstance(from_state, WorkflowState):
        from_state = from_state.value
    if isinstance(to_state, WorkflowState):
        to_state = to_state.value
        
    # Validate current state if from_state is provided
    if from_state is not None and current_state != from_state:
        return False
        
    # Get performer ID
    performer_id = agent_id or context.get("agent_id", "unknown")
    
    # Create transition record
    transition = {
        "timestamp": datetime.now().isoformat(),
        "from_state": current_state,
        "to_state": to_state,
        "performer_id": performer_id
    }
    
    if reason:
        transition["reason"] = reason
    if metadata:
        transition["metadata"] = metadata
        
    # Update workflow state
    workflows[workflow_id]["state"] = to_state
    workflows[workflow_id]["last_updated"] = transition["timestamp"]
    
    # Update workflow history
    if "history" not in workflows[workflow_id]:
        workflows[workflow_id]["history"] = []
        
    workflows[workflow_id]["history"].append(transition)
    
    # Update metadata if provided
    if metadata:
        if "metadata" not in workflows[workflow_id]:
            workflows[workflow_id]["metadata"] = {}
            
        for key, value in metadata.items():
            workflows[workflow_id]["metadata"][key] = value
    
    # Save back to session state
    session_state.set("workflows", workflows)
    
    # Update context
    context["session_state"] = session_state.to_dict()
    
    return True

def get_workflow_data(context: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
    """
    Get all data related to a specific workflow.
    
    Args:
        context: Agent context dictionary
        workflow_id: Workflow identifier
        
    Returns:
        Dict[str, Any]: Workflow data including state, history, and metadata
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    
    # Return workflow data or empty dict if not found
    return workflows.get(workflow_id, {})

def get_active_workflows(context: Dict[str, Any], agent_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all active workflows (not in terminal states).
    
    Args:
        context: Agent context dictionary
        agent_id: Optional agent ID to filter workflows by owner
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of active workflows
    """
    session_state = get_or_create_session_state(context)
    workflows = session_state.get("workflows", {})
    
    # Terminal states
    terminal_states = [
        WorkflowState.COMPLETED.value, 
        WorkflowState.FAILED.value, 
        WorkflowState.CANCELLED.value
    ]
    
    # Filter workflows
    active_workflows = {}
    for wf_id, workflow in workflows.items():
        # Skip workflow if in terminal state
        if workflow.get("state") in terminal_states:
            continue
            
        # Filter by agent_id if provided
        if agent_id and workflow.get("owner_id") != agent_id:
            continue
            
        active_workflows[wf_id] = workflow
        
    return active_workflows

# LLM-driven delegation patterns

class DelegationStrategy(str, Enum):
    """Enum for delegation strategies."""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    AVAILABILITY_BASED = "availability_based"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    LEARNING_BASED = "learning_based"  # New strategy using historical performance data

def create_delegation_request(
    context: Dict[str, Any],
    task_description: str,
    required_capabilities: List[str],
    priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
    deadline: Optional[str] = None,
    strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a delegation request for LLM-driven task delegation.
    
    Args:
        context: Agent context dictionary
        task_description: Description of the task to be delegated
        required_capabilities: List of capabilities required for the task
        priority: Priority of the task
        deadline: Optional deadline for task completion
        strategy: Delegation strategy to use
        metadata: Optional additional metadata
        
    Returns:
        Dict[str, Any]: Delegation request object
    """
    delegator_id = context.get("agent_id", "unknown")
    
    # Convert enum to string if needed
    if isinstance(strategy, DelegationStrategy):
        strategy = strategy.value
    if isinstance(priority, PriorityLevel):
        priority = priority.value
    
    # Create request
    delegation_request = {
        "request_id": str(uuid.uuid4()),
        "delegator_id": delegator_id,
        "task_description": task_description,
        "required_capabilities": required_capabilities,
        "priority": priority,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat(),
        "status": "pending"
    }
    
    if deadline:
        delegation_request["deadline"] = deadline
    
    if metadata:
        delegation_request["metadata"] = metadata
    
    # Add the request to delegation queue in session state
    session_state = get_or_create_session_state(context)
    delegation_queue = session_state.get("delegation_queue", [])
    delegation_queue.append(delegation_request)
    session_state.set("delegation_queue", delegation_queue)
    
    # Update the context
    context["session_state"] = session_state.to_dict()
    
    return delegation_request

def find_suitable_agent(
    context: Dict[str, Any],
    delegation_request: Dict[str, Any],
    available_agents: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    Find a suitable agent for a delegated task using LLM reasoning.
    
    Args:
        context: Agent context dictionary
        delegation_request: Delegation request object
        available_agents: Dictionary of available agents with their capabilities
        
    Returns:
        Optional[str]: ID of the selected agent or None if no suitable agent found
    """
    # Check if we have agents available
    if not available_agents:
        return None
    
    # Extract required capabilities from delegation request
    required_capabilities = delegation_request.get("required_capabilities", [])
    task_description = delegation_request.get("task_description", "")
    task_priority = delegation_request.get("priority", PriorityLevel.NORMAL)
    
    # Dictionary to track capability matches for each agent
    agent_matches = {}
    
    # First pass: Check for direct capability matches
    for agent_id, agent_data in available_agents.items():
        # Get agent capabilities
        agent_capabilities = agent_data.get("capabilities", [])
        
        # Calculate capability match score
        match_count = 0
        for req_capability in required_capabilities:
            if req_capability in agent_capabilities:
                match_count += 1
        
        # Store match information with match percentage
        match_percent = 0
        if required_capabilities:
            match_percent = (match_count / len(required_capabilities)) * 100
            
        agent_matches[agent_id] = {
            "agent_id": agent_id,
            "match_count": match_count,
            "match_percent": match_percent,
            "total_required": len(required_capabilities),
            "status": agent_data.get("status", {}),
            "agent_info": agent_data
        }
    
    # Find agents with highest match percentage
    max_match_percent = 0
    matched_agents = []
    
    for agent_id, match_data in agent_matches.items():
        if match_data["match_percent"] > max_match_percent:
            max_match_percent = match_data["match_percent"]
            matched_agents = [agent_id]
        elif match_data["match_percent"] == max_match_percent and max_match_percent > 0:
            matched_agents.append(agent_id)
    
    # If we have perfect matches, select based on agent load/availability
    if max_match_percent == 100 and matched_agents:
        # Choose agent with highest availability and lowest load
        best_agent_id = None
        best_score = -1
        
        for agent_id in matched_agents:
            agent_status = agent_matches[agent_id].get("status", {})
            availability = float(agent_status.get("availability", 0.0))
            load = float(agent_status.get("current_load", 1.0))
            
            # Calculate score (higher is better)
            score = availability * (1.0 - load)
            
            if score > best_score:
                best_score = score
                best_agent_id = agent_id
        
        return best_agent_id
    
    # If no perfect matches but some matches, use the best match
    if max_match_percent > 60 and matched_agents:  # Threshold for acceptable match
        return matched_agents[0]
    
    # Track delegation decisions to improve future matching
    track_delegation_decision(
        context=context,
        task_description=task_description,
        required_capabilities=required_capabilities,
        selected_agent=matched_agents[0] if matched_agents else None,
        match_percent=max_match_percent,
        success=matched_agents is not None and len(matched_agents) > 0
    )
    
    # If we have matches but below threshold, still return the best match
    if matched_agents:
        return matched_agents[0]
    
    # No suitable agent found
    return None

def track_delegation_decision(
    context: Dict[str, Any],
    task_description: str,
    required_capabilities: List[str],
    selected_agent: Optional[str],
    match_percent: float,
    success: bool
) -> None:
    """
    Track delegation decisions to improve future agent matching.
    
    Args:
        context: Agent context dictionary
        task_description: Description of the delegated task
        required_capabilities: List of required capabilities
        selected_agent: ID of the selected agent (if any)
        match_percent: Match percentage of the selected agent
        success: Whether delegation was successful
    """
    # Get the session state
    session_state = get_or_create_session_state(context)
    
    # Get delegation history
    delegation_history = session_state.get("delegation_history", [])
    
    # Create history entry
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_description": task_description,
        "required_capabilities": required_capabilities,
        "selected_agent": selected_agent,
        "match_percent": match_percent,
        "success": success
    }
    
    # Add entry to history
    delegation_history.append(history_entry)
    
    # Update session state (keep only the last 100 entries)
    if len(delegation_history) > 100:
        delegation_history = delegation_history[-100:]
        
    session_state.set("delegation_history", delegation_history)
    
    # Update context
    context["session_state"] = session_state.to_dict()

def delegate_task(
    context: Dict[str, Any],
    task_description: str,
    required_capabilities: List[str],
    available_agents: Dict[str, Dict[str, Any]],
    priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
    metadata: Optional[Dict[str, Any]] = None,
    strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED
) -> Tuple[bool, Dict[str, Any]]:
    """
    Delegate a task to a suitable agent using LLM reasoning.
    
    Args:
        context: Agent context dictionary
        task_description: Description of the task to delegate
        required_capabilities: List of capabilities required for the task
        available_agents: Dictionary of available agents with their capabilities
        priority: Priority of the task
        metadata: Optional additional metadata
        strategy: Delegation strategy to use
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and delegation result
    """
    # Create delegation request
    delegation_request = create_delegation_request(
        context=context,
        task_description=task_description,
        required_capabilities=required_capabilities,
        priority=priority,
        strategy=strategy,
        metadata=metadata
    )
    
    # Find suitable agent
    selected_agent_id = apply_delegation_strategy(
        context=context,
        delegation_request=delegation_request,
        available_agents=available_agents,
        strategy=strategy
    )
    
    if not selected_agent_id:
        # Update request status
        delegation_request["status"] = "failed"
        delegation_request["reason"] = "No suitable agent found"
        
        # Save updated request
        session_state = get_or_create_session_state(context)
        delegation_queue = session_state.get("delegation_queue", [])
        
        # Find and update the request in the queue
        for i, req in enumerate(delegation_queue):
            if req.get("request_id") == delegation_request["request_id"]:
                delegation_queue[i] = delegation_request
                break
                
        session_state.set("delegation_queue", delegation_queue)
        context["session_state"] = session_state.to_dict()
        
        return False, delegation_request
    
    # Create a workflow for this delegation
    workflow_id = create_workflow(
        context=context,
        workflow_type="task_delegation",
        owner_id=context.get("agent_id", "unknown"),
        initial_state=WorkflowState.DELEGATED,
        metadata={
            "delegation_request_id": delegation_request["request_id"],
            "delegatee_id": selected_agent_id,
            "task_description": task_description
        }
    )
    
    # Update delegation request
    delegation_request["status"] = "delegated"
    delegation_request["delegatee_id"] = selected_agent_id
    delegation_request["workflow_id"] = workflow_id
    
    # Save updated request
    session_state = get_or_create_session_state(context)
    delegation_queue = session_state.get("delegation_queue", [])
    
    # Find and update the request in the queue
    for i, req in enumerate(delegation_queue):
        if req.get("request_id") == delegation_request["request_id"]:
            delegation_queue[i] = delegation_request
            break
            
    session_state.set("delegation_queue", delegation_queue)
    context["session_state"] = session_state.to_dict()
    
    # Send a message to the selected agent
    message_content = {
        "action": "delegated_task",
        "data": {
            "task_description": task_description,
            "required_capabilities": required_capabilities,
            "priority": delegation_request["priority"],
            "workflow_id": workflow_id,
            "delegation_request_id": delegation_request["request_id"]
        }
    }
    
    if metadata:
        message_content["data"]["metadata"] = metadata
    
    send_message(
        context=context,
        sender_id=context.get("agent_id", "unknown"),
        recipient_id=selected_agent_id,
        message_type=MessageType.REQUEST.value,
        content=message_content,
        priority=delegation_request["priority"]
    )
    
    return True, delegation_request

def complete_delegated_task(
    context: Dict[str, Any],
    delegation_request_id: str,
    result: Dict[str, Any],
    status: StatusCode = StatusCode.OK
) -> bool:
    """
    Complete a delegated task by sending results back to the delegator.
    
    Args:
        context: Agent context dictionary
        delegation_request_id: ID of the original delegation request
        result: Task execution result
        status: Status code for the task execution
        
    Returns:
        bool: Success flag
    """
    # Get the delegation request
    session_state = get_or_create_session_state(context)
    delegation_queue = session_state.get("delegation_queue", [])
    
    delegation_request = None
    for req in delegation_queue:
        if req.get("request_id") == delegation_request_id:
            delegation_request = req
            break
    
    if not delegation_request:
        return False
    
    # Get the workflow
    workflow_id = delegation_request.get("workflow_id")
    if not workflow_id:
        return False
        
    workflow = get_workflow_data(context, workflow_id)
    if not workflow:
        return False
    
    # Update workflow state
    if isinstance(status, StatusCode):
        status_val = status.value
    else:
        status_val = status
        
    new_state = WorkflowState.COMPLETED if status_val == StatusCode.OK.value else WorkflowState.FAILED
    
    transition_workflow(
        context=context,
        workflow_id=workflow_id,
        from_state=None,  # No validation of current state
        to_state=new_state,
        agent_id=context.get("agent_id", "unknown"),
        reason="Task delegation completed",
        metadata={"result": result}
    )
    
    # Update delegation request
    delegation_request["status"] = "completed" if status_val == StatusCode.OK.value else "failed"
    delegation_request["completion_time"] = datetime.now().isoformat()
    delegation_request["result"] = result
    
    # Find and update the request in the queue
    for i, req in enumerate(delegation_queue):
        if req.get("request_id") == delegation_request_id:
            delegation_queue[i] = delegation_request
            break
            
    session_state.set("delegation_queue", delegation_queue)
    
    # Send completion message to delegator
    message_content = {
        "action": "delegated_task_completed",
        "data": {
            "delegation_request_id": delegation_request_id,
            "workflow_id": workflow_id,
            "result": result,
            "status": status_val
        }
    }
    
    send_message(
        context=context,
        sender_id=context.get("agent_id", "unknown"),
        recipient_id=delegation_request["delegator_id"],
        message_type=MessageType.RESPONSE.value,
        content=message_content
    )
    
    # Update context
    context["session_state"] = session_state.to_dict()
    
    return True

def get_agent_registry(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get the current agent registry from session state.
    
    Args:
        context: Agent context dictionary
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of registered agents and their capabilities
    """
    session_state = get_or_create_session_state(context)
    return session_state.get("agent_registry", {})

def register_agent_capabilities(
    context: Dict[str, Any],
    agent_id: str,
    capabilities: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register an agent and its capabilities in the registry.
    
    Args:
        context: Agent context dictionary
        agent_id: Agent identifier
        capabilities: List of agent capabilities
        metadata: Optional agent metadata
    """
    session_state = get_or_create_session_state(context)
    agent_registry = session_state.get("agent_registry", {})
    
    # Create or update agent entry
    if agent_id not in agent_registry:
        agent_registry[agent_id] = {}
    
    # Update capabilities
    agent_registry[agent_id]["capabilities"] = capabilities
    agent_registry[agent_id]["last_updated"] = datetime.now().isoformat()
    
    # Add metadata if provided
    if metadata:
        if "metadata" not in agent_registry[agent_id]:
            agent_registry[agent_id]["metadata"] = {}
            
        for key, value in metadata.items():
            agent_registry[agent_id]["metadata"][key] = value
    
    # Save back to session state
    session_state.set("agent_registry", agent_registry)
    context["session_state"] = session_state.to_dict()

def update_agent_status(
    context: Dict[str, Any],
    agent_id: str,
    status: str,
    current_load: float = 0.0,
    availability: float = 1.0
) -> None:
    """
    Update an agent's runtime status in the registry.
    
    Args:
        context: Agent context dictionary
        agent_id: Agent identifier
        status: Current status (e.g., "active", "busy", "idle")
        current_load: Current load factor (0.0-1.0)
        availability: Current availability factor (0.0-1.0)
    """
    session_state = get_or_create_session_state(context)
    agent_registry = session_state.get("agent_registry", {})
    
    # Create agent entry if it doesn't exist
    if agent_id not in agent_registry:
        agent_registry[agent_id] = {}
    
    # Update status
    agent_registry[agent_id]["status"] = status
    agent_registry[agent_id]["current_load"] = max(0.0, min(1.0, current_load))  # Clamp between 0 and 1
    agent_registry[agent_id]["availability"] = max(0.0, min(1.0, availability))  # Clamp between 0 and 1
    agent_registry[agent_id]["last_updated"] = datetime.now().isoformat()
    
    # Save back to session state
    session_state.set("agent_registry", agent_registry)
    context["session_state"] = session_state.to_dict()

def track_agent_metrics(
    context: Dict[str, Any],
    agent_id: str,
    execution_time: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Track agent performance metrics for adaptive delegation.
    
    Args:
        context: Agent context dictionary
        agent_id: Agent identifier
        execution_time: Task execution time in seconds
        success: Whether the task was successful
        metadata: Optional additional metrics
    """
    session_state = get_or_create_session_state(context)
    agent_registry = session_state.get("agent_registry", {})
    
    # Create agent entry if it doesn't exist
    if agent_id not in agent_registry:
        agent_registry[agent_id] = {}
    
    # Initialize metrics if they don't exist
    if "metrics" not in agent_registry[agent_id]:
        agent_registry[agent_id]["metrics"] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "success_rate": 0.0
        }
    
    metrics = agent_registry[agent_id]["metrics"]
    
    # Update metrics
    metrics["total_tasks"] += 1
    if success:
        metrics["successful_tasks"] += 1
    metrics["total_execution_time"] += execution_time
    metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["total_tasks"]
    metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"]
    
    # Add custom metrics if provided
    if metadata:
        for key, value in metadata.items():
            metrics[key] = value
    
    # Save back to session state
    session_state.set("agent_registry", agent_registry)
    context["session_state"] = session_state.to_dict()

def get_agent_metrics(
    context: Dict[str, Any],
    agent_id: str
) -> Dict[str, Any]:
    """
    Get performance metrics for an agent.
    
    Args:
        context: Agent context dictionary
        agent_id: Agent identifier
        
    Returns:
        Dict[str, Any]: Agent metrics
    """
    session_state = get_or_create_session_state(context)
    agent_registry = session_state.get("agent_registry", {})
    
    if agent_id in agent_registry and "metrics" in agent_registry[agent_id]:
        return agent_registry[agent_id]["metrics"]
    
    return {}

def broadcast_message(
    context: Dict[str, Any],
    sender_id: str,
    message_type: str,
    content: Dict[str, Any],
    recipient_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    priority: str = PRIORITY_NORMAL
) -> List[Dict[str, Any]]:
    """
    Broadcast a message to multiple agents.
    
    Args:
        context: Agent context dictionary
        sender_id: ID of the sending agent
        message_type: Type of message
        content: Message content
        recipient_filter: Optional filter function to determine recipients
        priority: Message priority
        
    Returns:
        List[Dict[str, Any]]: List of sent messages
    """
    session_state = get_or_create_session_state(context)
    agent_registry = session_state.get("agent_registry", {})
    
    messages = []
    
    # Iterate through registered agents
    for agent_id, agent_info in agent_registry.items():
        # Skip sender itself
        if agent_id == sender_id:
            continue
            
        # Apply filter if provided
        if recipient_filter and not recipient_filter(agent_id, agent_info):
            continue
            
        # Send message to this agent
        message = send_message(
            context=context,
            sender_id=sender_id,
            recipient_id=agent_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        messages.append(message)
    
    return messages

def agent_task_decorator(
    task_name: str, 
    required_capabilities: List[str] = None,
    track_metrics: bool = True
) -> Callable:
    """
    Decorator for agent task functions that handles communication protocol and metrics tracking.
    
    Args:
        task_name: Name of the task
        required_capabilities: List of capabilities required for this task
        track_metrics: Whether to track performance metrics
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(agent: Any, context: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
            # Get agent ID
            agent_id = getattr(agent, "name", str(id(agent)))
            context["agent_id"] = agent_id
            
            # Register this execution in the workflow if workflow_id exists
            workflow_id = context.get("workflow_id")
            if workflow_id:
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=None,  # No validation
                    to_state=WorkflowState.IN_PROGRESS,
                    agent_id=agent_id,
                    reason=f"Executing task {task_name}"
                )
            
            # Record start time if tracking metrics
            start_time = datetime.now() if track_metrics else None
            
            # Update agent status to busy
            update_agent_status(
                context=context,
                agent_id=agent_id,
                status="busy",
                current_load=0.8,
                availability=0.2
            )
            
            # Execute the task function
            try:
                result = func(agent, context, *args, **kwargs)
                success = True
                
                # Update workflow if applicable
                if workflow_id:
                    transition_workflow(
                        context=context,
                        workflow_id=workflow_id,
                        from_state=None,  # No validation
                        to_state=WorkflowState.COMPLETED,
                        agent_id=agent_id,
                        reason=f"Completed task {task_name}",
                        metadata={"result": result}
                    )
            except Exception as e:
                # Handle the exception
                if hasattr(agent, "handle_error") and callable(agent.handle_error):
                    result = agent.handle_error(e, context)
                else:
                    result = {"error": str(e), "status": "error"}
                
                success = False
                
                # Update workflow if applicable
                if workflow_id:
                    transition_workflow(
                        context=context,
                        workflow_id=workflow_id,
                        from_state=None,  # No validation
                        to_state=WorkflowState.FAILED,
                        agent_id=agent_id,
                        reason=f"Failed task {task_name}: {str(e)}",
                        metadata={"error": str(e)}
                    )
            
            # Record end time and track metrics if enabled
            if track_metrics and start_time:
                execution_time = (datetime.now() - start_time).total_seconds()
                track_agent_metrics(
                    context=context,
                    agent_id=agent_id,
                    execution_time=execution_time,
                    success=success,
                    metadata={
                        "task_name": task_name,
                        "task_completion_time": execution_time
                    }
                )
            
            # Update agent status back to available
            update_agent_status(
                context=context,
                agent_id=agent_id,
                status="active",
                current_load=0.2,
                availability=0.8
            )
            
            return result
        
        # Store task metadata in the function
        wrapper.__task_name__ = task_name
        wrapper.__required_capabilities__ = required_capabilities or []
        
        return wrapper
    
    return decorator

def apply_delegation_strategy(
    context: Dict[str, Any],
    delegation_request: Dict[str, Any],
    available_agents: Dict[str, Dict[str, Any]],
    strategy: Union[DelegationStrategy, str]
) -> Optional[str]:
    """
    Apply a specific delegation strategy to select an agent.
    
    Args:
        context: Agent context dictionary
        delegation_request: Delegation request object
        available_agents: Dictionary of available agents and their capabilities
        strategy: Delegation strategy to apply
        
    Returns:
        Optional[str]: Selected agent ID or None if no suitable agent found
    """
    # Convert enum to string if needed
    if isinstance(strategy, DelegationStrategy):
        strategy_value = strategy.value
    else:
        strategy_value = strategy
        
    # Get session state for tracking rounds
    session_state = get_or_create_session_state(context)
    
    # Apply the specified strategy
    if strategy_value == DelegationStrategy.ROUND_ROBIN.value:
        # Round-robin strategy: distribute tasks evenly among agents
        last_used_idx = session_state.get("round_robin_index", -1)
        agent_ids = list(available_agents.keys())
        
        if not agent_ids:
            return None
            
        next_idx = (last_used_idx + 1) % len(agent_ids)
        selected_agent = agent_ids[next_idx]
        
        # Update index in session state
        session_state.set("round_robin_index", next_idx)
        context["session_state"] = session_state.to_dict()
        
        return selected_agent
        
    elif strategy_value == DelegationStrategy.CAPABILITY_BASED.value:
        # Already implemented by find_suitable_agent
        return find_suitable_agent(context, delegation_request, available_agents)
        
    elif strategy_value == DelegationStrategy.AVAILABILITY_BASED.value:
        # Availability-based: choose the most available agent
        best_agent = None
        best_availability = -1
        
        for agent_id, agent_data in available_agents.items():
            status = agent_data.get("status", {})
            availability = float(status.get("availability", 0.0))
            
            if availability > best_availability:
                best_availability = availability
                best_agent = agent_id
                
        return best_agent
        
    elif strategy_value == DelegationStrategy.LOAD_BALANCED.value:
        # Load-balanced: choose the agent with the lowest load
        best_agent = None
        lowest_load = float('inf')
        
        for agent_id, agent_data in available_agents.items():
            status = agent_data.get("status", {})
            load = float(status.get("current_load", 1.0))
            
            if load < lowest_load:
                lowest_load = load
                best_agent = agent_id
                
        return best_agent
        
    elif strategy_value == DelegationStrategy.PRIORITY_BASED.value:
        # Priority-based: high priority tasks go to more capable agents
        priority = delegation_request.get("priority", PriorityLevel.NORMAL.value)
        required_capabilities = delegation_request.get("required_capabilities", [])
        
        # Different selection criteria based on priority
        if priority == PriorityLevel.URGENT.value or priority == PriorityLevel.HIGH.value:
            # For high priority, focus on capability matches
            return find_suitable_agent(context, delegation_request, available_agents)
        else:
            # For normal/low priority, focus on load balancing
            best_agent = None
            best_score = -1
            
            for agent_id, agent_data in available_agents.items():
                # Get agent capabilities and status
                agent_capabilities = agent_data.get("capabilities", [])
                status = agent_data.get("status", {})
                load = float(status.get("current_load", 1.0))
                
                # Calculate basic capability match
                match_count = sum(1 for cap in required_capabilities if cap in agent_capabilities)
                
                # For lower priority tasks, load is more important than perfect capability match
                score = match_count * 0.5 + (1.0 - load) * 0.5
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
                    
            return best_agent
            
    elif strategy_value == DelegationStrategy.ADAPTIVE.value:
        # Adaptive: consider past performance, current load, capability match
        required_capabilities = delegation_request.get("required_capabilities", [])
        best_agent = None
        best_score = -1
        
        for agent_id, agent_data in available_agents.items():
            # Get agent capabilities and status
            agent_capabilities = agent_data.get("capabilities", [])
            status = agent_data.get("status", {})
            metrics = agent_data.get("metrics", {})
            
            # Extract relevant metrics
            load = float(status.get("current_load", 1.0))
            availability = float(status.get("availability", 0.0))
            success_rate = float(metrics.get("success_rate", 0.5))
            avg_execution_time = float(metrics.get("avg_execution_time", 1.0))
            
            # Normalize execution time (lower is better) - assume 10s is a good baseline
            norm_exec_time = min(1.0, 10.0 / max(0.1, avg_execution_time))
            
            # Calculate capability match
            match_count = sum(1 for cap in required_capabilities if cap in agent_capabilities)
            match_percent = match_count / len(required_capabilities) if required_capabilities else 0
            
            # Calculate overall score using weighted factors
            score = (
                match_percent * 0.4 +          # Capability match: 40%
                success_rate * 0.3 +           # Success rate: 30%
                (1.0 - load) * 0.15 +          # Low load: 15%
                availability * 0.1 +           # Availability: 10%
                norm_exec_time * 0.05          # Execution time: 5%
            )
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
                
        return best_agent
    
    elif strategy_value == DelegationStrategy.LEARNING_BASED.value:
        # Learning-based: use historical task-agent matching data
        required_capabilities = delegation_request.get("required_capabilities", [])
        task_description = delegation_request.get("task_description", "")
        
        # Get historical delegation data
        delegation_history = session_state.get("delegation_history", [])
        
        # Find similar tasks in history
        similar_tasks = []
        for entry in delegation_history:
            # Simple similarity check - can be enhanced with embeddings/ML
            history_caps = set(entry.get("required_capabilities", []))
            current_caps = set(required_capabilities)
            
            # Calculate Jaccard similarity
            intersection = len(history_caps.intersection(current_caps))
            union = len(history_caps.union(current_caps))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.6:  # Threshold for similarity
                similar_tasks.append((entry, similarity))
        
        # If we have similar tasks, use the most successful agent
        if similar_tasks:
            # Sort by similarity
            similar_tasks.sort(key=lambda x: x[1], reverse=True)
            
            # Count agent frequencies weighted by success and similarity
            agent_scores = {}
            for entry, similarity in similar_tasks[:10]:  # Use top 10 similar tasks
                agent_id = entry.get("selected_agent")
                if not agent_id:
                    continue
                    
                # Skip if agent no longer available
                if agent_id not in available_agents:
                    continue
                    
                success = entry.get("success", False)
                
                # Calculate score based on similarity and success
                score = similarity * (1.0 if success else 0.2)
                
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = 0
                agent_scores[agent_id] += score
            
            # If we have scores, select the highest scoring agent
            if agent_scores:
                best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
                return best_agent
        
        # Fall back to capability-based if learning strategy doesn't yield results
        return find_suitable_agent(context, delegation_request, available_agents)
        
    else:
        # Default to capability-based matching
        return find_suitable_agent(context, delegation_request, available_agents)

class CommunicationProtocol:
    """
    Communication protocol handler for robust agent communication.
    
    This class provides a standardized way to handle robust communication between agents,
    including message delivery guarantees, retries, and acknowledgments.
    """
    
    def __init__(self, context: Dict[str, Any], agent_id: str):
        """
        Initialize the communication protocol handler.
        
        Args:
            context: Agent context dictionary
            agent_id: ID of the agent using this protocol handler
        """
        self.context = context
        self.agent_id = agent_id
        self.session_state = get_or_create_session_state(context)
        self.logger = logging.getLogger(f"finflow.communication.{agent_id}")
    
    def send_message(
        self, 
        recipient_id: str,
        message_type: Union[MessageType, str],
        content: Dict[str, Any],
        priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
        reference_id: Optional[str] = None,
        delivery_guarantees: bool = False,
        timeout: float = 30.0,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Send a message to another agent with robust delivery guarantees.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message
            content: Message content
            priority: Message priority
            reference_id: Optional reference to another message
            delivery_guarantees: Whether to ensure delivery
            timeout: Timeout period for delivery confirmation in seconds
            retry_count: Number of retries for delivery
            
        Returns:
            Dict[str, Any]: The sent message
        """
        # Convert enums to strings
        if isinstance(message_type, MessageType):
            message_type = message_type.value
        if isinstance(priority, PriorityLevel):
            priority = priority.value
            
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Create protocol-compliant message
        message = create_protocol_message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            message_id=message_id,
            priority=priority,
            reference_id=reference_id
        )
        
        # Send the message
        self.logger.debug(f"Sending message {message_id} to {recipient_id}")
        
        # Store in session state
        messages = self.session_state.get("messages", [])
        messages.append(message)
        self.session_state.set("messages", messages)
        
        # Add to recipient's inbox
        inbox_key = f"{recipient_id}_inbox"
        inbox = self.session_state.get(inbox_key, [])
        inbox.append(message)
        self.session_state.set(inbox_key, inbox)
        
        # If delivery guarantees are requested
        if delivery_guarantees:
            # Add to pending acknowledgments
            pending_acks = self.session_state.get("pending_acknowledgments", {})
            pending_acks[message_id] = {
                "message": message,
                "sent_time": datetime.now().isoformat(),
                "retries_left": retry_count,
                "timeout": timeout,
                "status": "pending"
            }
            self.session_state.set("pending_acknowledgments", pending_acks)
            
            # TODO: In a real implementation, you would set up a retry mechanism
            # For now, we'll just log this intent
            self.logger.debug(f"Message {message_id} set for delivery guarantees with {retry_count} retries")
        
        # Update context
        self.context["session_state"] = self.session_state.to_dict()
        
        return message
    
    def acknowledge_message(self, message_id: str, status: str = "received") -> None:
        """
        Acknowledge receipt of a message.
        
        Args:
            message_id: ID of the message to acknowledge
            status: Acknowledgment status
        """
        # Find the message
        messages = self.session_state.get("messages", [])
        for message in messages:
            if message.get("message_id") == message_id:
                # Send acknowledgment to the original sender
                sender_id = message.get("sender_id")
                
                # Create acknowledgment message
                ack_message = create_protocol_message(
                    sender_id=self.agent_id,
                    recipient_id=sender_id,
                    message_type=MessageType.RESPONSE,
                    content={
                        "action": "acknowledgment",
                        "status": status,
                        "original_message_id": message_id
                    },
                    priority=PriorityLevel.HIGH
                )
                
                # Store in session state
                ack_messages = self.session_state.get("messages", [])
                ack_messages.append(ack_message)
                self.session_state.set("messages", ack_messages)
                
                # Add to sender's inbox
                inbox_key = f"{sender_id}_inbox"
                inbox = self.session_state.get(inbox_key, [])
                inbox.append(ack_message)
                self.session_state.set(inbox_key, inbox)
                
                # Update original message status
                message["status"] = "acknowledged"
                
                # Update context
                self.context["session_state"] = self.session_state.to_dict()
                break
    
    def wait_for_response(
        self, 
        message_id: str, 
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a response to a specific message.
        
        Args:
            message_id: ID of the message to wait for response to
            timeout: Maximum time to wait in seconds
            
        Returns:
            Optional[Dict[str, Any]]: The response message or None if timeout
        """
        # In a real async implementation, this would use await or callbacks
        # For this simplified version, we'll just check the session state
        
        # Get all messages referencing the original message
        messages = self.session_state.get("messages", [])
        for message in messages:
            if message.get("reference_id") == message_id and message.get("message_type") == MessageType.RESPONSE.value:
                return message
        
        # If no response found
        return None
    
    def get_unread_messages(self) -> List[Dict[str, Any]]:
        """
        Get all unread messages for this agent.
        
        Returns:
            List[Dict[str, Any]]: List of unread messages
        """
        # Get the agent's inbox
        inbox_key = f"{self.agent_id}_inbox"
        inbox = self.session_state.get(inbox_key, [])
        
        # Get read messages
        read_key = f"{self.agent_id}_read_messages"
        read_message_ids = self.session_state.get(read_key, [])
        
        # Filter out read messages
        unread = [
            message for message in inbox 
            if message.get("message_id") not in read_message_ids
        ]
        
        return unread
    
    def mark_as_read(self, message_id: str) -> None:
        """
        Mark a message as read.
        
        Args:
            message_id: ID of the message to mark as read
        """
        # Get read messages
        read_key = f"{self.agent_id}_read_messages"
        read_message_ids = self.session_state.get(read_key, [])
        
        # Add to read messages if not already there
        if message_id not in read_message_ids:
            read_message_ids.append(message_id)
            self.session_state.set(read_key, read_message_ids)
            
            # Update context
            self.context["session_state"] = self.session_state.to_dict()

class TaskExecutionFramework:
    """
    Task execution framework for managing delegated tasks.
    
    This class provides utilities for tracking and executing tasks,
    with progress reporting and parallel execution capabilities.
    """
    
    def __init__(self, context: Dict[str, Any], agent_id: str):
        """
        Initialize the task execution framework.
        
        Args:
            context: Agent context dictionary
            agent_id: ID of the agent using this framework
        """
        self.context = context
        self.agent_id = agent_id
        self.session_state = get_or_create_session_state(context)
        self.logger = logging.getLogger(f"finflow.tasks.{agent_id}")
    
    def create_task(
        self,
        task_description: str,
        task_type: str,
        priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> str:
        """
        Create a new task for tracking and execution.
        
        Args:
            task_description: Description of the task
            task_type: Type of task (e.g., "document_processing", "validation")
            priority: Task priority
            metadata: Additional task metadata
            parent_task_id: Optional ID of the parent task
            workflow_id: Optional ID of the associated workflow
            
        Returns:
            str: Task ID
        """
        # Convert priority to string if needed
        if isinstance(priority, PriorityLevel):
            priority_value = priority.value
        else:
            priority_value = priority
            
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task object
        task = {
            "task_id": task_id,
            "description": task_description,
            "type": task_type,
            "priority": priority_value,
            "status": "created",
            "created_by": self.agent_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "progress": 0.0,
            "parent_task_id": parent_task_id,
            "workflow_id": workflow_id,
            "subtasks": []
        }
        
        # Store task in session state
        tasks = self.session_state.get("tasks", {})
        tasks[task_id] = task
        self.session_state.set("tasks", tasks)
        
        # If this is a subtask, add it to the parent's subtasks
        if parent_task_id and parent_task_id in tasks:
            tasks[parent_task_id]["subtasks"].append(task_id)
            self.session_state.set("tasks", tasks)
        
        # Update context
        self.context["session_state"] = self.session_state.to_dict()
        
        self.logger.debug(f"Created task {task_id}: {task_description}")
        
        return task_id
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status value
            progress: Optional progress value (0.0-1.0)
            result: Optional task result data
            error: Optional error message
        """
        # Get tasks from session state
        tasks = self.session_state.get("tasks", {})
        
        # Check if task exists
        if task_id not in tasks:
            self.logger.error(f"Task {task_id} not found")
            return
            
        # Update task
        task = tasks[task_id]
        task["status"] = status
        task["updated_at"] = datetime.now().isoformat()
        
        if progress is not None:
            task["progress"] = max(0.0, min(1.0, progress))  # Clamp to 0-1 range
        
        if result is not None:
            task["result"] = result
            
        if error is not None:
            task["error"] = error
            
        # Store updated task
        tasks[task_id] = task
        self.session_state.set("tasks", tasks)
        
        # Update context
        self.context["session_state"] = self.session_state.to_dict()
        
        self.logger.debug(f"Updated task {task_id} status to {status}, progress: {progress}")
        
        # Update parent task progress automatically if this is a subtask
        if task.get("parent_task_id"):
            self._update_parent_task_progress(task["parent_task_id"])
    
    def _update_parent_task_progress(self, parent_task_id: str) -> None:
        """
        Update parent task progress based on subtask progress.
        
        Args:
            parent_task_id: ID of the parent task
        """
        tasks = self.session_state.get("tasks", {})
        
        if parent_task_id not in tasks:
            return
            
        parent_task = tasks[parent_task_id]
        subtask_ids = parent_task.get("subtasks", [])
        
        if not subtask_ids:
            return
            
        # Calculate average progress of all subtasks
        total_progress = 0.0
        for subtask_id in subtask_ids:
            if subtask_id in tasks:
                subtask = tasks[subtask_id]
                total_progress += subtask.get("progress", 0.0)
        
        avg_progress = total_progress / len(subtask_ids)
        
        # Update parent task progress
        parent_task["progress"] = avg_progress
        tasks[parent_task_id] = parent_task
        self.session_state.set("tasks", tasks)
        
        # Update context
        self.context["session_state"] = self.session_state.to_dict()
        
        # Recursively update grandparent progress if needed
        if parent_task.get("parent_task_id"):
            self._update_parent_task_progress(parent_task["parent_task_id"])
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task details.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Optional[Dict[str, Any]]: Task details or None if not found
        """
        tasks = self.session_state.get("tasks", {})
        return tasks.get(task_id)
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all tasks with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List[Dict[str, Any]]: List of matching tasks
        """
        tasks = self.session_state.get("tasks", {})
        return [task for task in tasks.values() if task.get("status") == status]
    
    def get_subtasks(self, parent_task_id: str) -> List[Dict[str, Any]]:
        """
        Get all subtasks for a parent task.
        
        Args:
            parent_task_id: ID of the parent task
            
        Returns:
            List[Dict[str, Any]]: List of subtask details
        """
        tasks = self.session_state.get("tasks", {})
        
        if parent_task_id not in tasks:
            return []
            
        parent_task = tasks[parent_task_id]
        subtask_ids = parent_task.get("subtasks", [])
        
        return [tasks[subtask_id] for subtask_id in subtask_ids if subtask_id in tasks]
    
    def execute_task(
        self,
        task_id: str,
        executor_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a task and track its progress.
        
        Args:
            task_id: ID of the task to execute
            executor_func: Function that executes the task
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Get task details
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        # Update status to in-progress
        self.update_task_status(task_id, "in_progress", 0.0)
        
        try:
            # Execute the task
            start_time = datetime.now()
            result = executor_func(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update task with result
            self.update_task_status(
                task_id=task_id,
                status="completed",
                progress=1.0,
                result={
                    "data": result,
                    "execution_time": execution_time
                }
            )
            
            # Track agent metrics if agent_id is specified in task metadata
            if "assigned_to" in task.get("metadata", {}):
                agent_id = task["metadata"]["assigned_to"]
                track_agent_metrics(
                    context=self.context,
                    agent_id=agent_id,
                    execution_time=execution_time,
                    success=True,
                    metadata={
                        "task_id": task_id,
                        "task_type": task["type"]
                    }
                )
            
            return result
            
        except Exception as e:
            # Handle execution error
            error_msg = str(e)
            self.logger.error(f"Error executing task {task_id}: {error_msg}")
            
            # Update task with error
            self.update_task_status(
                task_id=task_id,
                status="failed",
                error=error_msg
            )
            
            # Track metrics for failed execution
            if "assigned_to" in task.get("metadata", {}):
                agent_id = task["metadata"]["assigned_to"]
                track_agent_metrics(
                    context=self.context,
                    agent_id=agent_id,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    metadata={
                        "task_id": task_id,
                        "task_type": task["type"],
                        "error": error_msg
                    }
                )
            
            # Re-raise the exception
            raise
    
    def create_and_execute_subtasks(
        self,
        parent_task_id: str,
        subtask_definitions: List[Dict[str, Any]],
        parallel: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create and execute multiple subtasks for a parent task.
        
        Args:
            parent_task_id: ID of the parent task
            subtask_definitions: List of subtask definitions
            parallel: Whether to execute subtasks in parallel
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of subtask results mapped by subtask ID
        """
        # Get parent task
        parent_task = self.get_task(parent_task_id)
        if not parent_task:
            raise ValueError(f"Parent task {parent_task_id} not found")
            
        # Create subtasks
        subtask_ids = []
        for subtask_def in subtask_definitions:
            subtask_id = self.create_task(
                task_description=subtask_def["description"],
                task_type=subtask_def["type"],
                priority=subtask_def.get("priority", parent_task["priority"]),
                metadata=subtask_def.get("metadata"),
                parent_task_id=parent_task_id,
                workflow_id=parent_task.get("workflow_id")
            )
            subtask_ids.append(subtask_id)
            
        # Update parent task with subtasks
        tasks = self.session_state.get("tasks", {})
        parent_task = tasks[parent_task_id]  # Get fresh copy
        parent_task["subtasks"] = subtask_ids
        tasks[parent_task_id] = parent_task
        self.session_state.set("tasks", tasks)
        
        # Update context
        self.context["session_state"] = self.session_state.to_dict()
        
        # Execute subtasks
        results = {}
        
        # TODO: In a real implementation, parallel execution would use async/await
        # For now, we'll simulate sequential execution only
        for i, subtask_id in enumerate(subtask_ids):
            subtask = self.get_task(subtask_id)
            if not subtask:
                continue
                
            # Get the executor function from the subtask definition
            executor_func = subtask_definitions[i].get("executor")
            if not executor_func:
                self.update_task_status(
                    subtask_id,
                    "failed",
                    error="No executor function provided"
                )
                continue
                
            try:
                # Execute the subtask
                result = self.execute_task(subtask_id, executor_func)
                results[subtask_id] = result
            except Exception as e:
                self.logger.error(f"Error executing subtask {subtask_id}: {e}")
                # Error already recorded in execute_task
        
        # Update parent task status based on subtask results
        failed_subtasks = self.get_tasks_by_status("failed")
        failed_subtasks = [t for t in failed_subtasks if t.get("parent_task_id") == parent_task_id]
        
        if failed_subtasks:
            # Some subtasks failed
            if len(failed_subtasks) == len(subtask_ids):
                # All subtasks failed
                self.update_task_status(
                    parent_task_id,
                    "failed",
                    error="All subtasks failed"
                )
            else:
                # Some subtasks succeeded
                self.update_task_status(
                    parent_task_id,
                    "partially_completed",
                    result={"subtask_results": results}
                )
        else:
            # All subtasks succeeded
            self.update_task_status(
                parent_task_id,
                "completed",
                progress=1.0,
                result={"subtask_results": results}
            )
            
        return results
