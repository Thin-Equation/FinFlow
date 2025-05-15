"""
Utilities for agent-to-agent communication in the FinFlow system.
"""

from typing import Any, Dict, List
from google.adk.tools.agent_tool import AgentTool # type: ignore

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
