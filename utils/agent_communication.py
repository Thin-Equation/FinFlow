"""
Utilities for agent-to-agent communication in the FinFlow system.
"""

from typing import Any, Dict, Optional, List
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import ToolContext

def create_agent_tool(agent_name: str, description: str) -> AgentTool:
    """
    Create an AgentTool for invoking another agent.
    
    Args:
        agent_name: Name of the agent to create a tool for
        description: Description of the agent's purpose
        
    Returns:
        AgentTool: Tool for invoking the specified agent
    """
    return AgentTool(
        name=f"{agent_name}Agent",
        description=description,
        agent_name=agent_name
    )

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

def create_agent_tools(agent_definitions: List[Dict[str, str]]) -> List[AgentTool]:
    """
    Create a list of AgentTools based on provided definitions.
    
    Args:
        agent_definitions: List of dictionaries with 'name' and 'description' keys
        
    Returns:
        List[AgentTool]: List of agent tools
    """
    return [
        create_agent_tool(agent["name"], agent["description"])
        for agent in agent_definitions
    ]
