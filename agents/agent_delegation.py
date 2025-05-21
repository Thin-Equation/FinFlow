"""
Agent delegation implementation for FinFlow.

This module implements LLM-driven delegation patterns using the enhanced agent communication framework.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

from google.adk.tools import ToolContext  # type: ignore

from agents.base_agent import BaseAgent
from utils.agent_communication import (
    create_enhanced_agent_tool,
    agent_task_decorator, delegate_task, register_agent_capabilities,
    transition_workflow, create_workflow, get_workflow_data,
    DelegationStrategy, WorkflowState,
    get_messages, create_response
)
from utils.agent_protocol import (
    MessageType, PriorityLevel, StatusCode, 
    create_response
)
from utils.session_state import get_or_create_session_state


class AgentDelegator(BaseAgent):
    """
    Agent delegator that implements LLM-driven delegation patterns.
    
    This agent can intelligently delegate tasks to other agents based on their capabilities,
    monitor task progress, and handle the delegation workflow.
    """
    
    def __init__(
        self,
        name: str = "FinFlow_Delegator",
        model: str = "gemini-2.0-pro",
        description: str = "Intelligent agent task delegator",
        worker_agents: Optional[Dict[str, BaseAgent]] = None
    ):
        """
        Initialize the agent delegator.
        
        Args:
            name: Agent name
            model: LLM model to use
            description: Agent description
            worker_agents: Dictionary of available worker agents
        """
        instruction = """You are the Delegator Agent for the FinFlow system.
Your role is to intelligently assign tasks to worker agents based on their capabilities,
monitor their progress, and ensure successful task completion.

You should:
1. Analyze incoming requests to identify required capabilities
2. Select the most appropriate agent for each task
3. Create and track delegation workflows
4. Monitor progress and handle failures
5. Aggregate and return results to requesters
"""

        # Initialize the base agent
        super().__init__(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            temperature=0.2
        )
        
        # Initialize worker agents dictionary
        self.worker_agents = worker_agents or {}
        
        # Set up logger
        self.logger = logging.getLogger(f"finflow.agents.{self.name}")
        
        # Register worker agents
        self._register_worker_agents()
    
    def _register_worker_agents(self) -> None:
        """Register available worker agents as tools and in the registry."""
        for agent_id, agent in self.worker_agents.items():
            if agent is not None:
                self.logger.info(f"Registering worker agent as tool: {agent_id}")
                
                # Create and add the enhanced agent tool
                self.add_tool(create_enhanced_agent_tool(agent, name=f"{agent_id}_invoke"))
                
                # Get agent capabilities if available
                capabilities = getattr(agent, "capabilities", [])
                if not capabilities and hasattr(agent, "description"):
                    # Extract capabilities from description if not explicitly defined
                    desc = agent.description.lower()
                    if "document" in desc or "extraction" in desc:
                        capabilities = ["document_processing", "information_extraction"]
                    elif "validation" in desc or "verify" in desc:
                        capabilities = ["validation", "rule_checking"]
                    elif "storage" in desc or "database" in desc:
                        capabilities = ["data_storage", "persistence"]
                    elif "analytics" in desc or "reporting" in desc:
                        capabilities = ["data_analysis", "reporting"]
                
                # Register in the agent registry (will be stored in session state)
                def register_in_context(context: Dict[str, Any]) -> None:
                    register_agent_capabilities(
                        context=context,
                        agent_id=agent_id,
                        capabilities=capabilities or ["general_purpose"],
                        metadata={
                            "description": getattr(agent, "description", ""),
                            "model": getattr(agent, "model", "unknown")
                        }
                    )
                
                # Store this function to be called when we have a context
                setattr(self, f"_register_{agent_id}", register_in_context)
    
    def initialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the context with registered agents and create delegation workflow.
        
        Args:
            context: Agent context
            
        Returns:
            Dict[str, Any]: Initialized context
        """
        # Register all agents in the registry
        for attr_name in dir(self):
            if attr_name.startswith("_register_") and callable(getattr(self, attr_name)):
                register_func = getattr(self, attr_name)
                register_func(context)
        
        # Create a delegation workflow if it doesn't exist
        if "delegation_workflow_id" not in context:
            workflow_id = create_workflow(
                context=context,
                workflow_type="agent_delegation",
                owner_id=self.name,
                initial_state=WorkflowState.INITIALIZED,
                metadata={
                    "delegator_id": self.name,
                    "description": "Agent delegation workflow"
                }
            )
            context["delegation_workflow_id"] = workflow_id
        
        return context
    
    @agent_task_decorator(task_name="delegate_task", required_capabilities=["delegation"], track_metrics=True)
    def delegate_agent_task(
        self,
        context: Dict[str, Any],
        task_description: str,
        required_capabilities: List[str],
        priority: Union[PriorityLevel, str] = PriorityLevel.NORMAL,
        strategy: Union[DelegationStrategy, str] = DelegationStrategy.CAPABILITY_BASED
    ) -> Dict[str, Any]:
        """
        Delegate a task to a suitable worker agent.
        
        Args:
            context: Agent context
            task_description: Description of the task to delegate
            required_capabilities: List of capabilities required for the task
            priority: Priority of the task
            strategy: Delegation strategy to use
            
        Returns:
            Dict[str, Any]: Delegation result
        """
        # Initialize context if needed
        context = self.initialize_context(context)
        
        # Log the delegation request
        self.logger.info(f"Received delegation request for task: {task_description}")
        self.logger.info(f"Required capabilities: {required_capabilities}")
        
        # Get the agent registry
        session_state = get_or_create_session_state(context)
        agent_registry = session_state.get("agent_registry", {})
        
        # Delegate the task
        success, delegation_result = delegate_task(
            context=context,
            task_description=task_description,
            required_capabilities=required_capabilities,
            available_agents=agent_registry,
            priority=priority,
            metadata={
                "delegator_agent": self.name,
                "delegation_time": datetime.now().isoformat()
            },
            strategy=strategy
        )
        
        # Create and store delegation result
        result = {
            "success": success,
            "delegation_request_id": delegation_result.get("request_id"),
            "selected_agent": delegation_result.get("delegatee_id", "none"),
            "workflow_id": delegation_result.get("workflow_id"),
            "status": delegation_result.get("status")
        }
        
        # Add error message if delegation failed
        if not success:
            result["error"] = delegation_result.get("reason", "Delegation failed")
        
        return result
    
    @agent_task_decorator(task_name="monitor_delegated_task", required_capabilities=["monitoring"], track_metrics=True)
    def monitor_delegated_task(
        self,
        context: Dict[str, Any],
        delegation_request_id: str
    ) -> Dict[str, Any]:
        """
        Monitor a delegated task and return its current status.
        
        Args:
            context: Agent context
            delegation_request_id: ID of the delegation request to monitor
            
        Returns:
            Dict[str, Any]: Task status and progress
        """
        # Initialize context if needed
        context = self.initialize_context(context)
        
        # Get the delegation request
        session_state = get_or_create_session_state(context)
        delegation_queue = session_state.get("delegation_queue", [])
        
        delegation_request = None
        for req in delegation_queue:
            if req.get("request_id") == delegation_request_id:
                delegation_request = req
                break
        
        if not delegation_request:
            return {
                "error": f"Delegation request {delegation_request_id} not found",
                "status": "unknown"
            }
        
        # Get the workflow if available
        workflow_id = delegation_request.get("workflow_id")
        workflow_data = {}
        if workflow_id:
            workflow_data = get_workflow_data(context, workflow_id)
        
        # Create monitoring result
        result = {
            "delegation_request_id": delegation_request_id,
            "status": delegation_request.get("status", "unknown"),
            "delegatee_id": delegation_request.get("delegatee_id"),
            "task_description": delegation_request.get("task_description"),
            "priority": delegation_request.get("priority"),
            "created_at": delegation_request.get("timestamp"),
            "workflow_state": workflow_data.get("state", "unknown") if workflow_data else "unknown"
        }
        
        # Add completion information if available
        if "completion_time" in delegation_request:
            result["completion_time"] = delegation_request["completion_time"]
            
        if "result" in delegation_request:
            result["result"] = delegation_request["result"]
        
        return result
    
    def process_incoming_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process all incoming messages for this agent.
        
        Args:
            context: Agent context
            
        Returns:
            List[Dict[str, Any]]: List of responses sent
        """
        # Initialize context if needed
        context = self.initialize_context(context)
        
        # Get unread messages
        messages = get_messages(context, self.name, only_unread=True)
        responses = []
        
        for message in messages:
            # Process the message based on type
            if message.get("message_type") == MessageType.REQUEST.value:
                response = self._handle_request_message(context, message)
                responses.append(response)
            elif message.get("message_type") == MessageType.RESPONSE.value:
                self._handle_response_message(context, message)
            elif message.get("message_type") == MessageType.NOTIFICATION.value:
                self._handle_notification_message(context, message)
        
        return responses
    
    def _handle_request_message(self, context: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request messages."""
        content = message.get("content", {})
        action = content.get("action", "")
        
        response_content = {
            "action": f"{action}_response",
            "data": {}
        }
        
        if action == "delegate_task":
            # Handle delegation request
            data = content.get("data", {})
            result = self.delegate_agent_task(
                context,
                task_description=data.get("task_description", ""),
                required_capabilities=data.get("required_capabilities", []),
                priority=data.get("priority", PriorityLevel.NORMAL),
                strategy=data.get("strategy", DelegationStrategy.CAPABILITY_BASED)
            )
            response_content["data"] = result
            
        elif action == "monitor_task":
            # Handle monitoring request
            data = content.get("data", {})
            result = self.monitor_delegated_task(
                context,
                delegation_request_id=data.get("delegation_request_id", "")
            )
            response_content["data"] = result
            
        else:
            # Unknown action
            response_content["data"] = {
                "error": f"Unknown action: {action}",
                "status": "error"
            }
        
        # Send response
        return create_response(
            context,
            message,
            response_content
        )
    
    def _handle_response_message(self, context: Dict[str, Any], message: Dict[str, Any]) -> None:
        """Handle response messages."""
        content = message.get("content", {})
        action = content.get("action", "")
        data = content.get("data", {})
        
        if action == "delegated_task_completed":
            # Handle task completion
            delegation_request_id = data.get("delegation_request_id")
            workflow_id = data.get("workflow_id")
            
            if workflow_id:
                # Update workflow state
                transition_workflow(
                    context=context,
                    workflow_id=workflow_id,
                    from_state=None,  # No validation
                    to_state=WorkflowState.COMPLETED,
                    agent_id=self.name,
                    reason="Task completed by delegate",
                    metadata={"result": data.get("result", {})}
                )
    
    def _handle_notification_message(self, context: Dict[str, Any], message: Dict[str, Any]) -> None:
        """Handle notification messages."""
        # Just log notifications for now
        content = message.get("content", {})
        data = content.get("data", {})
        subject = data.get("subject", "Unknown notification")
        
        self.logger.info(f"Received notification: {subject}")


class DelegatableAgent(BaseAgent):
    """
    Agent that can accept delegated tasks and report results back to the delegator.
    
    This is a mixin class that adds delegation support to any agent.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        description: str,
        capabilities: List[str],
        instruction: str = "",
        temperature: float = 0.2
    ):
        """
        Initialize a delegatable agent.
        
        Args:
            name: Agent name
            model: LLM model to use
            description: Agent description
            capabilities: List of agent capabilities
            instruction: Agent instruction prompt
            temperature: LLM temperature
        """
        # Store capabilities
        self.capabilities = capabilities
        
        # Initialize the base agent
        super().__init__(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            temperature=temperature
        )
        
        # Add capabilities for handling delegated tasks
        self.register_task_handlers()
    
    def register_task_handlers(self) -> None:
        """Register task handlers for this agent."""
        # Override in subclasses to register specific task handlers
        pass
    
    @agent_task_decorator(task_name="handle_delegated_task", track_metrics=True)
    def handle_delegated_task(
        self,
        context: Dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> Dict[str, Any]:
        """
        Handle a task delegated by another agent.
        
        Args:
            context: Agent context
            tool_context: Tool context
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Get the delegation request from the context
        delegation_request_id = context.get("delegation_request_id")
        workflow_id = context.get("workflow_id")
        task_description = context.get("task_description", "Unknown task")
        
        if not delegation_request_id:
            return {
                "error": "No delegation request ID provided",
                "status": "error"
            }
        
        self.logger.info(f"Handling delegated task: {task_description}")
        self.logger.info(f"Delegation request ID: {delegation_request_id}")
        self.logger.info(f"Workflow ID: {workflow_id}")
        
        try:
            # Process the task according to its description
            # This is a placeholder - actual implementation would depend on agent capabilities
            result = self.process_task(task_description, context, tool_context)
            
            # Report task completion
            from utils.agent_communication import complete_delegated_task
            complete_delegated_task(
                context=context,
                delegation_request_id=delegation_request_id,
                result=result,
                status=StatusCode.OK
            )
            
            return {
                "success": True,
                "delegation_request_id": delegation_request_id,
                "result": result
            }
            
        except Exception as e:
            # Report task failure
            error_message = f"Failed to process task: {str(e)}"
            self.logger.error(error_message)
            
            from utils.agent_communication import complete_delegated_task
            complete_delegated_task(
                context=context,
                delegation_request_id=delegation_request_id,
                result={"error": error_message},
                status=StatusCode.ERROR
            )
            
            return {
                "success": False,
                "error": error_message,
                "delegation_request_id": delegation_request_id
            }
    
    def process_task(
        self,
        task_description: str,
        context: Dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> Dict[str, Any]:
        """
        Process a delegated task based on its description.
        
        Args:
            task_description: Description of the task
            context: Agent context
            tool_context: Tool context
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Override this method in subclasses to provide actual task processing
        return {
            "message": f"Task '{task_description}' processed by {self.name}",
            "agent": self.name,
            "capabilities": self.capabilities,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_incoming_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process incoming messages for this agent.
        
        Args:
            context: Agent context
            
        Returns:
            List[Dict[str, Any]]: List of responses
        """
        # Get unread messages
        messages = get_messages(context, self.name, only_unread=True)
        responses = []
        
        for message in messages:
            # Check for delegated task messages
            if message.get("message_type") == MessageType.REQUEST.value:
                content = message.get("content", {})
                action = content.get("action")
                
                if action == "delegated_task":
                    # Handle delegated task
                    data = content.get("data", {})
                    
                    # Update context with task info
                    task_context = context.copy()
                    task_context.update({
                        "delegation_request_id": data.get("delegation_request_id"),
                        "workflow_id": data.get("workflow_id"),
                        "task_description": data.get("task_description", "Unknown task"),
                        "required_capabilities": data.get("required_capabilities", []),
                        "priority": data.get("priority", PriorityLevel.NORMAL.value)
                    })
                    
                    # Process the task
                    result = self.handle_delegated_task(task_context)
                    
                    # Create response
                    response = create_response(
                        context,
                        message,
                        {
                            "action": "delegated_task_response",
                            "data": result
                        }
                    )
                    responses.append(response)
        
        return responses
