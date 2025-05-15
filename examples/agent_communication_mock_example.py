"""
Example demonstrating two agents communicating with each other using mocks.
"""

from typing import Any, Dict
import json
import logging
import sys
import os

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent_communication_example")

# Import dependencies
from utils.agent_protocol import (
    MessageType, PriorityLevel, StatusCode, 
    create_request, create_response
)

class MockAgent:
    """Mock agent base class for the example."""
    
    def __init__(self, name: str):
        """Initialize a mock agent."""
        self.name = name
        self.logger = logging.getLogger(f"finflow.agents.mock.{name}")
    
    def log_activity(self, activity_type: str, details: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Log agent activity."""
        self.logger.info(f"Activity: {activity_type} - {json.dumps(details)}")
    
    def update_session_state(self, state_key: str, state_value: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update the session state."""
        if "session_state" not in context:
            context["session_state"] = {}
            
        if "data" not in context["session_state"]:
            context["session_state"]["data"] = {}
            
        context["session_state"]["data"][state_key] = state_value
        self.logger.debug(f"Updated session state: {state_key} = {state_value}")
        return context

class RequestAgent(MockAgent):
    """An agent that sends requests to other agents."""
    
    def __init__(self, name: str = "RequestAgent"):
        super().__init__(name=name)
    
    def request_data(self, context: Dict[str, Any], recipient_id: str, entity_type: str) -> Dict[str, Any]:
        """
        Send a data request to another agent.
        
        Args:
            context: Agent context
            recipient_id: ID of the recipient agent
            entity_type: Type of entity to request data for
            
        Returns:
            Dict: Updated context with request information
        """
        logger.info(f"{self.name} requesting data for {entity_type} from {recipient_id}")
        
        # Create a request
        request = create_request(
            sender_id=self.name,
            recipient_id=recipient_id,
            action="get_entity_data",
            data={
                "entity_type": entity_type,
                "include_details": True
            },
            priority=PriorityLevel.NORMAL
        )
        
        # Add the request to session state
        if "session_state" not in context:
            context["session_state"] = {}
        
        if "data" not in context["session_state"]:
            context["session_state"]["data"] = {}
            
        if "messages" not in context["session_state"]["data"]:
            context["session_state"]["data"]["messages"] = []
            
        # Add to messages
        context["session_state"]["data"]["messages"].append(request)
        
        # Add to recipient's inbox
        inbox_key = f"{recipient_id}_inbox"
        if inbox_key not in context["session_state"]["data"]:
            context["session_state"]["data"][inbox_key] = []
            
        context["session_state"]["data"][inbox_key].append(request)
        
        # Log the action
        self.log_activity(
            activity_type="send_request",
            details={
                "request_id": request["message_id"],
                "recipient": recipient_id,
                "entity_type": entity_type
            },
            context=context
        )
        
        logger.info(f"{self.name} sent request {request['message_id']} to {recipient_id}")
        
        return context

    def process_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process responses from other agents.
        
        Args:
            context: Agent context
            
        Returns:
            Dict: Updated context with processed information
        """
        # Get messages for this agent
        if "session_state" not in context or "data" not in context["session_state"]:
            logger.info(f"{self.name} has no session state")
            return context
            
        inbox_key = f"{self.name}_inbox"
        if inbox_key not in context["session_state"]["data"]:
            logger.info(f"{self.name} has no messages in inbox")
            return context
            
        inbox = context["session_state"]["data"][inbox_key]
        
        # Process each message
        for message in inbox:
            if message["message_type"] == MessageType.RESPONSE.value:
                logger.info(f"{self.name} processing response {message['message_id']} from {message['sender_id']}")
                
                # Get the response data
                response_data = message["content"]["data"]
                
                # Process the data (in a real application, this would do something meaningful)
                if "entity_data" in response_data:
                    entity_data = response_data["entity_data"]
                    logger.info(f"{self.name} received entity data: {json.dumps(entity_data, indent=2)}")
                    
                    # Store processed data in session state
                    processed_key = f"processed_{message['message_id']}"
                    self.update_session_state(processed_key, {
                        "source": message["sender_id"],
                        "processed_data": entity_data
                    }, context)
                
                # Mark as read
                read_key = f"{self.name}_read_messages"
                if read_key not in context["session_state"]["data"]:
                    context["session_state"]["data"][read_key] = []
                    
                context["session_state"]["data"][read_key].append(message["message_id"])
                
        return context


class ResponseAgent(MockAgent):
    """An agent that responds to requests from other agents."""
    
    def __init__(self, name: str = "ResponseAgent"):
        super().__init__(name=name)
        
        # Sample data this agent has access to
        self.entity_database = {
            "customer": {
                "id": "cust-123",
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "accounts": ["acc-1", "acc-2"]
            },
            "account": {
                "id": "acc-1",
                "type": "checking",
                "balance": 1500.00,
                "currency": "USD",
                "created_at": "2022-01-15"
            },
            "transaction": {
                "id": "txn-456",
                "amount": 250.00,
                "type": "debit",
                "description": "ATM Withdrawal",
                "timestamp": "2023-05-10T14:30:00Z"
            }
        }
    
    def process_requests(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming requests from other agents.
        
        Args:
            context: Agent context
            
        Returns:
            Dict: Updated context with response information
        """
        # Get messages for this agent
        if "session_state" not in context or "data" not in context["session_state"]:
            logger.info(f"{self.name} has no session state")
            return context
            
        inbox_key = f"{self.name}_inbox"
        if inbox_key not in context["session_state"]["data"]:
            logger.info(f"{self.name} has no messages in inbox")
            return context
            
        inbox = context["session_state"]["data"][inbox_key]
        
        # Find unread request messages
        read_key = f"{self.name}_read_messages"
        read_messages = context["session_state"]["data"].get(read_key, [])
        
        # Process each unread request
        for message in inbox:
            if message["message_id"] in read_messages:
                continue  # Skip already processed messages
                
            if message["message_type"] == MessageType.REQUEST.value:
                logger.info(f"{self.name} processing request {message['message_id']} from {message['sender_id']}")
                
                # Get the request details
                action = message["content"].get("action")
                request_data = message["content"].get("data", {})
                
                # Handle different types of actions
                if action == "get_entity_data":
                    entity_type = request_data.get("entity_type")
                    include_details = request_data.get("include_details", False)
                    
                    # Get data from database
                    entity_data = self.get_entity_data(entity_type, include_details)
                    
                    # Create response
                    response = create_response(
                        request_message=message,
                        data={
                            "entity_type": entity_type,
                            "entity_data": entity_data
                        },
                        status=StatusCode.OK
                    )
                    
                    # Add to messages
                    if "messages" not in context["session_state"]["data"]:
                        context["session_state"]["data"]["messages"] = []
                        
                    context["session_state"]["data"]["messages"].append(response)
                    
                    # Add to recipient's inbox
                    recipient_inbox_key = f"{message['sender_id']}_inbox"
                    if recipient_inbox_key not in context["session_state"]["data"]:
                        context["session_state"]["data"][recipient_inbox_key] = []
                        
                    context["session_state"]["data"][recipient_inbox_key].append(response)
                    
                    logger.info(f"{self.name} sent response {response['message_id']} to {message['sender_id']}")
                else:
                    logger.warning(f"{self.name} received unknown action: {action}")
                
                # Mark as read
                if read_key not in context["session_state"]["data"]:
                    context["session_state"]["data"][read_key] = []
                    
                context["session_state"]["data"][read_key].append(message["message_id"])
                
        return context
    
    def get_entity_data(self, entity_type: str, include_details: bool) -> Dict[str, Any]:
        """
        Get data for an entity from the database.
        
        Args:
            entity_type: Type of entity to get data for
            include_details: Whether to include all details
            
        Returns:
            Dict: Entity data
        """
        if entity_type not in self.entity_database:
            return {"error": f"Entity type {entity_type} not found"}
            
        data = self.entity_database[entity_type]
        
        if not include_details:
            # Only include basic info
            basic_info = {}
            for key in ["id", "name", "type"]:
                if key in data:
                    basic_info[key] = data[key]
            return basic_info
        
        return data


def test_agent_communication():
    """Test communication between request and response agents."""
    # Create agents
    request_agent = RequestAgent(name="RequestAgent")
    response_agent = ResponseAgent(name="ResponseAgent")
    
    # Create shared context
    context: Dict[str, Any] = {
        "session_state": {
            "data": {
                "messages": []
            }
        }
    }
    
    # RequestAgent sends a request
    logger.info("Starting agent communication test")
    logger.info("-" * 50)
    
    # Step 1: RequestAgent sends a request for customer data
    context = request_agent.request_data(context, "ResponseAgent", "customer")
    logger.info("RequestAgent has sent a request for customer data")
    
    # Step 2: ResponseAgent processes the request
    context = response_agent.process_requests(context)
    logger.info("ResponseAgent has processed the request and sent a response")
    
    # Step 3: RequestAgent processes the response
    context = request_agent.process_response(context)
    logger.info("RequestAgent has processed the response")
    
    # Step 4: RequestAgent sends another request for account data
    context = request_agent.request_data(context, "ResponseAgent", "account")
    logger.info("RequestAgent has sent a request for account data")
    
    # Step 5: ResponseAgent processes the new request
    context = response_agent.process_requests(context)
    logger.info("ResponseAgent has processed the request and sent a response")
    
    # Step 6: RequestAgent processes the response
    context = request_agent.process_response(context)
    logger.info("RequestAgent has processed the response")
    
    # Print summary
    logger.info("-" * 50)
    logger.info("Agent Communication Test Summary:")
    
    if "messages" in context["session_state"]["data"]:
        num_messages = len(context["session_state"]["data"]["messages"])
        logger.info(f"Total messages exchanged: {num_messages}")
    
    # Check processed data in RequestAgent's session
    processed_keys = [k for k in context["session_state"]["data"].keys() if k.startswith("processed_")]
    
    logger.info(f"RequestAgent processed {len(processed_keys)} responses")
    
    for key in processed_keys:
        processed_data = context["session_state"]["data"][key]
        logger.info(f"Processed data from {processed_data['source']}: {json.dumps(processed_data['processed_data'], indent=2)}")
    
    logger.info("Agent communication test completed successfully")


if __name__ == "__main__":
    print("Starting agent communication test...")
    test_agent_communication()
    print("Agent communication test completed!")
