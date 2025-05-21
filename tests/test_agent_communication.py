"""
Tests for the agent communication framework.
"""

import unittest
from typing import Any, Dict

from utils.agent_communication import (
    create_message, send_message, get_messages, mark_message_read,
    MESSAGE_TYPE_REQUEST, MESSAGE_TYPE_RESPONSE, PRIORITY_NORMAL
)
from utils.agent_protocol import (
    create_request, create_response,
    MessageType, PriorityLevel, StatusCode
)

class TestAgentCommunication(unittest.TestCase):
    """Test cases for agent communication."""

    def test_create_message(self):
        """Test creating a message."""
        sender_id = "agent1"
        recipient_id = "agent2"
        message_type = MESSAGE_TYPE_REQUEST
        content = {"action": "test", "data": {"key": "value"}}
        
        message = create_message(sender_id, recipient_id, message_type, content)
        
        self.assertEqual(message["sender_id"], sender_id)
        self.assertEqual(message["recipient_id"], recipient_id)
        self.assertEqual(message["message_type"], message_type)
        self.assertEqual(message["content"], content)
        self.assertEqual(message["priority"], PRIORITY_NORMAL)
        self.assertIn("message_id", message)
        self.assertIn("timestamp", message)
    
    def test_message_flow(self):
        """Test sending and receiving messages."""
        # Create a context with session state
        context: Dict[str, Any] = {}
        
        # Agent IDs
        sender_id = "agent1"
        recipient_id = "agent2"
        
        # Send a message
        content = {"action": "get_data", "data": {"entity": "customer"}}
        sent_message = send_message(
            context, sender_id, recipient_id, 
            MESSAGE_TYPE_REQUEST, content
        )
        
        # Verify message was sent
        self.assertIn("session_state", context)
        self.assertIn("messages", context["session_state"]["data"])
        self.assertEqual(len(context["session_state"]["data"]["messages"]), 1)
        
        # Get messages for recipient
        received_messages = get_messages(context, recipient_id)
        
        # Verify message was received
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0]["sender_id"], sender_id)
        self.assertEqual(received_messages[0]["recipient_id"], recipient_id)
        self.assertEqual(received_messages[0]["content"], content)
        
        # Mark message as read
        message_id = received_messages[0]["message_id"]
        mark_message_read(context, recipient_id, message_id)
        
        # Verify message is marked as read
        unread_messages = get_messages(context, recipient_id, only_unread=True)
        self.assertEqual(len(unread_messages), 0)
        
        # Send a response
        response_content = {
            "action": "get_data_response", 
            "data": {"customer": {"id": "123", "name": "John"}}
        }
        
        response_message = send_message(
            context, recipient_id, sender_id,
            MESSAGE_TYPE_RESPONSE, response_content,
            reference_id=message_id
        )
        
        # Verify response was sent
        self.assertEqual(len(context["session_state"]["data"]["messages"]), 2)
        
        # Get messages for original sender
        sender_messages = get_messages(context, sender_id)
        
        # Verify response was received
        self.assertEqual(len(sender_messages), 1)
        self.assertEqual(sender_messages[0]["sender_id"], recipient_id)
        self.assertEqual(sender_messages[0]["reference_id"], message_id)

    def test_protocol_message(self):
        """Test protocol message creation."""
        sender_id = "agent1"
        recipient_id = "agent2"
        
        # Test request creation
        request = create_request(
            sender_id, recipient_id, 
            "get_data", {"entity": "customer"},
            PriorityLevel.HIGH
        )
        
        self.assertEqual(request["sender_id"], sender_id)
        self.assertEqual(request["recipient_id"], recipient_id)
        self.assertEqual(request["message_type"], MessageType.REQUEST.value)
        self.assertEqual(request["priority"], PriorityLevel.HIGH.value)
        self.assertEqual(request["content"]["action"], "get_data")
        
        # Test response creation
        response = create_response(
            request, {"customer": {"id": "123"}}, StatusCode.OK
        )
        
        self.assertEqual(response["sender_id"], recipient_id)
        self.assertEqual(response["recipient_id"], sender_id)
        self.assertEqual(response["message_type"], MessageType.RESPONSE.value)
        self.assertEqual(response["content"]["status"], StatusCode.OK.value)
        self.assertEqual(response["reference_id"], request["message_id"])

if __name__ == "__main__":
    unittest.main()
