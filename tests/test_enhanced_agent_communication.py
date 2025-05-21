#!/usr/bin/env python3
"""
Test for the enhanced agent communication framework.

This test validates the functionality of the enhanced agent communication framework
including the communication protocol, task execution framework, and delegation strategies.
"""

import unittest


# Import the components to test
from utils.agent_communication import (
    CommunicationProtocol,
    TaskExecutionFramework,
    apply_delegation_strategy,
    DelegationStrategy,
    WorkflowState,
    create_workflow,
    transition_workflow,
    get_workflow_data
)

# Import relevant utilities
from utils.session_state import SessionState
from utils.agent_protocol import MessageType, PriorityLevel

class TestCommunicationProtocol(unittest.TestCase):
    """Tests for the CommunicationProtocol class."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = {"agent_id": "test_agent"}
        self.session_state = SessionState()
        self.context["session_state"] = self.session_state.to_dict()
        self.comms = CommunicationProtocol(self.context, "test_agent")
    
    def test_send_message(self):
        """Test sending a message."""
        message = self.comms.send_message(
            recipient_id="recipient_agent",
            message_type=MessageType.REQUEST,
            content={"action": "test", "data": {"key": "value"}},
            priority=PriorityLevel.NORMAL
        )
        
        # Check message properties
        self.assertEqual(message["sender_id"], "test_agent")
        self.assertEqual(message["recipient_id"], "recipient_agent")
        self.assertEqual(message["message_type"], MessageType.REQUEST.value)
        self.assertIn("message_id", message)
        self.assertIn("timestamp", message)
        self.assertEqual(message["content"]["action"], "test")
        self.assertEqual(message["content"]["data"]["key"], "value")
        
        # Check that message is stored in session state
        session_state = SessionState.from_dict(self.context["session_state"])
        messages = session_state.get("messages", [])
        self.assertEqual(len(messages), 1)
        
        # Check recipient inbox
        inbox = session_state.get("recipient_agent_inbox", [])
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["message_id"], message["message_id"])
    
    def test_get_unread_messages(self):
        """Test getting unread messages."""
        # Send a test message
        message = self.comms.send_message(
            recipient_id="test_agent",  # Send to self
            message_type=MessageType.NOTIFICATION,
            content={"action": "test_notification"}
        )
        
        # Get unread messages
        unread = self.comms.get_unread_messages()
        self.assertEqual(len(unread), 1)
        self.assertEqual(unread[0]["message_id"], message["message_id"])
        
        # Mark as read
        self.comms.mark_as_read(message["message_id"])
        
        # Check that message is now marked as read
        unread = self.comms.get_unread_messages()
        self.assertEqual(len(unread), 0)

class TestTaskExecutionFramework(unittest.TestCase):
    """Tests for the TaskExecutionFramework class."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = {"agent_id": "test_agent"}
        self.session_state = SessionState()
        self.context["session_state"] = self.session_state.to_dict()
        self.tasks = TaskExecutionFramework(self.context, "test_agent")
    
    def test_create_task(self):
        """Test creating a task."""
        task_id = self.tasks.create_task(
            task_description="Test task",
            task_type="test",
            metadata={"key": "value"}
        )
        
        # Check task was created
        task = self.tasks.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task["description"], "Test task")
        self.assertEqual(task["type"], "test")
        self.assertEqual(task["status"], "created")
        self.assertEqual(task["metadata"]["key"], "value")
    
    def test_update_task_status(self):
        """Test updating task status."""
        # Create a task
        task_id = self.tasks.create_task(
            task_description="Test task",
            task_type="test"
        )
        
        # Update status
        self.tasks.update_task_status(
            task_id=task_id,
            status="in_progress",
            progress=0.5
        )
        
        # Check task was updated
        task = self.tasks.get_task(task_id)
        self.assertEqual(task["status"], "in_progress")
        self.assertEqual(task["progress"], 0.5)
    
    def test_subtasks(self):
        """Test creating and managing subtasks."""
        # Create parent task
        parent_id = self.tasks.create_task(
            task_description="Parent task",
            task_type="parent"
        )
        
        # Create child task
        child_id = self.tasks.create_task(
            task_description="Child task",
            task_type="child",
            parent_task_id=parent_id
        )
        
        # Check parent contains child
        parent = self.tasks.get_task(parent_id)
        self.assertIn(child_id, parent["subtasks"])
        
        # Get subtasks
        subtasks = self.tasks.get_subtasks(parent_id)
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0]["task_id"], child_id)

class TestDelegationStrategies(unittest.TestCase):
    """Tests for delegation strategies."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = {"agent_id": "test_agent"}
        self.session_state = SessionState()
        self.context["session_state"] = self.session_state.to_dict()
        
        # Create test agents with capabilities
        self.available_agents = {
            "agent1": {
                "capabilities": ["capability1", "capability2"],
                "status": {"availability": 0.8, "current_load": 0.2}
            },
            "agent2": {
                "capabilities": ["capability2", "capability3"],
                "status": {"availability": 0.5, "current_load": 0.5}
            },
            "agent3": {
                "capabilities": ["capability1", "capability3", "capability4"],
                "status": {"availability": 0.2, "current_load": 0.8}
            }
        }
        
        # Create a test delegation request
        self.delegation_request = {
            "request_id": "test_request",
            "task_description": "Test task",
            "required_capabilities": ["capability1", "capability2"],
            "priority": PriorityLevel.NORMAL.value,
            "strategy": DelegationStrategy.CAPABILITY_BASED.value
        }
    
    def test_capability_based_strategy(self):
        """Test capability-based delegation strategy."""
        selected = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.CAPABILITY_BASED
        )
        
        # Agent1 should be selected as it has both required capabilities
        self.assertEqual(selected, "agent1")
    
    def test_availability_based_strategy(self):
        """Test availability-based delegation strategy."""
        selected = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.AVAILABILITY_BASED
        )
        
        # Agent1 should be selected as it has the highest availability
        self.assertEqual(selected, "agent1")
    
    def test_load_balanced_strategy(self):
        """Test load-balanced delegation strategy."""
        selected = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.LOAD_BALANCED
        )
        
        # Agent1 should be selected as it has the lowest load
        self.assertEqual(selected, "agent1")
    
    def test_round_robin_strategy(self):
        """Test round-robin delegation strategy."""
        # First call should select agent1
        selected1 = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.ROUND_ROBIN
        )
        
        # Second call should select agent2
        selected2 = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.ROUND_ROBIN
        )
        
        # Third call should select agent3
        selected3 = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.ROUND_ROBIN
        )
        
        # Fourth call should cycle back to agent1
        selected4 = apply_delegation_strategy(
            context=self.context,
            delegation_request=self.delegation_request,
            available_agents=self.available_agents,
            strategy=DelegationStrategy.ROUND_ROBIN
        )
        
        self.assertEqual(selected1, "agent1")
        self.assertEqual(selected2, "agent2")
        self.assertEqual(selected3, "agent3")
        self.assertEqual(selected4, "agent1")

class TestWorkflowManagement(unittest.TestCase):
    """Tests for workflow state management."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = {"agent_id": "test_agent"}
        self.session_state = SessionState()
        self.context["session_state"] = self.session_state.to_dict()
    
    def test_workflow_state_transitions(self):
        """Test workflow state transitions."""
        # Create a workflow
        workflow_id = create_workflow(
            context=self.context,
            workflow_type="test_workflow",
            owner_id="test_agent",
            initial_state=WorkflowState.INITIALIZED,
            metadata={"test_key": "test_value"}
        )
        
        # Check workflow exists
        workflow_data = get_workflow_data(self.context, workflow_id)
        self.assertEqual(workflow_data["state"], WorkflowState.INITIALIZED.value)
        self.assertEqual(workflow_data["type"], "test_workflow")
        self.assertEqual(workflow_data["metadata"]["test_key"], "test_value")
        
        # Transition to IN_PROGRESS
        success = transition_workflow(
            context=self.context,
            workflow_id=workflow_id,
            from_state=WorkflowState.INITIALIZED,
            to_state=WorkflowState.IN_PROGRESS,
            agent_id="test_agent",
            reason="Starting workflow"
        )
        
        self.assertTrue(success)
        
        # Check new state
        workflow_data = get_workflow_data(self.context, workflow_id)
        self.assertEqual(workflow_data["state"], WorkflowState.IN_PROGRESS.value)
        
        # Check history contains the transition
        self.assertEqual(len(workflow_data["history"]), 1)
        self.assertEqual(workflow_data["history"][0]["from_state"], WorkflowState.INITIALIZED.value)
        self.assertEqual(workflow_data["history"][0]["to_state"], WorkflowState.IN_PROGRESS.value)
        self.assertEqual(workflow_data["history"][0]["performer_id"], "test_agent")
        
        # Try invalid transition (invalid from_state)
        success = transition_workflow(
            context=self.context,
            workflow_id=workflow_id,
            from_state=WorkflowState.INITIALIZED,  # We're already in IN_PROGRESS
            to_state=WorkflowState.COMPLETED,
            agent_id="test_agent",
            reason="Completing workflow"
        )
        
        self.assertFalse(success)
        
        # Valid transition to COMPLETED
        success = transition_workflow(
            context=self.context,
            workflow_id=workflow_id,
            from_state=WorkflowState.IN_PROGRESS,
            to_state=WorkflowState.COMPLETED,
            agent_id="test_agent",
            reason="Completing workflow",
            metadata={"completion_data": "test"}
        )
        
        self.assertTrue(success)
        
        # Check final state
        workflow_data = get_workflow_data(self.context, workflow_id)
        self.assertEqual(workflow_data["state"], WorkflowState.COMPLETED.value)
        self.assertEqual(workflow_data["metadata"]["completion_data"], "test")
        self.assertEqual(len(workflow_data["history"]), 2)

if __name__ == "__main__":
    unittest.main()
