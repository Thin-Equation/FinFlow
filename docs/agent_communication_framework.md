# Agent Communication Framework

This document outlines the Agent Communication Framework implemented in the FinFlow system.

## Overview

The Agent Communication Framework provides a structured way for agents to exchange messages, share state, and coordinate actions within the FinFlow ecosystem. It implements a standardized protocol for agent-to-agent communication, session state management, and message handling.

## Components

### Session State Management

The session state manager (`utils/session_state.py`) has been extended to support:

- Agent-specific data compartments via `agent_data`
- Shared keys between agents
- Methods to control data sharing between agents

Key methods:
- `set_agent_data()` - Store data specific to an agent
- `get_agent_data()` - Retrieve agent-specific data
- `share_key()` / `unshare_key()` - Control which keys are shared between agents
- `is_key_shared()` / `get_shared_keys()` - Query shared key status

### Agent Communication Utilities

The agent communication utilities (`utils/agent_communication.py`) provide helper functions for:

- Creating structured messages for inter-agent communication
- Sending messages between agents via session state
- Retrieving and filtering messages for an agent
- Tracking read/unread message status

Key functions:
- `create_message()` - Create a structured message 
- `send_message()` - Send a message to another agent
- `get_messages()` - Retrieve messages for an agent with optional filtering
- `mark_message_read()` - Track message read status

### Agent Communication Protocol

The agent protocol module (`utils/agent_protocol.py`) defines:

- Message types (request, response, notification, error)
- Priority levels for messages (low, normal, high, urgent)
- Status codes for responses (ok, error, partial, processing)
- TypedDict definitions for message structure
- Helper functions for creating protocol-compliant messages

Key protocol features:
- Standard message format with sender, recipient, timestamp, and content
- Reference IDs for connecting related messages
- Content schema for different message types

## Usage Example

```python
# Import the protocol classes
from utils.agent_protocol import create_request, create_response, MessageType

# Agent A sends a request to Agent B
request = create_request(
    sender_id="AgentA",
    recipient_id="AgentB",
    action="get_data",
    data={"entity_type": "customer"}
)

# Store the request in session state
context["session_state"]["messages"].append(request)

# Agent B processes the request and sends a response
response = create_response(
    request_message=request,
    data={"customer_data": {"id": "123", "name": "John Doe"}},
    status="ok"
)

# Store the response in session state
context["session_state"]["messages"].append(response)
```

## Testing

The framework includes tests to verify:

- Message creation and structure
- Message sending and receiving
- Protocol compliance
- Session state interactions

Run the tests using:
```
./run_agent_test.sh
```

## Example Implementation

A complete example showing two agents communicating is provided in `examples/agent_communication_example.py`. This demonstrates:

- A RequestAgent that sends data requests
- A ResponseAgent that processes requests and sends back data
- Tracking of message state
- Processing of responses

Run the example using:
```
./run_agent_communication.sh
```
