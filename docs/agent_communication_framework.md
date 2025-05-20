# Agent Communication Framework

A comprehensive framework for agent-to-agent communication in the FinFlow system.

## Features

- **Full Agent Communication Protocol**: Standardized message formats, request/response patterns, and notification mechanisms
- **Direct Invocation via AgentTool**: Enhanced agent tool creation for transparent and efficient agent invocation
- **State-based Communication**: Workflow state tracking and management for complex multi-agent processes
- **LLM-driven Delegation Patterns**: Intelligent task delegation based on agent capabilities and performance metrics

## Core Components

- **AgentInvokeTool**: Enhanced tool for direct agent-to-agent communication
- **Workflow States**: State machine for tracking complex workflows
- **Delegation Strategies**: Multiple strategies for intelligent task delegation
- **Protocol Messages**: Standardized message formats for agent communication

## Usage

### Example: Delegating a Task

```python
from utils.agent_communication import delegate_task, DelegationStrategy
from utils.agent_protocol import PriorityLevel

# Create delegation request
success, delegation_result = delegate_task(
    context=context,
    task_description="Extract information from invoice document",
    required_capabilities=["document_processing", "information_extraction"],
    available_agents=agent_registry,
    priority=PriorityLevel.HIGH,
    strategy=DelegationStrategy.CAPABILITY_BASED
)

# Check result
if success:
    workflow_id = delegation_result.get("workflow_id")
    delegatee_id = delegation_result.get("delegatee_id")
else:
    error_reason = delegation_result.get("reason")
```

### Example: Using Workflow States

```python
from utils.agent_communication import create_workflow, transition_workflow, WorkflowState

# Create a workflow
workflow_id = create_workflow(
    context=context,
    workflow_type="document_processing",
    owner_id="master_orchestrator",
    initial_state=WorkflowState.INITIALIZED
)

# Update workflow state
transition_workflow(
    context=context,
    workflow_id=workflow_id,
    from_state=WorkflowState.INITIALIZED,
    to_state=WorkflowState.IN_PROGRESS,
    agent_id="master_orchestrator",
    reason="Starting document processing workflow"
)
```

## Integration

The framework is fully integrated with the FinFlow agent system. Key integration points:

- **MasterOrchestratorAgent**: Enhanced with delegation capabilities
- **AgentDelegator**: Specialized agent for LLM-driven delegation
- **DelegatableAgent**: Mixin for agents that can accept delegated tasks

See `examples/agent_communication_example.py` for complete usage examples.
