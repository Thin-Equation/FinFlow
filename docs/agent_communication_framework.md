# Agent Communication Framework

A comprehensive framework for agent-to-agent communication in the FinFlow system. The framework provides robust, production-level components for building complex multi-agent systems with reliable communication patterns.

## Features

- **Full Agent Communication Protocol**: Standardized message formats, request/response patterns, and notification mechanisms with delivery guarantees
- **Direct Invocation via AgentTool**: Enhanced agent tool creation for transparent and efficient agent invocation with state management
- **State-based Communication**: Workflow state tracking and management for complex multi-agent processes with history tracking
- **LLM-driven Delegation Patterns**: Intelligent task delegation based on agent capabilities, performance metrics, and adaptive learning
- **Task Execution Framework**: Robust system for task management, progress tracking, and parallel execution
- **Advanced Delegation Strategies**: Multiple delegation strategies including capability-based, load-balanced, priority-based, and learning-based approaches

## Core Components

### Communication Protocol

The `CommunicationProtocol` class provides a robust, production-level implementation for agent communication with:

- **Message Delivery Guarantees**: Configurable retries and acknowledgments
- **Standardized Message Formats**: Consistent protocol across all agents
- **Message History and Tracking**: Complete audit trail of all communications
- **Unread Message Management**: System for handling unread messages

```python
# Using the Communication Protocol
comms = CommunicationProtocol(context, agent_id)

# Send a message with delivery guarantees
message = comms.send_message(
    recipient_id="DocumentProcessor",
    message_type=MessageType.REQUEST,
    content={"action": "process_document", "document_path": "/path/to/doc.pdf"},
    delivery_guarantees=True,
    retry_count=3
)

# Check for unread messages
unread_messages = comms.get_unread_messages()

# Acknowledge message receipt
comms.acknowledge_message(message_id)
```

### Task Execution Framework

The `TaskExecutionFramework` manages task creation, execution, and monitoring with:

- **Task Hierarchy**: Support for parent tasks and subtasks
- **Progress Tracking**: Automatic propagation of progress to parent tasks
- **Parallel Execution**: Support for concurrent task execution
- **Execution Metrics**: Performance tracking for continuous improvement

```python
# Using the Task Execution Framework
tasks = TaskExecutionFramework(context, agent_id)

# Create a main task
task_id = tasks.create_task(
    task_description="Process financial document",
    task_type="document_processing"
)

# Create and execute subtasks
subtask_results = tasks.create_and_execute_subtasks(
    parent_task_id=task_id,
    subtask_definitions=[
        {
            "description": "Extract invoice data",
            "type": "data_extraction",
            "executor": extract_invoice_data_func
        },
        {
            "description": "Validate invoice data",
            "type": "data_validation",
            "executor": validate_invoice_data_func
        }
    ]
)

# Update task status
tasks.update_task_status(
    task_id=task_id, 
    status="completed", 
    progress=1.0,
    result={"processed_data": result_data}
)
```

### Advanced Delegation Strategies

The framework provides multiple delegation strategies to optimize agent selection:

- **CAPABILITY_BASED**: Find agents with matching capabilities
- **AVAILABILITY_BASED**: Select agents with highest availability
- **LOAD_BALANCED**: Distribute tasks to minimize load
- **PRIORITY_BASED**: High-priority tasks go to best-matched agents
- **ADAPTIVE**: Consider past performance, load, and capability
- **LEARNING_BASED**: Use historical performance data to improve selection
- **ROUND_ROBIN**: Distribute tasks evenly among agents
- **HIERARCHICAL**: Use hierarchical delegation patterns

```python
# Use specific delegation strategies
success, delegation_result = delegate_task(
    context=context,
    task_description="Extract information from invoice document",
    required_capabilities=["document_processing", "information_extraction"],
    available_agents=agent_registry,
    priority=PriorityLevel.HIGH,
    strategy=DelegationStrategy.ADAPTIVE  # Using the adaptive strategy
)

# Or apply a specific strategy directly
selected_agent_id = apply_delegation_strategy(
    context=context,
    delegation_request=delegation_request,
    available_agents=available_agents,
    strategy=DelegationStrategy.LEARNING_BASED
)
```

### Direct Agent Invocation

Enhanced tools for direct agent invocation with state management:

- **AgentInvokeTool**: Advanced tool for direct agent-to-agent communication
- **Enhanced Agent Tools**: With built-in state management
- **Transparent Invocation**: Call other agents like regular functions

```python
from utils.agent_communication import create_enhanced_agent_tool

# Create enhanced tool for agent
validation_tool = create_enhanced_agent_tool(validation_agent)

# Invoke agent with context
validation_result = validation_tool.execute(context)
```

### Workflow State Management

Comprehensive workflow state machine for complex multi-agent processes:

- **State Transitions**: Controlled transitions with validation
- **History Tracking**: Complete audit trail of all state changes
- **Metadata Management**: Associate metadata with workflows
- **Multi-agent Coordination**: Coordinate work across multiple agents

```python
from utils.agent_communication import WorkflowState, create_workflow, transition_workflow

# Create workflow
workflow_id = create_workflow(
    context=context,
    workflow_type="document_processing",
    initial_state=WorkflowState.INITIALIZED,
    metadata={"document_path": "/path/to/document.pdf"}
)

# Transition workflow from one state to another with validation
success = transition_workflow(
    context=context,
    workflow_id=workflow_id,
    from_state=WorkflowState.INITIALIZED,  # Must be in this state
    to_state=WorkflowState.IN_PROGRESS,
    agent_id="processor_agent",
    reason="Starting document processing"
)
```

### Agent Performance Tracking

Built-in metrics tracking for continuous improvement:

- **Execution Time**: Track how long tasks take
- **Success Rates**: Monitor agent reliability
- **Custom Metrics**: Add domain-specific metrics
- **Adaptive Selection**: Use metrics to improve agent selection

```python
from utils.agent_communication import track_agent_metrics, get_agent_metrics

# Track agent performance
track_agent_metrics(
    context=context,
    agent_id="DocumentProcessor",
    execution_time=1.25,  # seconds
    success=True,
    metadata={"document_type": "invoice"}
)

# Get agent performance metrics
metrics = get_agent_metrics(context, "DocumentProcessor")
```

## Advanced Usage Examples

See the comprehensive example in `examples/advanced_agent_communication_example.py` which demonstrates:

1. Full agent communication protocol usage
2. State-based workflow management
3. Task hierarchy with parent/subtasks
4. Multiple delegation strategies
5. Performance metrics tracking
6. Error handling and recovery

## Integration With Existing FinFlow Components

The framework is fully integrated with the FinFlow agent system. Key integration points:

- **MasterOrchestratorAgent**: Enhanced with delegation capabilities
- **AgentDelegator**: Specialized agent for LLM-driven delegation
- **DelegatableAgent**: Mixin for agents that can accept delegated tasks

## Best Practices

1. **Use Workflow IDs**: Always associate tasks with workflows for tracking
2. **Track Performance**: Use the metrics system to improve delegation
3. **Select Appropriate Strategies**: Choose delegation strategies based on task requirements
4. **Implement Error Handling**: Handle delegation failures appropriately
5. **Use Task Hierarchies**: Break complex tasks into subtasks for better management
