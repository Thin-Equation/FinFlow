# FinFlow Workflow System

The FinFlow Workflow System provides a robust, production-level framework for defining, executing, and managing financial process workflows. It supports sequential and parallel execution models, conditional branching, and rich error handling.

## Core Components

### Workflow Definitions

- **`WorkflowDefinition`**: Abstract base class for all workflows
- **`FinancialProcess`**: Specialized workflow for financial processes
- **`WorkflowTask`**: Individual tasks within a workflow
- **`WorkflowExecutionContext`**: Context for workflow execution with shared state
- **`WorkflowResult`**: Result of workflow execution

### Execution Engines

- **`SequentialAgent`**: Executes workflow tasks sequentially, respecting dependencies
- **`ParallelAgent`**: Executes tasks in parallel when dependencies allow
- **`ConditionalBranching`**: Provides dynamic path selection based on conditions

## Key Features

1. **Dependency Management**: Tasks can declare dependencies on other tasks, ensuring proper execution order
2. **State Management**: Shared context allows tasks to access results from previous tasks
3. **Parallel Execution**: Concurrent execution of independent tasks for improved performance
4. **Error Handling**: Comprehensive error handling with configurable failure behavior
5. **Conditional Branching**: Dynamic workflow paths based on runtime conditions
6. **Monitoring & Reporting**: Detailed execution status and results
7. **Timeouts & Retries**: Configurable timeouts and retry policies for tasks

## Usage Example

```python
from workflow.workflow_definitions import (
    Workflow,
    WorkflowTask,
    WorkflowExecutionContext
)
from workflow.sequential_agent import SequentialAgent

# Create a workflow definition
workflow = Workflow.create(
    name="Invoice Processing",
    description="Process invoices from receipt to payment",
    process_type="invoice_processing"
)

# Define workflow tasks
extract_task = WorkflowTask(
    id="extract_data",
    name="Extract Invoice Data",
    description="Extract structured data from invoice document",
    execute=lambda ctx: {"invoice_number": "INV-2025-1234", "amount": 5750.00}
)

validate_task = WorkflowTask(
    id="validate_data",
    name="Validate Invoice Data",
    description="Validate invoice data for completeness and correctness",
    execute=lambda ctx: {"valid": True, "errors": []},
    dependencies=[extract_task.id]  # This task depends on the extract_task
)

# Add tasks to the workflow
workflow.add_task(extract_task)
workflow.add_task(validate_task)

# Create a sequential agent to execute the workflow
agent = SequentialAgent()

# Execute the workflow
result = agent.run_workflow(
    workflow_definition=workflow,
    context={"start_time": "2025-05-21T10:00:00"},
    parameters={"document_id": "DOC-12345"}
)

# Check results
if result.is_successful:
    print(f"Workflow completed successfully in {result.execution_time} seconds")
else:
    print(f"Workflow failed: {result.error}")
```

## Conditional Branching Example

```python
from workflow.conditional import ConditionalBranching, Branch, BranchSelectionStrategy

# Define approval branches based on invoice amount
high_value_branch = Branch(
    name="high_value_approval",
    condition=lambda ctx: ctx.get_result("calculate_metrics")["amount"] >= 10000,
    tasks=[manager_approval_task, director_approval_task],
    priority=3
)

medium_value_branch = Branch(
    name="medium_value_approval",
    condition=lambda ctx: 1000 <= ctx.get_result("calculate_metrics")["amount"] < 10000,
    tasks=[manager_approval_task],
    priority=2
)

# Add conditional branching to the workflow
ConditionalBranching.add_switch(
    workflow=workflow,
    switch_name="approval_routing",
    branches=[high_value_branch, medium_value_branch, low_value_branch],
    parent_task_id=calculate_metrics_task.id,
    exit_task_id=payment_task.id,
    strategy=BranchSelectionStrategy.PRIORITY
)
```

## Parallel Processing Example

```python
from workflow.parallel_agent import ParallelAgent

# Create independent report section tasks
income_statement_task = WorkflowTask(
    id="income_statement",
    name="Generate Income Statement",
    description="Generate the income statement section",
    execute=lambda ctx: {"section": "income_statement", "status": "generated"},
    dependencies=[transform_data_task.id]
)

balance_sheet_task = WorkflowTask(
    id="balance_sheet",
    name="Generate Balance Sheet",
    description="Generate the balance sheet section",
    execute=lambda ctx: {"section": "balance_sheet", "status": "generated"},
    dependencies=[transform_data_task.id]
)

# Create a task that depends on both reports
consolidate_task = WorkflowTask(
    id="consolidate_reports",
    name="Consolidate Reports",
    description="Consolidate all report sections",
    execute=lambda ctx: {"consolidated": True, "status": "success"},
    dependencies=[income_statement_task.id, balance_sheet_task.id]
)

# Add tasks to the workflow
workflow.add_task(income_statement_task)
workflow.add_task(balance_sheet_task)
workflow.add_task(consolidate_task)

# Use ParallelAgent for execution
parallel_agent = ParallelAgent(max_workers=5)
result = parallel_agent.run_workflow(workflow_definition=workflow, context={})
```

## Best Practices

1. **Task Granularity**: Design tasks with appropriate granularity - not too small to create overhead, not too large to block parallelism
2. **Idempotency**: Design tasks to be idempotent when possible to support retries
3. **Error Handling**: Define clear error handling strategies for each workflow
4. **State Management**: Use the execution context to share state between tasks
5. **Monitoring**: Implement proper logging within tasks for observability
6. **Testing**: Create unit tests for individual tasks and integration tests for workflows
