# FinFlow Agent Architecture

## Agent Hierarchy

```
                    +----------------------------+
                    |                            |
                    |  Master Orchestrator Agent |
                    |                            |
                    +----------------------------+
                         |         |       |
             +-----------+         |       +------------+
             |                     |                    |
             v                     v                    v
+-------------------------+ +----------------+ +-------------------+
|                         | |                | |                   |
| Document Processor Agent| | Validation Agent| | Storage Agent     |
|                         | |                | |                   |
+-------------------------+ +----------------+ +-------------------+
             |                     |                    |
             |                     |                    |
             v                     v                    v
+-------------------------+ +----------------+ +-------------------+
|                         | |                | |                   |
| Rule Retrieval Agent    | | Analytics Agent| |                   |
|                         | |                | |                   |
+-------------------------+ +----------------+ +-------------------+
```

## Agent Responsibilities

### Master Orchestrator Agent
- Coordinates the overall document processing workflow
- Delegates tasks to specialized worker agents
- Manages the state of document processing
- Handles errors and retries
- Ensures end-to-end processing completion

### Document Processor Agent
- Extracts structured information from documents using Document AI
- Classifies document types (invoices, receipts, etc.)
- Normalizes extracted data to standard formats
- Performs initial data validation

### Rule Retrieval Agent
- Retrieves applicable compliance rules based on document type
- Filters rules based on jurisdiction and other attributes
- Provides rule details to the validation agent
- Updates rule repository with new rules

### Validation Agent
- Validates documents against compliance rules
- Performs mathematical validation for financial documents
- Checks for regulatory compliance
- Generates validation reports with issue details
- Categorizes issues by severity

### Storage Agent
- Manages data persistence in BigQuery
- Creates relationships between entities and documents
- Implements data versioning and audit trails
- Handles data retrieval requests
- Maintains data consistency and integrity

### Analytics Agent
- Calculates financial metrics and KPIs
- Identifies spending patterns and trends
- Generates financial reports and visualizations
- Provides forecast and budget analysis
- Detects unusual transactions

## Communication Pattern

Agents communicate primarily through:
1. Direct invocation via AgentTool
2. State-based communication via shared session state
3. LLM-driven delegation through transfer_to_agent

## Data Flow

1. Document ingestion via Document Processor
2. Rule retrieval for the specific document type
3. Validation against retrieved rules
4. Storage of validated document data
5. Analytics on stored documents
