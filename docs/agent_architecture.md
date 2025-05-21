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
| Document Processor Agent| |Validation Agent| |  Storage Agent    |
|                         | |                | |                   |
+-------------------------+ +----------------+ +-------------------+
                                   |                    
                                   |                    
                                   v                    
                            +----------------+ 
                            |                | 
                            | Analytics Agent| 
                            |                | 
                            +----------------+ 
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
2. Validation against the rules
3. Storage of validated document data
4. Analytics on stored documents
