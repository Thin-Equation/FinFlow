# FinFlow Storage Agent

## Overview

The Storage Agent is a core component of the FinFlow system responsible for managing all data persistence operations in BigQuery. It handles document storage, data retrieval, entity management, relationship tracking, and financial data analysis.

This production-level implementation provides:

- Comprehensive BigQuery dataset and schema management
- Complete data insertion, transformation, and querying capabilities
- Financial analysis functions for business intelligence
- Caching for frequent queries to improve performance
- Robust error handling and data validation

## Architecture

The StorageAgent is built on a modular architecture that separates concerns:

1. **Storage Agent Core**: Manages agent initialization, tool registration, and high-level API
2. **BigQuery Tool Layer**: Handles all direct interactions with the BigQuery API
3. **Data Transformation Layer**: Converts between application models and database schemas
4. **Query Construction Tools**: Builds optimized queries for different data access patterns
5. **Caching Layer**: Improves performance for frequently-accessed data

## Features

### BigQuery Integration

- Creates and manages datasets and tables with optimized schemas
- Handles all CRUD operations for financial documents and entities
- Supports both standard and legacy SQL query formats
- Implements proper error handling and retry mechanisms

### Data Schema

The agent uses the following table schemas:

- **documents**: Main storage for all financial documents
- **line_items**: Line items from financial documents
- **entities**: Vendors, customers, and other business entities
- **document_relationships**: Relationships between documents
- **account_codes**: Chart of accounts and financial categories
- **financial_summaries**: Aggregated financial metrics

### Financial Analysis

Built-in analysis functions include:

- Aging reports for accounts payable/receivable
- Spending analysis by vendor
- Monthly expense trends
- Category distribution analysis

### Performance Optimization

- Implements caching for frequent queries using LRU cache
- Optimized table schemas with appropriate clustering/partitioning
- Batch operations for improved throughput

## Usage

### Basic Usage

```python
from agents.storage_agent import StorageAgent

# Initialize the storage agent
agent = StorageAgent()

# Store a document
result = await agent.store_document(document_data)

# Retrieve a document
document = await agent.retrieve_document(document_id)

# Query documents
results = await agent.query_documents({
    "filters": {"document_type": "invoice"},
    "limit": 10
})
```

### Advanced Features

```python
# Run financial analysis
analysis_result = agent._tool_run_financial_analysis(
    analysis_type="aging_report",
    parameters={
        "doc_type": "invoice",
        "start_date": "2025-01-01",
        "end_date": "2025-03-31"
    }
)

# Create document relationships
relationship = agent._tool_create_document_relationship(
    source_document_id="doc-001",
    target_document_id="doc-002",
    relationship_type="invoice_to_receipt"
)

# Run a custom query with caching
result = bigquery.cached_query_financial_data(
    query="SELECT * FROM documents WHERE total_amount > 1000",
    project_id="my-project",
    cache_key="high_value_docs",
    max_age_seconds=300
)
```

## Configuration

The StorageAgent can be configured through:

1. YAML configuration files (development.yaml, staging.yaml, production.yaml)
2. Environment variables
3. Direct configuration during initialization

Key configuration options:

```yaml
bigquery:
  project_id: "your-project-id"
  dataset_id: "finflow_data"
  location: "US"

storage:
  enable_cache: true
  cache_ttl_seconds: 300
  max_cache_size: 100
  batch_size: 100
```

## Unit Testing

Comprehensive unit tests are provided in `tests/test_storage_agent.py`. Run them with:

```bash
pytest tests/test_storage_agent.py
```

Integration tests that interact with actual BigQuery services are in `tests/test_storage_agent_integration.py`. Run them with:

```bash
pytest -m integration tests/test_storage_agent_integration.py
```

## Error Handling

The StorageAgent implements comprehensive error handling:

- All database operations are wrapped in try/except blocks
- Detailed error messages are logged
- Appropriate error responses are returned to callers
- Critical errors trigger alerts through the logging system

## Future Improvements

- Implement distributed caching with Redis
- Add support for data versioning and audit trails
- Implement more advanced financial analysis functions
- Add support for data export to common formats
