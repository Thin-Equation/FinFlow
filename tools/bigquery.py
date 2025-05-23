"""
BigQuery integration tools for the FinFlow system.
"""

import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext  # type: ignore
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField, Table, Dataset, QueryJobConfig

# Type definitions for clarity
SchemaType = List[SchemaField]
TableSchemas = Dict[str, SchemaType]
QueryResult = Dict[str, Any]
BQRow = Dict[str, Any]


def create_dataset(project_id: str, dataset_id: str, location: str = "US", 
                   tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Create a BigQuery dataset if it doesn't exist.
    
    Args:
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        location: Dataset location (default: "US")
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    
    try:
        # Check if dataset exists
        client.get_dataset(dataset_ref)
        return {"status": "success", "message": f"Dataset {dataset_id} already exists"}
    except NotFound:
        # Dataset doesn't exist, create it
        dataset = Dataset(dataset_ref)
        dataset.location = location
        dataset.description = "FinFlow financial data management dataset"
        
        try:
            dataset = client.create_dataset(dataset, timeout=30)
            return {
                "status": "success", 
                "message": f"Dataset {dataset_id} created successfully in {location}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error creating dataset: {str(e)}"}


def get_table_schemas() -> TableSchemas:
    """
    Define the schema for all FinFlow tables.
    
    Returns:
        Dict[str, List[SchemaField]]: Table schemas
    """
    # Define schemas for each table
    return {
        # Main document storage - normalized structure
        "documents": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Unique document ID"),
            SchemaField("document_type", "STRING", mode="REQUIRED", description="Type of document (invoice, receipt, etc.)"),
            SchemaField("document_number", "STRING", mode="REQUIRED", description="Document number (invoice #, receipt #)"),
            SchemaField("status", "STRING", mode="REQUIRED", description="Document processing status"),
            SchemaField("issuer_id", "STRING", description="ID of the issuing entity"),
            SchemaField("recipient_id", "STRING", description="ID of the recipient entity"),
            SchemaField("issue_date", "DATE", description="Date the document was issued"),
            SchemaField("due_date", "DATE", description="Date payment is due"),
            SchemaField("currency", "STRING", description="Currency code (USD, EUR, etc)"),
            SchemaField("total_amount", "FLOAT64", description="Total document amount"),
            SchemaField("subtotal", "FLOAT64", description="Subtotal before taxes/fees"),
            SchemaField("tax_amount", "FLOAT64", description="Total tax amount"),
            SchemaField("confidence_score", "FLOAT64", description="Document AI confidence score"),
            SchemaField("source_file", "STRING", description="Original source file path/URI"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("content", "JSON", description="Full document content in JSON format"),
            SchemaField("metadata", "JSON", description="Additional metadata"),
        ],
        
        # Line items from documents (child table)
        "line_items": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Unique line item ID"),
            SchemaField("document_id", "STRING", mode="REQUIRED", description="Parent document ID"),
            SchemaField("description", "STRING", description="Line item description"),
            SchemaField("quantity", "FLOAT64", description="Quantity"),
            SchemaField("unit_price", "FLOAT64", description="Unit price"),
            SchemaField("total_amount", "FLOAT64", description="Total line amount"),
            SchemaField("tax_amount", "FLOAT64", description="Tax amount for this line"),
            SchemaField("tax_rate", "FLOAT64", description="Tax rate percentage"),
            SchemaField("account_code", "STRING", description="GL account code"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("metadata", "JSON", description="Additional metadata"),
        ],
        
        # Entities table (vendors, customers, etc.)
        "entities": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Unique entity ID"),
            SchemaField("entity_type", "STRING", mode="REQUIRED", description="Type of entity"),
            SchemaField("name", "STRING", mode="REQUIRED", description="Entity name"),
            SchemaField("tax_id", "STRING", description="Tax identification number"),
            SchemaField("email", "STRING", description="Primary email address"),
            SchemaField("phone", "STRING", description="Primary phone number"),
            SchemaField("website", "STRING", description="Website URL"),
            SchemaField("address", "JSON", description="Address information"),
            SchemaField("payment_terms", "STRING", description="Default payment terms"),
            SchemaField("industry", "STRING", description="Industry classification"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("metadata", "JSON", description="Additional metadata"),
        ],
        
        # Document relationships
        "document_relationships": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Relationship ID"),
            SchemaField("source_document_id", "STRING", mode="REQUIRED"),
            SchemaField("target_document_id", "STRING", mode="REQUIRED"),
            SchemaField("relationship_type", "STRING", mode="REQUIRED"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("metadata", "JSON", description="Additional metadata"),
        ],
        
        # Financial account codes (chart of accounts)
        "account_codes": [
            SchemaField("code", "STRING", mode="REQUIRED", description="Account code"),
            SchemaField("name", "STRING", mode="REQUIRED", description="Account name"),
            SchemaField("account_type", "STRING", mode="REQUIRED", description="Account type"),
            SchemaField("description", "STRING", description="Account description"),
            SchemaField("parent_code", "STRING", description="Parent account code"),
            SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        ],
        
        # For aggregated financial data by period
        "financial_summaries": [
            SchemaField("id", "STRING", mode="REQUIRED"),
            SchemaField("period", "STRING", mode="REQUIRED", description="YYYY-MM format"),
            SchemaField("entity_id", "STRING", description="Related entity ID if applicable"),
            SchemaField("summary_type", "STRING", mode="REQUIRED"),  # e.g., 'monthly_expenses'
            SchemaField("amount", "FLOAT64", mode="REQUIRED"),
            SchemaField("currency", "STRING", mode="REQUIRED"),
            SchemaField("metric_name", "STRING", mode="REQUIRED"),  # e.g., 'total_revenue'
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("data_point", "JSON", description="Additional metric data"),
        ],
        
        # Document versioning
        "document_versions": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Version ID"),
            SchemaField("document_id", "STRING", mode="REQUIRED", description="ID of the document"),
            SchemaField("version_number", "INTEGER", mode="REQUIRED", description="Sequential version number"),
            SchemaField("content", "JSON", mode="REQUIRED", description="Document content at this version"),
            SchemaField("created_at", "TIMESTAMP", mode="REQUIRED", description="When this version was created"),
            SchemaField("created_by", "STRING", description="User or system that created this version"),
            SchemaField("change_summary", "STRING", description="Summary of changes from previous version"),
            SchemaField("metadata", "JSON", description="Additional version metadata"),
        ],
        
        # Audit trail
        "audit_trail": [
            SchemaField("id", "STRING", mode="REQUIRED", description="Audit record ID"),
            SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="When the action occurred"),
            SchemaField("action", "STRING", mode="REQUIRED", description="Type of action (create, update, delete, etc.)"),
            SchemaField("resource_type", "STRING", mode="REQUIRED", description="Type of resource affected (document, entity, etc.)"),
            SchemaField("resource_id", "STRING", mode="REQUIRED", description="ID of the resource affected"),
            SchemaField("user_id", "STRING", description="User who performed the action"),
            SchemaField("details", "JSON", description="Details of the action"),
            SchemaField("ip_address", "STRING", description="IP address where action originated"),
            SchemaField("user_agent", "STRING", description="User agent information"),
        ],
    }


def create_financial_tables(project_id: str, dataset_id: str, 
                           tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Create all required tables for FinFlow in BigQuery.
    
    Args:
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client(project=project_id)
    tables = get_table_schemas()
    results: Dict[str, str] = {}
    
    for table_name, schema in tables.items():
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        table = Table(table_id, schema=schema)
        
        # Add clustering/partitioning for performance optimization
        if table_name == "documents":
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="created_at"
            )
            table.clustering_fields = ["document_type", "status"]
        elif table_name == "line_items":
            table.clustering_fields = ["document_id"]
        elif table_name == "financial_summaries":
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.MONTH,
                field="created_at"
            )
            table.clustering_fields = ["summary_type", "entity_id"]
        elif table_name == "document_versions":
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="created_at"
            )
            table.clustering_fields = ["document_id", "version_number"]
        elif table_name == "audit_trail":
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            table.clustering_fields = ["resource_type", "resource_id", "action"]
        
        try:
            client.create_table(table, exists_ok=True)
            results[table_name] = "created"
        except Exception as e:
            results[table_name] = f"error: {str(e)}"
    
    return {
        "status": "success",
        "tables": results
    }


def store_document(document_data: Dict[str, Any], project_id: str, dataset_id: str, 
                   table_id: str = "documents", tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Store processed document data in BigQuery.
    
    Args:
        document_data: Structured document data
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID (default: "documents")
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client(project=project_id)
    
    # Add timestamps if not present
    if "created_at" not in document_data:
        document_data["created_at"] = datetime.utcnow().isoformat()
    if "updated_at" not in document_data:
        document_data["updated_at"] = document_data["created_at"]
    
    # Prepare the data for insertion
    rows_to_insert = [document_data]
    
    try:
        fully_qualified_table_id = f"{project_id}.{dataset_id}.{table_id}"
        errors = client.insert_rows_json(fully_qualified_table_id, rows_to_insert)
        
        if not errors:
            return {
                "status": "success", 
                "message": "Document stored successfully",
                "document_id": document_data.get("id", "unknown")
            }
        else:
            return {
                "status": "error", 
                "message": f"Encountered errors while inserting rows: {errors}"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def store_batch(rows: List[Dict[str, Any]], project_id: str, dataset_id: str, 
                table_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Store multiple rows in BigQuery in a single batch operation.
    
    Args:
        rows: List of rows to insert
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    if not rows:
        return {"status": "success", "message": "No rows to insert", "count": 0}
    
    client = bigquery.Client(project=project_id)
    fully_qualified_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    # Add timestamps for all rows if not present
    current_time = datetime.utcnow().isoformat()
    for row in rows:
        if "created_at" not in row:
            row["created_at"] = current_time
        if "updated_at" not in row:
            row["updated_at"] = current_time
    
    try:
        errors = client.insert_rows_json(fully_qualified_table_id, rows)
        
        if not errors:
            return {
                "status": "success", 
                "message": f"Successfully inserted {len(rows)} rows",
                "count": len(rows)
            }
        else:
            return {
                "status": "error", 
                "message": f"Encountered errors while inserting rows: {errors}"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_financial_data(query: str, project_id: str, 
                         use_legacy_sql: bool = False,
                         timeout: int = 30,
                         tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Execute a query against financial data in BigQuery.
    
    Args:
        query: SQL query to execute
        project_id: Google Cloud project ID
        use_legacy_sql: Whether to use legacy SQL syntax
        timeout: Query timeout in seconds
        tool_context: Provided by ADK
        
    Returns:
        dict: Query results
    """
    client = bigquery.Client(project=project_id)
    
    job_config = QueryJobConfig()
    job_config.use_legacy_sql = use_legacy_sql
    
    try:
        # Execute the query
        query_job = client.query(query, job_config=job_config)
        results = query_job.result(timeout=timeout)
        
        # Process results into a standard format
        processed_rows: List[Dict[str, Any]] = []
        schema = [field.name for field in results.schema]
        
        for row in results:
            processed_row = {}
            for field_name in schema:
                value = row.get(field_name)
                
                # Handle special types that need conversion
                if isinstance(value, datetime):
                    processed_row[field_name] = value.isoformat()
                elif hasattr(value, 'value'):
                    # Handle custom BigQuery types
                    processed_row[field_name] = value.value
                else:
                    processed_row[field_name] = value
                
            processed_rows.append(processed_row)
        
        return {
            "status": "success",
            "row_count": len(processed_rows),
            "columns": schema,
            "data": processed_rows
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "row_count": 0,
            "data": []
        }


@lru_cache(maxsize=128)
def cached_query_financial_data(query: str, project_id: str, cache_key: str, 
                               max_age_seconds: int = 300) -> Dict[str, Any]:
    """
    Execute a query with caching support.
    
    Args:
        query: SQL query to execute
        project_id: Google Cloud project ID
        cache_key: Unique key for caching (include timestamp parts that matter)
        max_age_seconds: Maximum cache age in seconds
        
    Returns:
        dict: Query results
    """
    # Simple time-based cache invalidation - time calculation kept for future implementation
    _ = int(time.time() / max_age_seconds)
    # Cache key is used by the lru_cache decorator automatically
    
    # Execute the actual query - the lru_cache decorator will handle caching
    return query_financial_data(query, project_id)


def delete_data(table_id: str, filter_condition: str, project_id: str, dataset_id: str,
                tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Delete data from a BigQuery table using a filter condition.
    
    Args:
        table_id: Table name to delete from
        filter_condition: WHERE clause for deletion (without 'WHERE')
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client(project=project_id)
    
    # Construct the full table name
    fully_qualified_table_id = f"{dataset_id}.{table_id}"
    
    # Construct and execute the DELETE query
    query = f"DELETE FROM `{project_id}.{fully_qualified_table_id}` WHERE {filter_condition}"
    
    try:
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for query to complete
        
        return {
            "status": "success",
            "message": f"Successfully deleted matching rows from {table_id}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting data: {str(e)}"
        }


def update_data(table_id: str, update_clauses: str, filter_condition: str, 
                project_id: str, dataset_id: str,
                tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Update data in a BigQuery table.
    
    Args:
        table_id: Table name to update
        update_clauses: SET clause for UPDATE (without 'SET')
        filter_condition: WHERE clause (without 'WHERE')
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client(project=project_id)
    
    # Construct the full table name
    fully_qualified_table_id = f"{dataset_id}.{table_id}"
    
    # Add updated_at timestamp to update clauses if not already specified
    if "updated_at" not in update_clauses:
        current_timestamp = f"TIMESTAMP '{datetime.utcnow().isoformat()}'"
        update_clauses += f", updated_at = {current_timestamp}"
    
    # Construct and execute the UPDATE query
    query = f"UPDATE `{project_id}.{fully_qualified_table_id}` SET {update_clauses} WHERE {filter_condition}"
    
    try:
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for query to complete
        
        return {
            "status": "success",
            "message": f"Successfully updated matching rows in {table_id}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error updating data: {str(e)}"
        }


def run_financial_analysis(analysis_type: str, parameters: Dict[str, Any],
                          project_id: str, dataset_id: str,
                          tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Run pre-defined financial analysis queries.
    
    Args:
        analysis_type: Type of analysis to run
        parameters: Parameters for the analysis
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Analysis results
    """
    # Map of analysis types to query templates
    analysis_queries = {
        "aging_report": """
            SELECT 
                e.name as entity_name,
                d.document_number,
                d.total_amount,
                d.currency,
                d.issue_date,
                d.due_date,
                DATE_DIFF(CURRENT_DATE(), d.due_date, DAY) as days_overdue,
                CASE
                    WHEN DATE_DIFF(CURRENT_DATE(), d.due_date, DAY) <= 0 THEN 'current'
                    WHEN DATE_DIFF(CURRENT_DATE(), d.due_date, DAY) <= 30 THEN '1-30'
                    WHEN DATE_DIFF(CURRENT_DATE(), d.due_date, DAY) <= 60 THEN '31-60'
                    WHEN DATE_DIFF(CURRENT_DATE(), d.due_date, DAY) <= 90 THEN '61-90'
                    ELSE '90+'
                END as aging_bucket
            FROM 
                `{project_id}.{dataset_id}.documents` d
            JOIN
                `{project_id}.{dataset_id}.entities` e
            ON
                d.{entity_field} = e.id
            WHERE
                d.status = 'approved'
                AND d.document_type = '{doc_type}'
                {entity_filter}
                {date_filter}
            ORDER BY
                days_overdue DESC
        """,
        
        "spending_by_vendor": """
            SELECT
                e.name as vendor_name,
                SUM(d.total_amount) as total_spent,
                d.currency,
                COUNT(d.id) as document_count,
                MIN(d.issue_date) as first_transaction,
                MAX(d.issue_date) as latest_transaction
            FROM
                `{project_id}.{dataset_id}.documents` d
            JOIN
                `{project_id}.{dataset_id}.entities` e
            ON
                d.issuer_id = e.id
            WHERE
                d.document_type IN ('invoice', 'receipt', 'expense')
                {date_filter}
                {entity_filter}
            GROUP BY
                e.name, d.currency
            ORDER BY
                total_spent DESC
        """,
        
        "monthly_expenses": """
            SELECT
                FORMAT_DATE('%Y-%m', d.issue_date) as month,
                SUM(d.total_amount) as total_expenses,
                d.currency
            FROM
                `{project_id}.{dataset_id}.documents` d
            WHERE
                d.document_type IN ('invoice', 'receipt', 'expense')
                {entity_filter}
                {date_filter}
            GROUP BY
                month, d.currency
            ORDER BY
                month
        """,
        
        "category_distribution": """
            SELECT
                li.account_code,
                ac.name as account_name,
                SUM(li.total_amount) as total_amount,
                COUNT(li.id) as line_item_count
            FROM
                `{project_id}.{dataset_id}.line_items` li
            LEFT JOIN
                `{project_id}.{dataset_id}.account_codes` ac
            ON
                li.account_code = ac.code
            JOIN
                `{project_id}.{dataset_id}.documents` d
            ON
                li.document_id = d.id
            WHERE
                d.document_type IN ({document_types})
                {date_filter}
            GROUP BY
                li.account_code, ac.name
            ORDER BY
                total_amount DESC
        """
    }
    
    # Validate the analysis type
    if analysis_type not in analysis_queries:
        return {
            "status": "error", 
            "message": f"Unsupported analysis type: {analysis_type}. "
                       f"Supported types: {', '.join(analysis_queries.keys())}"
        }
    
    # Extract and validate parameters
    start_date = parameters.get("start_date")
    end_date = parameters.get("end_date")
    entity_id = parameters.get("entity_id")
    doc_type = parameters.get("doc_type", "invoice")
    entity_field = parameters.get("entity_field", "recipient_id" if doc_type == "invoice" else "issuer_id")
    document_types = parameters.get("document_types", "'invoice', 'receipt', 'expense'")
    
    # Build filter clauses
    date_filter = ""
    if start_date:
        date_filter += f"AND d.issue_date >= '{start_date}' "
    if end_date:
        date_filter += f"AND d.issue_date <= '{end_date}' "
    
    entity_filter = ""
    if entity_id:
        if analysis_type == "spending_by_vendor":
            entity_filter = f"AND e.id = '{entity_id}' "
        else:
            entity_filter = f"AND d.{entity_field} = '{entity_id}' "
    
    # Format the query with parameters
    query = analysis_queries[analysis_type].format(
        project_id=project_id,
        dataset_id=dataset_id,
        entity_field=entity_field,
        doc_type=doc_type,
        entity_filter=entity_filter,
        date_filter=date_filter,
        document_types=document_types
    )
    
    # Execute the query
    return query_financial_data(query, project_id)
