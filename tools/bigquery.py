"""
BigQuery integration tools for the FinFlow system.
"""

from typing import Any, Dict, List, Optional
from google.adk.tools import ToolContext, BaseTool
from google.cloud import bigquery

def store_document(document_data: Dict[str, Any], table_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Store processed document data in BigQuery.
    
    Args:
        document_data: Structured document data
        table_id: BigQuery table ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client()
    
    # Prepare the data for insertion
    rows_to_insert = [document_data]
    
    # Insert data
    errors = client.insert_rows_json(table_id, rows_to_insert)
    
    if errors:
        return {"status": "error", "errors": errors}
    else:
        return {"status": "success", "message": "Document stored successfully"}

def query_financial_data(query: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Execute a query against financial data in BigQuery.
    
    Args:
        query: SQL query to execute
        tool_context: Provided by ADK
        
    Returns:
        dict: Query results
    """
    client = bigquery.Client()
    
    # Execute the query
    query_job = client.query(query)
    results = query_job.result()
    
    # Convert results to a list of dictionaries
    rows = []
    for row in results:
        rows.append(dict(row.items()))
    
    return {
        "status": "success",
        "row_count": len(rows),
        "data": rows
    }

def create_financial_tables(dataset_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Create required tables for FinFlow in BigQuery.
    
    Args:
        dataset_id: BigQuery dataset ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Result of the operation
    """
    client = bigquery.Client()
    
    # Define table schemas
    tables = {
        "documents": [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("document_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("metadata", "JSON"),
            bigquery.SchemaField("content", "JSON"),
        ],
        "entities": [
            bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("entity_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("metadata", "JSON"),
        ],
    }
    
    results = {}
    
    # Create each table
    for table_name, schema in tables.items():
        table_id = f"{dataset_id}.{table_name}"
        table = bigquery.Table(table_id, schema=schema)
        try:
            created_table = client.create_table(table, exists_ok=True)
            results[table_name] = "created"
        except Exception as e:
            results[table_name] = f"error: {str(e)}"
    
    return {
        "status": "success",
        "tables": results
    }
