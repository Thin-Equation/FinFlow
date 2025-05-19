"""
Configuration for the StorageAgent.
"""

from typing import Dict, Any

# BigQuery configuration
BIGQUERY_CONFIG = {
    # Replace with your actual project ID in production
    "project_id": "finflow-project",
    "dataset_id": "finflow_financial_data",
    "location": "US",  # BigQuery dataset location
    "tables": {
        "documents": "documents",
        "line_items": "line_items",
        "entities": "entities",
        "document_relationships": "document_relationships",
        "account_codes": "account_codes",
        "financial_summaries": "financial_summaries"
    }
}

# Cache configuration
CACHE_CONFIG = {
    "enable_cache": True,
    "default_ttl_seconds": 300,  # 5 minutes
    "max_cache_size": 100,       # Maximum cache entries
    "cache_distributed": False   # Set to True to use Redis in production
}

# Storage configuration
STORAGE_CONFIG = {
    "batch_size": 100,           # Maximum batch size for batch operations
    "retry_attempts": 3,         # Number of retry attempts for failed operations
    "timeout_seconds": 30,       # Operation timeout in seconds
    "enable_audit_trail": True,  # Whether to create audit trail entries
    "schema_validation": True    # Whether to validate documents against schema
}

def get_storage_config() -> Dict[str, Any]:
    """
    Get the storage configuration.
    
    Returns:
        Dict[str, Any]: Storage configuration
    """
    return {
        "bigquery": BIGQUERY_CONFIG,
        "cache": CACHE_CONFIG,
        "storage": STORAGE_CONFIG
    }
