"""
Configuration settings for the Document Processor Agent.

This module defines configuration settings, constants, and defaults
for the Document Processor Agent.
"""

from typing import Dict, Any

# Document processing constants
DEFAULT_PROCESSOR_LOCATION = "us-central1"
MAX_BATCH_SIZE = 20
MAX_PARALLEL_WORKERS = 5
MAX_RETRY_COUNT = 3
DOCUMENT_CACHE_ENABLED = True
DEFAULT_OPTIMIZATION_LEVEL = "medium"  # low, medium, high
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Processor settings
PROCESSOR_CONFIGS = {
    "invoice": {
        "processor_name": "finflow-invoice-processor",
        "processor_version": "latest",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff"],
        "confidence_threshold": 0.7,
    },
    "receipt": {
        "processor_name": "finflow-receipt-processor",
        "processor_version": "latest",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff"],
        "confidence_threshold": 0.6,
    },
    "general": {
        "processor_name": "finflow-document-processor",
        "processor_version": "latest",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff", "text/plain"],
        "confidence_threshold": 0.5,
    },
}

# Document classification settings
CLASSIFICATION = {
    "enabled": True,
    "confidence_threshold": 0.6,
    "auto_route": True,  # Automatically route to optimal processor
    "use_enhanced_classifier": True,
}

# Supported document types and their priority order
DOCUMENT_TYPE_PRIORITY = [
    "invoice", 
    "receipt", 
    "bank_statement",
    "tax_document",
    "contract",
    "unknown"
]

# File validation
FILE_VALIDATION = {
    "enforce_strict_validation": False,  # Reject invalid files if True
    "max_file_size_mb": 20,
    "min_image_dimension": 300,
    "valid_extensions": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".txt"],
}

# Storage settings
STORAGE = {
    "raw_documents_path": "sample_data/raw",
    "processed_documents_path": "sample_data/processed",
    "rejected_documents_path": "sample_data/rejected",
    "cache_path": "sample_data/cache",
    "keep_raw_files": True,
}

# Error handling
ERROR_HANDLING = {
    "auto_retry": True,
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "log_detailed_errors": True,
    "fallback_to_default_processor": True,
}

# Telemetry and metrics
TELEMETRY = {
    "collect_metrics": True,
    "track_processing_time": True,
    "track_confidence_scores": True,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
}


def get_processor_config(document_type: str, environment: str = "development") -> Dict[str, Any]:
    """
    Get processor configuration for document type and environment.
    
    Args:
        document_type: Type of document (invoice, receipt, etc.)
        environment: Environment (development, staging, production)
        
    Returns:
        Dict with processor configuration
    """
    # Get base processor config
    config = PROCESSOR_CONFIGS.get(document_type, PROCESSOR_CONFIGS["general"]).copy()
    
    # Override location based on environment
    if environment == "production":
        config["location"] = "us-central1"  # Use production location
    elif environment == "staging":
        config["location"] = "us-central1"  # Use staging location
    else:
        config["location"] = DEFAULT_PROCESSOR_LOCATION
    
    return config


def get_processor_id(project_id: str, document_type: str, environment: str = "development") -> str:
    """
    Get full processor ID for document type and environment.
    
    Args:
        project_id: Google Cloud project ID
        document_type: Type of document (invoice, receipt, etc.)
        environment: Environment (development, staging, production)
        
    Returns:
        Full processor ID
    """
    config = get_processor_config(document_type, environment)
    location = config.get("location", DEFAULT_PROCESSOR_LOCATION)
    processor_name = config.get("processor_name")
    
    return f"projects/{project_id}/locations/{location}/processors/{processor_name}"