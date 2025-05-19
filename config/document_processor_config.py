"""
Configuration settings for the Document Processor Agent.

This module defines configuration settings, constants, and defaults
for the Document Processor Agent.
"""

from typing import Dict, Any
import os
from functools import lru_cache

# Document processing constants
DEFAULT_PROCESSOR_LOCATION = "us"
MAX_BATCH_SIZE = 50
MAX_PARALLEL_WORKERS = 10
MAX_RETRY_COUNT = 3
DOCUMENT_CACHE_ENABLED = True
DEFAULT_OPTIMIZATION_LEVEL = "high"  # low, medium, high
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Processor settings
PROCESSOR_CONFIGS = {
    "invoice": {
        "processor_name": "finflow-invoice-processor",
        "processor_version": "stable",
        "processor_type": "FORM_PARSER_PROCESSOR",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff", "image/webp"],
        "confidence_threshold": 0.75,
        "fields_of_interest": [
            "invoice_id", "invoice_date", "due_date", "total_amount", "tax_amount",
            "vendor_name", "vendor_address", "vendor_phone", "vendor_email", "vendor_website",
            "customer_name", "customer_address", "customer_id", "line_items", "payment_terms",
            "currency", "purchase_order", "shipping_amount", "discount_amount"
        ],
        "normalization": {
            "date_format": "YYYY-MM-DD",
            "currency_format": "standard",
            "amount_precision": 2,
            "extract_line_items": True
        },
        "auto_correction": True,
        "processor_timeout_seconds": 120
    },
    "receipt": {
        "processor_name": "finflow-receipt-processor",
        "processor_version": "stable",
        "processor_type": "FORM_PARSER_PROCESSOR",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff", "image/webp"],
        "confidence_threshold": 0.7,
        "fields_of_interest": [
            "merchant_name", "transaction_date", "receipt_time", "total_amount", 
            "tax_amount", "tip_amount", "payment_method", "currency", "line_items",
            "receipt_number", "store_address", "store_phone"
        ],
        "normalization": {
            "date_format": "YYYY-MM-DD",
            "time_format": "24H",
            "currency_format": "standard",
            "amount_precision": 2,
            "extract_line_items": True
        },
        "auto_correction": True,
        "processor_timeout_seconds": 90
    },
    "bank_statement": {
        "processor_name": "finflow-bank-statement-processor",
        "processor_version": "stable",
        "processor_type": "FORM_PARSER_PROCESSOR",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff"],
        "confidence_threshold": 0.8,
        "fields_of_interest": [
            "account_number", "account_holder", "statement_date", "period_start", "period_end",
            "opening_balance", "closing_balance", "total_deposits", "total_withdrawals", 
            "transactions", "bank_name", "bank_address", "bank_identifier"
        ],
        "normalization": {
            "date_format": "YYYY-MM-DD",
            "currency_format": "standard",
            "amount_precision": 2,
            "extract_transactions": True
        },
        "auto_correction": True,
        "processor_timeout_seconds": 180
    },
    "general": {
        "processor_name": "feb3031ec34b67f4",
        "processor_version": "stable",
        "processor_type": "DOCUMENT_PROCESSOR",
        "mime_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff", "image/webp", "text/plain"],
        "confidence_threshold": 0.6,
        "extract_tables": True,
        "extract_forms": True,
        "processor_timeout_seconds": 120
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
    "enforce_strict_validation": True,  # Reject invalid files if True
    "max_file_size_mb": 20,
    "min_image_dimension": 300,
    "min_dpi": 200,
    "valid_extensions": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".txt", ".doc", ".docx"],
    "scan_for_malware": True,
    "pdf_validation": {
        "check_corrupt": True,
        "min_text_length": 10,  # Minimum characters to consider a PDF valid for processing
        "enforce_text_extractability": True,
        "check_password_protection": True
    },
    "image_validation": {
        "min_quality_score": 30,  # 0-100 score
        "check_blurriness": True,
        "max_compression": 90,  # JPEG quality 0-100
        "preferred_color_mode": "rgb"  # rgb, grayscale
    }
}

# Storage settings
STORAGE = {
    "raw_documents_path": "sample_data/raw",
    "processed_documents_path": "sample_data/processed",
    "rejected_documents_path": "sample_data/rejected",
    "cache_path": "sample_data/cache",
    "archive_path": "sample_data/archive",
    "keep_raw_files": True,
    "storage_structure": "date_based",  # date_based, type_based, flat
    "retention_policy": {
        "raw_days": 90,
        "processed_days": 365,
        "rejected_days": 30,
        "cache_days": 7
    },
    "compression": {
        "enabled": True,
        "format": "zip",
        "threshold_days": 30  # Compress files older than this
    },
    "backup": {
        "enabled": True,
        "frequency": "daily",
        "remote_storage": False
    }
}

# Error handling
ERROR_HANDLING = {
    "auto_retry": True,
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "exponential_backoff": True,
    "log_detailed_errors": True,
    "fallback_to_default_processor": True,
    "error_notification": {
        "enabled": True,
        "threshold_percent": 10  # Alert if error rate exceeds this percentage
    },
    "circuit_breaker": {
        "enabled": True,
        "failure_threshold": 5,
        "reset_timeout_seconds": 300
    }
}

# Telemetry and metrics
TELEMETRY = {
    "collect_metrics": True,
    "track_processing_time": True,
    "track_confidence_scores": True,
    "track_error_rates": True,
    "track_throughput": True,
    "track_resource_usage": True,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "prometheus_export": False,
    "periodic_reporting": {
        "enabled": True,
        "interval_minutes": 60
    }
}

# Performance optimization
PERFORMANCE = {
    "cache_enabled": True,
    "cache_ttl_hours": 24,
    "preprocess_images": True,  # Apply image optimizations before processing
    "batch_optimize": True,  # Group documents for batch processing
    "parallel_processing": True,
    "max_concurrent_batches": 3,
    "resource_limits": {
        "max_memory_mb": 2048,
        "max_cpu_percent": 80
    }
}

@lru_cache(maxsize=32)
def get_processor_config(document_type: str, environment: str = "development") -> Dict[str, Any]:
    """
    Get processor configuration for document type and environment.
    
    Args:
        document_type: Type of document (invoice, receipt, etc.)
        environment: Environment (development, staging, production)
        
    Returns:
        Dict containing processor configuration
    """
    # Default to "general" processor if specified type not found
    processor_config = PROCESSOR_CONFIGS.get(document_type, PROCESSOR_CONFIGS["general"]).copy()
    
    # Add environment-specific settings
    if environment == "production":
        # Increase thresholds and enforce stricter validation in production
        processor_config["confidence_threshold"] += 0.05
        processor_config["processor_version"] = "stable"  # Use stable in production
    elif environment == "development":
        # Lower thresholds for development
        processor_config["confidence_threshold"] = max(0.5, processor_config["confidence_threshold"] - 0.1)
        processor_config["processor_version"] = "latest"  # Use latest in development
    
    return processor_config

def get_processor_id(document_type: str, project_id: str, environment: str = "development") -> str:
    """
    Get fully qualified processor ID for the specified document type.
    
    Args:
        document_type: Type of document (invoice, receipt, etc.)
        project_id: Google Cloud project ID
        environment: Environment (development, staging, production)
        
    Returns:
        Fully qualified processor ID
    """
    config = get_processor_config(document_type, environment)
    location = DEFAULT_PROCESSOR_LOCATION
    processor_name = config["processor_name"]
    
    # Format the processor ID - Note: We're using just the processor name without version
    # because the Document AI API may be using a different format than expected
    return f"projects/{project_id}/locations/{location}/processors/{processor_name}"

def get_validation_settings(environment: str = "development") -> Dict[str, Any]:
    """
    Get validation settings based on environment.
    
    Args:
        environment: Environment (development, staging, production)
        
    Returns:
        Dict containing validation settings
    """
    validation_settings = FILE_VALIDATION.copy()
    
    # Adjust settings based on environment
    if environment == "production":
        validation_settings["enforce_strict_validation"] = True
        validation_settings["scan_for_malware"] = True
    elif environment == "development":
        validation_settings["enforce_strict_validation"] = False
        validation_settings["scan_for_malware"] = False
    
    return validation_settings

def get_storage_path(document_type: str, status: str = "raw", environment: str = "development") -> str:
    """
    Get storage path for documents based on type and status.
    
    Args:
        document_type: Type of document (invoice, receipt, etc.)
        status: Document status (raw, processed, rejected, cache)
        environment: Environment (development, staging, production)
        
    Returns:
        Storage path for the document
    """
    base_path = STORAGE.get(f"{status}_documents_path", "sample_data/raw")
    
    # Use environment as subfolder to separate data
    if environment != "development":
        base_path = os.path.join(base_path, environment)
    
    # Create structure based on document type if using type_based structure
    if STORAGE.get("storage_structure") == "type_based":
        return os.path.join(base_path, document_type)
    
    # Create structure based on date if using date_based structure
    if STORAGE.get("storage_structure") == "date_based":
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(base_path, date_str, document_type)
    
    return base_path
