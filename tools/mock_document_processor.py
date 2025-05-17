"""
Mock implementations for document processor testing.

This module provides mock implementations of the Document AI and ingestion
tools to allow testing without actual API calls.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import random
import uuid

# Patch external modules to avoid real API calls
def patch_modules():
    """Patch external modules to use mock implementations."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock modules if they don't exist
    mock_modules = [
        'google.cloud.documentai_v1',
        'google.cloud.storage',
        'google.api_core.exceptions',
        'google.adk.tools',
    ]
    
    for module_name in mock_modules:
        # Check if module exists
        if module_name not in sys.modules:
            parts = module_name.split('.')
            current_module = sys.modules.setdefault(parts[0], MagicMock())
            
            # Build nested modules
            for i in range(1, len(parts)):
                parent_name = '.'.join(parts[:i])
                child_name = parts[i]
                parent = sys.modules[parent_name]
                
                if not hasattr(parent, child_name):
                    child_module = MagicMock()
                    setattr(parent, child_name, child_module)
                    sys.modules[f"{parent_name}.{child_name}"] = child_module
    
    # Patch specific classes and exceptions
    from google.cloud import documentai_v1, storage
    from google.api_core import exceptions
    
    # Create DocumentAI exceptions
    exceptions.GoogleAPIError = type('GoogleAPIError', (Exception,), {})
    exceptions.RetryError = type('RetryError', (exceptions.GoogleAPIError,), {})
    exceptions.ResourceExhausted = type('ResourceExhausted', (exceptions.GoogleAPIError,), {})
    
    # Log the patching
    logging.info("Mock modules patched successfully")

class MockToolContext:
    """Mock implementation of the ToolContext class."""
    
    def __init__(self, processor_id: str, user: str, session: str):
        """Initialize the mock tool context."""
        self.processor_id = processor_id
        self.user = user
        self.session = session
        self.properties = {
            "processor_id": processor_id,
            "user": user,
            "session": session
        }
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property from the context."""
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property in the context."""
        self.properties[key] = value

def generate_mock_invoice_data() -> Dict[str, Any]:
    """Generate mock invoice data for testing."""
    vendors = ["Acme Corp.", "TechSupplies Inc.", "Office Solutions Ltd.", 
               "Global Services Co.", "Innovate Systems"]
    
    # Generate random invoice data
    invoice_number = f"INV-{random.randint(10000, 99999)}"
    invoice_date = datetime.now().strftime("%Y-%m-%d")
    due_date = datetime.now().strftime("%Y-%m-%d")
    total_amount = round(random.uniform(100, 5000), 2)
    tax_amount = round(total_amount * 0.1, 2)
    subtotal = round(total_amount - tax_amount, 2)
    
    # Generate line items
    line_items = []
    for i in range(1, random.randint(2, 5)):
        line_items.append({
            "description": f"Item {i}",
            "quantity": random.randint(1, 10),
            "unit_price": round(random.uniform(10, 500), 2),
            "amount": round(random.uniform(10, 500) * random.randint(1, 10), 2)
        })
    
    # Create invoice data
    return {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "vendor": random.choice(vendors),
        "buyer": "Sample Company",
        "subtotal": subtotal,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "currency": "USD",
        "line_items": line_items
    }

def process_mock_document(file_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """Process a document with mock implementation."""
    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error_message": f"File not found: {file_path}",
            "error_type": "file_not_found"
        }
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
        return {
            "status": "error",
            "error_message": f"Unsupported file type: {ext}",
            "error_type": "invalid_file_type"
        }
    
    # Add a small delay to simulate processing time
    time.sleep(random.uniform(0.2, 1.0))
    
    # Generate mock extraction results
    if not document_type:
        document_type = "invoice" if random.random() < 0.7 else "receipt"
    
    confidence = round(random.uniform(0.65, 0.95), 2)
    
    # Generate appropriate data based on document type
    if document_type == "invoice":
        extracted_data = generate_mock_invoice_data()
    else:
        # Generic receipt data
        extracted_data = {
            "merchant": random.choice(["Retail Store", "Restaurant", "Office Store", "Electronics Shop"]),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_amount": round(random.uniform(10, 200), 2),
            "payment_method": random.choice(["Credit Card", "Cash", "Debit Card"])
        }
    
    # Return mock result
    return {
        "status": "success",
        "document_type": document_type,
        "confidence": confidence,
        "extraction_time": round(random.uniform(0.5, 2.0), 2),
        "extracted_data": extracted_data,
        "metadata": {
            "file_name": os.path.basename(file_path),
            "process_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
    }