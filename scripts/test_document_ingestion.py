#!/usr/bin/env python
"""
Test script for document ingestion tool.
"""

import os
import sys
# Add parent directory to path so we can import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.document_ingestion import (
    validate_document,
    preprocess_document,
    upload_document,
    batch_upload_documents
)

def test_document_ingestion():
    """Test document ingestion with the provided sample files."""
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Sample files directory
    sample_dir = os.path.join(base_dir, "sample_data", "invoices", "test")
    
    # Check if sample directory exists
    if not os.path.exists(sample_dir):
        print(f"Error: Sample directory not found: {sample_dir}")
        sys.exit(1)
    
    # Get all files in the sample directory
    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))]
    
    if not files:
        print(f"Error: No sample files found in: {sample_dir}")
        sys.exit(1)
    
    print(f"Found {len(files)} sample files for testing")
    
    # Test validation
    print("\n=== Testing Document Validation ===")
    for file_path in files:
        result = validate_document(file_path)
        print(f"Validating {os.path.basename(file_path)}: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error: {result['message']}")
    
    # Test preprocessing
    print("\n=== Testing Document Preprocessing ===")
    for file_path in files:
        result = preprocess_document(file_path)
        print(f"Preprocessing {os.path.basename(file_path)}: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error: {result['message']}")
        else:
            print(f"  Processed file: {os.path.basename(result['processed_file_path'])}")
    
    # Test upload
    print("\n=== Testing Document Upload ===")
    destination_folder = os.path.join(base_dir, "sample_data", "invoices", "uploaded")
    for file_path in files:
        result = upload_document(file_path, destination_folder)
        print(f"Uploading {os.path.basename(file_path)}: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error: {result['message']}")
        else:
            print(f"  Uploaded to: {os.path.basename(result['file_path'])}")
    
    # Test batch upload
    print("\n=== Testing Batch Document Upload ===")
    batch_result = batch_upload_documents(files, destination_folder)
    print(f"Batch upload status: {batch_result['status']}")
    print(f"Message: {batch_result['message']}")
    
    print("\nDocument ingestion testing completed!")

if __name__ == "__main__":
    try:
        test_document_ingestion()
    except Exception as e:
        import traceback
        print(f"Error running test: {e}")
        traceback.print_exc()
