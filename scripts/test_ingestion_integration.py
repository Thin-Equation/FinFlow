#!/usr/bin/env python
"""
Basic test of the document ingestion tool functionality.

This script directly tests the document ingestion tools without
requiring the full agent initialization.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_integration():
    """Test the integration of document ingestion with document processor tools."""
    
    print("Testing document ingestion integration...")
    
    # Import tools
    from tools.document_ingestion import (
        validate_document,
        preprocess_document,
        upload_document,
        batch_upload_documents
    )
    
    # Get sample directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "sample_data", "invoices", "test")
    
    # Get a sample file
    sample_files = [
        os.path.join(sample_dir, f) 
        for f in os.listdir(sample_dir) 
        if os.path.isfile(os.path.join(sample_dir, f))
    ]
    
    if not sample_files:
        print("No sample files found!")
        return
        
    sample_file = sample_files[0]
    print(f"Using sample file: {os.path.basename(sample_file)}")
    
    # Simulate processor flow
    print("\n1. Validating document...")
    validation_result = validate_document(sample_file)
    print(f"   Validation result: {validation_result['status']}")
    
    if validation_result['valid']:
        print("\n2. Preprocessing document...")
        preprocess_result = preprocess_document(sample_file)
        print(f"   Preprocessing result: {preprocess_result['status']}")
        
        if preprocess_result['status'] == 'success':
            processed_file = preprocess_result['processed_file_path']
            print(f"   Processed file: {os.path.basename(processed_file)}")
            
            # Now simulate document AI processing
            print("\n3. Simulating Document AI processing...")
            print("   Extracting fields from document...")
            
            # Create mock document AI result
            ai_result = {
                "status": "success",
                "text": "This is a sample invoice",
                "entities": {
                    "invoice_number": f"INV-{datetime.now().strftime('%Y%m%d')}",
                    "total_amount": "$1,234.56",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "vendor": "Sample Corp"
                }
            }
            print(f"   Extracted fields: {list(ai_result['entities'].keys())}")
            
            print("\n4. Integration test completed successfully!")
        else:
            print(f"   Error preprocessing: {preprocess_result['message']}")
    else:
        print(f"   Error validating: {validation_result['message']}")

if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        import traceback
        print(f"Error during integration test: {e}")
        traceback.print_exc()
