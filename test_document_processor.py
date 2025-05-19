#!/usr/bin/env python3
"""
Test script for the DocumentProcessorAgent.
This script processes a sample invoice and displays the results.
"""

import os
from datetime import datetime
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finflow.test")

# Import the DocumentProcessorAgent
from agents.document_processor import DocumentProcessorAgent
from utils.logging_config import TraceContext
from config.config_loader import load_config

def process_sample_document(document_path: str) -> Dict[str, Any]:
    """Process a single document and return the results."""
    # Initialize document processor agent
    config = load_config()
    document_processor = DocumentProcessorAgent(config)
    
    # Create processing context
    context = {
        "document_path": document_path,
        "processing_id": f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "processing_start_time": datetime.now().isoformat()
    }
    
    # Process the document with classification for best results
    try:
        with TraceContext() as trace:
            result = document_processor.process_document_with_classification(context)
        return result
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"status": "error", "error": str(e), "document_path": document_path}

def batch_process_documents(document_paths: list) -> Dict[str, Any]:
    """Process multiple documents in batch and return results."""
    # Initialize document processor agent
    config = load_config()
    document_processor = DocumentProcessorAgent(config)
    
    # Create processing context
    context = {
        "document_paths": document_paths,
        "batch_id": f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "processing_start_time": datetime.now().isoformat()
    }
    
    # Process documents in batch
    try:
        with TraceContext() as trace:
            result = document_processor.batch_process_documents(context)
        return result
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return {"status": "error", "error": str(e), "document_count": len(document_paths)}

def print_result(result: Dict[str, Any]) -> None:
    """Pretty print the processing result."""
    print("\n" + "="*80)
    print(f"Document Processing Result:")
    print("="*80)
    
    # Print status and document type
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Document type: {result.get('document_type', 'unknown')}")
    
    # Print classification info if available
    classification = result.get("classification", {})
    if classification:
        confidence = classification.get("confidence", "N/A")
        print(f"Classification confidence: {confidence}")
    
    # Print extracted data if available
    extracted_data = result.get("extracted_data", {})
    if extracted_data:
        print("\nExtracted Data:")
        print("-" * 40)
        
        # Print common fields
        for field in ["invoice_number", "date", "due_date", "total_amount"]:
            if field in extracted_data:
                print(f"{field.replace('_', ' ').title()}: {extracted_data[field]}")
        
        # Print vendor info
        vendor = extracted_data.get("vendor", {})
        if isinstance(vendor, dict):
            print("\nVendor Information:")
            for key, value in vendor.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"\nVendor: {vendor}")
        
        # Print line items if available
        line_items = extracted_data.get("line_items", [])
        if line_items:
            print("\nLine Items:")
            for i, item in enumerate(line_items):
                print(f"  Item {i+1}:")
                for key, value in item.items():
                    print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Print processing time if available
    if "processing_time" in result:
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
    
    print("="*80)

def main():
    """Main function to test the document processor."""
    sample_dir = os.path.join(os.getcwd(), "sample_data", "invoices")
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                   if os.path.isfile(os.path.join(sample_dir, f)) and f.endswith(".pdf")]
    
    if not sample_files:
        logger.error("No sample invoice files found in the invoices directory!")
        return
    
    # Test single document processing
    print(f"\nTesting single document processing with: {os.path.basename(sample_files[0])}")
    result = process_sample_document(sample_files[0])
    print_result(result)
    
    # Test batch document processing if multiple files exist
    if len(sample_files) >= 2:
        print(f"\nTesting batch document processing with {len(sample_files)} documents")
        batch_result = batch_process_documents(sample_files)
        
        print("\n" + "="*80)
        print(f"Batch Processing Result:")
        print("="*80)
        print(f"Status: {batch_result.get('status', 'unknown')}")
        print(f"Total documents: {len(sample_files)}")
        print(f"Successful: {batch_result.get('successful_count', 0)}")
        print(f"Failed: {batch_result.get('failed_count', 0)}")
        
        # Print individual document results
        individual_results = batch_result.get("results", [])
        for i, doc_result in enumerate(individual_results):
            print(f"\nDocument {i+1}: {os.path.basename(doc_result.get('document_path', 'unknown'))}")
            print(f"Status: {doc_result.get('status', 'unknown')}")
            print(f"Document Type: {doc_result.get('document_type', 'unknown')}")
            
            # Show error if failed
            if doc_result.get("status") == "error":
                print(f"Error: {doc_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
