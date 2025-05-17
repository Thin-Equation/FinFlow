#!/usr/bin/env python
"""
Test script for invoice processing using the DocumentProcessor agent.

This script tests the DocumentProcessor agent's ability to extract 
structured information from invoice documents using Document AI integration.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_processor import DocumentProcessorAgent
from utils.logging_config import configure_logging
from config.config_loader import load_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test invoice processing with DocumentProcessor agent")
    parser.add_argument("--invoice", help="Path to invoice file to process")
    parser.add_argument("--dir", help="Directory containing invoice files to process")
    parser.add_argument("--output", help="Output JSON file for results", default="invoice_results.json")
    parser.add_argument("--log-level", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()

def process_invoice(agent: DocumentProcessorAgent, invoice_path: str) -> Dict[str, Any]:
    """
    Process a single invoice using the document processor agent.
    
    Args:
        agent: Initialized document processor agent
        invoice_path: Path to the invoice file
        
    Returns:
        Processing results
    """
    print(f"Processing invoice: {invoice_path}")
    
    # Create context for processing
    context = {
        "document_path": invoice_path,
        "document_type": "invoice",
        "session_id": "test_session",
        "user_id": "test_user"
    }
    
    try:
        # Process the document
        result = agent.process_document(context)
        
        # Print a summary of the extracted data
        if result.get("status") == "success":
            extracted = result.get("extracted_data", {})
            print(f"✅ Successfully processed invoice {os.path.basename(invoice_path)}")
            print(f"   Invoice Number: {extracted.get('invoice_number', 'N/A')}")
            print(f"   Date: {extracted.get('date', 'N/A')}")
            print(f"   Vendor: {extracted.get('vendor', {}).get('name', 'N/A')}")
            print(f"   Total Amount: {extracted.get('total_amount', 'N/A')}")
            
            # Check for line items
            line_items = extracted.get("line_items", [])
            if line_items:
                print(f"   Line Items: {len(line_items)}")
        else:
            print(f"❌ Failed to process invoice: {result.get('error_message', 'Unknown error')}")
        
        return result
    except Exception as e:
        print(f"❌ Error processing invoice: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": str(e),
            "file": os.path.basename(invoice_path)
        }
    
    return result

def process_directory(agent: DocumentProcessorAgent, directory: str, output_file: str) -> None:
    """
    Process all invoices in a directory.
    
    Args:
        agent: Initialized document processor agent
        directory: Directory containing invoice files
        output_file: Path to save results JSON
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return
    
    results: List[Dict[str, Any]] = []
    
    # Find all PDF, JPG, and PNG files in the directory
    invoice_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            invoice_files.append(os.path.join(directory, filename))
    
    if not invoice_files:
        print(f"No invoice files found in {directory}")
        return
    
    print(f"Found {len(invoice_files)} invoice files to process")
    
    # Process each invoice
    for invoice_path in invoice_files:
        result = process_invoice(agent, invoice_path)
        results.append({
            "file": os.path.basename(invoice_path),
            "result": result
        })
    
    # Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    success_count = sum(1 for r in results if r["result"].get("status") == "success")
    print(f"\nSummary: Successfully processed {success_count} out of {len(results)} invoices")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    
    # Load configuration
    config = load_config()
    
    print("Initializing document processor agent...")
    
    # Create and initialize document processor agent
    agent = DocumentProcessorAgent()
    
    # Register document processing tools
    agent.register_tools()
    
    print("Document processor agent initialized with tools:")
    for tool in agent._tools if hasattr(agent, '_tools') else []:
        print(f"  - {tool.name}: {tool.description}")
    
    # Process invoices
    if args.invoice:
        # Process single invoice
        process_invoice(agent, args.invoice)
    elif args.dir:
        # Process all invoices in directory
        process_directory(agent, args.dir, args.output)
    else:
        print("Error: Please provide either --invoice or --dir argument")
        print("Example: python test_invoice_processing.py --invoice /path/to/invoice.pdf")
        print("Example: python test_invoice_processing.py --dir /path/to/invoices")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error running test: {e}")
        traceback.print_exc()
