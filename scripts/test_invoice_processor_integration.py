#!/usr/bin/env python
"""
Invoice Processor Integration Test script.

Tests the DocumentProcessorAgent's ability to process invoices as part of Day 10 implementation.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DocumentProcessorAgent
from agents.document_processor import DocumentProcessorAgent

def main():
    """Main function to test the document processing agent with invoices."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test DocumentProcessorAgent with invoices")
    parser.add_argument("--invoice", help="Path to invoice file to process")
    parser.add_argument("--dir", help="Directory containing invoices to process")
    parser.add_argument("--output", help="Output JSON file for results", default="invoice_processor_results.json")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("invoice_processor_test")
    
    # Initialize DocumentProcessorAgent
    logger.info("Initializing DocumentProcessorAgent...")
    agent = DocumentProcessorAgent()
    agent.register_tools()
    logger.info("Agent initialized with tools")
    
    results = []
    
    # Process a single invoice
    if args.invoice:
        if not os.path.isfile(args.invoice):
            logger.error(f"File not found: {args.invoice}")
            return
            
        logger.info(f"Processing invoice: {args.invoice}")
        
        # Process the invoice
        context = {
            "document_path": args.invoice,
            "document_type": "invoice",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        result = agent.process_document(context)
        
        # Display results
        if result.get("status") == "success":
            extracted = result.get("extracted_data", {})
            logger.info("✓ Document processed successfully")
            logger.info(f"  Invoice Number: {extracted.get('invoice_number', 'N/A')}")
            logger.info(f"  Date: {extracted.get('date', 'N/A')}")
            logger.info(f"  Vendor: {extracted.get('vendor', {}).get('name', 'N/A')}")
            logger.info(f"  Total Amount: {extracted.get('total_amount', 'N/A')}")
            
            # Add to results
            results.append({
                "filename": os.path.basename(args.invoice),
                "status": "success",
                "data": extracted
            })
        else:
            logger.error(f"✗ Error processing document: {result.get('error_message', 'Unknown error')}")
            results.append({
                "filename": os.path.basename(args.invoice),
                "status": "error",
                "error": result.get('error_message', 'Unknown error')
            })
    
    # Process all invoices in a directory
    elif args.dir:
        if not os.path.isdir(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            return
            
        # Find all potential invoice files
        invoice_files = []
        for filename in os.listdir(args.dir):
            if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff')):
                invoice_files.append(os.path.join(args.dir, filename))
        
        if not invoice_files:
            logger.error(f"No invoice files found in {args.dir}")
            return
            
        logger.info(f"Found {len(invoice_files)} potential invoice files")
        
        # Process each invoice
        for i, invoice_file in enumerate(invoice_files):
            logger.info(f"Processing invoice {i+1}/{len(invoice_files)}: {os.path.basename(invoice_file)}")
            
            # Process the invoice
            context = {
                "document_path": invoice_file,
                "document_type": "invoice",
                "session_id": "test_session",
                "user_id": "test_user"
            }
            
            try:
                result = agent.process_document(context)
                
                if result.get("status") == "success":
                    extracted = result.get("extracted_data", {})
                    logger.info("✓ Document processed successfully")
                    logger.info(f"  Invoice Number: {extracted.get('invoice_number', 'N/A')}")
                    logger.info(f"  Date: {extracted.get('date', 'N/A')}")
                    logger.info(f"  Vendor: {extracted.get('vendor', {}).get('name', 'N/A')}")
                    logger.info(f"  Total Amount: {extracted.get('total_amount', 'N/A')}")
                    
                    # Add to results
                    results.append({
                        "filename": os.path.basename(invoice_file),
                        "status": "success",
                        "data": extracted
                    })
                else:
                    logger.error(f"✗ Error processing document: {result.get('error_message', 'Unknown error')}")
                    results.append({
                        "filename": os.path.basename(invoice_file),
                        "status": "error",
                        "error": result.get('error_message', 'Unknown error')
                    })
            except Exception as e:
                logger.error(f"✗ Exception while processing {os.path.basename(invoice_file)}: {str(e)}")
                results.append({
                    "filename": os.path.basename(invoice_file),
                    "status": "error",
                    "error": str(e)
                })
    
    else:
        logger.error("No input provided. Use --invoice or --dir to specify input.")
        parser.print_help()
        return
    
    # Write results to output file
    if results:
        output_data = {
            "timestamp": str(import_datetime().datetime.now()),
            "total_files": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "results": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
            
        logger.info(f"Results saved to {args.output}")
        logger.info(f"Processed {len(results)} files, {output_data['successful']} successful")

def import_datetime():
    """Import datetime module and return it."""
    import datetime
    return datetime

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
