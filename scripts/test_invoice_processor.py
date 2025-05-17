"""
Test script for Document AI invoice processor.

This script tests both the direct Document AI invoice processor API
and the DocumentProcessorAgent integration with Document AI.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DocumentProcessorAgent
from agents.document_processor import DocumentProcessorAgent

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_invoice_processor_api(project_id: str, test_file: str) -> Dict[str, Any]:
    """
    Test the invoice processor API directly with a sample invoice file.
    
    Args:
        project_id: Google Cloud project ID
        test_file: Path to sample invoice file for testing
        
    Returns:
        Processing result
    """
    from tools.document_ai import test_invoice_processor
    
    logger.info(f"Testing direct Document AI invoice processor API with file: {test_file}")
    
    # Construct processor ID
    processor_id = f"projects/{project_id}/locations/us-central1/processors/finflow-invoice-processor"
    
    # Test the processor
    result = test_invoice_processor(processor_id, test_file)
    
    # Display results
    if result.get("status") == "success":
        logger.info("✅ Document AI invoice processor test successful")
        logger.info(f"Processor ID: {result['processor_id']}")
        
        invoice_data = result.get("invoice_data", {})
        logger.info("\nExtracted Invoice Data (Direct API):")
        logger.info(f"Invoice Number: {invoice_data.get('invoice_number')}")
        logger.info(f"Issue Date: {invoice_data.get('issue_date')}")
        logger.info(f"Due Date: {invoice_data.get('due_date')}")
        logger.info(f"Vendor: {invoice_data.get('vendor', {}).get('name')}")
        logger.info(f"Total Amount: {invoice_data.get('total_amount')}")
        logger.info(f"Confidence Score: {invoice_data.get('confidence_score')}")
    else:
        logger.error(f"❌ Document AI invoice processor test failed: {result.get('message')}")
        
    return result

def test_document_processor_agent(test_file: str) -> Dict[str, Any]:
    """
    Test the DocumentProcessorAgent with a sample invoice file.
    
    Args:
        test_file: Path to sample invoice file for testing
        
    Returns:
        Processing result
    """
    logger.info(f"Testing DocumentProcessorAgent with file: {test_file}")
    
    # Initialize agent
    agent = DocumentProcessorAgent()
    agent.register_tools()
    
    # Create context
    context = {
        "document_path": test_file,
        "document_type": "invoice",
        "session_id": "test_session",
        "user_id": "test_user"
    }
    
    # Process document
    result = agent.process_document(context)
    
    # Display results
    if result.get("status") == "success":
        logger.info("✅ DocumentProcessorAgent test successful")
        
        extracted_data = result.get("extracted_data", {})
        logger.info("\nExtracted Invoice Data (Agent):")
        logger.info(f"Invoice Number: {extracted_data.get('invoice_number')}")
        logger.info(f"Date: {extracted_data.get('date')}")
        logger.info(f"Due Date: {extracted_data.get('due_date')}")
        logger.info(f"Vendor: {extracted_data.get('vendor', {}).get('name')}")
        logger.info(f"Total Amount: {extracted_data.get('total_amount')}")
        
        if "metadata" in extracted_data and "confidence" in extracted_data["metadata"]:
            logger.info(f"Confidence Score: {extracted_data['metadata']['confidence']}")
        
        # Count line items
        line_items = extracted_data.get("line_items", [])
        if line_items:
            logger.info(f"Line Items: {len(line_items)}")
    else:
        logger.error(f"❌ DocumentProcessorAgent test failed: {result.get('error_message', 'Unknown error')}")
    
    return result

def generate_sample_invoice() -> str:
    """
    Generate a sample invoice for testing.
    
    Returns:
        str: Path to the generated invoice
    """
    import subprocess
    
    # Create output directory for test invoice
    test_dir = "./sample_data/invoices/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate one sample invoice
    test_file = os.path.join(test_dir, "sample_invoice.pdf")
    
    # Run the generate_sample_invoices.py script
    cmd = ["python", "./tools/generate_sample_invoices.py", "-n", "1", "-o", test_dir]
    try:
        logger.info("Generating sample invoice for testing...")
        subprocess.run(cmd, check=True)
        logger.info(f"Sample invoice generated at {test_file}")
        return test_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate sample invoice: {e}")
        raise

def main():
    """Run the invoice processor test."""
    parser = argparse.ArgumentParser(description="Test Document AI invoice processor")
    parser.add_argument("--project-id", type=str, required=True, help="Google Cloud project ID")
    parser.add_argument("--test-file", type=str, help="Path to sample invoice file for testing")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample invoice for testing")
    
    args = parser.parse_args()
    
    # Generate sample invoice if requested or if no test file provided
    if args.generate_sample or not args.test_file:
        test_file = generate_sample_invoice()
    else:
        test_file = args.test_file
    
    # Test the invoice processor
    test_invoice_processor(args.project_id, test_file)

if __name__ == "__main__":
    main()
