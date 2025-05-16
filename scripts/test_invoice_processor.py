"""
Test script for Document AI invoice processor.
"""

import os
import argparse
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_invoice_processor(project_id: str, test_file: str) -> None:
    """
    Test the invoice processor with a sample invoice file.
    
    Args:
        project_id: Google Cloud project ID
        test_file: Path to sample invoice file for testing
    """
    from tools.document_ai import test_invoice_processor
    
    logger.info(f"Testing invoice processor with file: {test_file}")
    
    # Construct processor ID
    processor_id = f"projects/{project_id}/locations/us-central1/processors/finflow-invoice-processor"
    
    # Test the processor
    result = test_invoice_processor(processor_id, test_file)
    
    # Display results
    if result.get("status") == "success":
        logger.info("✅ Invoice processor test successful")
        logger.info(f"Processor ID: {result['processor_id']}")
        
        invoice_data = result.get("invoice_data", {})
        logger.info("\nExtracted Invoice Data:")
        logger.info(f"Invoice Number: {invoice_data.get('invoice_number')}")
        logger.info(f"Issue Date: {invoice_data.get('issue_date')}")
        logger.info(f"Due Date: {invoice_data.get('due_date')}")
        logger.info(f"Vendor: {invoice_data.get('vendor', {}).get('name')}")
        logger.info(f"Total Amount: {invoice_data.get('total_amount')}")
        logger.info(f"Confidence Score: {invoice_data.get('confidence_score')}")
    else:
        logger.error(f"❌ Invoice processor test failed: {result.get('message')}")

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
