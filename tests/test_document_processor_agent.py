#!/usr/bin/env python3
"""
Advanced test script for the enhanced document processor.
This script verifies the full functionality of the document processor
with real Google Document AI integration.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("finflow.test_enhanced")

def setup_environment() -> bool:
    """Set up environment for testing."""
    try:
        # Load configuration
        from config.config_loader import load_config
        config = load_config(environment="development")
        
        # Check if google_cloud configuration exists
        if "google_cloud" not in config:
            logger.error("Missing 'google_cloud' section in configuration.")
            return False
        
        # Get credentials path
        credentials_path = config["google_cloud"].get("credentials_path")
        if not credentials_path or not os.path.exists(credentials_path):
            logger.error(f"Credentials file not found at: {credentials_path}")
            return False
        
        # Set environment variable for Google Cloud authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        logger.info(f"Using credentials from: {credentials_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False

def test_document_classification(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Test document classification functionality with a real document."""
    try:
        from tools.document_classification import classify_document
        
        logger.info(f"Classifying document: {file_path}")
        start_time = time.time()
        
        # Classify document
        result = classify_document(file_path)
        
        classification_time = time.time() - start_time
        logger.info(f"Classification completed in {classification_time:.2f} seconds")
        
        # Print classification results
        print("\n" + "="*80)
        print("Document Classification Result:")
        print("="*80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Document type: {result.get('document_type', 'unknown')}")
        print(f"Document category: {result.get('document_category', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Classification method: {result.get('classification_method', 'N/A')}")
        print(f"Processing time: {classification_time:.2f} seconds")
        
        # Print document metadata
        metadata = result.get("metadata", {})
        if metadata and verbose:
            print("\nDocument Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        return result
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        return {"status": "error", "error": str(e)}

def test_document_processing(file_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """Test document processing with the enhanced processor."""
    try:
        from tools.enhanced_document_ai import DocumentProcessor
        from config.config_loader import load_config
        
        # Load configuration
        config = load_config()
        project_id = config.get("google_cloud", {}).get("project_id")
        
        if not project_id:
            logger.error("Project ID not found in configuration.")
            return {"status": "error", "message": "Missing project ID"}
        
        logger.info(f"Initializing document processor for project: {project_id}")
        
        # Create document processor
        processor = DocumentProcessor(
            project_id=project_id,
            environment="development",
            location="us-central1"
        )
        
        # Process the document
        if not document_type:
            # First classify to get document type
            from tools.document_classification import classify_document
            classification = classify_document(file_path)
            document_type = classification.get("document_type", "general")
            logger.info(f"Auto-detected document type: {document_type}")
        
        logger.info(f"Processing document as type: {document_type}")
        start_time = time.time()
        
        # Use process_single_document (this is the actual method name in the DocumentProcessor class)
        result = processor.process_single_document(
            content=file_path,
            document_type=document_type
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Document processed in {processing_time:.2f} seconds")
        
        # Print processing result
        print("\n" + "="*80)
        print("Document Processing Result:")
        print("="*80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Document type: {result.get('document_type', document_type)}")
        print(f"Processing ID: {result.get('processing_id', 'N/A')}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Print extracted data
        extracted_data = result.get("extracted_data", {})
        if extracted_data:
            print("\nExtracted Data:")
            print(json.dumps(extracted_data, indent=2))
        
        return result
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def test_batch_processing(file_paths: List[str]) -> Dict[str, Any]:
    """Test batch processing with multiple documents."""
    try:
        from tools.enhanced_document_ai import DocumentProcessor
        from config.config_loader import load_config
        
        # Load configuration
        config = load_config()
        project_id = config.get("google_cloud", {}).get("project_id")
        
        if not project_id:
            logger.error("Project ID not found in configuration.")
            return {"status": "error", "message": "Missing project ID"}
        
        if not file_paths:
            logger.error("No files provided for batch processing.")
            return {"status": "error", "message": "No files to process"}
        
        logger.info(f"Initializing document processor for project: {project_id}")
        
        # Create document processor
        processor = DocumentProcessor(
            project_id=project_id,
            environment="development",
            location="us-central1"
        )
        
        # Process documents in batch
        logger.info(f"Batch processing {len(file_paths)} documents...")
        start_time = time.time()
        
        result = processor.batch_process_documents(
            document_paths=file_paths,
            max_workers=3,
            batch_size=min(5, len(file_paths))
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
        
        # Print batch processing result
        print("\n" + "="*80)
        print("Batch Processing Result:")
        print("="*80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Processing ID: {result.get('processing_id', 'N/A')}")
        print(f"Total documents: {len(file_paths)}")
        print(f"Successful: {result.get('success_count', 0)}")
        print(f"Failed: {result.get('failed_count', 0)}")
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        if "avg_processing_time" in result:
            print(f"Average time per document: {result.get('avg_processing_time', 0):.2f} seconds")
        
        # Print individual results
        individual_results = result.get("results", {})
        for file_path, doc_result in individual_results.items():
            print(f"\nDocument: {os.path.basename(file_path)}")
            print(f"  Status: {doc_result.get('status', 'unknown')}")
            
            if doc_result.get("status") == "error":
                print(f"  Error: {doc_result.get('message', 'Unknown error')}")
        
        return result
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the enhanced document processor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--batch', '-b', action='store_true', help='Test batch processing')
    parser.add_argument('--document-type', '-t', type=str, help='Force document type (invoice, receipt, etc.)')
    parser.add_argument('--file', '-f', type=str, help='Specific file to process')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Enhanced Document Processor Test")
    print("="*80)
    
    # Set up environment
    if not setup_environment():
        print("Failed to set up environment. Exiting.")
        return
    
    # Get sample document paths
    sample_dir = os.path.join(os.getcwd(), "sample_data", "invoices")
    all_files = [f for f in os.listdir(sample_dir) 
                if os.path.isfile(os.path.join(sample_dir, f)) and f.endswith(".pdf")]
    
    if not all_files:
        print("\nNo sample files found in the sample_data/invoices directory!")
        return
    
    if args.file:
        # Process specific file
        if os.path.exists(args.file):
            file_path = args.file
        else:
            file_path = os.path.join(sample_dir, args.file)
            if not os.path.exists(file_path):
                print(f"\nFile not found: {args.file}")
                return
        
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # First classify the document
        classification = test_document_classification(file_path, args.verbose)
        
        # Then process it
        document_type = args.document_type or classification.get("document_type")
        test_document_processing(file_path, document_type)
    
    elif args.batch:
        # Batch processing test
        max_batch_size = 3  # Limit to 3 files for testing
        file_paths = [os.path.join(sample_dir, f) for f in all_files[:max_batch_size]]
        
        print(f"\nBatch processing {len(file_paths)} files:")
        for i, file_path in enumerate(file_paths):
            print(f"  {i+1}. {os.path.basename(file_path)}")
        
        test_batch_processing(file_paths)
    
    else:
        # Interactive mode
        print(f"\nFound {len(all_files)} sample files:")
        for i, file in enumerate(all_files):
            print(f"  {i+1}. {file}")
        
        # Select a file to process
        while True:
            try:
                choice = int(input("\nEnter the number of the file to process (or 0 to exit): "))
                if choice == 0:
                    print("Exiting.")
                    return
                elif 1 <= choice <= len(all_files):
                    file_path = os.path.join(sample_dir, all_files[choice-1])
                    print(f"\nSelected: {all_files[choice-1]}")
                    
                    # First classify the document
                    classification = test_document_classification(file_path, args.verbose)
                    
                    # Then process it
                    document_type = args.document_type or classification.get("document_type")
                    test_document_processing(file_path, document_type)
                    
                    # Ask if user wants to process another document
                    another = input("\nProcess another document? (y/n): ").lower() == 'y'
                    if not another:
                        break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

if __name__ == "__main__":
    main()
