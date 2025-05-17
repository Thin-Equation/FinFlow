#!/usr/bin/env python
"""
Comprehensive test script for the enhanced document processor implementation.

This script tests all the advanced features of the document processor agent:
1. Document classification
2. Batch processing
3. Error handling and recovery
4. Performance optimization

Usage:
    python test_enhanced_document_processor.py [options]

Options:
    --test-single          Run single document processing tests
    --test-batch           Run batch processing tests
    --test-classification  Run document classification tests
    --test-error-handling  Run error handling and recovery tests
    --test-performance     Run performance optimization tests
    --test-all             Run all tests (default)
    --verbose              Enable verbose output
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List
import uuid
import shutil
import json
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Import mock implementations if needed for testing
try:
    from tools.mock_document_processor import patch_modules, MockToolContext
    MOCK_ENABLED = True
    # Patch modules with mock implementations
    patch_modules()
except ImportError:
    MOCK_ENABLED = False

# Import the agent and tools
from agents.document_processor import DocumentProcessorAgent
from tools.enhanced_document_ai import DocumentProcessor
from tools.document_classification import DocumentClassifier
from config.document_processor_config import get_processor_config
from utils.logging_config import configure_logging

# Global constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SAMPLE_DATA_DIR = os.path.join(PROJECT_DIR, "sample_data")
INVOICES_DIR = os.path.join(SAMPLE_DATA_DIR, "invoices")
TEST_OUTPUT_DIR = os.path.join(PROJECT_DIR, "test_output")
TEST_TEMP_DIR = os.path.join(TEST_OUTPUT_DIR, "temp")

# Test helper functions
def setup_test_environment():
    """Set up the test environment and directories."""
    # Create test output directory if it doesn't exist
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_TEMP_DIR, exist_ok=True)
    
    # Clean up any previous test files
    for file in os.listdir(TEST_TEMP_DIR):
        file_path = os.path.join(TEST_TEMP_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    print(f"\n{'=' * 80}\nSetting up test environment at {TEST_OUTPUT_DIR}\n{'=' * 80}")
    return TEST_OUTPUT_DIR

def get_sample_documents(sample_dir: str = None) -> List[str]:
    """Get a list of sample documents to test."""
    # Use default directory if none provided
    if not sample_dir:
        sample_dir = INVOICES_DIR
    
    # Check if directory exists
    if not os.path.isdir(sample_dir):
        print(f"Error: Sample directory not found: {sample_dir}")
        return []
    
    # Get list of PDF files in the directory
    sample_files = []
    for file in os.listdir(sample_dir):
        if file.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            sample_files.append(os.path.join(sample_dir, file))
    
    return sample_files

def create_tool_context(processor_id: str = "mock-processor-id") -> Any:
    """Create a tool context for testing."""
    if MOCK_ENABLED:
        return MockToolContext(
            processor_id=processor_id,
            user="test-user",
            session=f"test-session-{uuid.uuid4()}"
        )
    else:
        # Use a simple dict as a stand-in for testing
        return {
            "processor_id": processor_id,
            "user": "test-user",
            "session": f"test-session-{uuid.uuid4()}"
        }

def create_test_context(document_path: str = None, batch_paths: List[str] = None) -> Dict[str, Any]:
    """Create a context dictionary for testing."""
    context = {
        "session_id": f"test-session-{uuid.uuid4()}",
        "user_id": "test-user",
        "session_state": {},
        "test_mode": True
    }
    
    if document_path:
        context["document_path"] = document_path
    
    if batch_paths:
        context["batch_documents"] = batch_paths
    
    return context

def print_result_summary(result: Dict[str, Any], verbose: bool = False):
    """Print a summary of the processing result."""
    print("\nResult Summary:")
    print(f"  Status: {result.get('status', 'unknown')}")
    
    # Handle error results
    if result.get("status") == "error":
        print(f"  Error: {result.get('error_message', 'Unknown error')}")
        if result.get("error_type"):
            print(f"  Error Type: {result.get('error_type')}")
        return
    
    # Handle successful results
    if "document_type" in result:
        print(f"  Document Type: {result.get('document_type', 'unknown')}")
    
    if "confidence" in result:
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
    
    if "extraction_time" in result:
        print(f"  Processing Time: {result.get('extraction_time', 'N/A')} seconds")
    
    # Show extracted data summary
    if "extracted_data" in result:
        extracted = result.get("extracted_data", {})
        if isinstance(extracted, dict):
            fields = len(extracted.keys())
            print(f"  Extracted Fields: {fields}")
            
            # Print key fields for invoices if present
            if "total_amount" in extracted:
                print(f"  Total Amount: {extracted.get('total_amount', 'N/A')}")
            if "invoice_date" in extracted:
                print(f"  Invoice Date: {extracted.get('invoice_date', 'N/A')}")
            if "vendor" in extracted:
                print(f"  Vendor: {extracted.get('vendor', 'N/A')}")
        
        # Print all extracted data in verbose mode
        if verbose:
            print("\nExtracted Data:")
            print(json.dumps(extracted, indent=2, default=str))
    
    # Batch processing results
    if "batch_results" in result:
        batch = result.get("batch_results", {})
        total = batch.get("total", 0)
        success = batch.get("successful", 0)
        failed = batch.get("failed", 0)
        print(f"\nBatch Processing Summary:")
        print(f"  Total Documents: {total}")
        print(f"  Successfully Processed: {success}")
        print(f"  Failed: {failed}")
        
        if verbose and "documents" in batch:
            print("\nIndividual Document Results:")
            for doc in batch.get("documents", []):
                print(f"  - {doc.get('file_name', 'unknown')}: {doc.get('status', 'unknown')}")

# Test implementations
def test_single_document_processing(agent: DocumentProcessorAgent, verbose: bool = False):
    """Test single document processing functionality."""
    print(f"\n{'=' * 80}\nTesting Single Document Processing\n{'=' * 80}")
    
    # Get sample documents
    sample_files = get_sample_documents()
    if not sample_files:
        print("No sample documents found for testing.")
        return False
    
    # Create tool context
    tool_context = create_tool_context()
    
    # Process a single document
    test_file = sample_files[0]
    print(f"Processing: {os.path.basename(test_file)}")
    
    context = create_test_context(document_path=test_file)
    start_time = time.time()
    
    try:
        result = agent.process_document(context, tool_context)
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print_result_summary(result, verbose)
        
        # Validate the result structure
        if "status" not in result:
            print("Error: Missing status in result")
            return False
        
        if result.get("status") == "success":
            if "extracted_data" not in result:
                print("Error: Missing extracted_data in successful result")
                return False
            return True
        else:
            print(f"Processing failed: {result.get('error_message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Error during single document processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing(agent: DocumentProcessorAgent, verbose: bool = False):
    """Test batch document processing functionality."""
    print(f"\n{'=' * 80}\nTesting Batch Document Processing\n{'=' * 80}")
    
    # Get sample documents
    sample_files = get_sample_documents()
    if len(sample_files) < 2:
        print("Not enough sample documents found for batch testing.")
        return False
    
    # Use up to 3 documents for batch testing
    batch_files = sample_files[:3]
    print(f"Batch processing {len(batch_files)} documents")
    
    # Create tool context
    tool_context = create_tool_context()
    
    # Create context with batch documents
    context = create_test_context(batch_paths=batch_files)
    start_time = time.time()
    
    try:
        result = agent.process_document_batch(context, tool_context)
        processing_time = time.time() - start_time
        print(f"Batch processing completed in {processing_time:.2f} seconds")
        print_result_summary(result, verbose)
        
        # Validate batch result structure
        if "batch_results" not in result:
            print("Error: Missing batch_results in result")
            return False
        
        batch_results = result.get("batch_results", {})
        if "total" not in batch_results or "successful" not in batch_results:
            print("Error: Missing batch statistics in result")
            return False
        
        # Check if any documents were processed successfully
        success_count = batch_results.get("successful", 0)
        if success_count == 0:
            print("Error: No documents were processed successfully")
            return False
            
        return True
    except Exception as e:
        print(f"Error during batch document processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_classification(verbose: bool = False):
    """Test document classification functionality."""
    print(f"\n{'=' * 80}\nTesting Document Classification\n{'=' * 80}")
    
    # Get sample documents
    sample_files = get_sample_documents()
    if not sample_files:
        print("No sample documents found for classification testing.")
        return False
    
    # Initialize document classifier
    classifier = DocumentClassifier()
    
    # Test classification on multiple documents
    success_count = 0
    
    for i, sample_file in enumerate(sample_files[:3]):  # Test up to 3 documents
        print(f"\nClassifying document {i+1}: {os.path.basename(sample_file)}")
        
        try:
            # Classify document
            start_time = time.time()
            result = classifier.classify_document(sample_file)
            classification_time = time.time() - start_time
            
            # Print classification results
            print(f"Classification completed in {classification_time:.2f} seconds")
            print(f"Document Type: {result.get('document_type', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            
            if "category" in result:
                print(f"Category: {result.get('category', 'N/A')}")
            
            # Print detailed results in verbose mode
            if verbose and "keyword_matches" in result:
                print("\nKeyword Matches:")
                for doc_type, score in result["keyword_matches"].items():
                    if score > 0:
                        print(f"  - {doc_type}: {score}")
            
            # Count successful classifications
            if result.get("document_type") != "unknown" and result.get("confidence", 0) > 0.3:
                success_count += 1
                
        except Exception as e:
            print(f"Error during document classification: {e}")
            import traceback
            traceback.print_exc()
    
    # Check overall success
    print(f"\nSuccessfully classified {success_count} out of {min(3, len(sample_files))} documents")
    return success_count > 0

def test_error_handling(agent: DocumentProcessorAgent, verbose: bool = False):
    """Test error handling and recovery mechanisms."""
    print(f"\n{'=' * 80}\nTesting Error Handling and Recovery\n{'=' * 80}")
    
    # Create a non-existent file path
    non_existent_file = os.path.join(TEST_TEMP_DIR, "non_existent_file.pdf")
    
    # Create an invalid file
    invalid_file = os.path.join(TEST_TEMP_DIR, "invalid_file.pdf")
    with open(invalid_file, "w") as f:
        f.write("This is not a valid PDF file")
    
    # Create tool context
    tool_context = create_tool_context()
    
    # Test case 1: Non-existent file
    print("\nTest Case 1: Non-existent file")
    context = create_test_context(document_path=non_existent_file)
    
    try:
        result = agent.process_document(context, tool_context)
        print("Agent handled non-existent file correctly")
        print(f"Error message: {result.get('error_message', 'None')}")
        
        # Verify error handling
        if result.get("status") != "error":
            print("Error: Expected status 'error' for non-existent file")
            test1_success = False
        else:
            test1_success = True
            
    except Exception as e:
        print(f"Uncaught exception for non-existent file: {e}")
        test1_success = False
    
    # Test case 2: Invalid file content
    print("\nTest Case 2: Invalid file content")
    context = create_test_context(document_path=invalid_file)
    
    try:
        result = agent.process_document(context, tool_context)
        print("Agent handled invalid file correctly")
        print(f"Error message: {result.get('error_message', 'None')}")
        
        # Verify error handling
        if result.get("status") != "error":
            print("Error: Expected status 'error' for invalid file")
            test2_success = False
        else:
            test2_success = True
            
    except Exception as e:
        print(f"Uncaught exception for invalid file: {e}")
        test2_success = False
    
    # Test case 3: Retry mechanism (using the Document AI processor directly)
    print("\nTest Case 3: Retry mechanism")
    
    # Get a real sample document
    sample_files = get_sample_documents()
    if not sample_files:
        print("No sample documents available for retry test")
        test3_success = False
    else:
        test_file = sample_files[0]
        
        # Create document processor with limited retries for testing
        doc_processor = DocumentProcessor(
            project_id="test-project",
            location="us-central1"
        )
        doc_processor.max_retries = 2
        
        # Create a custom exception to simulate API error
        class SimulatedAPIError(Exception):
            pass
        
        # Mock the document AI client to simulate failures
        original_client = doc_processor.client
        
        failure_count = 0
        def mock_process(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count < 2:  # Fail first attempt, succeed on 2nd
                raise SimulatedAPIError("Simulated API failure")
            return {"text": "Processed after retry"}
        
        try:
            # Patch the processor
            doc_processor.client = MagicMock()
            doc_processor.client.process_document.side_effect = mock_process
            
            # Try processing
            result = doc_processor.process_single_document(test_file)
            
            # Check if retry worked
            print(f"Retry attempts: {failure_count}")
            test3_success = failure_count > 1 and 'retry_count' in result
            
            if test3_success:
                print("Retry mechanism working correctly")
            else:
                print("Retry mechanism not working as expected")
                
        except Exception as e:
            print(f"Error during retry test: {e}")
            test3_success = False
            
        finally:
            # Restore the original client
            doc_processor.client = original_client
    
    # Overall test success
    return test1_success and test2_success and test3_success

def test_performance_optimizations(agent: DocumentProcessorAgent, verbose: bool = False):
    """Test performance optimization features."""
    print(f"\n{'=' * 80}\nTesting Performance Optimizations\n{'=' * 80}")
    
    # Get sample documents
    sample_files = get_sample_documents()
    if len(sample_files) < 2:
        print("Not enough sample documents found for performance testing.")
        return False
    
    # Create tool context
    tool_context = create_tool_context()
    
    # Test 1: Document caching
    print("\nTest Case 1: Document Caching")
    
    # Process the same document twice to test caching
    test_file = sample_files[0]
    context = create_test_context(document_path=test_file)
    
    try:
        # First run
        print("First processing run...")
        start_time = time.time()
        result1 = agent.process_document(context, tool_context)
        first_run_time = time.time() - start_time
        print(f"First processing completed in {first_run_time:.2f} seconds")
        
        # Add cache hint to context
        context["use_cache"] = True
        
        # Second run with same document
        print("Second processing run (with caching)...")
        start_time = time.time()
        result2 = agent.process_document(context, tool_context)
        second_run_time = time.time() - start_time
        print(f"Second processing completed in {second_run_time:.2f} seconds")
        
        # Check if caching improved performance
        if "cache_hit" in result2 and result2["cache_hit"] == True:
            print("Cache hit detected!")
        
        # Should be faster or have cache indicator
        test1_success = "cache_hit" in result2 or second_run_time < first_run_time
        print(f"Caching test {'successful' if test1_success else 'failed'}")
        
    except Exception as e:
        print(f"Error during caching test: {e}")
        import traceback
        traceback.print_exc()
        test1_success = False
    
    # Test 2: Parallel processing in batch mode
    print("\nTest Case 2: Parallel Processing")
    
    # Process multiple documents in batch
    batch_files = sample_files[:3]  # Use up to 3 documents
    context = create_test_context(batch_paths=batch_files)
    
    try:
        # Process in batch mode
        start_time = time.time()
        result = agent.process_document_batch(context, tool_context)
        batch_time = time.time() - start_time
        print(f"Batch processing completed in {batch_time:.2f} seconds")
        
        # Get batch metrics
        batch_results = result.get("batch_results", {})
        total_docs = batch_results.get("total", 0)
        parallel_info = batch_results.get("parallel_processing", False)
        workers_used = batch_results.get("workers_used", 1)
        
        print(f"Processed {total_docs} documents")
        print(f"Parallel processing: {'Yes' if parallel_info else 'No'}")
        print(f"Workers used: {workers_used}")
        
        # Individual timing would be better than this simple check
        if total_docs > 0:
            avg_time_per_doc = batch_time / total_docs
            print(f"Average time per document: {avg_time_per_doc:.2f} seconds")
        
        # Simple criteria for success
        test2_success = total_docs > 0 and (parallel_info or workers_used > 1)
        print(f"Parallel processing test {'successful' if test2_success else 'failed'}")
        
    except Exception as e:
        print(f"Error during parallel processing test: {e}")
        import traceback
        traceback.print_exc()
        test2_success = False
    
    # Overall test success
    return test1_success or test2_success  # At least one optimization should work

def main():
    """Main function to run the document processor tests."""
    parser = argparse.ArgumentParser(description="Test the Enhanced Document Processor")
    parser.add_argument("--test-single", action="store_true", help="Run single document processing tests")
    parser.add_argument("--test-batch", action="store_true", help="Run batch processing tests")
    parser.add_argument("--test-classification", action="store_true", help="Run document classification tests")
    parser.add_argument("--test-error-handling", action="store_true", help="Run error handling tests")
    parser.add_argument("--test-performance", action="store_true", help="Run performance optimization tests")
    parser.add_argument("--test-all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-level", "-l", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")
    
    args = parser.parse_args()
    
    # If no specific tests are specified, run all tests
    run_all = args.test_all or not (args.test_single or args.test_batch or 
                                   args.test_classification or args.test_error_handling or 
                                   args.test_performance)
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    
    # Set up test environment
    setup_test_environment()
    
    # Create mock imports if needed
    if not MOCK_ENABLED:
        from unittest.mock import MagicMock
    
    # Initialize document processor agent
    print("Initializing Document Processor Agent...")
    try:
        # For testing purposes, create agent with test config
        test_config = {
            "telemetry": {"collect_metrics": True, "log_level": args.log_level},
            "document_cache_enabled": True
        }
        agent = DocumentProcessorAgent(config=test_config)
        agent.register_tools()
        
        # If verbose, print agent tools
        if args.verbose:
            print("\nAvailable tools:")
            if hasattr(agent, '_tools'):
                for tool in agent._tools:
                    print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Track test results
    test_results = {}
    
    # Run tests based on arguments
    try:
        # Single document processing test
        if run_all or args.test_single:
            test_results["single"] = test_single_document_processing(agent, args.verbose)
        
        # Batch processing test
        if run_all or args.test_batch:
            test_results["batch"] = test_batch_processing(agent, args.verbose)
        
        # Document classification test
        if run_all or args.test_classification:
            test_results["classification"] = test_document_classification(args.verbose)
        
        # Error handling test
        if run_all or args.test_error_handling:
            test_results["error_handling"] = test_error_handling(agent, args.verbose)
        
        # Performance optimization test
        if run_all or args.test_performance:
            test_results["performance"] = test_performance_optimizations(agent, args.verbose)
    
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print overall summary
    print(f"\n{'=' * 80}\nTest Summary\n{'=' * 80}")
    for test_name, result in test_results.items():
        print(f"{test_name.replace('_', ' ').title()}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Overall success
    all_passed = all(test_results.values())
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
