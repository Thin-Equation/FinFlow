#!/usr/bin/env python
"""
Validation script for the invoice processor.

This script evaluates the performance of the Document Processing Agent
on a set of sample invoices to measure extraction accuracy.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_processor import DocumentProcessorAgent
from utils.logging_config import configure_logging
from config.config_loader import load_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate and evaluate invoice processor performance")
    parser.add_argument("--input-dir", help="Directory containing sample invoices with known data")
    parser.add_argument("--truth-file", help="JSON file containing ground truth data")
    parser.add_argument("--output", help="Output report file", default="invoice_validation_results.json")
    parser.add_argument("--log-level", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()

def load_ground_truth(truth_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth data for invoices.
    
    Args:
        truth_file: Path to the JSON file with ground truth data
        
    Returns:
        Dictionary mapping filename to expected extraction values
    """
    if not os.path.exists(truth_file):
        print(f"Error: Ground truth file not found: {truth_file}")
        sys.exit(1)
    
    with open(truth_file, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in ground truth file: {truth_file}")
            sys.exit(1)

def evaluate_extraction(
    actual: Dict[str, Any], 
    expected: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Evaluate extraction accuracy by comparing actual vs expected values.
    
    Args:
        actual: Extracted data from invoice processor
        expected: Ground truth data for the invoice
        
    Returns:
        Tuple containing (field accuracy scores, field error messages)
    """
    accuracy_scores = {}
    error_messages = {}
    
    # Top-level fields to evaluate
    fields = ["invoice_number", "date", "due_date", "total_amount", "subtotal", "tax_amount"]
    
    # Evaluate each field
    for field in fields:
        actual_value = actual.get(field)
        expected_value = expected.get(field)
        
        if actual_value is None and expected_value is None:
            accuracy_scores[field] = 1.0
            continue
            
        if actual_value is None and expected_value is not None:
            accuracy_scores[field] = 0.0
            error_messages[field] = f"Missing field: expected {expected_value}"
            continue
            
        if actual_value is not None and expected_value is None:
            accuracy_scores[field] = 0.5  # Partial credit for extraction without ground truth
            error_messages[field] = f"Unexpected value: {actual_value}"
            continue
        
        # Special handling for monetary values
        if field in ["total_amount", "subtotal", "tax_amount"]:
            try:
                actual_float = float(str(actual_value).replace(',', ''))
                expected_float = float(str(expected_value).replace(',', ''))
                
                if abs(actual_float - expected_float) < 0.01:  # Allow small difference
                    accuracy_scores[field] = 1.0
                else:
                    diff_pct = abs(actual_float - expected_float) / max(abs(expected_float), 0.01) * 100
                    if diff_pct < 5:  # Within 5% difference
                        accuracy_scores[field] = 0.8
                    elif diff_pct < 10:  # Within 10% difference
                        accuracy_scores[field] = 0.6
                    else:
                        accuracy_scores[field] = 0.3
                        
                    error_messages[field] = f"Value mismatch: got {actual_float}, expected {expected_float}"
            except (ValueError, TypeError):
                accuracy_scores[field] = 0.0
                error_messages[field] = f"Invalid monetary value: {actual_value}"
        
        # String fields like invoice number
        else:
            if str(actual_value).strip() == str(expected_value).strip():
                accuracy_scores[field] = 1.0
            else:
                accuracy_scores[field] = 0.0
                error_messages[field] = f"Value mismatch: got '{actual_value}', expected '{expected_value}'"
    
    # Nested fields
    if "vendor" in actual and "vendor" in expected:
        if actual["vendor"].get("name") == expected["vendor"].get("name"):
            accuracy_scores["vendor.name"] = 1.0
        else:
            accuracy_scores["vendor.name"] = 0.0
            error_messages["vendor.name"] = f"Value mismatch: got '{actual['vendor'].get('name')}', expected '{expected['vendor'].get('name')}'"
    
    # Calculate line items accuracy if present
    if "line_items" in actual and "line_items" in expected:
        actual_items = actual["line_items"]
        expected_items = expected["line_items"]
        
        if len(actual_items) == len(expected_items):
            accuracy_scores["line_items.count"] = 1.0
        else:
            accuracy_scores["line_items.count"] = min(len(actual_items), len(expected_items)) / max(len(actual_items), len(expected_items))
            error_messages["line_items.count"] = f"Count mismatch: got {len(actual_items)}, expected {len(expected_items)}"
    
    return accuracy_scores, error_messages

def process_and_evaluate(
    agent: DocumentProcessorAgent, 
    input_dir: str, 
    ground_truth: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process invoices and evaluate extraction accuracy.
    
    Args:
        agent: Document processor agent
        input_dir: Directory with invoice files
        ground_truth: Dictionary with expected values
        
    Returns:
        Evaluation results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": 0,
        "successful_extractions": 0,
        "average_accuracy": 0.0,
        "field_accuracies": {},
        "document_results": []
    }
    
    # Track aggregate field accuracies
    field_accuracy_sums = {}
    field_counts = {}
    
    # Find all invoice files
    invoice_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            invoice_files.append(os.path.join(input_dir, filename))
    
    if not invoice_files:
        print(f"No invoice files found in {input_dir}")
        return results
    
    results["total_documents"] = len(invoice_files)
    print(f"Processing and evaluating {len(invoice_files)} invoices...")
    
    # Process each invoice
    for i, invoice_path in enumerate(invoice_files):
        basename = os.path.basename(invoice_path)
        print(f"Processing invoice {i+1}/{len(invoice_files)}: {basename}")
        
        # Create processing context
        context = {
            "document_path": invoice_path,
            "document_type": "invoice"
        }
        
        # Extract ground truth if available
        expected_data = ground_truth.get(basename, {})
        if not expected_data and basename.endswith('.pdf'):
            # Try without extension
            expected_data = ground_truth.get(basename[:-4], {})
        
        # Process invoice
        try:
            result = agent.process_document(context)
            document_result = {
                "filename": basename,
                "status": result.get("status"),
                "document_type": result.get("document_type", "unknown"),
            }
            
            if result.get("status") == "success":
                results["successful_extractions"] += 1
                extracted_data = result.get("extracted_data", {})
                
                # Only evaluate if we have ground truth
                if expected_data:
                    accuracy_scores, error_messages = evaluate_extraction(extracted_data, expected_data)
                    
                    # Calculate overall accuracy for this document
                    if accuracy_scores:
                        doc_accuracy = sum(accuracy_scores.values()) / len(accuracy_scores)
                    else:
                        doc_accuracy = 0.0
                        
                    # Add to document results
                    document_result.update({
                        "accuracy": doc_accuracy,
                        "field_accuracies": accuracy_scores,
                        "errors": error_messages,
                        "extracted_data": extracted_data
                    })
                    
                    # Update field accuracy sums
                    for field, score in accuracy_scores.items():
                        if field not in field_accuracy_sums:
                            field_accuracy_sums[field] = 0.0
                            field_counts[field] = 0
                        
                        field_accuracy_sums[field] += score
                        field_counts[field] += 1
                else:
                    document_result["extracted_data"] = extracted_data
                    document_result["note"] = "No ground truth available for evaluation"
            else:
                document_result["error_message"] = result.get("error_message", "Unknown error")
            
            results["document_results"].append(document_result)
                
        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")
            results["document_results"].append({
                "filename": basename,
                "status": "error",
                "error_message": str(e)
            })
    
    # Calculate average field accuracies
    for field, total in field_accuracy_sums.items():
        count = field_counts[field]
        if count > 0:
            results["field_accuracies"][field] = total / count
    
    # Calculate overall average accuracy
    if field_accuracy_sums:
        total_accuracy = sum(results["field_accuracies"].values())
        results["average_accuracy"] = total_accuracy / len(results["field_accuracies"])
    
    return results

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    
    print("Initializing document processor agent...")
    
    # Create and initialize document processor agent
    agent = DocumentProcessorAgent()
    agent.register_tools()
    
    # Load ground truth data if provided
    ground_truth = {}
    if args.truth_file:
        ground_truth = load_ground_truth(args.truth_file)
        print(f"Loaded ground truth data for {len(ground_truth)} invoices")
    
    # Process and evaluate invoices
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
            
        results = process_and_evaluate(agent, args.input_dir, ground_truth)
        
        # Save results to output file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {args.output}")
        
        # Print summary
        success_rate = results["successful_extractions"] / results["total_documents"] if results["total_documents"] > 0 else 0
        print("\nEvaluation Summary:")
        print(f"  Total Invoices: {results['total_documents']}")
        print(f"  Successful Extractions: {results['successful_extractions']} ({success_rate:.1%})")
        
        if ground_truth:
            print(f"  Average Accuracy: {results['average_accuracy']:.1%}")
            print("\nField Accuracies:")
            for field, accuracy in results["field_accuracies"].items():
                print(f"  {field}: {accuracy:.1%}")
    else:
        print("Error: No input directory specified")
        print("Example: python validate_invoice_processor.py --input-dir sample_data/invoices/validation --truth-file ground_truth.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
