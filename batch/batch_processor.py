"""
Batch document processing module for FinFlow.

This module provides functionality for batch processing of financial documents.
"""

import os
import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)

# Document formats we can process
SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".docx"]


def process_document(
    agents: Dict[str, Any],
    config: Dict[str, Any],
    document_path: str,
    workflow_type: str = "standard"
) -> Dict[str, Any]:
    """
    Process a single document with the agent system.
    
    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        document_path: Path to the document to process
        workflow_type: Type of workflow to use
        
    Returns:
        Dict[str, Any]: Processing result
    """
    start_time = time.time()
    
    try:
        # Create processing context
        context = {
            "document_path": document_path,
            "workflow_type": workflow_type,
            "user_id": "batch_processor",
            "session_id": f"batch_{datetime.now().timestamp()}",
        }
        
        # Get the master orchestrator
        master_orchestrator = agents["master_orchestrator"]
        
        # Process the document
        result = master_orchestrator.process_document(context)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing metadata
        result["processing_time"] = processing_time
        result["timestamp"] = datetime.now().isoformat()
        result["status"] = "success"
        
        logger.info(f"Processed document {document_path} in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document {document_path}: {e}")
        
        # Return error information
        return {
            "document_path": document_path,
            "error": str(e),
            "status": "failed",
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }


def process_batch(
    agents: Dict[str, Any],
    config: Dict[str, Any],
    batch_dir: str,
    workflow_type: str = "standard",
    max_workers: int = 4,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a batch of documents in a directory.
    
    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        batch_dir: Directory containing documents to process
        workflow_type: Type of workflow to use
        max_workers: Maximum number of concurrent workers
        output_dir: Directory for output results (defaults to batch_dir/results)
        
    Returns:
        Dict[str, Any]: Batch processing results
    """
    # Validate batch directory
    if not os.path.isdir(batch_dir):
        raise ValueError(f"Batch directory not found: {batch_dir}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(batch_dir, "results")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of documents to process
    documents = []
    
    for root, _, files in os.walk(batch_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in SUPPORTED_EXTENSIONS:
                documents.append(file_path)
    
    if not documents:
        logger.warning(f"No supported documents found in {batch_dir}")
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "documents": []
        }
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Process documents with thread pool
    results = []
    success_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_document = {
            executor.submit(
                process_document, agents, config, doc_path, workflow_type
            ): doc_path
            for doc_path in documents
        }
        
        for future in concurrent.futures.as_completed(future_to_document):
            document_path = future_to_document[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # Save individual result to output directory
                document_filename = os.path.basename(document_path)
                result_filename = f"{os.path.splitext(document_filename)[0]}_result.json"
                result_path = os.path.join(output_dir, result_filename)
                
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                if result.get("status") == "success":
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error getting result for {document_path}: {e}")
                failed_count += 1
    
    # Save batch summary
    summary = {
        "total": len(documents),
        "success": success_count,
        "failed": failed_count,
        "timestamp": datetime.now().isoformat(),
        "batch_dir": batch_dir,
        "workflow_type": workflow_type
    }
    
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch processing complete. Processed {len(documents)} documents, "
               f"{success_count} successful, {failed_count} failed.")
    
    return summary
