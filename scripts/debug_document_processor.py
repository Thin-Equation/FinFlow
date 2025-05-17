#!/usr/bin/env python
"""
Debug the document processor agent's document processing capabilities.
This script provides detailed debugging information for document processing.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import configure_logging
from agents.document_processor import DocumentProcessorAgent

def main():
    """Main function to run the debug tool."""
    # Set up logging
    configure_logging(log_level="DEBUG")
    
    # Check for input file
    if len(sys.argv) < 2:
        print("Usage: python debug_document_processor.py <document_path>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    if not os.path.isfile(document_path):
        print(f"Error: File not found: {document_path}")
        sys.exit(1)
    
    print(f"Debugging document processing for: {document_path}")
    
    try:
        # Initialize document processor agent
        agent = DocumentProcessorAgent()
        
        # Register tools
        print("Registering document processing tools...")
        agent.register_tools()
        
        # Print available tools
        print("\nAvailable tools:")
        if hasattr(agent, '_tools'):
            for tool in agent._tools:
                print(f"  - {tool.name}: {tool.description}")
        
        # Process document
        print(f"\nProcessing document: {document_path}...")
        context = {
            "document_path": document_path,
            "session_id": "debug_session",
            "user_id": "debug_user"
        }
        
        # Check if file exists and readable
        print(f"File exists: {os.path.exists(document_path)}")
        print(f"File size: {os.path.getsize(document_path)} bytes")
        
        # Try to process
        result = agent.process_document(context)
        
        # Print full result
        print("\nProcessing Result:")
        print(json.dumps(result, indent=2, default=str))
        
        # Print summary
        if result.get("status") == "success":
            print("\n✅ Document processing succeeded")
            extracted = result.get("extracted_data", {})
            print(f"Document Type: {extracted.get('document_type', 'unknown')}")
            print(f"Confidence: {extracted.get('metadata', {}).get('confidence', 'N/A')}")
        else:
            print(f"\n❌ Document processing failed: {result.get('error_message', 'unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
