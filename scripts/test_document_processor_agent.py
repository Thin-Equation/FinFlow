#!/usr/bin/env python
"""
Test script for document processor agent's integration with the document ingestion tool.
"""

import os
import sys
# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_processor import DocumentProcessorAgent

def test_document_processor_agent():
    """Test the document processor agent with the document ingestion tool."""
    print("Initializing document processor agent...")
    
    # Create document processor agent
    agent = DocumentProcessorAgent()
    
    # Register document ingestion tools
    agent.register_tools()
    
    print("\nDocument processor agent initialized with tools:")
    for tool in agent._tools if hasattr(agent, '_tools') else []:
        print(f"  - {tool.name}: {tool.description}")
    
    print("\nDocument processor agent test completed!")

if __name__ == "__main__":
    try:
        test_document_processor_agent()
    except Exception as e:
        import traceback
        print(f"Error running test: {e}")
        traceback.print_exc()
