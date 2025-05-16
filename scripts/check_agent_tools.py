#!/usr/bin/env python
"""
A simple script to list tools registered in the document processor agent.
"""

import os
import sys
# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_processor import DocumentProcessorAgent

# Initialize the agent
agent = DocumentProcessorAgent()

# Get class attributes and print tools if any
print("Document processor agent attributes:")
for attr in dir(agent):
    if not attr.startswith('_') or attr == '_tools':
        try:
            attr_value = getattr(agent, attr)
            print(f"- {attr}: {type(attr_value)}")
        except:
            print(f"- {attr}: <error retrieving>")

print("\nTrying to register tools...")
try:
    agent.register_tools()
    print("Tools registered successfully")
except Exception as e:
    print(f"Error registering tools: {e}")

print("\nAgent has _register_document_ingestion_tool method:", 
      hasattr(agent, '_register_document_ingestion_tool'))
