#!/usr/bin/env bash
# Script to test the document ingestion tool

# Set the correct Python path
export PYTHONPATH=$PYTHONPATH:.

# Run the document ingestion tests
python -m pytest tests/test_document_ingestion.py -v
