#!/usr/bin/env python3
"""
Script to test Document AI processor with a real document.
This script will:
1. Load credentials from configuration
2. Connect to Document AI API
3. Process a sample document
4. Display the extracted information
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finflow.document_ai_test")

def load_configuration() -> Dict[str, Any]:
    """Load configuration from the development config file."""
    try:
        # Import config loader from project
        from config.config_loader import load_config
        return load_config()  # The function doesn't take an environment parameter
    except ImportError:
        logger.error("Failed to import config_loader. Make sure you're in the project root directory.")
        return {}

def setup_credentials(config: Dict[str, Any]) -> bool:
    """Set up Google Cloud credentials from config."""
    try:
        # Check if google_cloud section exists in config
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
        logger.error(f"Error setting up credentials: {e}")
        return False

def get_processor_id(config: Dict[str, Any], document_type: str = "invoice") -> Optional[str]:
    """Get the processor ID for a document type."""
    try:
        # Import processor configuration
        from config.document_processor_config import get_processor_id as get_id
        
        # Get project ID from config
        project_id = config["google_cloud"].get("project_id")
        if not project_id:
            logger.error("Project ID not found in configuration.")
            return None
        
        # Get processor ID for the document type
        return get_id(document_type, project_id, "development")
    except Exception as e:
        logger.error(f"Error getting processor ID: {e}")
        return None

def process_document(file_path: str, processor_id: str) -> Dict[str, Any]:
    """Process a document with Document AI."""
    try:
        from google.cloud import documentai_v1 as documentai
        
        # Initialize Document AI client
        client = documentai.DocumentProcessorServiceClient()
        
        # Read the file
        with open(file_path, "rb") as file:
            content = file.read()
        
        # Create the document object
        raw_document = documentai.RawDocument(
            content=content,
            mime_type="application/pdf"
        )
        
        # Configure the process request
        request = documentai.ProcessRequest(
            name=processor_id,
            raw_document=raw_document
        )
        
        # Process the document
        logger.info(f"Processing document: {file_path}")
        start_time = time.time()
        result = client.process_document(request=request)
        processing_time = time.time() - start_time
        
        logger.info(f"Document processed in {processing_time:.2f} seconds")
        
        # Return the result
        return {
            "status": "success",
            "document": result.document,
            "text": result.document.text,
            "pages": len(result.document.pages),
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def extract_entities(document) -> Dict[str, Any]:
    """Extract entities from a processed document."""
    entities = {}
    
    try:
        # Extract entities from the document
        for entity in document.entities:
            entity_type = entity.type_
            mention_text = entity.mention_text
            confidence = entity.confidence
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            entities[entity_type].append({
                "value": mention_text,
                "confidence": confidence
            })
        
        # Convert lists with single items to single values for cleaner output
        for entity_type, values in entities.items():
            if len(values) == 1:
                entities[entity_type] = values[0]
        
        return entities
    
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return {"error": str(e)}

def print_document_entities(entities: Dict[str, Any]) -> None:
    """Print extracted document entities in a readable format."""
    print("\nExtracted Entities:")
    print("=" * 50)
    
    for entity_type, value in entities.items():
        print(f"\n{entity_type.title()}:")
        
        if isinstance(value, list):
            # Multiple values for this entity type
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    print(f"  {i+1}. {item.get('value', 'N/A')} (Confidence: {item.get('confidence', 'N/A'):.2f})")
                else:
                    print(f"  {i+1}. {item}")
        elif isinstance(value, dict):
            # Single value with confidence
            print(f"  {value.get('value', 'N/A')} (Confidence: {value.get('confidence', 'N/A'):.2f})")
        else:
            # Simple value
            print(f"  {value}")

def main():
    print("\n" + "=" * 80)
    print("Document AI Processor Test")
    print("=" * 80)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_configuration()
    if not config:
        print("Failed to load configuration.")
        return
    
    # Set up credentials
    print("\nSetting up Google Cloud credentials...")
    if not setup_credentials(config):
        print("Failed to set up credentials.")
        return
    
    # Get sample document path
    sample_dir = os.path.join(os.getcwd(), "sample_data", "invoices")
    
    # List available sample files
    sample_files = [f for f in os.listdir(sample_dir) 
                    if os.path.isfile(os.path.join(sample_dir, f)) and f.endswith(".pdf")]
    
    if not sample_files:
        print("\nNo sample invoice files found in the sample_data/invoices directory!")
        return
    
    print(f"\nFound {len(sample_files)} sample invoice files:")
    for i, file in enumerate(sample_files):
        print(f"  {i+1}. {file}")
    
    # Select a file to process
    while True:
        try:
            choice = int(input("\nEnter the number of the file to process (or 0 to exit): "))
            if choice == 0:
                print("Exiting.")
                return
            elif 1 <= choice <= len(sample_files):
                file_path = os.path.join(sample_dir, sample_files[choice-1])
                print(f"\nSelected: {sample_files[choice-1]}")
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get processor ID
    print("\nGetting Document AI processor ID...")
    processor_id = get_processor_id(config, "invoice")
    if not processor_id:
        print("Failed to get processor ID.")
        return
    
    print(f"Using processor: {processor_id}")
    
    # Process the document
    print("\nProcessing document with Document AI...")
    result = process_document(file_path, processor_id)
    
    if result["status"] == "error":
        print(f"Error: {result.get('message', 'Unknown error')}")
        return
    
    # Extract and print entities
    print("\nDocument processed successfully!")
    print(f"Pages: {result['pages']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    # Extract and print entities
    entities = extract_entities(result["document"])
    print_document_entities(entities)
    
    # Print text sample
    text = result["text"]
    print("\nExtracted Text Sample:")
    print("=" * 50)
    print(text[:500] + "..." if len(text) > 500 else text)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
