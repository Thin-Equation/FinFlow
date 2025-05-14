"""
Document AI integration tools for the FinFlow system.
"""

from typing import Any, Dict, Optional
from google.adk.tools import ToolContext, BaseTool
from google.cloud import documentai

def process_document(file_path: str, processor_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Process a document using Document AI.
    
    Args:
        file_path: Path to the document file
        processor_id: Document AI processor ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Extracted document information
    """
    client = documentai.DocumentProcessorServiceClient()
    
    # Read the file into memory
    with open(file_path, "rb") as file:
        content = file.read()
        
    # Configure the process request
    document = {"content": content, "mime_type": "application/pdf"}
    request = {"name": processor_id, "document": document}
    
    # Process the document
    result = client.process_document(request=request)
    document = result.document
    
    # Extract and structure the document information
    extracted_data = _extract_document_entities(document)
    
    return {
        "status": "success",
        "extracted_data": extracted_data,
        "document_type": _determine_document_type(extracted_data),
        "confidence": document.text_detection_params.confidence
    }

def _extract_document_entities(document) -> Dict[str, Any]:
    """
    Extract entities from processed document.
    
    Args:
        document: Document AI processed document
        
    Returns:
        dict: Structured document data
    """
    # Mock implementation - would be expanded in full implementation
    entities = {}
    for entity in document.entities:
        entities[entity.type_] = entity.mention_text
    
    return {
        "text": document.text,
        "entities": entities,
        "pages": len(document.pages)
    }

def _determine_document_type(extracted_data: Dict[str, Any]) -> str:
    """
    Determine document type based on extracted data.
    
    Args:
        extracted_data: Extracted document data
        
    Returns:
        str: Document type (invoice, receipt, etc.)
    """
    # Mock implementation - would use more sophisticated logic in production
    entities = extracted_data.get("entities", {})
    
    if "invoice_id" in entities or "invoice_number" in entities:
        return "invoice"
    elif "receipt_id" in entities or "receipt_number" in entities:
        return "receipt"
    else:
        return "unknown"