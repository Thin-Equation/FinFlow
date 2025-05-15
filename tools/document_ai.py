"""
Document AI integration tools for the FinFlow system.
"""

from typing import Any, Dict, Optional
from google.adk.tools import ToolContext  # type: ignore
from google.cloud import documentai_v1 as documentai
import base64

def process_document(content: bytes, processor_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Process a document using Google Document AI.
    
    Args:
        content: Document bytes content
        processor_id: Document AI processor ID
        tool_context: Provided by ADK
        
    Returns:
        dict: Processed document information
    """
    # Initialize Document AI client
    client = documentai.DocumentProcessorServiceClient()
    
    # Prepare the document for processing with type annotations
    document: Dict[str, Any] = {"content": content, "mime_type": "application/pdf"}
    
    # Create the process request with type annotations
    request: Dict[str, Any] = {"name": processor_id, "document": document}
    
    try:
        # Process the document with type ignore for complex Google API
        result = client.process_document(request=request)  # type: ignore
        
        # Extract structured information
        return _extract_document_entities(result.document)  # type: ignore
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def _extract_document_entities(document: Any) -> Dict[str, Any]:
    """
    Extract entity information from Document AI result.
    
    Args:
        document: Document AI document result
        
    Returns:
        dict: Extracted entities and document info
    """
    # Extract entities with careful handling of potential type issues
    entities: Dict[str, Any] = {}
    
    try:
        # Handle document entities with type safety
        if hasattr(document, 'entities'):
            for entity in document.entities:  # type: ignore
                if hasattr(entity, 'type_') and hasattr(entity, 'mention_text'):
                    entity_type = str(entity.type_)
                    entity_text = str(entity.mention_text)
                    entities[entity_type] = entity_text
    except Exception as e:
        entities["error"] = str(e)
    
    # Extract basic document info
    doc_info: Dict[str, Any] = {
        "text": str(document.text) if hasattr(document, 'text') else "",
        "pages": len(document.pages) if hasattr(document, 'pages') else 0,
        "entities": entities,
    }
    
    return doc_info

def analyze_financial_document(file_content: bytes, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Analyze financial document to extract structured information.
    
    Args:
        file_content: Document bytes content
        tool_context: Provided by ADK
        
    Returns:
        dict: Extracted financial information
    """
    # Configure your Document AI processor ID here
    processor_id = "projects/YOUR_PROJECT/locations/YOUR_LOCATION/processors/YOUR_PROCESSOR_ID"
    
    # Process the document
    doc_info = process_document(file_content, processor_id, tool_context)
    
    # Add document type detection
    doc_info["document_type"] = _detect_document_type(doc_info)
    
    return doc_info

def _detect_document_type(doc_info: Dict[str, Any]) -> str:
    """
    Detect document type based on extracted content.
    
    Args:
        doc_info: Document information
        
    Returns:
        str: Document type
    """
    text = doc_info.get("text", "").lower()
    
    # Basic document type detection based on text content
    if "invoice" in text or "bill" in text:
        return "invoice"
    elif "statement" in text and ("account" in text or "bank" in text):
        return "bank_statement"
    elif "tax" in text and ("return" in text or "form" in text):
        return "tax_document"
    elif "policy" in text and "insurance" in text:
        return "insurance_policy"
    else:
        return "unknown"

def base64_to_bytes(base64_string: str) -> bytes:
    """
    Convert base64 string to bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        bytes: Decoded bytes
    """
    return base64.b64decode(base64_string)
