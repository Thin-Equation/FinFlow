"""
Document AI integration tools for the FinFlow system.
"""

import os
import logging
from dataclasses import dataclass
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
    # Default to invoice processor
    processor_id = "projects/YOUR_PROJECT/locations/us-central1/processors/finflow-invoice-processor"
    
    # Try to get processor_id from tool_context if available
    if tool_context:
        # Check if tool_context has get method (dictionary-like)
        if hasattr(tool_context, "get") and callable(getattr(tool_context, "get")):
            ctx_processor_id = tool_context.get("processor_id")  # type: ignore
            if ctx_processor_id:
                processor_id = ctx_processor_id
    
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


# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class InvoiceProcessorConfig:
    """Configuration for Document AI Invoice processor."""
    project_id: str
    location: str = "us-central1" 
    display_name: str = "FinFlow Invoice Processor"
    
    @property
    def parent(self) -> str:
        """Get the full resource name of the location."""
        return f"projects/{self.project_id}/locations/{self.location}"
    
    @property
    def processor_id(self) -> str:
        """Get the processor ID format."""
        return f"projects/{self.project_id}/locations/{self.location}/processors/{self.display_name.replace(' ', '-').lower()}"


def setup_invoice_processor(project_id: str, location: str = "us-central1", 
                           display_name: str = "FinFlow Invoice Processor") -> Dict[str, Any]:
    """
    Creates a new Document AI invoice processor.
    
    Args:
        project_id: Google Cloud project ID
        location: Location for the processor
        display_name: Display name for the processor
        
    Returns:
        Dict with processor ID or error
    """
    try:
        config = InvoiceProcessorConfig(
            project_id=project_id,
            location=location,
            display_name=display_name
        )
        
        # Initialize Document AI client
        client = documentai.DocumentProcessorServiceClient()
        
        # Log the creation attempt
        logger.info(f"Creating invoice processor: {config.display_name} in {config.location}")
        
        # Create processor with proper type
        processor = documentai.Processor(
            display_name=config.display_name,
            type_="INVOICE_PROCESSOR"  # This is the officially supported type
        )
        
        # Create processor request
        response = client.create_processor(
            parent=config.parent,
            processor=processor
        )
        
        # Log the success
        logger.info(f"Successfully created processor: {response.name}")
        
        # Return the processor ID
        return {
            "status": "success",
            "processor_id": response.name,
            "display_name": response.display_name
        }
    except Exception as e:
        logger.error(f"Error creating invoice processor: {str(e)}")
        return {
            "status": "error",
            "message": f"Error creating invoice processor: {str(e)}"
        }


def import_training_documents(processor_id: str, training_dir: str) -> Dict[str, Any]:
    """
    Import training documents for Document AI processor.
    
    Note: This is a simplified implementation. In production,
    you would typically upload documents to Google Cloud Storage
    and use the batch import API for larger datasets.
    
    Args:
        processor_id: Full resource name of the processor
        training_dir: Directory containing training documents
        
    Returns:
        Dict with import status information
    """
    try:
        # Check if directory exists
        if not os.path.isdir(training_dir):
            return {"status": "error", "message": f"Directory not found: {training_dir}"}
        
        # List training files
        valid_files = []
        for filename in os.listdir(training_dir):
            if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff')):
                file_path = os.path.join(training_dir, filename)
                valid_files.append(file_path)
        
        if not valid_files:
            return {"status": "error", "message": "No valid training files found"}
        
        logger.info(f"Found {len(valid_files)} valid files for training in {training_dir}")
        
        # For now, we'll simulate the import process since the real implementation
        # requires a full GCS integration and Document AI dataset creation
        return {
            "status": "success",
            "processor_id": processor_id,
            "files_imported": len(valid_files),
            "message": "In a production environment, these files would be uploaded to GCS and imported to Document AI"
        }
    except Exception as e:
        logger.error(f"Error importing training documents: {str(e)}")
        return {
            "status": "error", 
            "message": f"Error importing training documents: {str(e)}"
        }


def train_invoice_processor(processor_id: str, training_data_path: str) -> Dict[str, Any]:
    """
    Train an invoice processor with sample documents.
    
    Args:
        processor_id: The full resource name of the processor
        training_data_path: Path to the directory containing training documents
        
    Returns:
        dict: Training result information
    """
    try:
        logger.info(f"Starting training for processor {processor_id}")
        
        # Import training documents
        import_result = import_training_documents(processor_id, training_data_path)
        
        if import_result.get("status") == "error":
            return import_result
        
        # For now, simulate the training process since the real implementation
        # requires the Document AI trainProcessor API and waiting for completion
        return {
            "status": "success",
            "processor_id": processor_id,
            "import_result": import_result,
            "message": "In production, the Document AI processor would be trained with these documents"
        }
    except Exception as e:
        logger.error(f"Error training invoice processor: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


def test_invoice_processor(processor_id: str, test_file_path: str) -> Dict[str, Any]:
    """
    Test an invoice processor with a validation document.
    
    Args:
        processor_id: The full resource name of the processor
        test_file_path: Path to the test document
        
    Returns:
        dict: Structured invoice data from the test
    """
    try:
        # Check if file exists
        if not os.path.isfile(test_file_path):
            return {"status": "error", "message": f"Test file not found: {test_file_path}"}
        
        logger.info(f"Testing processor {processor_id} with file {test_file_path}")
        
        # Read test document
        with open(test_file_path, 'rb') as f:
            file_content = f.read()
        
        # Process document
        result = process_document(file_content, processor_id)
        
        # Extract invoice-specific fields and format the response
        if result.get("status") == "error":
            return result
        
        # Structure invoice data according to our data model
        entities = result.get("entities", {})
        
        invoice_data = {
            "document_type": "invoice",
            "invoice_number": entities.get("invoice_id"),
            "issue_date": entities.get("invoice_date"),
            "due_date": entities.get("due_date"),
            "vendor": {
                "name": entities.get("supplier_name"),
                "tax_id": entities.get("supplier_tax_id")
            },
            "customer": {
                "name": entities.get("customer_name"),
                "tax_id": entities.get("customer_tax_id")
            },
            "total_amount": entities.get("total_amount"),
            "tax_amount": entities.get("tax_amount"),
            "line_items": entities.get("line_items", []),
            "currency": entities.get("currency"),
            "payment_terms": entities.get("payment_terms"),
            "confidence_score": result.get("confidence", 0.0)
        }
        
        logger.info(f"Successfully tested processor with file {test_file_path}")
        
        return {
            "status": "success",
            "processor_id": processor_id,
            "test_file": test_file_path,
            "invoice_data": invoice_data
        }
    except Exception as e:
        logger.error(f"Error testing invoice processor: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


def evaluate_processor_performance(processor_id: str, evaluation_data_path: str) -> Dict[str, Any]:
    """
    Evaluate an invoice processor with a set of validation documents.
    
    Args:
        processor_id: The full resource name of the processor
        evaluation_data_path: Path to the directory containing validation documents
        
    Returns:
        dict: Evaluation metrics
    """
    import statistics
    from typing import List
    
    try:
        # Check if directory exists
        if not os.path.isdir(evaluation_data_path):
            return {"status": "error", "message": f"Directory not found: {evaluation_data_path}"}
        
        # Collect evaluation files
        evaluation_files: List[str] = []
        for filename in os.listdir(evaluation_data_path):
            if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff')):
                file_path = os.path.join(evaluation_data_path, filename)
                evaluation_files.append(file_path)
        
        if not evaluation_files:
            return {"status": "error", "message": "No valid evaluation files found"}
        
        logger.info(f"Evaluating processor {processor_id} with {len(evaluation_files)} files")
        
        # Process each file and collect metrics
        results: List[Dict[str, Any]] = []
        confidence_scores: List[float] = []
        field_extraction_rates: List[float] = []
        
        for file_path in evaluation_files:
            test_result = test_invoice_processor(processor_id, file_path)
            
            if test_result.get("status") == "success":
                results.append(test_result)
                
                # Track confidence scores
                confidence = float(test_result.get("invoice_data", {}).get("confidence_score", 0.0))
                confidence_scores.append(confidence)
                
                # Calculate field extraction rate (percentage of non-null fields)
                invoice_data = test_result.get("invoice_data", {})
                total_fields = len(invoice_data)
                non_null_fields = sum(1 for v in invoice_data.values() if v is not None and v != "")
                extraction_rate = non_null_fields / total_fields if total_fields > 0 else 0.0
                field_extraction_rates.append(extraction_rate)
        
        # Calculate aggregate metrics
        if not confidence_scores:
            return {"status": "error", "message": "No successful evaluations"}
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        avg_extraction_rate = sum(field_extraction_rates) / len(field_extraction_rates) if field_extraction_rates else 0.0
        
        # Calculate min and max without using built-in functions to avoid type issues
        min_confidence = confidence_scores[0]
        max_confidence = confidence_scores[0]
        for score in confidence_scores:
            if score < min_confidence:
                min_confidence = score
            if score > max_confidence:
                max_confidence = score
        
        # Calculate standard deviation manually if needed
        std_dev = 0.0
        if len(confidence_scores) > 1:
            try:
                std_dev = statistics.stdev(confidence_scores)
            except (TypeError, ValueError, statistics.StatisticsError):
                # Fallback in case of type issues
                mean = avg_confidence
                variance_sum = sum((x - mean) ** 2 for x in confidence_scores)
                std_dev = (variance_sum / (len(confidence_scores) - 1)) ** 0.5
        
        logger.info(f"Evaluation complete with {len(results)} successful tests")
        
        return {
            "status": "success",
            "processor_id": processor_id,
            "files_evaluated": len(evaluation_files),
            "successful_evaluations": len(results),
            "avg_confidence_score": avg_confidence,
            "avg_field_extraction_rate": avg_extraction_rate,
            "evaluation_summary": {
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "std_dev_confidence": std_dev,
            }
        }
    except Exception as e:
        logger.error(f"Error evaluating processor performance: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


def get_invoice_processor_capabilities() -> Dict[str, Any]:
    """
    Returns the capabilities and limitations of the invoice processor.
    
    Returns:
        dict: Capabilities and limitations information
    """
    return {
        "capabilities": {
            "document_types": ["Invoice", "Receipt"],
            "supported_formats": ["PDF", "JPEG", "PNG", "TIFF"],
            "extractable_fields": {
                "invoice_id": "Invoice number or identifier",
                "invoice_date": "Date the invoice was issued",
                "due_date": "Payment due date",
                "supplier_name": "Name of the supplier/vendor",
                "supplier_tax_id": "Tax ID or business registration of supplier",
                "customer_name": "Name of the customer",
                "customer_tax_id": "Tax ID of the customer",
                "line_items": "Individual items on the invoice with description, quantity, price",
                "subtotal": "Subtotal amount before taxes and fees",
                "tax_amount": "Total tax amount",
                "total_amount": "Total invoice amount",
                "currency": "Currency used for the invoice",
                "payment_terms": "Terms of payment"
            },
            "field_confidence": "Confidence score for each extracted field",
            "multi_page": "Support for multi-page invoices",
            "multi_language": "Support for multiple languages (varies by processor)",
        },
        "limitations": {
            "document_quality": "Low quality scans or handwritten documents may have lower extraction accuracy",
            "complex_layouts": "Non-standard or highly customized invoice layouts may have reduced accuracy",
            "field_types": "Custom fields unique to specific vendors may not be recognized",
            "language_support": "Best performance with English invoices, other languages may have varied results",
            "processing_time": "Complex multi-page documents may take longer to process",
            "training_data": "Performance improves with more diverse training examples",
            "minimum_training": "Requires at least 20 sample invoices for basic model training"
        }
    }
