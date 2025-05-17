"""
Document classification module for the FinFlow system.

This module provides advanced document classification capabilities beyond
basic type detection, identifying specific document categories and subtypes.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import json
from collections import Counter
from datetime import datetime

# For text extraction and classification
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import hashlib

# Configure logging
logger = logging.getLogger("finflow.tools.document_classification")

# Document categories and types
DOCUMENT_CATEGORIES = {
    "financial": [
        "invoice", "receipt", "bank_statement", "credit_card_statement",
        "purchase_order", "quote", "tax_form", "payslip"
    ],
    "legal": [
        "contract", "agreement", "nda", "terms_of_service", 
        "policy", "certificate"
    ],
    "identification": [
        "passport", "drivers_license", "id_card", "visa"
    ],
    "correspondence": [
        "letter", "email", "memo", "notification"
    ],
    "report": [
        "financial_report", "audit_report", "credit_report",
        "account_statement"
    ]
}

# Flattened document types for easy lookup
ALL_DOCUMENT_TYPES = {doc_type for types in DOCUMENT_CATEGORIES.values() for doc_type in types}

# Keywords for document type classification
DOCUMENT_TYPE_KEYWORDS = {
    # Financial documents
    "invoice": [
        "invoice", "bill", "billing", "invoice #", "invoice no", "invoice number", 
        "invoice date", "due date", "payment terms", "amount due", "total due"
    ],
    "receipt": [
        "receipt", "sales receipt", "purchase receipt", "payment received", 
        "thank you for your purchase", "merchant", "transaction"
    ],
    "bank_statement": [
        "bank statement", "account statement", "account summary", "balance", 
        "deposits", "withdrawals", "transactions", "beginning balance", "ending balance"
    ],
    "credit_card_statement": [
        "credit card statement", "statement", "card number", "card ending in", 
        "payment due date", "minimum payment", "credit limit"
    ],
    "purchase_order": [
        "purchase order", "p.o.", "po #", "po number", "order date", 
        "ship to", "bill to", "requisition"
    ],
    "quote": [
        "quote", "quotation", "estimate", "proposal", "validity", "price quote"
    ],
    "tax_form": [
        "tax form", "tax return", "tax statement", "form 1040", "form w-2", 
        "form 1099", "tax year", "federal tax"
    ],
    "payslip": [
        "payslip", "pay stub", "salary", "wages", "earnings", "deductions", 
        "net pay", "gross pay", "pay period"
    ],
    
    # Legal documents
    "contract": [
        "contract", "agreement", "terms and conditions", "parties", "hereby agree", 
        "effective date", "obligations", "termination", "governing law"
    ],
    "agreement": [
        "agreement", "between the parties", "hereby", "whereas", "term", 
        "executed", "signed and delivered"
    ],
    "nda": [
        "non-disclosure agreement", "nda", "confidential information", 
        "confidentiality", "trade secret", "proprietary information"
    ],
    "terms_of_service": [
        "terms of service", "terms and conditions", "terms of use", 
        "user agreement", "service agreement"
    ],
    "policy": [
        "policy", "insurance policy", "policy number", "coverage", 
        "premium", "insured", "policyholder", "effective date"
    ],
    "certificate": [
        "certificate", "certification", "certified", "attested", "validity", 
        "authorized signatory"
    ],
    
    # Identification documents
    "passport": [
        "passport", "nationality", "date of issue", "date of expiry", 
        "passport number", "issuing authority"
    ],
    "drivers_license": [
        "driver's license", "driving licence", "driver license", "class", 
        "restrictions", "endorsements", "donor"
    ],
    "id_card": [
        "identification card", "id card", "identity", "identification number"
    ],
    "visa": [
        "visa", "entry permit", "duration of stay", "valid until", 
        "issuing authority", "visa type"
    ],
    
    # Other document types
    "letter": [
        "dear", "sincerely", "regards", "to whom it may concern", 
        "letter", "date", "re:", "subject:"
    ],
    "email": [
        "from:", "to:", "cc:", "bcc:", "subject:", "sent:", "received:", 
        "forwarded", "replied"
    ],
    "report": [
        "report", "summary", "findings", "analysis", "conclusion", 
        "recommendation", "executive summary"
    ]
}

class DocumentClassifier:
    """Document classifier for advanced document type detection."""
    
    def __init__(self, 
                model_path: Optional[str] = None,
                confidence_threshold: float = 0.6,
                use_html_classifier: bool = False):
        """
        Initialize document classifier.
        
        Args:
            model_path: Path to optional pre-trained model data
            confidence_threshold: Minimum confidence for classification
            use_html_classifier: Whether to use HTML-specific classifier
        """
        self.confidence_threshold = confidence_threshold
        self.use_html_classifier = use_html_classifier
        
        # Load model data if available
        self.model_data = {}
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    self.model_data = json.load(f)
                    logger.info(f"Loaded classifier model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        
        # Track classification history for this session
        self.classification_history: List[Dict[str, Any]] = []
    
    def classify_document(self, 
                         file_path: str, 
                         extract_text: bool = True) -> Dict[str, Any]:
        """
        Classify a document to determine its type.
        
        Args:
            file_path: Path to the document
            extract_text: Whether to extract and include text in the result
            
        Returns:
            Dict with document classification
        """
        try:
            # Get file info and determine basic type
            mime_type = self._get_mime_type(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Extract text based on file type
            text = ""
            if extract_text:
                text = self._extract_text(file_path, mime_type)
            
            # Get basic document metadata
            metadata = self._extract_metadata(file_path, mime_type)
            
            # Classify document based on content and metadata
            classification = self._classify_by_content(text, metadata, mime_type)
            
            # Add confidence scores
            classification = self._add_confidence_scores(classification, text)
            
            # Format the result
            result = {
                "status": "success",
                "file_path": file_path,
                "mime_type": mime_type,
                "file_extension": file_ext,
                "document_type": classification["document_type"],
                "document_category": classification["document_category"],
                "confidence": classification["confidence"],
                "classification_method": classification["method"],
                "metadata": metadata
            }
            
            # Add text if extracted
            if extract_text:
                # Truncate text if it's too large
                if len(text) > 5000:
                    result["text"] = text[:5000] + "..."
                else:
                    result["text"] = text
            
            # Add to classification history
            self._add_to_history(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error classifying document {file_path}: {str(e)}")
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e),
                "document_type": "unknown",
                "confidence": 0.0
            }
    
    def batch_classify_documents(self, 
                                file_paths: List[str], 
                                extract_text: bool = False) -> Dict[str, Any]:
        """
        Classify multiple documents in batch.
        
        Args:
            file_paths: List of document paths
            extract_text: Whether to extract and include text
            
        Returns:
            Dict with batch classification results
        """
        start_time = datetime.now()
        results = []
        
        for file_path in file_paths:
            try:
                result = self.classify_document(file_path, extract_text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying {file_path}: {str(e)}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "error": str(e),
                    "document_type": "unknown",
                    "confidence": 0.0
                })
        
        # Count by document type
        doc_types = Counter([r.get("document_type", "unknown") for r in results])
        
        return {
            "status": "success",
            "total_documents": len(file_paths),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "document_types": dict(doc_types),
            "results": results
        }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get statistics about classifications performed in this session.
        
        Returns:
            Dict with classification statistics
        """
        if not self.classification_history:
            return {
                "status": "info",
                "message": "No classifications have been performed yet",
                "total_documents": 0
            }
        
        # Count by document type and category
        doc_types = Counter([c.get("document_type", "unknown") for c in self.classification_history])
        categories = Counter([c.get("document_category", "unknown") for c in self.classification_history])
        
        # Calculate average confidence
        avg_confidence = sum(c.get("confidence", 0) for c in self.classification_history) / len(self.classification_history)
        
        # Count successful vs. error classifications
        successful = sum(1 for c in self.classification_history if c.get("status") == "success")
        
        return {
            "status": "success",
            "total_documents": len(self.classification_history),
            "successful_classifications": successful,
            "failed_classifications": len(self.classification_history) - successful,
            "document_types": dict(doc_types),
            "document_categories": dict(categories),
            "average_confidence": avg_confidence
        }
    
    # Private helper methods
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type of file."""
        import magic
        return magic.from_file(file_path, mime=True)
    
    def _extract_text(self, file_path: str, mime_type: str) -> str:
        """Extract text from document based on mime type."""
        if mime_type == "application/pdf":
            return self._extract_pdf_text(file_path)
        elif mime_type.startswith("image/"):
            # For images, we'd ideally use OCR but that would require external dependencies
            # In a production system, integrate with Vision API or Tesseract
            return f"[Image file - OCR would be applied in production: {file_path}]"
        elif mime_type.startswith("text/"):
            # For text files, read directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Could not read text file {file_path}: {e}")
                    return ""
        else:
            # For other file types, return empty text
            return ""
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF document."""
        text = ""
        try:
            with fitz.open(file_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text += page.get_text()
            return text
        except Exception as e:
            logger.warning(f"Could not extract text from PDF {file_path}: {e}")
            return text
    
    def _extract_metadata(self, file_path: str, mime_type: str) -> Dict[str, Any]:
        """Extract file metadata based on mime type."""
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }
        
        # Extract additional metadata based on file type
        if mime_type == "application/pdf":
            try:
                with fitz.open(file_path) as pdf:
                    if hasattr(pdf, "metadata") and pdf.metadata:
                        pdf_meta = pdf.metadata
                        metadata.update({
                            "title": pdf_meta.get("title", ""),
                            "author": pdf_meta.get("author", ""),
                            "subject": pdf_meta.get("subject", ""),
                            "keywords": pdf_meta.get("keywords", ""),
                            "creator": pdf_meta.get("creator", ""),
                            "producer": pdf_meta.get("producer", ""),
                            "page_count": len(pdf)
                        })
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata from {file_path}: {e}")
        
        elif mime_type.startswith("image/"):
            try:
                with Image.open(file_path) as img:
                    metadata.update({
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode
                    })
                    
                    # Extract EXIF data if available
                    if hasattr(img, "info") and "exif" in img.info:
                        metadata["has_exif"] = True
            except Exception as e:
                logger.warning(f"Could not extract image metadata from {file_path}: {e}")
        
        return metadata
    
    def _classify_by_content(self, text: str, metadata: Dict[str, Any], 
                            mime_type: str) -> Dict[str, Any]:
        """Classify document based on content and metadata."""
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Default classification info
        classification = {
            "document_type": "unknown",
            "document_category": "unknown",
            "confidence": 0.0,
            "method": "default"
        }
        
        # Calculate keyword matches for each document type
        type_scores = {}
        for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
            # Count occurrences of each keyword in the text
            matches = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            # Calculate score based on matches and number of keywords
            if matches > 0:
                type_scores[doc_type] = matches / len(keywords)
        
        # Find document type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            doc_type = best_type[0]
            confidence = min(best_type[1], 1.0)  # Cap confidence at 1.0
            
            # Only assign if confidence is above threshold
            if confidence >= self.confidence_threshold:
                classification["document_type"] = doc_type
                classification["confidence"] = confidence
                classification["method"] = "keyword_matching"
                
                # Determine category from type
                for category, types in DOCUMENT_CATEGORIES.items():
                    if doc_type in types:
                        classification["document_category"] = category
                        break
        
        # Special case for PDF metadata-based classification
        if mime_type == "application/pdf" and classification["document_type"] == "unknown":
            if "title" in metadata and metadata["title"]:
                title_lower = metadata["title"].lower()
                
                # Check if title contains document type indicator
                for doc_type in ALL_DOCUMENT_TYPES:
                    if doc_type.replace("_", " ") in title_lower:
                        classification["document_type"] = doc_type
                        classification["confidence"] = 0.7  # Moderate confidence
                        classification["method"] = "pdf_metadata"
                        
                        # Determine category
                        for category, types in DOCUMENT_CATEGORIES.items():
                            if doc_type in types:
                                classification["document_category"] = category
                                break
                        break
        
        # If still unknown but filename has hints, use filename
        if classification["document_type"] == "unknown":
            filename_lower = metadata["file_name"].lower()
            
            for doc_type in ALL_DOCUMENT_TYPES:
                if doc_type.replace("_", "") in filename_lower.replace("_", "").replace(" ", ""):
                    classification["document_type"] = doc_type
                    classification["confidence"] = 0.6  # Lower confidence
                    classification["method"] = "filename"
                    
                    # Determine category
                    for category, types in DOCUMENT_CATEGORIES.items():
                        if doc_type in types:
                            classification["document_category"] = category
                            break
                    break
        
        return classification
    
    def _add_confidence_scores(self, classification: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Add detailed confidence scores for the classification."""
        doc_type = classification["document_type"]
        
        # If unknown, no detailed scores
        if doc_type == "unknown":
            return classification
        
        # Calculate additional confidence metrics
        confidence_details = {}
        
        # Text length metric: longer text generally gives more reliable classification
        text_length = len(text)
        if text_length > 0:
            text_length_score = min(text_length / 1000, 1.0)  # Cap at 1.0
            confidence_details["text_length_score"] = text_length_score
        
        # Keyword density: percentage of text composed of keywords for this type
        if doc_type in DOCUMENT_TYPE_KEYWORDS and text_length > 0:
            keywords = DOCUMENT_TYPE_KEYWORDS[doc_type]
            keyword_chars = sum(len(kw) * text.lower().count(kw.lower()) for kw in keywords)
            keyword_density = keyword_chars / text_length if text_length > 0 else 0
            confidence_details["keyword_density"] = keyword_density
        
        # Add confidence details to classification
        classification["confidence_details"] = confidence_details
        
        return classification
    
    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """Add classification result to history."""
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        # Add to history, limiting size to prevent memory issues
        self.classification_history.append(result)
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]


# Standalone functions for document classification

def classify_document(file_path: str) -> Dict[str, Any]:
    """
    Classify a document to determine its type.
    This is a wrapper function for compatibility.
    """
    classifier = DocumentClassifier()
    return classifier.classify_document(file_path)

def batch_classify_documents(file_paths: List[str]) -> Dict[str, Any]:
    """
    Classify multiple documents in batch.
    This is a wrapper function for compatibility.
    """
    classifier = DocumentClassifier()
    return classifier.batch_classify_documents(file_paths)

def create_document_classifier(confidence_threshold: float = 0.6) -> DocumentClassifier:
    """
    Create a document classifier instance.
    This is a helper function to create a classifier with custom settings.
    """
    return DocumentClassifier(confidence_threshold=confidence_threshold)