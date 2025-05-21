"""
Production-level document classification for the FinFlow system.

This module provides sophisticated document classification capabilities to identify
document types accurately and route them to appropriate processors.
"""

import os
import logging
import re
from typing import Any, Dict, List, Optional, Union
from collections import Counter
import json
import time
from datetime import datetime

# For text extraction and classification
import fitz  # PyMuPDF
from PIL import Image


# Configure logging
logger = logging.getLogger("finflow.tools.document_classification")

# Document categories with comprehensive document types
DOCUMENT_CATEGORIES = {
    "financial": [
        "invoice", "receipt", "bank_statement", "credit_card_statement",
        "purchase_order", "quote", "tax_form", "payslip", "balance_sheet",
        "profit_loss", "cash_flow_statement", "expense_report", "remittance_advice",
        "bill_of_lading", "packing_slip", "credit_note", "debit_note"
    ],
    "legal": [
        "contract", "agreement", "nda", "terms_of_service", "policy", "certificate",
        "power_of_attorney", "incorporation_document", "patent", "trademark_registration",
        "license", "regulatory_filing", "deed", "will", "trust_document"
    ],
    "identification": [
        "passport", "drivers_license", "id_card", "visa", "birth_certificate",
        "social_security_card", "health_insurance_card", "vehicle_registration",
        "professional_license", "membership_card", "student_id"
    ],
    "correspondence": [
        "letter", "email", "memo", "notification", "invoice_cover_letter",
        "demand_letter", "formal_request", "complaint", "response_letter",
        "acknowledgment", "invitation"
    ],
    "report": [
        "financial_report", "audit_report", "credit_report", "account_statement",
        "business_plan", "market_analysis", "project_report", "valuation_report",
        "compliance_report", "assessment_report", "technical_report"
    ],
    "other": [
        "application_form", "questionnaire", "survey", "brochure", "catalog",
        "newsletter", "manual", "transcript", "certificate_of_completion",
        "resume", "cv", "proposal", "specification", "white_paper"
    ]
}

# Flattened document types for easy lookup
ALL_DOCUMENT_TYPES = {doc_type for types in DOCUMENT_CATEGORIES.values() for doc_type in types}

# Keywords for document type classification - comprehensive set of identifying terms
DOCUMENT_TYPE_KEYWORDS = {
    # Financial documents - comprehensive detection patterns
    "invoice": [
        "invoice", "bill", "billing", "invoice #", "invoice no", "invoice number", 
        "invoice date", "due date", "payment terms", "amount due", "total due",
        "invoice total", "bill to", "remittance", "tax invoice", "vat invoice",
        "balance due", "payment details", "account number", "customer id",
        "item description", "quantity", "unit price", "subtotal", "tax rate",
        "purchase order number", "order number", "shipping", "shipping address",
        "billing address", "terms", "net", "net 30", "payment due", "payment method"
    ],
    "receipt": [
        "receipt", "sales receipt", "purchase receipt", "payment received", 
        "thank you for your purchase", "merchant", "transaction", "cashier",
        "store", "item", "qty", "amount", "total", "paid", "change", "cash",
        "credit card", "debit card", "transaction id", "terminal id", "card ending",
        "return policy", "customer copy", "merchant copy", "tax", "subtotal",
        "discount", "store number", "register", "sale", "item count", "auth code"
    ],
    "bank_statement": [
        "bank statement", "account statement", "account summary", "balance", 
        "deposits", "withdrawals", "transactions", "beginning balance", "ending balance",
        "account number", "statement period", "statement date", "interest earned",
        "fees charged", "overdraft", "available balance", "direct deposit", "atm withdrawal",
        "check", "transfer", "electronic", "fee", "interest", "branch", "routing number",
        "credit", "debit", "withdrawal", "deposit", "transaction details", "date posted"
    ],
    "credit_card_statement": [
        "credit card statement", "statement", "card number", "card ending in", 
        "payment due date", "minimum payment", "credit limit", "available credit",
        "annual percentage rate", "apr", "previous balance", "new balance",
        "finance charge", "late payment fee", "cash advance", "purchase",
        "billing period", "billing date", "credit line", "rewards", "points earned",
        "foreign transaction fee", "transaction date", "posting date", "merchant name"
    ],
    "purchase_order": [
        "purchase order", "p.o.", "po #", "po number", "order date", 
        "ship to", "bill to", "requisition", "delivery date", "shipping method",
        "requested by", "authorized by", "vendor", "supplier", "item number",
        "description", "quantity ordered", "unit cost", "extended cost", "subtotal",
        "shipping instructions", "terms and conditions", "confirmation number",
        "buyer", "shipping terms", "payment terms", "delivery instructions"
    ],
    "quote": [
        "quote", "quotation", "estimate", "proposal", "validity", "price quote",
        "quote number", "quote date", "valid until", "expiration date", "prepared by",
        "discount", "terms and conditions", "acceptance", "project scope", "pricing",
        "rate", "hourly rate", "quantity", "description", "unit price", "total price",
        "subtotal", "tax", "grand total", "optional items", "line items", "proposal for"
    ],
    "tax_form": [
        "tax form", "tax return", "tax statement", "form 1040", "form w-2", 
        "form 1099", "tax year", "federal tax", "state tax", "income tax",
        "taxable income", "tax deduction", "tax credit", "filing status",
        "social security number", "employer identification number", "ein",
        "withholding", "exemptions", "adjusted gross income", "agi", "schedule",
        "signature", "preparer", "tax due", "refund", "tax liability"
    ],
    "payslip": [
        "payslip", "pay stub", "salary", "wages", "earnings", "deductions", 
        "net pay", "gross pay", "pay period", "employee id", "employee name",
        "employer name", "tax withholding", "fica", "medicare", "social security",
        "overtime", "regular hours", "overtime hours", "ytd", "year to date",
        "federal tax", "state tax", "local tax", "benefits", "retirement", "401k",
        "health insurance", "dental insurance", "vision insurance", "fsa", "hsa"
    ],
    
    # Legal documents - comprehensive patterns
    "contract": [
        "contract", "agreement", "terms and conditions", "parties", "hereby agree", 
        "effective date", "obligations", "termination", "governing law",
        "warranties", "representations", "indemnification", "force majeure", 
        "severability", "entire agreement", "binding effect", "counterparts",
        "witness whereof", "executed", "breach", "remedies", "jurisdiction",
        "venue", "arbitration", "confidentiality", "term", "termination"
    ],
    "agreement": [
        "agreement", "between the parties", "hereby", "whereas", "term", 
        "executed", "signed and delivered", "now therefore", "in consideration of",
        "consent", "assigns", "successors", "herein", "hereof", "hereunder",
        "provisions", "covenant", "recitals", "witnesseth", "date of execution"
    ],
    "terms_of_service": [
        "terms of service", "terms and conditions", "terms of use", 
        "user agreement", "service agreement"
    ]
}

# Regular expressions for identifying document types
DOCUMENT_TYPE_PATTERNS = {
    "invoice": re.compile(r'(?i)(invoice\s+(?:no|number|#)|tax\s+invoice|original\s+invoice|billing\s+statement)'),
    "receipt": re.compile(r'(?i)(receipt|sales\s+receipt|payment\s+receipt|purchase\s+receipt)'),
    "bank_statement": re.compile(r'(?i)(bank\s+statement|account\s+statement|statement\s+of\s+account|statement\s+period)'),
    "credit_card_statement": re.compile(r'(?i)(credit\s+card\s+statement|card\s+ending|card\s+member\s+statement|card\s+account|credit\s+line)'),
    "purchase_order": re.compile(r'(?i)(purchase\s+order|p\.?o\.?\s+(?:no|number|#))'),
    "tax_form": re.compile(r'(?i)(form\s+(?:1040|1099|w-2|w-4)|tax\s+(?:form|return|statement))'),
    "payslip": re.compile(r'(?i)(pay\s+(?:slip|stub|statement)|earnings\s+statement|salary\s+statement)'),
    "contract": re.compile(r'(?i)(contract|agreement|between\s+parties|terms\s+and\s+conditions)'),
    "quote": re.compile(r'(?i)(quotation|quote|estimate|proposal|price\s+quote)')
}

# Common layouts and structures for document types
DOCUMENT_LAYOUTS = {
    "invoice": ["header_with_invoice", "bill_to_ship_to", "line_items_table", "totals_section"],
    "receipt": ["merchant_header", "items_list", "total_and_payment"],
    "bank_statement": ["bank_header", "account_info", "transactions_table", "summary_section"],
    "credit_card_statement": ["card_issuer_header", "account_summary", "transactions_table", "payment_info"],
    "purchase_order": ["po_header", "vendor_info", "ship_to_info", "items_table"],
    "tax_form": ["tax_authority_header", "taxpayer_info", "income_sections", "deductions_section"],
    "payslip": ["employer_header", "employee_info", "earnings_section", "deductions_section"]
}

def extract_document_features(text: str) -> Dict[str, Any]:
    """
    Extract features from document text for classification.
    
    Args:
        text: Document text content
        
    Returns:
        Dictionary of document features
    """
    if not text:
        return {
            "word_count": 0,
            "line_count": 0,
            "has_table": False,
            "has_amount": False,
            "has_date": False,
            "keywords": {},
            "patterns": {}
        }
    
    # Basic text features
    lines = text.split('\n')
    words = text.split()
    
    # Check for common features
    has_amount = bool(re.search(r'\$\s*[\d,]+\.\d{2}', text) or re.search(r'(?i)(total|amount|sum):?\s*[\d,]+\.\d{2}', text))
    has_date = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) or re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b', text, re.I))
    has_table = bool(
        # Check for tabular structures with multiple columns
        len([line for line in lines if line.count("  ") >= 2]) >= 3 or  
        len([line for line in lines if line.count("\t") >= 2]) >= 3 or
        len([line for line in lines if line.count("|") >= 2]) >= 3
    )
    
    # Check for keywords by document type
    keyword_matches = {}
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        if matches > 0:
            keyword_matches[doc_type] = matches
    
    # Check for regex patterns
    pattern_matches = {}
    for doc_type, pattern in DOCUMENT_TYPE_PATTERNS.items():
        matches = len(pattern.findall(text))
        if matches > 0:
            pattern_matches[doc_type] = matches
    
    # Return all extracted features
    return {
        "word_count": len(words),
        "line_count": len(lines),
        "has_table": has_table,
        "has_amount": has_amount,
        "has_date": has_date,
        "keywords": keyword_matches,
        "patterns": pattern_matches
    }

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from PDF content.
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        Extracted text from PDF
    """
    try:
        # Create a temporary file to use with PyMuPDF
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(pdf_content)
            temp_path = temp.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(temp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Remove the temporary file
        os.unlink(temp_path)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_image(image_content: bytes) -> str:
    """
    Extract text from image using OCR.
    
    Args:
        image_content: Image file content as bytes
        
    Returns:
        Extracted text from image
    """
    try:
        # For production, we'd integrate with a proper OCR service
        # Here's a stub that would use Google Cloud Vision or similar
        logger.warning("OCR functionality requires Google Cloud Vision or similar service")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""

def extract_text_from_document(document_path: str) -> str:
    """
    Extract text content from a document.
    
    Args:
        document_path: Path to the document
        
    Returns:
        Extracted text content
    """
    try:
        if not os.path.exists(document_path):
            logger.error(f"Document not found: {document_path}")
            return ""
            
        # Get file extension
        _, ext = os.path.splitext(document_path.lower())
        
        # PDF extraction
        if ext == '.pdf':
            try:
                pdf_text = []
                with fitz.open(document_path) as pdf:
                    for page_num in range(len(pdf)):
                        page = pdf[page_num]
                        pdf_text.append(page.get_text())
                return "\n".join(pdf_text)
            except Exception as e:
                logger.error(f"Error extracting text from PDF {document_path}: {str(e)}")
                return ""
                
        # Image extraction (just returning placeholder - in production would use OCR)
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
            logger.warning(f"Text extraction from images requires OCR, returning placeholder for {document_path}")
            return f"IMAGE_CONTENT_{document_path}"
            
        # Text file extraction
        elif ext in ['.txt', '.csv', '.json', '.xml']:
            with open(document_path, 'r', errors='ignore') as f:
                return f.read()
        
        # Office documents - in production would use appropriate libraries
        elif ext in ['.doc', '.docx', '.xls', '.xlsx']:
            logger.warning(f"Office document extraction not implemented for {document_path}")
            return f"OFFICE_DOCUMENT_CONTENT_{document_path}"
            
        else:
            logger.warning(f"Unsupported file type for text extraction: {ext}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {document_path}: {str(e)}")
        return ""

# Field extraction patterns for different document types
FIELD_EXTRACTION_PATTERNS = {
    "invoice": {
        "invoice_number": [
            r"invoice\s*#?\s*[:]?\s*([\w\-]+)",
            r"invoice number\s*[:]?\s*([\w\-]+)",
            r"inv[./#]\s*([\w\-]+)",
            r"bill number[:]?\s*([\w\-]+)"
        ],
        "invoice_date": [
            r"invoice date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"invoice date[:]?\s*(\w+\s+\d{1,2},?\s*\d{4})"
        ],
        "due_date": [
            r"due date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"payment due[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"due date[:]?\s*(\w+\s+\d{1,2},?\s*\d{4})"
        ],
        "total_amount": [
            r"total[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"amount due[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"total amount[:]?\s*[\$€£¥]?([\d,]+\.\d{2})"
        ],
        "vendor": [
            r"from[:]?\s*([A-Za-z0-9\s,\.]+)",
            r"vendor[:]?\s*([A-Za-z0-9\s,\.]+)",
            r"supplier[:]?\s*([A-Za-z0-9\s,\.]+)"
        ]
    },
    "receipt": {
        "receipt_number": [
            r"receipt\s*#?\s*[:]?\s*([\w\-]+)",
            r"receipt number[:]?\s*([\w\-]+)",
            r"transaction\s*#?\s*[:]?\s*([\w\-]+)"
        ],
        "date": [
            r"date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"receipt date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"transaction date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
        ],
        "total_amount": [
            r"total[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"amount[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"total due[:]?\s*[\$€£¥]?([\d,]+\.\d{2})"
        ],
        "merchant": [
            r"merchant[:]?\s*([A-Za-z0-9\s,\.]+)",
            r"store[:]?\s*([A-Za-z0-9\s,\.]+)",
            r"business[:]?\s*([A-Za-z0-9\s,\.]+)"
        ]
    },
    "bank_statement": {
        "account_number": [
            r"account\s*#?\s*[:]?\s*([\w\-]+)",
            r"account number[:]?\s*([\w\-]+)"
        ],
        "statement_date": [
            r"statement date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"date[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
        ],
        "opening_balance": [
            r"opening balance[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"beginning balance[:]?\s*[\$€£¥]?([\d,]+\.\d{2})"
        ],
        "closing_balance": [
            r"closing balance[:]?\s*[\$€£¥]?([\d,]+\.\d{2})",
            r"ending balance[:]?\s*[\$€£¥]?([\d,]+\.\d{2})"
        ]
    }
}

def extract_document_fields(document_path: str, document_type: str) -> Dict[str, Any]:
    """
    Extract key fields from a document based on its type using regex patterns.
    
    Args:
        document_path: Path to the document
        document_type: Type of document (invoice, receipt, etc.)
        
    Returns:
        Dictionary of extracted fields
    """
    extracted_fields = {}
    
    try:
        # Extract text from document
        text = extract_text_from_document(document_path)
        if not text:
            logger.warning(f"Could not extract text from {document_path}")
            return extracted_fields
            
        # Get patterns for this document type
        patterns = FIELD_EXTRACTION_PATTERNS.get(document_type, {})
        if not patterns:
            logger.warning(f"No extraction patterns defined for document type {document_type}")
            return extracted_fields
            
        # Extract fields using regex patterns
        for field, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match and match.group(1):
                        # Take first match and clean it
                        value = match.group(1).strip()
                        
                        # Clean and normalize values
                        if "amount" in field.lower() or "balance" in field.lower() or "total" in field.lower():
                            # Remove commas from amounts
                            value = value.replace(',', '')
                            
                        extracted_fields[field] = value
                        break  # Stop after first match for this field
                        
                # If we found a value for this field, move to next field
                if field in extracted_fields:
                    break
                    
        logger.debug(f"Extracted {len(extracted_fields)} fields from {document_path}")
                    
    except Exception as e:
        logger.error(f"Error extracting fields from {document_path}: {str(e)}")
        
    return extracted_fields

def normalize_extracted_data(data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
    """
    Normalize extracted data for consistency.
    
    Args:
        data: Dictionary of extracted fields
        document_type: Type of document for specialized normalization
        
    Returns:
        Normalized data dictionary
    """
    normalized = data.copy()
    
    try:
        # Normalize date fields
        date_fields = ["date", "invoice_date", "due_date", "receipt_date", "statement_date",
                      "transaction_date", "period_start", "period_end"]
        
        for field in [f for f in date_fields if f in normalized]:
            try:
                # Handle various date formats
                date_str = normalized[field]
                # This is where you'd implement proper date parsing
                # For now we'll just ensure the field exists
                pass
            except Exception:
                # If normalization fails, keep original
                pass
                
        # Normalize amount fields
        amount_fields = ["total_amount", "subtotal", "tax_amount", "tip_amount", 
                        "opening_balance", "closing_balance", "amount"]
                        
        for field in [f for f in amount_fields if f in normalized]:
            try:
                # Convert to float and format properly
                amount_str = normalized[field].replace('$', '').replace(',', '')
                normalized[field] = format(float(amount_str), '.2f')
            except Exception:
                # If normalization fails, keep original
                pass
                
        # Document-specific normalization
        if document_type == "invoice":
            # Ensure vendor info is properly structured
            if "vendor" in normalized:
                vendor_name = normalized.pop("vendor")
                normalized["vendor"] = {"name": vendor_name}
                
        elif document_type == "receipt":
            # Ensure merchant info is properly structured
            if "merchant" in normalized:
                merchant_name = normalized.pop("merchant")
                normalized["merchant"] = {"name": merchant_name}
                
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        
    return normalized

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


def classify_document_type(
    content: Union[bytes, str],
    top_n: int = 1,
    confidence_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Classify a document to determine its type.
    
    Args:
        content: Document content as bytes or file path
        top_n: Number of top classifications to return
        confidence_threshold: Minimum confidence threshold for classification
        
    Returns:
        Classification results with document types and confidence scores
    """
    start_time = time.time()
    
    try:
        # Convert file path to content if needed
        if isinstance(content, str) and os.path.exists(content):
            with open(content, 'rb') as f:
                content = f.read()
                
        # Determine file type
        from magic import Magic
        mime = Magic(mime=True)
        mime_type = mime.from_buffer(content)
        
        # Extract text based on file type
        text = ""
        if mime_type == 'application/pdf':
            text = extract_text_from_pdf(content)
        elif mime_type.startswith('image/'):
            text = extract_text_from_image(content)
        elif mime_type.startswith('text/'):
            text = content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else content
        else:
            # Unsupported file type
            return {
                "status": "error",
                "message": f"Unsupported mime type: {mime_type}",
                "classifications": []
            }
        
        # Extract features for classification
        features = extract_document_features(text)
        
        # Calculate scores for each document type
        scores = {}
        for doc_type in ALL_DOCUMENT_TYPES:
            # Initialize score
            score = 0.0
            
            # Score based on keyword matches
            keyword_matches = features["keywords"].get(doc_type, 0)
            score += keyword_matches * 0.15  # 15% weight for keyword matches
            
            # Score based on regex pattern matches
            pattern_matches = features["patterns"].get(doc_type, 0)
            score += pattern_matches * 0.25  # 25% weight for regex patterns
            
            # Additional features specific to document types
            if doc_type in ["invoice", "receipt", "purchase_order"] and features["has_amount"]:
                score += 0.15  # Financial documents typically have amounts
            
            if doc_type in ["invoice", "bank_statement", "credit_card_statement"] and features["has_date"]:
                score += 0.15  # These documents typically have dates
                
            if doc_type in ["invoice", "bank_statement", "credit_card_statement", "purchase_order"] and features["has_table"]:
                score += 0.15  # These documents typically have tabular data
            
            # Only keep scores above threshold
            if score > 0:
                scores[doc_type] = min(score, 0.95)  # Cap at 0.95
        
        # Get top N classifications
        top_classifications = sorted(
            [{"type": doc_type, "confidence": score} for doc_type, score in scores.items()],
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        # Filter by confidence threshold
        classifications = [c for c in top_classifications if c["confidence"] >= confidence_threshold]
        
        # Limit to top N
        classifications = classifications[:top_n]
        
        # Return result
        processing_time = time.time() - start_time
        return {
            "status": "success" if classifications else "unknown",
            "classifications": classifications,
            "mime_type": mime_type,
            "features_extracted": features["word_count"] > 0,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error classifying document: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "classifications": [],
            "processing_time": time.time() - start_time
        }