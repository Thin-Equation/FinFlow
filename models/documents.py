"""
Data models for financial documents in the FinFlow system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class DocumentStatus(str, Enum):
    """Status of a financial document."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"
    OVERDUE = "overdue"
    VOID = "void"


class DocumentType(str, Enum):
    """Types of financial documents supported."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    EXPENSE = "expense"
    BANK_STATEMENT = "bank_statement"
    FINANCIAL_STATEMENT = "financial_statement"
    TAX_FORM = "tax_form"
    PURCHASE_ORDER = "purchase_order"
    CONTRACT = "contract"
    OTHER = "other"


class Currency(str, Enum):
    """Common currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    # Add more currencies as needed


class Address(BaseModel):
    """Physical address model."""
    street_address: str
    city: str
    state: Optional[str] = None
    postal_code: str
    country: str


class Organization(BaseModel):
    """Organization or company data model."""
    name: str
    tax_id: Optional[str] = None
    address: Optional[Address] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None


class LineItem(BaseModel):
    """Line item in a financial document."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total_amount: float = Field(ge=0)
    tax_amount: Optional[float] = Field(default=0, ge=0)
    tax_rate: Optional[float] = Field(default=0, ge=0)
    account_code: Optional[str] = None
    
    @field_validator('total_amount', pre=True, always=True)
    def calculate_total_amount(cls, v, values):
        """Calculate total amount if not provided."""
        if v == 0 and 'quantity' in values and 'unit_price' in values:
            return values['quantity'] * values['unit_price']
        return v


class PaymentTerms(BaseModel):
    """Payment terms for a document."""
    due_days: int = Field(ge=0)
    discount_days: Optional[int] = None
    discount_percent: Optional[float] = None
    payment_method: Optional[str] = None
    
    @property
    def description(self) -> str:
        """Generate a human-readable description of payment terms."""
        result = f"Net {self.due_days}"
        if self.discount_days is not None and self.discount_percent is not None:
            result += f", {self.discount_percent}% {self.discount_days}"
        return result


class TaxInfo(BaseModel):
    """Tax information for a document."""
    tax_id: Optional[str] = None
    tax_type: Optional[str] = None  # VAT, GST, Sales Tax, etc.
    total_tax_amount: float = Field(default=0, ge=0)
    tax_breakdown: Optional[Dict[str, float]] = None  # Tax type to amount


class FinancialDocument(BaseModel):
    """Base financial document model."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_type: DocumentType
    document_number: str
    status: DocumentStatus = DocumentStatus.DRAFT
    issue_date: datetime
    due_date: Optional[datetime] = None
    currency: Currency = Currency.USD
    total_amount: float = Field(ge=0)
    subtotal: float = Field(ge=0)
    
    issuer: Organization
    recipient: Optional[Organization] = None
    
    line_items: List[LineItem] = []
    payment_terms: Optional[PaymentTerms] = None
    tax_info: Optional[TaxInfo] = None
    
    notes: Optional[str] = None
    attachments: List[str] = []  # List of file URIs
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source_file: Optional[str] = None  # Original file path/URI
    confidence_score: Optional[float] = None  # Document AI confidence
    
    @validator('total_amount', pre=True, always=True)
    def validate_total_amount(cls, v, values):
        """Validate that total_amount equals sum of line items if line items exist."""
        if v == 0 and 'line_items' in values and values['line_items']:
            return sum(item.total_amount for item in values['line_items'])
        return v
    
    class Config:
        validate_assignment = True


class Invoice(FinancialDocument):
    """Invoice-specific document model."""
    document_type: DocumentType = DocumentType.INVOICE
    purchase_order_ref: Optional[str] = None
    shipping_info: Optional[dict] = None
    payment_instructions: Optional[str] = None


class Receipt(FinancialDocument):
    """Receipt-specific document model."""
    document_type: DocumentType = DocumentType.RECEIPT
    payment_method_used: Optional[str] = None
    transaction_id: Optional[str] = None


class Expense(FinancialDocument):
    """Expense report document model."""
    document_type: DocumentType = DocumentType.EXPENSE
    employee_id: Optional[str] = None
    expense_category: Optional[str] = None
    reimbursable: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class BankStatement(FinancialDocument):
    """Bank statement document model."""
    document_type: DocumentType = DocumentType.BANK_STATEMENT
    account_number: str
    period_start: datetime
    period_end: datetime
    opening_balance: float
    closing_balance: float
    transactions: List[Dict] = []


class DocumentRelationship(BaseModel):
    """Relationship between two financial documents."""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_document_id: str
    target_document_id: str
    relationship_type: str  # e.g., "invoice_to_receipt", "po_to_invoice"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = {}
