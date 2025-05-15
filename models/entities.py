"""
Data models for financial entities in the FinFlow system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
import uuid

from models.documents import Address


class EntityType(str, Enum):
    """Types of financial entities."""
    VENDOR = "vendor"
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    CONTRACTOR = "contractor"
    BANK = "bank"
    TAX_AUTHORITY = "tax_authority"
    OTHER = "other"


class PaymentMethod(str, Enum):
    """Payment methods."""
    CREDIT_CARD = "credit_card"
    ACH = "ach"
    WIRE = "wire"
    CHECK = "check"
    CASH = "cash"
    OTHER = "other"


class FinancialEntity(BaseModel):
    """Financial entity base model (vendor, customer, etc.)."""
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    name: str
    tax_id: Optional[str] = None
    
    # Contact information
    address: Optional[Address] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    
    # Banking information
    payment_methods: List[PaymentMethod] = []
    bank_accounts: List[Dict] = []
    
    # Classification and metadata
    tags: Set[str] = set()
    industry: Optional[str] = None
    notes: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Document history
    document_ids: List[str] = []
    
    class Config:
        validate_assignment = True


class VendorEntity(FinancialEntity):
    """Vendor-specific entity model."""
    entity_type: EntityType = EntityType.VENDOR
    payment_terms: Optional[str] = None
    vendor_category: Optional[str] = None
    preferred: bool = False
    approved: bool = True
    

class CustomerEntity(FinancialEntity):
    """Customer-specific entity model."""
    entity_type: EntityType = EntityType.CUSTOMER
    credit_limit: Optional[float] = None
    payment_terms: Optional[str] = None
    customer_category: Optional[str] = None


class AccountCode(BaseModel):
    """Chart of accounts code model."""
    code: str
    name: str
    description: Optional[str] = None
    account_type: str  # asset, liability, equity, revenue, expense
    parent_code: Optional[str] = None
    is_active: bool = True


class FiscalPeriod(BaseModel):
    """Fiscal period model."""
    period_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_date: datetime
    end_date: datetime
    is_closed: bool = False
    year: int
    quarter: Optional[int] = None
    month: Optional[int] = None
