"""
Base data model for all FinFlow models.
"""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field
import uuid


class FinFlowModel(BaseModel):
    """Base model with common fields for all FinFlow models."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    metadata: Dict = {}
    
    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
        arbitrary_types_allowed = True
