"""
Session state management utilities for the FinFlow system.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
import json
import uuid
import time
from datetime import datetime

T = TypeVar('T')

class SessionState:
    """
    Manages state persistence across agent interactions.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session state with optional ID.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.data = {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            key: Data key
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with key or default
        """
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in session state.
        
        Args:
            key: Data key
            value: Data value
        """
        self.data[key] = value
        self.last_updated = datetime.now().isoformat()
    
    def update(self, values: Dict[str, Any]) -> None:
        """
        Update multiple values in session state.
        
        Args:
            values: Dictionary of key-value pairs to update
        """
        self.data.update(values)
        self.last_updated = datetime.now().isoformat()
    
    def delete(self, key: str) -> None:
        """
        Delete key from session state.
        
        Args:
            key: Data key to delete
        """
        if key in self.data:
            del self.data[key]
            self.last_updated = datetime.now().isoformat()
    
    def clear(self) -> None:
        """Clear all session data."""
        self.data = {}
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session state to dictionary.
        
        Returns:
            Dict containing session state
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """
        Create SessionState from dictionary.
        
        Args:
            data: Dictionary containing session data
            
        Returns:
            SessionState instance
        """
        session = cls(session_id=data.get("session_id"))
        session.created_at = data.get("created_at", session.created_at)
        session.last_updated = data.get("last_updated", session.last_updated)
        session.data = data.get("data", {})
        return session
    
    def to_json(self) -> str:
        """
        Convert session state to JSON.
        
        Returns:
            JSON string representation of session state
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionState':
        """
        Create SessionState from JSON string.
        
        Args:
            json_str: JSON string containing session data
            
        Returns:
            SessionState instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


def get_or_create_session_state(context: Dict[str, Any]) -> SessionState:
    """
    Get existing or create new SessionState from context.
    
    Args:
        context: Context dictionary
        
    Returns:
        SessionState instance
    """
    if "session_state" not in context:
        context["session_state"] = SessionState().to_dict()
    
    if isinstance(context["session_state"], dict) and "session_id" in context["session_state"]:
        return SessionState.from_dict(context["session_state"])
    
    # If session_state exists but is not in the expected format, create new
    session = SessionState()
    context["session_state"] = session.to_dict()
    return session


def update_context_session_state(context: Dict[str, Any], session: SessionState) -> Dict[str, Any]:
    """
    Update context with session state.
    
    Args:
        context: Context dictionary
        session: SessionState instance
        
    Returns:
        Updated context dictionary
    """
    context["session_state"] = session.to_dict()
    return context
