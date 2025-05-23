"""
Session state management utilities for the FinFlow system.
"""

from typing import Any, Dict, List, Optional, Set, TypeVar, cast
import json
import uuid
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
        self.session_id: str = session_id or str(uuid.uuid4())
        self.created_at: str = datetime.now().isoformat()
        self.last_updated: str = self.created_at
        self.data: Dict[str, Any] = {}
        self.agent_data: Dict[str, Dict[str, Any]] = {}  # Agent-specific data compartments
        self.shared_keys: Set[str] = set()  # Keys that are shared between agents
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            key: Data key
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
        self.data = {}  # Reset to empty dictionary
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
            "data": self.data,
            "agent_data": self.agent_data,
            "shared_keys": list(self.shared_keys)
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
        # Create new session instance
        session = cls()
            
        # Handle session_id
        if "session_id" in data and data["session_id"] is not None:
            try:
                # Try to convert to string
                session.session_id = str(data["session_id"])
            except Exception:
                # Keep original if conversion fails
                pass
                
        # Handle created_at
        if "created_at" in data and data["created_at"] is not None:
            try:
                # Try to get as string
                session.created_at = str(data["created_at"])
            except Exception:
                # Keep original if conversion fails
                pass
                
        # Handle last_updated
        if "last_updated" in data and data["last_updated"] is not None:
            try:
                # Try to get as string
                session.last_updated = str(data["last_updated"])
            except Exception:
                # Keep original if conversion fails
                pass
            
        # Manual data handling to avoid type checking issues
        session.data = {}
        if "data" in data and isinstance(data["data"], dict):
            raw_data = cast(Dict[Any, Any], data["data"])  # Force cast to avoid type checking issues
            
            # Create a copy as we rebuild with string keys
            try:
                # Get keys as a list to iterate over
                keys = [k for k in raw_data.keys()]
                
                # Rebuild with string keys
                for k in keys:
                    if k is not None:
                        key_str = str(k)
                        session.data[key_str] = raw_data[k]
            except Exception:
                # On any error, use empty dict (already initialized above)
                pass
        
        # Handle agent_data
        session.agent_data = {}
        if "agent_data" in data and isinstance(data["agent_data"], dict):
            raw_agent_data = cast(Dict[Any, Any], data["agent_data"])
            
            try:
                # Get agent IDs as a list to iterate over
                agent_ids = [a for a in raw_agent_data.keys()]
                
                # Rebuild with string keys
                for agent_id in agent_ids:
                    if agent_id is not None:
                        agent_id_str = str(agent_id)
                        agent_dict = raw_agent_data[agent_id]
                        
                        if isinstance(agent_dict, dict):
                            session.agent_data[agent_id_str] = {}
                            
                            # Get keys for this agent
                            keys = [k for k in agent_dict.keys()]
                            
                            # Rebuild with string keys
                            for k in keys:
                                if k is not None:
                                    key_str = str(k)
                                    session.agent_data[agent_id_str][key_str] = agent_dict[k]
            except Exception:
                # On any error, use empty dict (already initialized above)
                pass
        
        # Handle shared_keys
        session.shared_keys = set()
        if "shared_keys" in data and isinstance(data["shared_keys"], list):
            raw_shared_keys = cast(List[Any], data["shared_keys"])
            
            try:
                # Convert each item to string and add to set
                for k in raw_shared_keys:
                    if k is not None:
                        key_str = str(k)
                        session.shared_keys.add(key_str)
            except Exception:
                # On any error, use empty set (already initialized above)
                pass
                
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
        try:
            data = json.loads(json_str)
            # We can safely pass this to from_dict as it handles non-dict values
            return cls.from_dict(cast(Dict[str, Any], data))
        except Exception:
            # On any error, return a new empty session
            return cls()
    
    def set_agent_data(self, agent_id: str, key: str, value: Any) -> None:
        """
        Set data specific to an agent.
        
        Args:
            agent_id: ID of the agent
            key: Data key
            value: Data value
        """
        if agent_id not in self.agent_data:
            self.agent_data[agent_id] = {}
            
        self.agent_data[agent_id][key] = value
        self.last_updated = datetime.now().isoformat()
    
    def get_agent_data(self, agent_id: str, key: str, default: Any = None) -> Any:
        """
        Get data specific to an agent.
        
        Args:
            agent_id: ID of the agent
            key: Data key
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with key or default
        """
        if agent_id not in self.agent_data:
            return default
            
        return self.agent_data[agent_id].get(key, default)
    
    def share_key(self, key: str) -> None:
        """
        Mark a key as shared between agents.
        
        Args:
            key: Key to mark as shared
        """
        self.shared_keys.add(key)
    
    def unshare_key(self, key: str) -> None:
        """
        Remove a key from shared keys.
        
        Args:
            key: Key to unmark as shared
        """
        if key in self.shared_keys:
            self.shared_keys.remove(key)
    
    def is_key_shared(self, key: str) -> bool:
        """
        Check if a key is shared between agents.
        
        Args:
            key: Key to check
            
        Returns:
            True if the key is shared, False otherwise
        """
        return key in self.shared_keys
    
    def get_shared_keys(self) -> List[str]:
        """
        Get all shared keys.
        
        Returns:
            List of shared keys
        """
        return list(self.shared_keys)
    
    def get_agent_keys(self, agent_id: str) -> List[str]:
        """
        Get all keys specific to an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of keys specific to the agent
        """
        if agent_id not in self.agent_data:
            return []
            
        return list(self.agent_data[agent_id].keys())


def get_or_create_session_state(context: Dict[str, Any]) -> SessionState:
    """
    Get existing or create new SessionState from context.
    
    Args:
        context: Context dictionary
        
    Returns:
        SessionState instance
    """
    # Create new session if not in context or context is invalid
    if not context or "session_state" not in context:
        session = SessionState()
        context["session_state"] = session.to_dict()
        return session
        
    # Try to use existing session state (with type safety)
    session_state_any = context.get("session_state")
    
    # If it's not a dict or is None, create a new one
    if not isinstance(session_state_any, dict):
        session = SessionState()
        context["session_state"] = session.to_dict()
        return session
        
    # Type hint to help the type checker
    session_state = cast(Dict[str, Any], session_state_any)
        
    # Create from dict (from_dict handles further validation)
    return SessionState.from_dict(session_state)


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
