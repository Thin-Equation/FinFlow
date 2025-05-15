"""
Unit tests for the SessionState class.
"""

import unittest
from typing import Dict, Any

from utils.session_state import SessionState, get_or_create_session_state, update_context_session_state

class TestSessionState(unittest.TestCase):
    """Test cases for the SessionState class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.session = SessionState()
        self.context: Dict[str, Any] = {}

    def test_initialization(self) -> None:
        """Test session state initialization."""
        session = SessionState("test-id")
        self.assertEqual(session.session_id, "test-id")
        self.assertEqual(session.data, {})

        # Auto-generated ID should be a UUID string
        default_session = SessionState()
        self.assertIsNotNone(default_session.session_id)
        self.assertIsInstance(default_session.session_id, str)
        self.assertGreater(len(default_session.session_id), 0)

    def test_get_set_methods(self) -> None:
        """Test get and set methods."""
        # Initial state should not have the key
        self.assertIsNone(self.session.get("test_key"))
        self.assertEqual(self.session.get("test_key", "default"), "default")

        # After setting, should return the value
        self.session.set("test_key", "test_value")
        self.assertEqual(self.session.get("test_key"), "test_value")

    def test_update_method(self) -> None:
        """Test update method."""
        self.session.update({
            "key1": "value1",
            "key2": "value2"
        })
        
        self.assertEqual(self.session.get("key1"), "value1")
        self.assertEqual(self.session.get("key2"), "value2")

    def test_delete_method(self) -> None:
        """Test delete method."""
        self.session.set("test_key", "test_value")
        self.assertEqual(self.session.get("test_key"), "test_value")
        
        self.session.delete("test_key")
        self.assertIsNone(self.session.get("test_key"))

    def test_clear_method(self) -> None:
        """Test clear method."""
        self.session.update({
            "key1": "value1",
            "key2": "value2"
        })
        
        self.session.clear()
        self.assertEqual(self.session.data, {})

    def test_to_dict_method(self) -> None:
        """Test to_dict method."""
        self.session.set("test_key", "test_value")
        session_dict = self.session.to_dict()
        
        self.assertEqual(session_dict["session_id"], self.session.session_id)
        self.assertEqual(session_dict["data"]["test_key"], "test_value")

    def test_from_dict_method(self) -> None:
        """Test from_dict method."""
        original_dict: Dict[str, Any] = {
            "session_id": "test-session-id",
            "created_at": "2023-01-01T00:00:00",
            "last_updated": "2023-01-01T01:00:00",
            "data": {"key": "value"}
        }
        
        session = SessionState.from_dict(original_dict)
        
        self.assertEqual(session.session_id, "test-session-id")
        self.assertEqual(session.created_at, "2023-01-01T00:00:00")
        self.assertEqual(session.last_updated, "2023-01-01T01:00:00")
        self.assertEqual(session.data, {"key": "value"})

    def test_json_serialization(self) -> None:
        """Test JSON serialization and deserialization."""
        self.session.set("test_key", "test_value")
        
        # Serialize to JSON
        json_str = self.session.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from JSON
        new_session = SessionState.from_json(json_str)
        
        # Compare properties
        self.assertEqual(new_session.session_id, self.session.session_id)
        self.assertEqual(new_session.data, self.session.data)

    def test_get_or_create_session_state(self) -> None:
        """Test get_or_create_session_state function."""
        # Empty context
        session1 = get_or_create_session_state(self.context)
        self.assertIsNotNone(session1)
        self.assertIsInstance(session1, SessionState)
        
        # Context already has session_state
        session_dict = self.session.to_dict()
        self.context["session_state"] = session_dict
        
        session2 = get_or_create_session_state(self.context)
        self.assertEqual(session2.session_id, self.session.session_id)

    def test_update_context_session_state(self) -> None:
        """Test update_context_session_state function."""
        session = SessionState("test-session")
        session.set("test_key", "test_value")
        
        updated_context = update_context_session_state(self.context, session)
        
        self.assertIn("session_state", updated_context)
        self.assertEqual(updated_context["session_state"]["session_id"], "test-session")
        self.assertEqual(updated_context["session_state"]["data"]["test_key"], "test_value")

if __name__ == "__main__":
    unittest.main()
