"""
Unit tests for Document AI tools.
"""

import unittest
from unittest.mock import patch, MagicMock

# Import only the public function
from tools.document_ai import process_document

class TestDocumentAI(unittest.TestCase):
    """Test cases for the Document AI tools."""

    @patch('tools.document_ai.documentai.DocumentProcessorServiceClient')
    def test_process_document(self, mock_client_class: MagicMock) -> None:
        """Test the process_document function."""
        # Mock the DocumentAI client response
        mock_client = mock_client_class.return_value
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_document.entities = [
            MagicMock(type_="invoice_id", mention_text="INV-12345"),
            MagicMock(type_="amount", mention_text="100.00")
        ]
        mock_document.text = "Sample Invoice"
        mock_document.pages = [MagicMock(), MagicMock()]
        # type detection confidence
        mock_document.text_detection_params = MagicMock(confidence=0.95)
        mock_result.document = mock_document
        
        # Set up the mock client to return our mock result
        mock_client.process_document.return_value = mock_result
        
        # Test document content
        test_content = b'test document content'
        
        # Call the function
        result = process_document(
            content=test_content, 
            processor_id="test-processor-id"
        )
        
        # Verify client process_document was called
        mock_client.process_document.assert_called_once()
        
        # Verify result structure based on the actual implementation
        self.assertEqual(result["text"], "Sample Invoice")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["entities"]["invoice_id"], "INV-12345")
        self.assertEqual(result["entities"]["amount"], "100.00")

    @patch('tools.document_ai.documentai.DocumentProcessorServiceClient')
    def test_extract_document_entities_through_process(self, mock_client_class: MagicMock) -> None:
        """Test entity extraction through the public process_document function."""
        # Create a mock document and result
        mock_document = MagicMock()
        mock_document.text = "Sample text"
        mock_document.pages = [MagicMock(), MagicMock(), MagicMock()]
        mock_document.entities = [
            MagicMock(type_="invoice_id", mention_text="INV-12345"),
            MagicMock(type_="date", mention_text="2025-05-14")
        ]
        
        # Set up the mock client
        mock_client = mock_client_class.return_value
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_client.process_document.return_value = mock_result
        
        # Call the public function
        result = process_document(b"test content", "test-processor-id")
        
        # Verify result contains the extracted entities
        self.assertEqual(result["text"], "Sample text")
        self.assertEqual(result["pages"], 3)
        self.assertEqual(result["entities"]["invoice_id"], "INV-12345")
        self.assertEqual(result["entities"]["date"], "2025-05-14")

    @patch('tools.document_ai.documentai.DocumentProcessorServiceClient')
    def test_document_type_detection(self, mock_client_class: MagicMock) -> None:
        """Test document type detection through public API."""
        # Set up common mocking
        mock_client = mock_client_class.return_value
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_client.process_document.return_value = mock_result
        mock_result.document = mock_document
        
        # Test invoice detection
        mock_document.text = "invoice details for customer"
        mock_document.pages = [MagicMock()]
        mock_document.entities = [
            MagicMock(type_="invoice_id", mention_text="INV-12345"),
            MagicMock(type_="amount", mention_text="100.00")
        ]
        
        result = process_document(b"test content", "test-processor-id")
        self.assertEqual(result["document_type"], "invoice")
        
        # Test bank statement detection
        mock_document.text = "bank statement account summary"
        mock_document.entities = [
            MagicMock(type_="account_number", mention_text="123456789"),
            MagicMock(type_="balance", mention_text="5000.00")
        ]
        
        result = process_document(b"test content", "test-processor-id")
        self.assertEqual(result["document_type"], "bank_statement")
        
        # Test unknown document type
        mock_document.text = "some random document content"
        mock_document.entities = [
            MagicMock(type_="some_field", mention_text="some value")
        ]
        
        result = process_document(b"test content", "test-processor-id")
        self.assertEqual(result["document_type"], "unknown")

if __name__ == "__main__":
    unittest.main()
