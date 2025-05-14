"""
Unit tests for Document AI tools.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

from tools.document_ai import process_document, _extract_document_entities, _determine_document_type

class TestDocumentAI(unittest.TestCase):
    """Test cases for the Document AI tools."""

    @patch('google.cloud.documentai.DocumentProcessorServiceClient')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test document content')
    def test_process_document(self, mock_file, mock_client_class):
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
        mock_document.text_detection_params.confidence = 0.95
        mock_result.document = mock_document
        
        # Set up the mock client to return our mock result
        mock_client.process_document.return_value = mock_result
        
        # Call the function
        result = process_document(
            file_path="/path/to/document.pdf", 
            processor_id="test-processor-id"
        )
        
        # Verify file was read
        mock_file.assert_called_once_with("/path/to/document.pdf", "rb")
        
        # Verify client was called correctly
        mock_client.process_document.assert_called_once()
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document_type"], "invoice")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["extracted_data"]["text"], "Sample Invoice")
        self.assertEqual(result["extracted_data"]["pages"], 2)
        self.assertEqual(result["extracted_data"]["entities"]["invoice_id"], "INV-12345")
        self.assertEqual(result["extracted_data"]["entities"]["amount"], "100.00")

    def test_extract_document_entities(self):
        """Test the _extract_document_entities function."""
        # Create a mock document
        mock_document = MagicMock()
        mock_document.text = "Sample text"
        mock_document.pages = [MagicMock(), MagicMock(), MagicMock()]
        mock_document.entities = [
            MagicMock(type_="invoice_id", mention_text="INV-12345"),
            MagicMock(type_="date", mention_text="2025-05-14")
        ]
        
        # Call the function
        result = _extract_document_entities(mock_document)
        
        # Verify result
        self.assertEqual(result["text"], "Sample text")
        self.assertEqual(result["pages"], 3)
        self.assertEqual(result["entities"]["invoice_id"], "INV-12345")
        self.assertEqual(result["entities"]["date"], "2025-05-14")

    def test_determine_document_type(self):
        """Test the _determine_document_type function."""
        # Test invoice detection
        invoice_data = {
            "entities": {
                "invoice_id": "INV-12345",
                "amount": "100.00"
            }
        }
        self.assertEqual(_determine_document_type(invoice_data), "invoice")
        
        # Test receipt detection
        receipt_data = {
            "entities": {
                "receipt_number": "R-67890",
                "total": "50.00"
            }
        }
        self.assertEqual(_determine_document_type(receipt_data), "receipt")
        
        # Test unknown document type
        unknown_data = {
            "entities": {
                "some_field": "some value"
            }
        }
        self.assertEqual(_determine_document_type(unknown_data), "unknown")

if __name__ == "__main__":
    unittest.main()
