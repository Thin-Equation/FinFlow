"""
Test cases for document ingestion tools.
"""

import os
import unittest
import tempfile
from PIL import Image
from PyPDF2 import PdfWriter
import io
import shutil
from pathlib import Path

from tools.document_ingestion import (
    validate_document,
    validate_pdf,
    validate_image,
    preprocess_document,
    preprocess_pdf,
    preprocess_image,
    upload_document,
    batch_upload_documents,
    SUPPORTED_FILE_TYPES
)

class TestDocumentIngestion(unittest.TestCase):
    """Test cases for document ingestion tools."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.pdf_path = self._create_test_pdf()
        self.jpg_path = self._create_test_image('jpg')
        self.png_path = self._create_test_image('png')
        self.text_path = self._create_test_text()
        
        # Define a temporary destination folder
        self.dest_folder = os.path.join(self.test_dir, "destination")
        os.makedirs(self.dest_folder, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory and files
        shutil.rmtree(self.test_dir)
    
    def _create_test_pdf(self):
        """Create a test PDF file."""
        pdf_path = os.path.join(self.test_dir, "test.pdf")
        
        # Create a simple PDF
        pdf_writer = PdfWriter()
        pdf_writer.add_blank_page(width=800, height=600)
        
        with open(pdf_path, "wb") as f:
            pdf_writer.write(f)
        
        return pdf_path
    
    def _create_test_image(self, format_type):
        """Create a test image file."""
        if format_type == 'jpg':
            img_path = os.path.join(self.test_dir, "test.jpg")
            format_name = "JPEG"
        else:
            img_path = os.path.join(self.test_dir, "test.png")
            format_name = "PNG"
        
        # Create a simple image
        img = Image.new('RGB', (200, 100), color=(73, 109, 137))
        img.save(img_path, format=format_name)
        
        return img_path
    
    def _create_test_text(self):
        """Create a test text file."""
        text_path = os.path.join(self.test_dir, "test.txt")
        
        # Create a simple text file
        with open(text_path, "w") as f:
            f.write("This is a test file.")
        
        return text_path
    
    def test_validate_document_pdf(self):
        """Test PDF document validation."""
        result = validate_document(self.pdf_path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "success")
    
    def test_validate_document_image(self):
        """Test image document validation."""
        result = validate_document(self.jpg_path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "success")
        
        result = validate_document(self.png_path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "success")
    
    def test_validate_document_unsupported(self):
        """Test unsupported document validation."""
        result = validate_document(self.text_path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "error")
    
    def test_validate_document_not_found(self):
        """Test non-existent document validation."""
        result = validate_document(os.path.join(self.test_dir, "nonexistent.pdf"))
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "error")
    
    def test_preprocess_document_pdf(self):
        """Test PDF document preprocessing."""
        result = preprocess_document(self.pdf_path)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(result["processed_file_path"]))
    
    def test_preprocess_document_image(self):
        """Test image document preprocessing."""
        result = preprocess_document(self.jpg_path)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(result["processed_file_path"]))
        
        result = preprocess_document(self.png_path)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(result["processed_file_path"]))
    
    def test_upload_document(self):
        """Test document upload."""
        result = upload_document(self.pdf_path, self.dest_folder)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(result["file_path"]))
        
        # Also verify preprocessing was done
        self.assertIsNotNone(result["preprocessing_details"])
        self.assertEqual(result["preprocessing_details"]["status"], "success")
    
    def test_batch_upload_documents(self):
        """Test batch document upload."""
        file_paths = [self.pdf_path, self.jpg_path, self.png_path]
        result = batch_upload_documents(file_paths, self.dest_folder)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["results"]), 3)
        
        # Check if all uploads were successful
        success_count = sum(1 for r in result["results"] if r["status"] == "success")
        self.assertEqual(success_count, 3)
    
    def test_batch_upload_with_invalid(self):
        """Test batch upload with invalid document."""
        file_paths = [self.pdf_path, self.text_path, self.jpg_path]
        result = batch_upload_documents(file_paths, self.dest_folder)
        
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(len(result["results"]), 3)
        
        # Check if correct number of uploads were successful
        success_count = sum(1 for r in result["results"] if r["status"] == "success")
        self.assertEqual(success_count, 2)

if __name__ == "__main__":
    unittest.main()