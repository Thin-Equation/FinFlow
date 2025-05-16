"""
Document ingestion tools for the FinFlow system.

This module provides tools for uploading, validating, and preprocessing documents
before they are processed by Document AI.
"""

import os
import logging
import mimetypes
import magic
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import shutil
from datetime import datetime

from google.adk.tools import ToolContext  # type: ignore
from PIL import Image
import fitz  # PyMuPDF for PDF validation and preprocessing

# Configure logging
logger = logging.getLogger("finflow.tools.document_ingestion")

# Define supported document types
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/tiff': '.tiff',
    'image/bmp': '.bmp',
}

class DocumentValidationError(Exception):
    """Exception raised for document validation errors."""
    pass

def validate_document(file_path: str) -> Dict[str, Any]:
    """
    Validate if a document is of supported type and format.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        dict: Validation result with status and details
    """
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "valid": False,
            "message": f"File not found: {file_path}"
        }
    
    try:
        # Get file size in MB
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        
        # Check if file size is under 20MB (Document AI limit)
        if file_size > 20:
            return {
                "status": "error",
                "valid": False,
                "message": f"File size ({file_size:.2f}MB) exceeds 20MB limit"
            }
        
        # Use python-magic to determine file type
        mime_type = magic.from_file(file_path, mime=True)
        
        if mime_type not in SUPPORTED_FILE_TYPES:
            return {
                "status": "error",
                "valid": False,
                "message": f"Unsupported file type: {mime_type}. Supported types: {', '.join(SUPPORTED_FILE_TYPES.keys())}"
            }
        
        # Additional validation based on file type
        if mime_type == 'application/pdf':
            return validate_pdf(file_path)
        elif mime_type.startswith('image/'):
            return validate_image(file_path, mime_type)
        else:
            # Should not reach here due to previous check
            return {
                "status": "error",
                "valid": False,
                "message": f"Unsupported file type: {mime_type}"
            }
    
    except Exception as e:
        logger.error(f"Error validating document {file_path}: {str(e)}")
        return {
            "status": "error",
            "valid": False,
            "message": f"Validation error: {str(e)}"
        }

def validate_pdf(file_path: str) -> Dict[str, Any]:
    """
    Validate a PDF document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        dict: Validation result
    """
    try:
        # Open the PDF file to check if it's valid
        with fitz.open(file_path) as pdf:
            # Check if the PDF has at least one page
            if len(pdf) == 0:
                return {
                    "status": "error",
                    "valid": False,
                    "message": "PDF file has no pages"
                }
            
            # Get basic PDF info
            info = pdf.metadata
            
            return {
                "status": "success",
                "valid": True,
                "message": "PDF is valid",
                "details": {
                    "pages": len(pdf),
                    "title": info.get("title", "Untitled"),
                    "author": info.get("author", "Unknown"),
                    "creation_date": info.get("creationDate", "Unknown"),
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "valid": False,
            "message": f"Invalid PDF file: {str(e)}"
        }

def validate_image(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Validate an image document.
    
    Args:
        file_path: Path to the image file
        mime_type: MIME type of the image
        
    Returns:
        dict: Validation result
    """
    try:
        # Open the image to check if it's valid
        with Image.open(file_path) as img:
            width, height = img.size
            format_name = img.format
            
            # Check if the image is too small to be useful
            if width < 100 or height < 100:
                return {
                    "status": "warning",
                    "valid": True,  # Still valid but with warning
                    "message": f"Image resolution is low: {width}x{height}",
                    "details": {
                        "width": width,
                        "height": height,
                        "format": format_name,
                    }
                }
            
            return {
                "status": "success",
                "valid": True,
                "message": "Image is valid",
                "details": {
                    "width": width,
                    "height": height,
                    "format": format_name,
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "valid": False,
            "message": f"Invalid image file: {str(e)}"
        }

def preprocess_document(file_path: str) -> Dict[str, Any]:
    """
    Preprocess document for improved Document AI processing.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        dict: Preprocessing result with status and processed file path
    """
    try:
        # Validate document first
        validation_result = validate_document(file_path)
        
        if not validation_result["valid"]:
            return {
                "status": "error",
                "message": validation_result["message"],
                "processed_file_path": None
            }
        
        # Get mime type
        mime_type = magic.from_file(file_path, mime=True)
        
        # Create processed file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
        
        # Create processed directory if it doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        processed_file_path = os.path.join(
            processed_dir, f"{base_name}_processed_{timestamp}{ext}"
        )
        
        # Apply preprocessing based on file type
        if mime_type == 'application/pdf':
            result = preprocess_pdf(file_path, processed_file_path)
        elif mime_type.startswith('image/'):
            result = preprocess_image(file_path, processed_file_path)
        else:
            # Simply copy the file if no specific preprocessing is needed
            shutil.copy2(file_path, processed_file_path)
            result = {
                "status": "success",
                "message": "File copied without specific preprocessing",
                "processed_file_path": processed_file_path
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error preprocessing document {file_path}: {str(e)}")
        return {
            "status": "error",
            "message": f"Preprocessing error: {str(e)}",
            "processed_file_path": None
        }

def preprocess_pdf(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Preprocess PDF document.
    
    Args:
        input_path: Path to the input PDF file
        output_path: Path to save the processed PDF file
        
    Returns:
        dict: Preprocessing result
    """
    try:
        with fitz.open(input_path) as pdf:
            # PDF optimization can be done here
            # For example, we could remove annotations, optimize images, etc.
            # For now, we'll just save a copy with default settings
            pdf.save(output_path)
            
            return {
                "status": "success",
                "message": "PDF processed successfully",
                "processed_file_path": output_path,
                "details": {
                    "pages": len(pdf),
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error preprocessing PDF: {str(e)}",
            "processed_file_path": None
        }

def preprocess_image(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Preprocess image document.
    
    Args:
        input_path: Path to the input image file
        output_path: Path to save the processed image file
        
    Returns:
        dict: Preprocessing result
    """
    try:
        with Image.open(input_path) as img:
            # Check if image needs to be converted (e.g., BMP to PNG)
            output_ext = os.path.splitext(output_path)[1].lower()
            original_format = img.format.lower() if img.format else ""
            
            # Convert image formats if needed
            if (output_ext == '.png' and original_format != 'png') or \
               (output_ext == '.jpg' and original_format != 'jpeg'):
                # Convert to RGB mode if not already (needed for some formats)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            
            # Save the processed image
            img.save(output_path, quality=95, optimize=True)
            
            return {
                "status": "success",
                "message": "Image processed successfully",
                "processed_file_path": output_path,
                "details": {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error preprocessing image: {str(e)}",
            "processed_file_path": None
        }

def upload_document(
    file_path: str, 
    destination_folder: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Upload a document to the FinFlow system.
    
    Args:
        file_path: Path to the document file
        destination_folder: Optional folder to store the uploaded file
        tool_context: Provided by ADK
        
    Returns:
        dict: Upload result with status and details
    """
    try:
        # Validate the document first
        validation_result = validate_document(file_path)
        
        if not validation_result["valid"]:
            return {
                "status": "error",
                "message": validation_result["message"]
            }
        
        # Determine destination folder
        if not destination_folder:
            destination_folder = os.path.join(
                os.path.dirname(os.path.dirname(file_path)), 
                "sample_data", "invoices"
            )
        
        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # Generate a unique filename
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        unique_filename = f"{base_name}_{timestamp}{ext}"
        destination_path = os.path.join(destination_folder, unique_filename)
        
        # Copy the file to the destination
        shutil.copy2(file_path, destination_path)
        
        # Preprocess the document
        preprocess_result = preprocess_document(destination_path)
        
        # Return success result with file details
        return {
            "status": "success",
            "message": "Document uploaded and processed successfully",
            "file_path": destination_path,
            "file_name": unique_filename,
            "validation_details": validation_result.get("details", {}),
            "preprocessing_details": preprocess_result
        }
    
    except Exception as e:
        logger.error(f"Error uploading document {file_path}: {str(e)}")
        return {
            "status": "error",
            "message": f"Upload error: {str(e)}"
        }

def batch_upload_documents(
    file_paths: List[str], 
    destination_folder: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Upload multiple documents to the FinFlow system.
    
    Args:
        file_paths: List of paths to document files
        destination_folder: Optional folder to store the uploaded files
        tool_context: Provided by ADK
        
    Returns:
        dict: Batch upload result with status and details
    """
    results = []
    success_count = 0
    
    for file_path in file_paths:
        result = upload_document(file_path, destination_folder, tool_context)
        results.append({
            "file_path": file_path,
            "status": result["status"],
            "message": result["message"]
        })
        
        if result["status"] == "success":
            success_count += 1
    
    return {
        "status": "success" if success_count == len(file_paths) else "partial_success" if success_count > 0 else "error",
        "message": f"Uploaded {success_count} out of {len(file_paths)} documents successfully",
        "results": results
    }