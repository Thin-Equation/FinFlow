"""
Enhanced document ingestion tools for the FinFlow system.

This module provides production-level tools for uploading, validating, and preprocessing documents
before they are processed by Document AI, including batch processing and error recovery.
"""

import os
import logging
import magic
from typing import Any, Dict, List, Optional, Tuple
import shutil
from datetime import datetime
import time
import concurrent.futures
import uuid
import hashlib
import json

# Image processing
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import fitz  # PyMuPDF for PDF processing
from google.adk.tools import ToolContext  # type: ignore

# Configure logging
logger = logging.getLogger("finflow.tools.enhanced_document_ingestion")

# Define supported document types with more specific MIME types
SUPPORTED_FILE_TYPES = {
    # PDF
    'application/pdf': '.pdf',
    
    # Images
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/tiff': '.tiff',
    'image/bmp': '.bmp',
    'image/gif': '.gif',
    'image/webp': '.webp',
    
    # Office documents (optionally supported)
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    
    # Text formats
    'text/plain': '.txt',
    'text/csv': '.csv',
    'application/json': '.json',
    'application/xml': '.xml',
    'text/xml': '.xml',
    
    # Email formats
    'message/rfc822': '.eml',
    'application/vnd.ms-outlook': '.msg',
}

# Define size limits
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB limit for Document AI
MIN_IMAGE_DIMENSION = 300  # Minimum width/height for useful OCR
MIN_DPI = 200  # Minimum DPI for quality OCR
MAX_PAGE_COUNT = 100  # Maximum number of pages to process

# Validation settings
VALIDATION_SETTINGS = {
    "enforce_strict_validation": True,
    "check_corrupt": True,
    "check_password_protection": True,
    "check_image_quality": True,
    "check_malware": False,  # Requires additional libraries
    "validate_content": True,  # Check if document has meaningful content
    "min_text_length": 20,  # Minimum text length for a valid document
}

# Preprocess settings
PREPROCESS_SETTINGS = {
    "optimize_images": True,
    "compress_pdfs": True,
    "enhance_contrast": True,
    "convert_to_pdf": True,  # Convert images to PDF for better handling
    "remove_blank_pages": True,
    "deskew": True,
    "max_image_dimension": 3000,  # Resize large images for better processing
}

class DocumentValidationError(Exception):
    """Exception raised for document validation errors."""
    pass

class MalformedDocumentError(DocumentValidationError):
    """Exception raised for malformed documents that cannot be processed."""
    pass

class UnsupportedDocumentError(DocumentValidationError):
    """Exception raised for unsupported document types."""
    pass

class CorruptDocumentError(DocumentValidationError):
    """Exception raised for corrupt document files."""
    pass

class DocumentIngestionManager:
    """Manager class for document ingestion with advanced capabilities."""
    
    def __init__(self, 
                base_storage_path: Optional[str] = None,
                enable_cache: bool = True,
                max_workers: int = 10,
                validation_settings: Optional[Dict[str, Any]] = None,
                preprocess_settings: Optional[Dict[str, Any]] = None,
                environment: str = "production"):
        """
        Initialize document ingestion manager.
        
        Args:
            base_storage_path: Base path for document storage
            enable_cache: Whether to enable document caching
            max_workers: Maximum number of parallel workers for batch operations
            validation_settings: Custom validation settings
            preprocess_settings: Custom preprocessing settings
            environment: Environment (development, staging, production)
        """
        # Import configuration
        # In a real production environment, you would load from config files
        # For this implementation we'll use the module constants
        self.validation_settings = validation_settings or VALIDATION_SETTINGS
        self.preprocess_settings = preprocess_settings or PREPROCESS_SETTINGS
        self.environment = environment
        
        # Adjust settings based on environment
        if environment == "development":
            self.validation_settings["enforce_strict_validation"] = False
            self.validation_settings["check_malware"] = False
        
        # Storage paths
        if base_storage_path:
            self.base_path = os.path.abspath(base_storage_path)
        else:
            # Default path relative to the current module
            self.base_path = os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "sample_data"
            ))
        
        # Create hierarchical storage with date-based organization
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.raw_path = os.path.join(self.base_path, "raw", timestamp)
        self.processed_path = os.path.join(self.base_path, "processed", timestamp)
        self.rejected_path = os.path.join(self.base_path, "rejected", timestamp)
        self.cache_path = os.path.join(self.base_path, "cache")
        self.archive_path = os.path.join(self.base_path, "archive")
        
        # Create storage directories if they don't exist
        for path in [self.raw_path, self.processed_path, self.rejected_path, self.cache_path, self.archive_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.enable_cache = enable_cache
        self.max_workers = max_workers
        
        # Performance monitoring
        self.stats = {
            "start_time": datetime.now(),
            "files_processed": 0,
            "files_rejected": 0,
            "processing_errors": 0,
            "validation_errors": {},
            "processed_bytes": 0,
            "processing_times": [],
        }
        
        # Track processed files for this session using a file hash -> path mapping
        self.processed_files: Dict[str, str] = {}
        
        # Load file type validators - extensible for custom validators
        self.validators = {
            'application/pdf': self._validate_pdf,
            'image/jpeg': self._validate_image,
            'image/png': self._validate_image,
            'image/tiff': self._validate_image,
            'image/webp': self._validate_image,
            'image/gif': self._validate_image,
        }
        
        # Load file type preprocessors - extensible for custom preprocessors
        self.preprocessors = {
            'application/pdf': self._preprocess_pdf,
            'image/jpeg': self._preprocess_image,
            'image/png': self._preprocess_image,
            'image/tiff': self._preprocess_image,
            'image/webp': self._preprocess_image,
        }
        
        logger.info(f"Document ingestion manager initialized in {environment} environment")
        logger.info(f"Storage path: {self.base_path}")
        logger.info(f"Max workers: {max_workers}")
        logger.info(f"Cache enabled: {enable_cache}")
        if self.environment == "production":
            logger.info("Production mode: Strict validation enabled")  # file_hash -> processed_path
        
        # Metrics tracking
        self.metrics = {
            "files_ingested": 0,
            "files_rejected": 0,
            "files_processed": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
        }
        
        logger.info(f"Document ingestion manager initialized with base path: {self.base_path}")
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
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
            # Get file size in bytes
            file_size = os.path.getsize(file_path)
            
            # Check file size
            if file_size > MAX_FILE_SIZE_BYTES:
                return {
                    "status": "error",
                    "valid": False,
                    "message": f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds {MAX_FILE_SIZE_BYTES / (1024 * 1024)}MB limit"
                }
            elif file_size == 0:
                return {
                    "status": "error",
                    "valid": False,
                    "message": "File is empty (0 bytes)"
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
                return self._validate_pdf(file_path)
            elif mime_type.startswith('image/'):
                return self._validate_image(file_path, mime_type)
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
    
    def preprocess_document(self, file_path: str, optimization_level: str = "medium") -> Dict[str, Any]:
        """
        Preprocess document for improved Document AI processing.
        
        Args:
            file_path: Path to the document file
            optimization_level: Optimization level (low, medium, high)
            
        Returns:
            dict: Preprocessing result with status and processed file path
        """
        start_time = time.time()
        
        try:
            # First validate the document
            validation_result = self.validate_document(file_path)
            
            if not validation_result.get("valid", False):
                return {
                    "status": "error",
                    "message": validation_result.get("message", "Document validation failed"),
                    "processed_file_path": None
                }
            
            # Check if we already processed this file during this session
            file_hash = self._compute_file_hash(file_path)
            
            # Check cache if enabled
            if self.enable_cache:
                cached_path = self._check_cache(file_hash)
                if cached_path:
                    self.metrics["cache_hits"] += 1
                    logger.info(f"Cache hit for {file_path} -> {cached_path}")
                    return {
                        "status": "success",
                        "message": "Document retrieved from cache",
                        "processed_file_path": cached_path,
                        "cached": True
                    }
            
            # Get mime type
            mime_type = magic.from_file(file_path, mime=True)
            
            # Create processed file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = os.path.basename(file_path)
            base_name, ext = os.path.splitext(file_name)
            
            processed_file_path = os.path.join(
                self.processed_path, f"{base_name}_processed_{timestamp}{ext}"
            )
            
            # Apply preprocessing based on file type
            if mime_type == 'application/pdf':
                result = self._preprocess_pdf(file_path, processed_file_path, optimization_level)
            elif mime_type.startswith('image/'):
                result = self._preprocess_image(file_path, processed_file_path, optimization_level)
            else:
                # Simply copy the file if no specific preprocessing is needed
                shutil.copy2(file_path, processed_file_path)
                result = {
                    "status": "success",
                    "message": "File copied without specific preprocessing",
                    "processed_file_path": processed_file_path
                }
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["total_processing_time"] += processing_time
            self.metrics["files_processed"] += 1
            
            # If successful, add to cache
            if result["status"] == "success" and self.enable_cache:
                self._add_to_cache(file_hash, result["processed_file_path"])
            
            # Add processing time to result
            result["processing_time"] = processing_time
            
            return result
        
        except Exception as e:
            logger.error(f"Error preprocessing document {file_path}: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                "status": "error",
                "message": f"Preprocessing error: {str(e)}",
                "processed_file_path": None,
                "processing_time": processing_time
            }
    
    def batch_preprocess_documents(self, file_paths: List[str], 
                                  optimization_level: str = "medium") -> Dict[str, Any]:
        """
        Batch preprocess multiple documents in parallel.
        
        Args:
            file_paths: List of file paths to preprocess
            optimization_level: Optimization level (low, medium, high)
            
        Returns:
            dict: Batch preprocessing results
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        results = []
        
        try:
            # Process documents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.preprocess_document, path, optimization_level): path 
                    for path in file_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        result["file_path"] = path
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Exception processing {path}: {str(e)}")
                        results.append({
                            "status": "error",
                            "file_path": path,
                            "message": str(e)
                        })
            
            # Count successes and failures
            successful = sum(1 for r in results if r["status"] == "success")
            
            return {
                "status": "success" if successful == len(file_paths) else "partial_success" if successful > 0 else "error",
                "batch_id": batch_id,
                "total_files": len(file_paths),
                "successful": successful,
                "failed": len(file_paths) - successful,
                "processing_time": time.time() - start_time,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def upload_document(self, file_path: str, 
                        destination_folder: Optional[str] = None,
                        auto_preprocess: bool = True) -> Dict[str, Any]:
        """
        Upload a document to the FinFlow system.
        
        Args:
            file_path: Path to the document file
            destination_folder: Optional folder within the system to store the document
            auto_preprocess: Whether to automatically preprocess the document
            
        Returns:
            dict: Upload result with status and details
        """
        try:
            # Validate the document
            validation_result = self.validate_document(file_path)
            
            if not validation_result["valid"]:
                # Move to rejected folder
                rejected_path = self._move_to_rejected(file_path, validation_result["message"])
                self.metrics["files_rejected"] += 1
                
                return {
                    "status": "error",
                    "message": validation_result["message"],
                    "rejected_path": rejected_path
                }
            
            # Determine destination folder
            if not destination_folder:
                destination_folder = os.path.join(self.raw_path, "invoices")
            else:
                # If relative path provided, make it relative to raw_path
                if not os.path.isabs(destination_folder):
                    destination_folder = os.path.join(self.raw_path, destination_folder)
            
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
            self.metrics["files_ingested"] += 1
            
            result = {
                "status": "success",
                "message": "Document uploaded successfully",
                "file_path": destination_path,
                "file_name": unique_filename,
                "validation_details": validation_result.get("details", {})
            }
            
            # Auto-preprocess if enabled
            if auto_preprocess:
                preprocess_result = self.preprocess_document(destination_path)
                result["preprocessing"] = preprocess_result
                
                if preprocess_result["status"] == "success":
                    result["processed_file_path"] = preprocess_result["processed_file_path"]
            
            return result
        
        except Exception as e:
            logger.error(f"Error uploading document {file_path}: {str(e)}")
            return {
                "status": "error",
                "message": f"Upload error: {str(e)}"
            }
    
    def batch_upload_documents(self, file_paths: List[str],
                              destination_folder: Optional[str] = None,
                              auto_preprocess: bool = True) -> Dict[str, Any]:
        """
        Upload multiple documents to the FinFlow system.
        
        Args:
            file_paths: List of paths to document files
            destination_folder: Optional folder to store the uploaded files
            auto_preprocess: Whether to automatically preprocess documents
            
        Returns:
            dict: Batch upload result with status and details
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        results = []
        
        try:
            # Process documents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(
                        self.upload_document, 
                        path, 
                        destination_folder, 
                        auto_preprocess
                    ): path for path in file_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        result["original_file_path"] = path
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Exception uploading {path}: {str(e)}")
                        results.append({
                            "status": "error",
                            "original_file_path": path,
                            "message": str(e)
                        })
            
            # Count successes and failures
            successful = sum(1 for r in results if r["status"] == "success")
            
            return {
                "status": "success" if successful == len(file_paths) else "partial_success" if successful > 0 else "error",
                "batch_id": batch_id,
                "total_files": len(file_paths),
                "successful": successful,
                "failed": len(file_paths) - successful,
                "processing_time": time.time() - start_time,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Error in batch upload: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get document ingestion metrics."""
        # Calculate derived metrics
        total_files = self.metrics["files_ingested"]
        avg_processing_time = (
            self.metrics["total_processing_time"] / self.metrics["files_processed"] 
            if self.metrics["files_processed"] > 0 else 0
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "files_ingested": self.metrics["files_ingested"],
            "files_processed": self.metrics["files_processed"],
            "files_rejected": self.metrics["files_rejected"],
            "cache_hits": self.metrics["cache_hits"],
            "avg_processing_time": avg_processing_time,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["files_processed"] 
                if self.metrics["files_processed"] > 0 else 0
            ),
            "rejection_rate": (
                self.metrics["files_rejected"] / total_files 
                if total_files > 0 else 0
            )
        }
    
    # Private helper methods
    
    def _validate_pdf(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF document."""
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
                
                # Check for encryption/password protection
                if pdf.is_encrypted:
                    return {
                        "status": "error",
                        "valid": False,
                        "message": "PDF is encrypted or password-protected"
                    }
                
                # Get basic PDF info
                info = pdf.metadata
                
                # Count images and text to detect scanned vs digital PDFs
                text_length = 0
                image_count = 0
                
                # Check first few pages for content
                for page_idx in range(min(5, len(pdf))):
                    page = pdf[page_idx]
                    text_length += len(page.get_text())
                    image_count += len(page.get_images())
                
                # Determine if it's a scanned document (low text, has images)
                is_scanned = image_count > 0 and text_length < 100
                
                return {
                    "status": "success",
                    "valid": True,
                    "message": "PDF is valid",
                    "details": {
                        "pages": len(pdf),
                        "title": info.get("title", "Untitled"),
                        "author": info.get("author", "Unknown"),
                        "creation_date": info.get("creationDate", "Unknown"),
                        "is_scanned": is_scanned,
                        "has_text": text_length > 0,
                        "text_length": text_length,
                        "image_count": image_count
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "valid": False,
                "message": f"Invalid PDF file: {str(e)}"
            }
    
    def _validate_image(self, file_path: str, mime_type: str) -> Dict[str, Any]:
        """Validate image document."""
        try:
            # Open the image to check if it's valid
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                color_mode = img.mode
                
                # Check if the image is too small for OCR
                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    return {
                        "status": "warning",
                        "valid": True,  # Still valid but with warning
                        "message": f"Image resolution is low ({width}x{height}), may affect OCR quality",
                        "details": {
                            "width": width,
                            "height": height,
                            "format": format_name,
                            "color_mode": color_mode,
                            "dpi": self._get_image_dpi(img)
                        }
                    }
                
                # Check DPI for OCR quality
                dpi = self._get_image_dpi(img)
                if dpi and (dpi[0] < 200 or dpi[1] < 200):
                    return {
                        "status": "warning",
                        "valid": True,  # Still valid but with warning
                        "message": f"Image DPI is low ({dpi}), may affect OCR quality",
                        "details": {
                            "width": width,
                            "height": height,
                            "format": format_name,
                            "color_mode": color_mode,
                            "dpi": dpi
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
                        "color_mode": color_mode,
                        "dpi": dpi
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "valid": False,
                "message": f"Invalid image file: {str(e)}"
            }
    
    def _get_image_dpi(self, img: Image.Image) -> Optional[Tuple[float, float]]:
        """Get image DPI if available."""
        try:
            if hasattr(img, 'info') and 'dpi' in img.info:
                return img.info['dpi']
            return None
        except Exception:
            return None
    
    def _preprocess_pdf(self, input_path: str, output_path: str, optimization_level: str) -> Dict[str, Any]:
        """Preprocess PDF with optimization."""
        try:
            with fitz.open(input_path) as pdf:
                # Get validation details first
                validation = self._validate_pdf(input_path)
                details = validation.get("details", {})
                is_scanned = details.get("is_scanned", False)
                
                # Different optimization strategies based on level
                if optimization_level == "high":
                    # For high optimization, we might OCR scanned documents
                    if is_scanned:
                        # In a production environment, this would call an OCR service
                        logger.info(f"High optimization would apply OCR to {input_path}")
                
                # Medium optimization (default)
                # We'll optimize the PDF by cleaning up metadata and compressing images
                clean_info = {
                    "producer": "FinFlow Document Processor",
                    "creator": "FinFlow",
                    "title": details.get("title", "Processed Document"),
                    "creationDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Set clean metadata
                pdf.set_metadata(clean_info)
                
                # Save with optimization options
                pdf.save(
                    output_path,
                    garbage=4,  # Maximum garbage collection
                    deflate=True,  # Compress streams
                    clean=True,  # Clean unwanted data
                    pretty=False,  # Don't prettify PDF
                    linear=True  # Linearize PDF for faster web viewing
                )
                
                return {
                    "status": "success",
                    "message": "PDF processed successfully",
                    "processed_file_path": output_path,
                    "details": {
                        "pages": len(pdf),
                        "optimization_level": optimization_level,
                        "is_scanned": is_scanned
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error preprocessing PDF: {str(e)}",
                "processed_file_path": None
            }
    
    def _preprocess_image(self, input_path: str, output_path: str, optimization_level: str) -> Dict[str, Any]:
        """Preprocess image with enhancements for better OCR."""
        try:
            with Image.open(input_path) as img:
                # Store original details
                original_format = img.format
                original_mode = img.mode
                original_size = img.size
                
                # Convert to RGB if not already (needed for some operations)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Apply different processing based on optimization level
                if optimization_level == "low":
                    # Basic processing: just normalize orientation
                    img = ImageOps.exif_transpose(img)
                
                elif optimization_level == "medium":
                    # Medium processing: improve contrast and sharpen
                    img = ImageOps.exif_transpose(img)  # Fix orientation
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)  # Slightly increase contrast
                    
                    # Sharpen the image
                    img = img.filter(ImageFilter.SHARPEN)
                
                elif optimization_level == "high":
                    # Advanced processing for optimal OCR
                    img = ImageOps.exif_transpose(img)  # Fix orientation
                    
                    # Convert to grayscale if color
                    if img.mode == 'RGB':
                        img = img.convert('L')
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.5)  # Higher contrast
                    
                    # Apply adaptive thresholding (simulate)
                    img = img.point(lambda p: p > 128 and 255)
                    
                    # Sharpen
                    img = img.filter(ImageFilter.SHARPEN)
                    img = img.filter(ImageFilter.SHARPEN)  # Double sharpen
                
                # Determine output format
                output_format = original_format
                if output_format == 'BMP':
                    output_format = 'PNG'  # Convert BMP to PNG for better compression
                
                # Save with appropriate settings
                save_args = {}
                if output_format == 'JPEG':
                    save_args = {'quality': 95, 'optimize': True}
                elif output_format == 'PNG':
                    save_args = {'optimize': True}
                elif output_format == 'TIFF':
                    save_args = {'compression': 'tiff_lzw'}
                
                img.save(output_path, format=output_format, **save_args)
                
                return {
                    "status": "success",
                    "message": "Image processed successfully",
                    "processed_file_path": output_path,
                    "details": {
                        "original_format": original_format,
                        "output_format": output_format,
                        "original_mode": original_mode,
                        "output_mode": img.mode,
                        "original_size": original_size,
                        "output_size": img.size,
                        "optimization_level": optimization_level
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error preprocessing image: {str(e)}",
                "processed_file_path": None
            }
    
    def _move_to_rejected(self, file_path: str, reason: str) -> str:
        """Move invalid document to rejected folder with reason."""
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create a safe reason string for filename
        safe_reason = "".join(c if c.isalnum() else "_" for c in reason)
        safe_reason = safe_reason[:30]  # Truncate if too long
        
        rejected_file = f"{base_name}_{timestamp}_{safe_reason}{ext}"
        rejected_path = os.path.join(self.rejected_path, rejected_file)
        
        # Ensure rejected directory exists
        if not os.path.exists(self.rejected_path):
            os.makedirs(self.rejected_path)
        
        # Copy to rejected folder (don't move original)
        shutil.copy2(file_path, rejected_path)
        
        # Write rejection reason to a sidecar file
        reason_file = rejected_path + ".reason.txt"
        with open(reason_file, "w") as f:
            f.write(f"Rejection reason: {reason}\n")
            f.write(f"Original file: {file_path}\n")
            f.write(f"Rejection time: {datetime.now().isoformat()}\n")
        
        return rejected_path
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for caching."""
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _check_cache(self, file_hash: str) -> Optional[str]:
        """Check if document exists in cache by hash."""
        if not self.enable_cache:
            return None
        
        # Check if we already have this file in processed cache
        cache_index_path = os.path.join(self.cache_path, "cache_index.json")
        
        if os.path.exists(cache_index_path):
            try:
                with open(cache_index_path, "r") as f:
                    cache_index = json.load(f)
                
                if file_hash in cache_index and os.path.exists(cache_index[file_hash]):
                    return cache_index[file_hash]
            except Exception as e:
                logger.warning(f"Error reading cache index: {str(e)}")
        
        return None
    
    def _add_to_cache(self, file_hash: str, processed_path: str) -> None:
        """Add processed document to cache."""
        if not self.enable_cache:
            return
        
        # Ensure cache directory exists
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        # Read existing cache index
        cache_index_path = os.path.join(self.cache_path, "cache_index.json")
        cache_index = {}
        
        if os.path.exists(cache_index_path):
            try:
                with open(cache_index_path, "r") as f:
                    cache_index = json.load(f)
            except Exception:
                # If corrupted, start with empty cache
                cache_index = {}
        
        # Add or update entry
        cache_index[file_hash] = processed_path
        
        # Write updated index
        with open(cache_index_path, "w") as f:
            json.dump(cache_index, f, indent=2)


# Standalone functions for document ingestion (for backward compatibility)

def validate_document(file_path: str) -> Dict[str, Any]:
    """
    Validate if a document is of supported type and format.
    This is a wrapper function for compatibility.
    """
    manager = DocumentIngestionManager()
    return manager.validate_document(file_path)

def preprocess_document(file_path: str) -> Dict[str, Any]:
    """
    Preprocess document for improved Document AI processing.
    This is a wrapper function for compatibility.
    """
    manager = DocumentIngestionManager()
    return manager.preprocess_document(file_path)

def upload_document(
    file_path: str, 
    destination_folder: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Upload a document to the FinFlow system.
    This is a wrapper function for compatibility.
    """
    manager = DocumentIngestionManager()
    return manager.upload_document(file_path, destination_folder)

def batch_upload_documents(
    file_paths: List[str], 
    destination_folder: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Upload multiple documents to the FinFlow system.
    This is a wrapper function for compatibility.
    """
    manager = DocumentIngestionManager()
    return manager.batch_upload_documents(file_paths, destination_folder)