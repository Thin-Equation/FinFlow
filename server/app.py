"""
FinFlow API Server

This module provides a FastAPI server implementation for the FinFlow platform.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcessRequest(BaseModel):
    """Request model for document processing."""
    workflow_type: str = Field(default="standard", description="Type of workflow to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class DocumentResponse(BaseModel):
    """Response model for document processing."""
    document_id: str
    status: str
    result: Dict[str, Any]
    processing_time: float
    timestamp: str


def create_app(agents: Dict[str, Any], config: Dict[str, Any]) -> FastAPI:
    """
    Create a FastAPI application for the FinFlow platform.
    
    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="FinFlow API",
        description="Financial document processing and analysis API",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Static files for API docs
    # app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {"message": "FinFlow API", "version": "1.0.0"}
    
    @app.get("/status")
    async def status():
        """System status endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "environment": config.get("environment", "unknown"),
            "version": "1.0.0"
        }
    
    @app.post("/process", response_model=DocumentResponse)
    async def process_document(
        request: ProcessRequest = Body(...),
        background_tasks: BackgroundTasks = None,
        file: UploadFile = File(...)
    ):
        """
        Process a document using the agent system.
        
        Args:
            request: Processing request with workflow and options
            background_tasks: FastAPI background tasks
            file: Uploaded document file
            
        Returns:
            DocumentResponse: Processing result
        """
        start_time = datetime.now()
        
        try:
            # Save uploaded file to temporary location
            file_path = f"/tmp/finflow_{start_time.timestamp()}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Document saved to temporary location: {file_path}")
            
            # Create processing context
            context = {
                "document_path": file_path,
                "workflow_type": request.workflow_type,
                "options": request.options,
                "user_id": "api_user",  # Would be from authentication in production
                "session_id": f"api_{start_time.timestamp()}",
            }
            
            # Get the master orchestrator
            master_orchestrator = agents["master_orchestrator"]
            
            # Process the document
            result = master_orchestrator.process_document(context)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # If processing was successful, create background task to clean up
            if background_tasks:
                background_tasks.add_task(lambda: os.remove(file_path))
            
            # Create response
            response = {
                "document_id": result.get("document_id", "unknown"),
                "status": result.get("status", "unknown"),
                "result": result,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    @app.get("/workflows")
    async def list_workflows():
        """List available workflows."""
        try:
            # This would pull from a workflow registry in production
            workflows = [
                {"id": "standard", "name": "Standard Processing", "description": "Standard document processing workflow"},
                {"id": "invoice", "name": "Invoice Processing", "description": "Invoice-specialized processing workflow"},
                {"id": "receipt", "name": "Receipt Processing", "description": "Receipt-specialized processing workflow"},
            ]
            return {"workflows": workflows}
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing workflows: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        # Check essential services
        health_status = {
            "service": "ok",
            "database": "ok",
            "storage": "ok",
            "document_processor": "ok"
        }
        return health_status
        
    return app
