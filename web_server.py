#!/usr/bin/env python3
"""
FinFlow Web Server
Entry point for running the FinFlow web application with static file serving
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import the existing server application
from server.enhanced_app import create_app
from initialize_agents import initialize_system
from config.config_loader import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_web_app():
    """
    Create the complete FinFlow web application with API and static file serving.
    
    Returns:
        FastAPI: The complete web application
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize agents
        logger.info("Initializing FinFlow agents...")
        agents = initialize_system(config)
        logger.info("Agents initialized successfully")
        
        # Create the FastAPI app with API endpoints
        app = create_app(agents, config)
        
        # Define paths
        web_dir = project_root / "web"
        static_dir = web_dir / "static"
        
        # Ensure directories exist
        if not web_dir.exists():
            logger.error(f"Web directory not found: {web_dir}")
            raise FileNotFoundError(f"Web directory not found: {web_dir}")
            
        if not static_dir.exists():
            logger.error(f"Static directory not found: {static_dir}")
            raise FileNotFoundError(f"Static directory not found: {static_dir}")
        
        # Mount static files
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Serve the main HTML file at the root
        @app.get("/", include_in_schema=False)
        async def serve_index():
            """Serve the main web interface."""
            index_path = web_dir / "index.html"
            if not index_path.exists():
                logger.error(f"Index file not found: {index_path}")
                raise FileNotFoundError(f"Index file not found: {index_path}")
            return FileResponse(str(index_path))
        
        # Serve the web interface for any path that doesn't match API routes
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_spa(path: str):
            """
            Serve the single page application for client-side routing.
            This catches all routes not handled by API endpoints.
            """
            # List of API prefixes that should not serve the SPA
            api_prefixes = ['/docs', '/redoc', '/openapi.json', '/status', '/health', 
                          '/metrics', '/diagnostics', '/process', '/batch', '/workflows']
            
            # If the path starts with an API prefix, don't serve SPA
            if any(path.startswith(prefix.lstrip('/')) for prefix in api_prefixes):
                return {"error": "Not found"}, 404
            
            # For all other paths, serve the main index.html (SPA)
            index_path = web_dir / "index.html"
            return FileResponse(str(index_path))
        
        logger.info("FinFlow web application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Error creating web application: {e}")
        raise

def main():
    """Main entry point for the web server."""
    try:
        # Create the web application
        app = create_web_app()
        
        # Configure server settings
        host = os.getenv("FINFLOW_HOST", "127.0.0.1")
        port = int(os.getenv("FINFLOW_PORT", "8000"))
        reload = os.getenv("FINFLOW_RELOAD", "false").lower() == "true"
        
        logger.info(f"Starting FinFlow web server on {host}:{port}")
        logger.info(f"Web interface will be available at: http://{host}:{port}")
        logger.info(f"API documentation will be available at: http://{host}:{port}/docs")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
