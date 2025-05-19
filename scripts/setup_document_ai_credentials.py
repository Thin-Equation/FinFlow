#!/usr/bin/env python3
"""
Script to set up and verify Google Document AI credentials.
This script will:
1. Check for existing credentials
2. Guide through setting up credentials if needed
3. Test connection to Document AI APIs
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finflow.setup")

def check_google_cloud_sdk_installed() -> bool:
    """Check if Google Cloud SDK is installed."""
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
        logger.info("Google Cloud SDK is installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Google Cloud SDK not found. Please install it from https://cloud.google.com/sdk/docs/install")
        return False

def check_authenticated() -> bool:
    """Check if user is authenticated with Google Cloud."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list"], 
            capture_output=True, 
            check=True, 
            text=True
        )
        if "No credentialed accounts." in result.stdout:
            logger.warning("No Google Cloud authenticated accounts found.")
            return False
        else:
            logger.info("Google Cloud authentication detected.")
            return True
    except subprocess.SubprocessError:
        logger.warning("Error checking Google Cloud authentication status.")
        return False

def get_active_project() -> Optional[str]:
    """Get the active Google Cloud project."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"], 
            capture_output=True, 
            check=True, 
            text=True
        )
        project_id = result.stdout.strip()
        if project_id:
            logger.info(f"Active Google Cloud project: {project_id}")
            return project_id
        else:
            logger.warning("No active Google Cloud project set.")
            return None
    except subprocess.SubprocessError:
        logger.warning("Error getting active Google Cloud project.")
        return None

def create_credentials_file(credentials_path: str) -> bool:
    """Create a service account key file for Document AI."""
    try:
        # First, create a service account if needed
        service_account_name = "finflow-document-ai"
        project_id = get_active_project()
        
        if not project_id:
            logger.error("No active project. Please set a project with: gcloud config set project YOUR_PROJECT_ID")
            return False
        
        # Check if service account exists
        result = subprocess.run(
            ["gcloud", "iam", "service-accounts", "list", "--format=json"], 
            capture_output=True, 
            check=True, 
            text=True
        )
        
        service_accounts = json.loads(result.stdout)
        service_account_email = None
        
        for account in service_accounts:
            if service_account_name in account.get("displayName", ""):
                service_account_email = account.get("email")
                break
        
        # Create service account if it doesn't exist
        if not service_account_email:
            logger.info(f"Creating service account {service_account_name}...")
            subprocess.run([
                "gcloud", "iam", "service-accounts", "create", service_account_name,
                "--display-name", "FinFlow Document AI Service Account"
            ], check=True)
            
            service_account_email = f"{service_account_name}@{project_id}.iam.gserviceaccount.com"
        
        # Grant Document AI permissions to the service account
        logger.info(f"Granting Document AI permissions to service account {service_account_email}...")
        subprocess.run([
            "gcloud", "projects", "add-iam-policy-binding", project_id,
            "--member", f"serviceAccount:{service_account_email}",
            "--role", "roles/documentai.admin"
        ], check=True)
        
        # Create and download key
        logger.info("Creating and downloading service account key...")
        key_path = os.path.dirname(credentials_path)
        os.makedirs(key_path, exist_ok=True)
        
        subprocess.run([
            "gcloud", "iam", "service-accounts", "keys", "create",
            credentials_path,
            "--iam-account", service_account_email
        ], check=True)
        
        logger.info(f"Service account key created and saved to {credentials_path}")
        return True
    
    except subprocess.SubprocessError as e:
        logger.error(f"Error creating credentials: {e}")
        return False

def check_document_ai_api_enabled() -> bool:
    """Check if Document AI API is enabled for the current project."""
    try:
        project_id = get_active_project()
        if not project_id:
            return False
        
        result = subprocess.run(
            ["gcloud", "services", "list", "--format=json"], 
            capture_output=True, 
            check=True, 
            text=True
        )
        
        services = json.loads(result.stdout)
        for service in services:
            if "documentai.googleapis.com" in service.get("config", {}).get("name", ""):
                logger.info("Document AI API is enabled.")
                return True
        
        logger.warning("Document AI API is not enabled.")
        return False
    
    except subprocess.SubprocessError:
        logger.warning("Error checking Document AI API status.")
        return False

def enable_document_ai_api() -> bool:
    """Enable the Document AI API."""
    try:
        project_id = get_active_project()
        if not project_id:
            return False
        
        logger.info("Enabling Document AI API...")
        subprocess.run([
            "gcloud", "services", "enable", "documentai.googleapis.com"
        ], check=True)
        
        logger.info("Document AI API enabled successfully.")
        return True
    
    except subprocess.SubprocessError as e:
        logger.error(f"Error enabling Document AI API: {e}")
        return False

def test_document_ai_connection(credentials_path: str) -> bool:
    """Test connection to Document AI API using credentials."""
    try:
        # Set environment variable for credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        from google.cloud import documentai_v1 as documentai
        
        # Try to list processors to verify connection
        client = documentai.DocumentProcessorServiceClient()
        
        # Get project ID from the credentials file
        with open(credentials_path, 'r') as f:
            credentials_data = json.load(f)
            project_id = credentials_data.get("project_id")
        
        # List locations
        parent = f"projects/{project_id}/locations"
        request = documentai.ListLocationsRequest(name=parent)
        
        # This will throw an exception if credentials are invalid
        locations_client = client.transport.parent.locations_client
        locations_response = locations_client.list_locations(request)
        
        logger.info("Successfully connected to Document AI API!")
        logger.info("Available locations:")
        for location in locations_response.locations:
            logger.info(f" - {location.location_id}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing Document AI connection: {e}")
        return False

def update_config_file(credentials_path: str, project_id: str) -> bool:
    """Update the project configuration with Document AI settings."""
    try:
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        
        for env in ["development", "staging", "production"]:
            config_path = os.path.join(config_dir, f"{env}.yaml")
            
            if os.path.exists(config_path):
                # Read the current config
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # Check if google_cloud section exists and update it
                if "google_cloud:" in config_content:
                    import re
                    # Update project_id
                    config_content = re.sub(
                        r'google_cloud:(?:\s+.*?)*?\s+project_id:.*', 
                        f'google_cloud:\n  project_id: "{project_id}"', 
                        config_content, 
                        flags=re.DOTALL
                    )
                    
                    # Update credentials_path
                    config_content = re.sub(
                        r'google_cloud:(?:\s+.*?)*?\s+credentials_path:.*', 
                        f'google_cloud:\n  project_id: "{project_id}"\n  credentials_path: "{credentials_path}"', 
                        config_content, 
                        flags=re.DOTALL
                    )
                else:
                    # Add google_cloud section if it doesn't exist
                    config_content += f'\n\ngoogle_cloud:\n  project_id: "{project_id}"\n  credentials_path: "{credentials_path}"\n'
                
                # Write the updated config
                with open(config_path, 'w') as f:
                    f.write(config_content)
                
                logger.info(f"Updated {env} configuration with Document AI settings")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating config files: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("Google Document AI Setup Assistant")
    print("=" * 80)
    
    # Define default paths
    home_dir = str(Path.home())
    credentials_dir = os.path.join(home_dir, ".config", "finflow")
    credentials_path = os.path.join(credentials_dir, "document-ai-credentials.json")
    
    # Check if credentials already exist
    if os.path.exists(credentials_path):
        print(f"\nCredentials file already exists at: {credentials_path}")
        use_existing = input("Do you want to use the existing credentials? (y/n): ").lower() == 'y'
        
        if not use_existing:
            print("\nLet's set up new credentials.")
        else:
            # Test existing credentials
            print("\nTesting existing credentials...")
            if test_document_ai_connection(credentials_path):
                print("\nExisting credentials are working correctly!")
                
                # Update config with existing credentials
                with open(credentials_path, 'r') as f:
                    credentials_data = json.load(f)
                    project_id = credentials_data.get("project_id")
                
                if project_id and update_config_file(credentials_path, project_id):
                    print("\nConfiguration updated successfully!")
                return
            else:
                print("\nExisting credentials are not working. Let's set up new ones.")
    
    # Check if Google Cloud SDK is installed
    if not check_google_cloud_sdk_installed():
        print("\nPlease install Google Cloud SDK and run this script again.")
        print("Installation instructions: https://cloud.google.com/sdk/docs/install")
        return
    
    # Check if user is authenticated
    if not check_authenticated():
        print("\nYou need to authenticate with Google Cloud.")
        print("Run the following command and follow the instructions:")
        print("  gcloud auth login")
        return
    
    # Get active project
    project_id = get_active_project()
    if not project_id:
        print("\nPlease set a Google Cloud project:")
        print("  gcloud config set project YOUR_PROJECT_ID")
        return
    
    # Check and enable Document AI API if needed
    if not check_document_ai_api_enabled():
        print("\nDocument AI API needs to be enabled.")
        enable = input("Do you want to enable the Document AI API now? (y/n): ").lower() == 'y'
        
        if enable:
            if not enable_document_ai_api():
                print("\nFailed to enable Document AI API. Please enable it manually in the Google Cloud Console.")
                return
        else:
            print("\nPlease enable the Document AI API manually and run this script again.")
            return
    
    # Create credentials file
    print(f"\nCreating credentials file at: {credentials_path}")
    if not create_credentials_file(credentials_path):
        print("\nFailed to create credentials file.")
        return
    
    # Test connection
    print("\nTesting connection to Document AI API...")
    if test_document_ai_connection(credentials_path):
        print("\nDocument AI credentials set up successfully!")
        
        # Update config files
        if update_config_file(credentials_path, project_id):
            print("\nConfiguration updated successfully!")
            
            print("\nNext steps:")
            print("1. Create a Document AI processor in the Google Cloud Console")
            print("2. Update the processor ID in config/document_processor_config.py")
            print("3. Run the document processor tests with real documents")
    else:
        print("\nFailed to connect to Document AI API. Please check your credentials.")

if __name__ == "__main__":
    main()
