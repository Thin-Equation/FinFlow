"""
CLI interface for the FinFlow system.

This module provides an interactive command-line interface for the FinFlow system.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import cmd
import argparse
import shlex

logger = logging.getLogger(__name__)


class FinFlowCLI(cmd.Cmd):
    """Interactive CLI for the FinFlow system."""
    
    intro = "Welcome to the FinFlow CLI. Type help or ? to list commands.\n"
    prompt = "finflow> "
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the CLI.
        
        Args:
            agents: Dictionary of initialized agents
            config: Configuration dictionary
        """
        super().__init__()
        self.agents = agents
        self.config = config
        self.current_session_id = f"cli_{datetime.now().timestamp()}"
        self.master_orchestrator = agents["master_orchestrator"]
        self.env = config.get("environment", "development")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def do_exit(self, arg):
        """Exit the FinFlow CLI."""
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the FinFlow CLI."""
        return self.do_exit(arg)
    
    def do_config(self, arg):
        """Display or modify configuration."""
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(prog="config")
        parser.add_argument("action", choices=["show", "set", "reload"], help="Action to perform")
        parser.add_argument("key", nargs="?", help="Configuration key")
        parser.add_argument("value", nargs="?", help="Value to set")
        
        try:
            parsed_args = parser.parse_args(args)
            
            if parsed_args.action == "show":
                if parsed_args.key:
                    # Show specific key
                    keys = parsed_args.key.split(".")
                    value = self.config
                    for key in keys:
                        if key in value:
                            value = value[key]
                        else:
                            print(f"Configuration key not found: {parsed_args.key}")
                            return
                    print(f"{parsed_args.key} = {json.dumps(value, indent=2)}")
                else:
                    # Show all
                    print(json.dumps(self.config, indent=2))
                    
            elif parsed_args.action == "reload":
                # Reload configuration
                from config.config_loader import load_config
                self.config = load_config(reload=True)
                print("Configuration reloaded.")
                
            elif parsed_args.action == "set":
                # Set configuration value
                if not parsed_args.key or not parsed_args.value:
                    print("Both key and value are required for set action.")
                    return
                    
                # Parse the value as JSON if possible
                try:
                    value = json.loads(parsed_args.value)
                except json.JSONDecodeError:
                    value = parsed_args.value
                
                # Set the value
                keys = parsed_args.key.split(".")
                target = self.config
                for key in keys[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                target[keys[-1]] = value
                
                print(f"Set {parsed_args.key} = {value}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def do_process(self, arg):
        """Process a document."""
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(prog="process")
        parser.add_argument("document_path", help="Path to document to process")
        parser.add_argument("--workflow", help="Workflow type", default="standard")
        
        try:
            parsed_args = parser.parse_args(args)
            
            if not os.path.exists(parsed_args.document_path):
                print(f"Document not found: {parsed_args.document_path}")
                return
            
            print(f"Processing document: {parsed_args.document_path}")
            print(f"Using workflow: {parsed_args.workflow}")
            
            # Create context
            context = {
                "document_path": parsed_args.document_path,
                "workflow_type": parsed_args.workflow,
                "user_id": "cli_user",
                "session_id": self.current_session_id,
            }
            
            # Process document
            start_time = time.time()
            result = self.master_orchestrator.process_document(context)
            processing_time = time.time() - start_time
            
            print(f"Processing completed in {processing_time:.2f} seconds")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
    
    def do_agents(self, arg):
        """List or inspect agents."""
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(prog="agents")
        parser.add_argument("action", choices=["list", "info"], help="Action to perform")
        parser.add_argument("agent", nargs="?", help="Agent name")
        
        try:
            parsed_args = parser.parse_args(args)
            
            if parsed_args.action == "list":
                # List agents
                print("Available agents:")
                for name in self.agents:
                    print(f"  - {name}")
                    
            elif parsed_args.action == "info":
                # Show agent info
                if not parsed_args.agent:
                    print("Agent name is required for info action.")
                    return
                    
                if parsed_args.agent not in self.agents:
                    print(f"Agent not found: {parsed_args.agent}")
                    return
                
                agent = self.agents[parsed_args.agent]
                print(f"Agent: {parsed_args.agent}")
                print(f"  Type: {type(agent).__name__}")
                
                # Get agent attributes
                if hasattr(agent, "name"):
                    print(f"  Name: {agent.name}")
                if hasattr(agent, "description"):
                    print(f"  Description: {agent.description}")
                if hasattr(agent, "model"):
                    print(f"  Model: {agent.model}")
                if hasattr(agent, "tools") and agent.tools:
                    print("  Tools:")
                    for tool in agent.tools:
                        print(f"    - {tool.name}: {tool.description}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def do_workflow(self, arg):
        """Run or list workflows."""
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(prog="workflow")
        parser.add_argument("action", choices=["run", "list"], help="Action to perform")
        parser.add_argument("workflow", nargs="?", help="Workflow name")
        parser.add_argument("--document", help="Document path for workflow")
        
        try:
            parsed_args = parser.parse_args(args)
            
            if parsed_args.action == "list":
                # List workflows
                print("Available workflows:")
                print("  - standard: Standard document processing workflow")
                print("  - invoice: Invoice processing workflow")
                print("  - receipt: Receipt processing workflow")
                
            elif parsed_args.action == "run":
                # Run workflow
                if not parsed_args.workflow:
                    print("Workflow name is required for run action.")
                    return
                
                if not parsed_args.document:
                    print("Document path is required for workflow run.")
                    return
                
                if not os.path.exists(parsed_args.document):
                    print(f"Document not found: {parsed_args.document}")
                    return
                
                from workflow.workflow_runner import run_workflow
                
                print(f"Running workflow: {parsed_args.workflow}")
                result = run_workflow(
                    workflow_name=parsed_args.workflow,
                    agents=self.agents,
                    config=self.config,
                    document_path=parsed_args.document
                )
                
                print("Workflow result:")
                print(json.dumps(result, indent=2))
                
        except Exception as e:
            print(f"Error: {e}")
    
    def do_status(self, arg):
        """Show system status."""
        print(f"FinFlow Status ({self.env} environment)")
        print("-" * 50)
        print(f"Environment: {self.env}")
        print(f"Session ID: {self.current_session_id}")
        print(f"Agents: {len(self.agents)}")
        # Add more status info as needed
        print("-" * 50)


def run_cli(agents: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Run the FinFlow CLI.
    
    Args:
        agents: Dictionary of initialized agents
        config: Configuration dictionary
    """
    cli = FinFlowCLI(agents=agents, config=config)
    cli.cmdloop()
