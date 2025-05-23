#!/usr/bin/env python3
"""
Dependency checker script for FinFlow.

This script scans the project for imported Python modules and checks if they are properly
listed in the requirements.txt file. It helps ensure the requirements.txt file
stays up-to-date as development progresses.
"""

import os
import re
import sys
import ast
from typing import Dict, List, Set
import importlib.metadata


def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the project."""
    python_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                python_files.append(os.path.join(dirpath, filename))
    return python_files


def extract_imports(file_path: str) -> Set[str]:
    """Extract imported modules from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports = set()
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Standard imports: import X, import X.Y
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            
            # From imports: from X import Y
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:  # Handles: from . import X
                    imports.add(node.module.split('.')[0])
    except SyntaxError:
        print(f"Error parsing {file_path}. Skipping.")
    
    return imports


def parse_requirements(req_path: str) -> Dict[str, str]:
    """Parse requirements.txt file and extract package names and versions."""
    requirements = {}
    with open(req_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line != '\n':
                # Handle complex requirements like package[extra]>=1.0
                match = re.match(r'^([a-zA-Z0-9_\-.]+)(?:\[[^\]]+\])?(.*)$', line)
                if match:
                    package, version = match.groups()
                    requirements[package.lower()] = version
    return requirements


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of the Python standard library."""
    stdlib_modules = sys.stdlib_module_names
    return module_name in stdlib_modules


def get_installed_packages() -> Dict[str, str]:
    """Get a mapping of all installed packages to their module names."""
    package_to_module = {}
    for dist in importlib.metadata.distributions():
        try:
            top_level_modules = dist.read_text('top_level.txt')
            if top_level_modules:
                for module in top_level_modules.splitlines():
                    if module:  # Skip empty lines
                        package_to_module[module] = dist.metadata['Name']
        except Exception:
            # Some packages may not have top_level.txt
            pass
    
    return package_to_module


def main():
    """Main entry point for the dependency checker."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    requirements_path = os.path.join(project_root, 'requirements.txt')
    
    # Get all Python files
    python_files = find_python_files(project_root)
    
    # Parse requirements file
    requirements = parse_requirements(requirements_path)
    
    # Get mapping of modules to package names
    module_to_package = get_installed_packages()
    
    # All imports from project files
    all_imports = set()
    for file_path in python_files:
        file_imports = extract_imports(file_path)
        all_imports.update(file_imports)
    
    # Filter out standard library modules
    external_imports = {module for module in all_imports if not is_stdlib_module(module)}
    
    # Check if imports are covered by requirements
    missing_requirements = []
    for module in external_imports:
        package_name = module_to_package.get(module, module).lower()
        if package_name not in requirements and module != 'finflow':
            missing_requirements.append((module, package_name))
    
    # Print results
    if missing_requirements:
        print("The following imports may be missing from requirements.txt:")
        for module, package in missing_requirements:
            print(f"  - {module} (package: {package})")
        print("\nConsider adding them to maintain proper dependency documentation.")
        return 1
    else:
        print("All external imports appear to be properly listed in requirements.txt.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
