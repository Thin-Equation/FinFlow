#!/usr/bin/env python3
"""
Script to check for outdated dependencies in the FinFlow project.

This script:
1. Analyzes the current requirements.txt file
2. Checks for available updates to dependencies
3. Suggests updates based on semantic versioning rules
4. Generates a report of outdated packages
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Tuple
import re
from datetime import datetime

# ANSI color codes for better readability
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
RESET = '\033[0m'  # Reset color


def get_installed_packages() -> Dict[str, str]:
    """Get current installed packages with their versions."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True, check=True
    )
    packages = json.loads(result.stdout)
    return {pkg["name"]: pkg["version"] for pkg in packages}


def get_outdated_packages() -> List[Dict[str, str]]:
    """Get outdated packages information."""
    print(f"{BLUE}Checking for outdated packages...{RESET}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        capture_output=True, text=True, check=True
    )
    
    try:
        outdated = json.loads(result.stdout)
        return outdated
    except json.JSONDecodeError:
        print(f"{RED}Error parsing pip output. No outdated packages found or pip error.{RESET}")
        return []


def parse_requirements(req_path: str) -> Dict[str, str]:
    """Parse requirements.txt file and extract package constraints."""
    requirements = {}
    
    with open(req_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line != '\n':
                # Handle complex requirements like package[extra]>=1.0
                match = re.match(r'^([a-zA-Z0-9_\-.]+)(?:\[[^\]]+\])?(.*)$', line)
                if match:
                    package, version_constraint = match.groups()
                    requirements[package.lower()] = version_constraint
    
    return requirements


def is_safe_update(current: str, latest: str, constraint: str) -> Tuple[bool, str]:
    """
    Determine if update is safe based on semantic versioning and constraints.
    
    Returns:
        Tuple of (is_safe, reason)
    """
    # Parse version components
    try:
        current_parts = list(map(int, current.split('.')))
        latest_parts = list(map(int, latest.split('.')))
    except ValueError:
        # If versions contain non-numeric parts, consider it unsafe
        return False, "Complex version format"
    
    # Add zeros for shorter versions
    while len(current_parts) < 3:
        current_parts.append(0)
    while len(latest_parts) < 3:
        latest_parts.append(0)
    
    # Check for constraint compatibility
    if '>=' in constraint:
        # Should be safe as we're only specifying minimum
        return True, "Meets minimum version constraint"
    elif '==' in constraint:
        # Exact version required, only patch updates might be safe
        if current_parts[0] == latest_parts[0] and current_parts[1] == latest_parts[1]:
            return True, "Patch update only"
        return False, "Exact version constraint"
    elif '~=' in constraint or current_parts[0] != latest_parts[0]:
        # Major version change or compatible release constraint
        if current_parts[0] != latest_parts[0]:
            return False, "Major version change"
        if current_parts[1] != latest_parts[1]:
            return True, "Minor version change"
        return True, "Patch update only"
    
    # Default safe
    return True, "No specific constraint"


def generate_updated_requirements(current_reqs: Dict[str, str], 
                                 outdated: List[Dict[str, str]]) -> str:
    """Generate updated requirements file content."""
    new_content = "# Updated requirements.txt - Generated on {}\n".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Create mapping of package names to latest versions
    updates = {pkg["name"].lower(): pkg["latest_version"] for pkg in outdated}
    
    for pkg_name, constraint in current_reqs.items():
        if pkg_name.lower() in updates:
            # Get the original constraint operator
            match = re.search(r'([<>=~!]+)', constraint)
            operator = match.group(1) if match else ">="
            
            # Use the original package name to preserve casing
            original_name = next((p["name"] for p in outdated if p["name"].lower() == pkg_name.lower()), pkg_name)
            new_content += f"{original_name}{operator}{updates[pkg_name.lower()]}\n"
        else:
            # Find original case of the package name from the constraint
            new_content += f"{pkg_name}{constraint}\n"
    
    return new_content


def main():
    """Main entry point for the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    requirements_path = os.path.join(project_root, 'requirements.txt')
    
    print(f"{GREEN}=== FinFlow Dependency Update Checker ==={RESET}")
    print(f"{BLUE}Analyzing requirements in: {requirements_path}{RESET}")
    
    # Get current constraints
    current_requirements = parse_requirements(requirements_path)
    
    # Get outdated packages
    outdated_packages = get_outdated_packages()
    
    if not outdated_packages:
        print(f"{GREEN}All packages are up to date!{RESET}")
        return 0
    
    # Print outdated packages with update safety analysis
    print(f"\n{YELLOW}Found {len(outdated_packages)} outdated packages:{RESET}")
    print(f"\n{'Package':<30} {'Current':<15} {'Latest':<15} {'Update Safety':<20}")
    print("-" * 80)
    
    safe_updates = []
    unsafe_updates = []
    
    for pkg in outdated_packages:
        name = pkg["name"]
        current = pkg["version"]
        latest = pkg["latest_version"]
        
        # Get constraint for package
        constraint = current_requirements.get(name.lower(), ">=")
        
        # Check if update is safe
        is_safe, reason = is_safe_update(current, latest, constraint)
        
        safety_info = f"{GREEN}Safe{RESET}" if is_safe else f"{RED}Review{RESET}"
        
        print(f"{name:<30} {current:<15} {latest:<15} {safety_info} - {reason}")
        
        if is_safe:
            safe_updates.append(pkg)
        else:
            unsafe_updates.append(pkg)
    
    # Generate suggested updates file
    if safe_updates or unsafe_updates:
        suggested_path = os.path.join(project_root, 'requirements.suggested.txt')
        with open(suggested_path, 'w', encoding='utf-8') as f:
            f.write(generate_updated_requirements(current_requirements, outdated_packages))
        
        print(f"\n{GREEN}Suggested updates written to: requirements.suggested.txt{RESET}")
        print(f"\n{BLUE}To apply suggested updates:{RESET}")
        print("1. Review the suggested file")
        print("2. If satisfied: cp requirements.suggested.txt requirements.txt")
        print("3. Then run: pip install -r requirements.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
