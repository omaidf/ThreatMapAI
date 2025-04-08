"""
Common utility functions for the AI Threat Model Map Generator.

This module provides general-purpose utility functions used across the application.
"""

import os
import re
import logging
import click
from pathlib import Path
from typing import Optional, Dict, Any, List

# Define colored output styles
SUCCESS_STYLE = {'fg': 'green', 'bold': True}
ERROR_STYLE = {'fg': 'red', 'bold': True}
WARNING_STYLE = {'fg': 'yellow', 'bold': False}
INFO_STYLE = {'fg': 'blue', 'bold': False}

# Configure logger
logger = logging.getLogger(__name__)

def success_msg(message: str) -> None:
    """Print a success message in green with a checkmark."""
    click.secho(f"✅ {message}", **SUCCESS_STYLE)

def error_msg(message: str) -> None:
    """Print an error message in red."""
    click.secho(f"❌ {message}", **ERROR_STYLE)

def warning_msg(message: str) -> None:
    """Print a warning message in yellow."""
    click.secho(f"⚠️ {message}", **WARNING_STYLE)

def info_msg(message: str) -> None:
    """Print an info message in blue."""
    click.secho(f"ℹ️ {message}", **INFO_STYLE)

def get_env_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable value, optionally reading from .env file.
    
    Args:
        name: Name of the environment variable
        default: Default value if variable is not found
        
    Returns:
        Value of the environment variable or default
    """
    value = os.environ.get(name)
    
    # If not in environment but .env file exists, try to read from there
    if not value and os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
                for line in env_content.splitlines():
                    if line.strip().startswith(f'{name}='):
                        value = line.strip().split('=', 1)[1].strip()
                        # Strip quotes if present
                        if value and (value.startswith('"') and value.endswith('"') or 
                                    value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        # Set in environment for current session
                        os.environ[name] = value
                        logger.debug(f"Found {name} in .env file")
                        break
        except Exception as e:
            logger.warning(f"Error reading .env file: {str(e)}")
    
    return value if value else default

def update_env_file(name: str, value: str) -> bool:
    """
    Update or add a variable in the .env file.
    
    Args:
        name: Name of the environment variable
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create .env file if it doesn't exist
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write(f"{name}={value}\n")
            return True
            
        # Read current contents
        with open('.env', 'r') as f:
            env_content = f.read()
        
        # Check if variable is already set
        if f"{name}=" in env_content:
            # Replace existing variable
            env_content = re.sub(rf'{name}=.*', f'{name}={value}', env_content)
        else:
            # Add variable
            env_content += f"\n{name}={value}\n"
        
        # Write updated contents
        with open('.env', 'w') as f:
            f.write(env_content)
            
        # Update current session
        os.environ[name] = value
        return True
        
    except Exception as e:
        logger.error(f"Failed to update .env file: {str(e)}")
        return False

def find_files_by_pattern(directory: str, pattern: str) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files against
        
    Returns:
        List of paths to matching files
    """
    matching_files = []
    dir_path = Path(directory)
    
    if dir_path.exists() and dir_path.is_dir():
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                matching_files.append(file_path)
    
    return matching_files 