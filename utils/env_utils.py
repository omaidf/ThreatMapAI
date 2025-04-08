"""
Environment variable utilities for the AI Threat Model Map Generator.

This module provides utilities for managing environment variables and
the .env file used for configuration.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Configure logger
logger = logging.getLogger(__name__)

def get_env_variable(name: str, default: str = None) -> str:
    """
    Get an environment variable, with optional default value.
    
    Args:
        name: Name of the environment variable
        default: Default value if not found
        
    Returns:
        Value of the environment variable, or default if not found
    """
    value = os.environ.get(name, default)
    return value

def load_dotenv(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dictionary of environment variables loaded from the file
    """
    env_vars = {}
    
    try:
        env_file = Path(env_path)
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        env_vars[key] = value
                        os.environ[key] = value
        
        return env_vars
    
    except Exception as e:
        logger.warning(f"Failed to load .env file: {str(e)}")
        return {}

def update_env_file(key: str, value: str, env_path: str = ".env") -> bool:
    """
    Update or add an environment variable in the .env file.
    
    Args:
        key: Key to update
        value: New value to set
        env_path: Path to the .env file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        env_file = Path(env_path)
        
        # Read existing file if it exists
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                logger.warning(f"Failed to read .env file: {str(e)}")
                lines = []
        else:
            # File doesn't exist, create an empty file
            lines = []
            # Ensure parent directories exist
            env_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update or add the variable
        key_found = False
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                line_key = line.split('=', 1)[0].strip()
                if line_key == key:
                    lines[i] = f"{key}={value}\n"
                    key_found = True
                    break
        
        # If key not found, add it
        if not key_found:
            lines.append(f"{key}={value}\n")
        
        # Write back to file
        try:
            with open(env_file, 'w') as f:
                f.writelines(lines)
        except Exception as e:
            logger.error(f"Failed to write to .env file: {str(e)}")
            return False
        
        # Update the current environment
        os.environ[key] = value
        
        logger.info(f"Updated environment variable {key} in .env file")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update .env file: {str(e)}")
        return False

def get_all_env_vars() -> Dict[str, str]:
    """
    Get all environment variables as a dictionary.
    
    Returns:
        Dictionary of all environment variables
    """
    return dict(os.environ) 