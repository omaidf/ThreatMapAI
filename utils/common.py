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