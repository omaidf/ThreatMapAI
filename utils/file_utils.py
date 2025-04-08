"""
File Utilities for the AI Threat Model Map Generator.

This module provides utilities for file management, including
cleaning up output files from previous runs.
"""

import os
import logging
import click
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from importlib.metadata import version, PackageNotFoundError
from tqdm import tqdm

# Import from within utils package
from utils.common import success_msg, error_msg, warning_msg, info_msg

# Configure logger
logger = logging.getLogger(__name__)

# Define common file patterns
ANALYSIS_RESULT_FILES = [
    "analysis_results.json",
    "threat_model.json",
    "class_diagram.mmd",
    "flow_diagram.mmd",
    "threat_diagram.mmd",
    "threat_analysis_report.html"
]

EMBEDDING_STORE_FILES = [
    "embeddings.index",
    "embeddings_mapping.json"
]

def check_output_directory(output_dir: str) -> bool:
    """
    Check if output directory exists and create it if it doesn't.
    
    Args:
        output_dir: Directory to check and create
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.warning(f"Couldn't create output directory {output_dir}: {str(e)}")
        
        # Try to use a default directory if the specified one can't be created
        try:
            Path("output").mkdir(exist_ok=True)
            return True
        except Exception:
            return False

def find_previous_run_files(output_dir: str, include_embeddings: bool = False) -> Tuple[List[str], List[str]]:
    """
    Find files from previous runs in the output directory.
    
    Args:
        output_dir: Directory to check for previous run files
        include_embeddings: Whether to also check for embedding store files
        
    Returns:
        Tuple of (analysis_files, embedding_files) found
    """
    output_path = Path(output_dir)
    
    # Check if output directory exists
    if not output_path.exists():
        return [], []
    
    # Find analysis result files
    found_files = []
    for indicator in ANALYSIS_RESULT_FILES:
        file_path = output_path / indicator
        if file_path.exists():
            found_files.append(indicator)
    
    # Find embedding store files if requested
    found_embeddings = []
    if include_embeddings:
        for emb_file in EMBEDDING_STORE_FILES:
            file_path = output_path / emb_file
            if file_path.exists():
                found_embeddings.append(emb_file)
    
    return found_files, found_embeddings

def clean_previous_run(output_dir: str, force_clean: bool = False, clear_embeddings: bool = False) -> None:
    """
    Clean up data from previous runs while preserving model files.
    
    Args:
        output_dir: Directory containing output files
        force_clean: Force cleaning even without confirmation
        clear_embeddings: Whether to also clear the embedding store
    """
    output_path = Path(output_dir)
    
    # Check if output directory exists
    if not output_path.exists():
        return
    
    # Find files from previous runs
    found_files, found_embeddings = find_previous_run_files(output_dir, clear_embeddings)
    
    if not found_files and not found_embeddings:
        return
    
    # Ask for confirmation if not forced
    if not force_clean:
        if found_files:
            info_msg(f"Found {len(found_files)} files from previous run in {output_dir}:")
            for file in found_files:
                click.echo(f"  - {file}")
        
        if found_embeddings:
            info_msg(f"Found embedding store files that can be cleared:")
            for file in found_embeddings:
                click.echo(f"  - {file}")
        
        if not click.confirm("Do you want to clean up these files before running a new analysis?", default=True):
            return
    
    # Clean output files
    if found_files:
        with tqdm(total=len(found_files), desc="Cleaning previous run files") as progress:
            cleaned_count = 0
            for indicator in ANALYSIS_RESULT_FILES:
                file_path = output_path / indicator
                if file_path.exists():
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {str(e)}")
                    progress.update(1)
        success_msg(f"Cleaned {cleaned_count} files from previous run")
    
    # Clean embedding store if requested
    if clear_embeddings and found_embeddings:
        try:
            # Two options: delete the files or clear the store
            if force_clean:
                # Delete the files
                with tqdm(total=len(found_embeddings), desc="Clearing embedding store") as progress:
                    for emb_file in EMBEDDING_STORE_FILES:
                        file_path = output_path / emb_file
                        if file_path.exists():
                            file_path.unlink()
                            progress.update(1)
                success_msg(f"Removed embedding store files")
            else:
                # Just clear the store contents
                info_msg("Clearing embedding store contents...")
                try:
                    from repository_analyzer.embedding_store import EmbeddingStore
                    embedding_store = EmbeddingStore()
                    embedding_store.load()
                    embedding_store.clear()
                    success_msg("Embedding store cleared successfully")
                except Exception as e:
                    error_msg(f"Failed to clear embedding store with API: {str(e)}")
                    
                    # Fallback to directly deleting the files
                    with tqdm(total=len(found_embeddings), desc="Clearing embedding store") as progress:
                        for emb_file in EMBEDDING_STORE_FILES:
                            file_path = output_path / emb_file
                            if file_path.exists():
                                file_path.unlink()
                                progress.update(1)
                    success_msg(f"Removed embedding store files using fallback method")
        except Exception as e:
            warning_msg(f"Failed to clear embedding store: {str(e)}")

def check_required_files(output_dir: str, required_files: List[str]) -> bool:
    """
    Check if required files exist in the output directory.
    
    Args:
        output_dir: Directory to check for required files
        required_files: List of required file names
        
    Returns:
        True if all required files exist, False otherwise
    """
    output_path = Path(output_dir)
    
    for filename in required_files:
        file_path = output_path / filename
        if not file_path.exists():
            return False
    
    return True

def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    try:
        with open('requirements.txt') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            missing_deps = []
            
            for req in requirements:
                try:
                    # Parse the requirement string
                    req_name = req.split('>=')[0].split('==')[0].strip()
                    if ';' in req_name:  # Handle environment markers
                        req_name = req_name.split(';')[0].strip()
                    
                    # Try to get the version - will raise an exception if missing
                    version(req_name)
                except PackageNotFoundError:
                    missing_deps.append(req)
            
            if missing_deps:
                logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                
                if not click.confirm("Install missing dependencies?", default=True):
                    logger.warning("Continuing without installing dependencies. This may cause errors.")
                    return False
                
                for req in missing_deps:
                    logger.info(f"Installing missing dependency: {req}")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to install {req}. Continuing anyway.")
                        return False
                
                return True
            
            return True
    except Exception as e:
        logger.error(f"Failed to check/install requirements: {str(e)}")
        logger.warning("Continuing without checking dependencies. This may cause errors.")
        return False 