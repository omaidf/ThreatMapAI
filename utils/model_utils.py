"""
Model Utilities for the AI Threat Model Map Generator.

This module provides utilities for downloading and validating the LLM model files.
Functions for architecture detection and model configuration have been moved to model_config.py.
"""

import os
import logging
import shutil
import click
import sys
import subprocess
import requests
import dotenv
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import torch

# Import from within utils package
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable, update_env_file
from utils.model_config import detect_architecture, get_model_info, get_default_model_path

# Import the function from the new gpu_utils module for backward compatibility
# This ensures existing code will continue to work when importing from model_utils
from utils.gpu_utils import detect_gpu_capabilities

# Configure logger
logger = logging.getLogger(__name__)

def validate_model_path(ctx=None, param=None, value=None) -> str:
    """
    Validate the model path exists. Can be used as a Click callback or directly.
    
    Args:
        ctx: Click context (for Click callback compatibility)
        param: Click parameter (for Click callback compatibility)
        value: The model path value to validate
        
    Returns:
        Validated model path
    """
    # Support both usage patterns:
    # 1. As a Click callback: validate_model_path(ctx, param, value)
    # 2. Direct function call: validate_model_path(model_path)
    
    # Handle direct function call with positional arg
    if ctx is not None and param is None and value is None:
        # If only one arg is provided, assume it's the model path
        model_path = ctx
        ctx = None
    else:
        # Normal Click callback usage
        model_path = value
    
    if not model_path:
        # Use default model path
        model_path = get_default_model_path()
        
    if not Path(model_path).exists():
        # Warning but don't raise error
        logger.warning(f"Model not found at {model_path}. We'll try to use it anyway or provide fallback.")
    
    return model_path

def check_model_file(model_path: str) -> Tuple[bool, float]:
    """
    Check if the model file exists and is a valid size.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (exists, size_in_gb)
    """
    path = Path(model_path)
    
    if not path.exists():
        return False, 0.0
        
    # Check file size
    file_size = path.stat().st_size
    file_size_gb = file_size / (1024**3)
    
    return True, file_size_gb

def set_token_interactive() -> str:
    """
    Set Hugging Face token interactively.
    
    Returns:
        Token string or empty string if not set
    """
    try:
        # Prompt for token
        token = click.prompt(
            "Enter your Hugging Face token from https://huggingface.co/settings/tokens\n"
            "1. Create a free account at huggingface.co if you don't have one\n"
            "2. Go to https://huggingface.co/settings/tokens\n"
            "3. Create a new token (read access is sufficient)",
            hide_input=True, 
            default="",
            show_default=False
        )
        
        if not token:
            error_msg("No token provided. Cannot download the model.")
            return ""
            
        # Update environment and .env file
        if update_env_file("HF_TOKEN", token):
            success_msg("Hugging Face token set successfully!")
        
        return token
        
    except Exception as e:
        logger.error(f"Failed to set token: {str(e)}")
        return ""

def download_model(model_path: str, force: bool = False) -> str:
    """
    Download the model using huggingface_hub or direct download as fallback.
    
    Args:
        model_path: Path to save the model to
        force: Force re-download even if the model exists
        
    Returns:
        Path to the downloaded model
    """
    try:
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Store the original requested path
        original_model_path = model_path
            
        # First check if model already exists and is valid
        exists, file_size_gb = check_model_file(model_path)
        if exists and not force and file_size_gb >= 1.0:
            logger.info(f"Model already exists at {model_path} ({file_size_gb:.2f} GB)")
            return model_path
        
        # Get model name from the path
        if "70b" in model_path.lower():
            model_name = "codellama-70b-instruct"
        elif "7b" in model_path.lower():
            model_name = "codellama-7b-instruct"
        else:
            model_name = "codellama-7b-instruct"  # Default fallback
            
        model_variant = None
        
        # Try to determine variant from path
        if "Q4_0" in model_path:
            model_variant = "Q4_0"
        elif "Q4_K_M" in model_path:
            model_variant = "Q4_K_M"
        elif "Q5_K_M" in model_path:
            model_variant = "Q5_K_M"
        
        # Get model information from centralized config
        model_info = get_model_info(model_name, model_variant)
        repo_id = model_info["repo_id"]
        filename = model_info["filename"]
        
        # If we didn't determine a variant from the path, update model_path
        if not model_variant:
            model_path = os.path.join(os.path.dirname(original_model_path), filename)
            logger.info(f"Using model variant {model_info['variant']}, path updated to: {model_path}")
        
        logger.info(f"Downloading model {model_info['name']} ({model_info['variant']}) from {repo_id}")
        
        # Check if we need Hugging Face authentication token
        # First try to get it from the .env file directly
        dotenv.load_dotenv()  # Reload .env file to get the most current token
        
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            warning_msg("⚠️ No Hugging Face token found in environment or .env file")
            warning_msg("The model requires authentication to download")
            
            # Only ask for token if it's not already set
            if click.confirm("Would you like to set your Hugging Face token now?", default=True):
                hf_token = set_token_interactive()
                if not hf_token:
                    raise Exception("Cannot download model without a valid Hugging Face token")
            else:
                raise Exception("Cannot download model without a valid Hugging Face token. Run: python -m cli set_token")
        else:
            info_msg("Using Hugging Face token found in .env file")
        
        download_successful = False
        
        # First attempt: Try using huggingface_hub
        try:
            # Import huggingface_hub here to avoid requiring it at module level
            from huggingface_hub import hf_hub_download
            
            with tqdm(total=1, desc=f"Downloading {filename} with HF Hub") as progress:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=hf_token,
                    force_download=force,
                    cache_dir=os.path.dirname(model_path)
                )
                progress.update(1)
            
            # Copy from cache to our target location if different
            if downloaded_path != model_path:
                shutil.copy(downloaded_path, model_path)
                logger.info(f"Copied model from {downloaded_path} to {model_path}")
            
            download_successful = True
            
        except Exception as hf_error:
            warning_msg(f"HuggingFace Hub download failed: {str(hf_error)}")
            warning_msg("Trying alternative direct download method...")
        
        # Second attempt: Direct download if huggingface_hub fails
        if not download_successful:
            # Construct the direct download URL
            direct_url = model_info["url"]
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            
            info_msg(f"Attempting direct download from: {direct_url}")
            
            try:
                # Stream download with progress bar
                with requests.get(direct_url, headers=headers, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(model_path, 'wb') as f, tqdm(
                        desc=f"Downloading {filename} directly",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(len(chunk))
                
                download_successful = True
                
            except Exception as direct_error:
                error_msg(f"Direct download failed: {str(direct_error)}")
                raise Exception(f"All download methods failed. First error: {str(hf_error)}. Second error: {str(direct_error)}")
        
        # Verify file size
        exists, file_size_gb = check_model_file(model_path)
        if not exists or file_size_gb < 0.5:  # At least 500MB
            error_msg(f"Model file is too small or doesn't exist: {file_size_gb:.1f}GB")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception("Model download appears incomplete, file is too small")
                
        success_msg(f"Model downloaded successfully to {model_path} ({file_size_gb:.2f}GB)")
        
        # Update environment variable and .env file
        update_env_file("MODEL_PATH", model_path)
        
        return model_path
    except Exception as e:
        error_msg(f"Failed to download model: {str(e)}")
        raise Exception(f"Failed to download model: {str(e)}")

def test_model_loading(model_path: str) -> bool:
    """
    Test if the model can be loaded.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if the model can be loaded, False otherwise
    """
    try:
        # First, check if the file exists
        if not os.path.exists(model_path):
            warning_msg(f"Model file not found at {model_path}")
            return False
            
        # Check file size to avoid trying to load empty files
        file_size = os.path.getsize(model_path)
        if file_size < 100_000:  # Less than 100KB
            warning_msg(f"Model file is too small ({file_size/1024:.1f} KB)")
            return False
            
        # Try to initialize the model using LlamaCpp
        # This is a lightweight way to test without loading the full model
        try:
            from langchain_community.llms import LlamaCpp
            
            # Get optimal GPU configuration
            gpu_config = detect_gpu_capabilities()
            n_gpu_layers = gpu_config['n_gpu_layers'] if gpu_config['use_gpu'] else 0
            
            # Special handling for -1 value (all layers)
            if n_gpu_layers == -1:
                n_gpu_layers = 100  # LlamaCpp needs a high number to use all layers
            
            # Try to load the model with minimal settings and verify it works
            model = LlamaCpp(
                model_path=model_path,
                n_ctx=512,          # Use small context to load faster
                n_batch=1,          # Minimal batch size
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=False,      # Don't log everything
                f16_kv=True,        # Use FP16 for key/value cache
                use_mlock=True,     # Keep model in memory
                n_threads=1         # Use minimal threads
            )
            
            # Verify model works by generating a small test output
            result = model.invoke("test")
            
            # If we got a result, even if empty, model is working
            success_msg("Model loaded successfully")
            return True
        except Exception as llama_err:
            warning_msg(f"LlamaCpp loading failed: {str(llama_err)}")
            
            # Try with Hugging Face approach
            try:
                warning_msg("Trying to load with Hugging Face transformers...")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Try to load tokenizer as a minimal test
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # If we have GPU, try minimal load
                if torch.cuda.is_available():
                    # Try loading with low memory usage
                    AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        load_in_8bit=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        max_memory={0: "1GB"},  # Limit memory usage
                    )
                
                # If we got here, things worked
                success_msg("Model loaded successfully with Hugging Face approach")
                return True
            except Exception as hf_err:
                warning_msg(f"Hugging Face loading failed: {str(hf_err)}")
                
                # Final fallback - just check if the file is a valid GGUF file
                if model_path.endswith(".gguf"):
                    try:
                        with open(model_path, "rb") as f:
                            header = f.read(4)
                            # Check GGUF header
                            if header.startswith(b'GGUF'):
                                success_msg("File appears to be a valid GGUF model, but loading failed")
                                return True
                            else:
                                warning_msg("File does not appear to be a valid GGUF model")
                    except Exception as e:
                        warning_msg(f"Failed to read model file header: {str(e)}")
                
                # All loading attempts failed
                return False
    except Exception as e:
        warning_msg(f"Error testing model loading: {str(e)}")
        return False 