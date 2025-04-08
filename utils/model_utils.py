"""
Model Utilities for the AI Threat Model Map Generator.

This module provides utilities for downloading and validating the LLM model files.
Functions for architecture detection and model configuration have been moved to model_config.py.
"""

import os
import logging
import shutil
import click
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

# Import from within utils package
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable, update_env_file
from utils.model_config import detect_architecture, get_model_info, get_default_model_path

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
        
        # Extract model information from the path and get proper config
        model_name = "codellama-7b-instruct"  # Default
        model_variant = None
        
        # Try to determine variant from path
        if "Q4_0" in model_path:
            model_variant = "Q4_0"
        elif "Q4_K_M" in model_path:
            model_variant = "Q4_K_M"
        
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
        hf_token = get_env_variable("HF_TOKEN")
        if not hf_token:
            warning_msg("⚠️ No Hugging Face token found in environment or .env file")
            warning_msg("The model requires authentication to download")
            
            if click.confirm("Would you like to set your Hugging Face token now?", default=True):
                hf_token = set_token_interactive()
                if not hf_token:
                    raise Exception("Cannot download model without a valid Hugging Face token")
            else:
                raise Exception("Cannot download model without a valid Hugging Face token. Run: python -m cli set_token")
        
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
            import requests
            
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
    Test if the model can be loaded without segmentation faults.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if the model can be loaded, False otherwise
    """
    try:
        # First check if we have LlamaCpp available
        try:
            # Only import the needed module rather than the full processor
            from langchain_community.llms import LlamaCpp
            
            # Check if the file exists and has a good size before attempting to load
            exists, file_size_gb = check_model_file(model_path)
            if not exists:
                logger.error(f"Model file does not exist: {model_path}")
                return False
                
            if file_size_gb < 1.0:
                logger.warning(f"Model file seems too small ({file_size_gb:.2f} GB). It may be incomplete or corrupted.")
                
            # Set minimal context window and parameters to avoid memory issues
            os.environ["LLM_N_CTX"] = "512"  # Very small context window just for testing
            os.environ["LLM_N_BATCH"] = "128"  # Small batch size
            os.environ["LLM_N_GPU_LAYERS"] = "0"  # No GPU offloading in test
            
            # Create subprocess to test loading - safer approach to avoid segfaults in main process
            import subprocess
            import sys
            
            # Create a simple Python command to try loading the model in a separate process
            cmd = [
                sys.executable, 
                "-c", 
                f"from langchain_community.llms import LlamaCpp; print('Loading model...'); LlamaCpp(model_path='{model_path}', n_ctx=512, n_batch=128, verbose=False); print('Success!')"
            ]
            
            # Run with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check if loading succeeded
            if "Success!" in result.stdout and result.returncode == 0:
                logger.info("Model loading test succeeded")
                return True
            else:
                logger.warning(f"Model loading test failed with output: {result.stdout}")
                return False
                
        except (ImportError, ModuleNotFoundError):
            # LlamaCpp not available, try another approach
            logger.warning("LlamaCpp not available, trying alternative approach")
            
            # Try to check GGUF file sanity if possible
            if model_path.endswith('.gguf'):
                # Just check file header to see if it's a valid GGUF format
                with open(model_path, 'rb') as f:
                    header = f.read(4)
                    if header == b'GGUF':
                        logger.info("Model file has valid GGUF header")
                        return True
            
            # If we can't do a proper test, assume it's okay if the file exists and is large enough
            exists, file_size_gb = check_model_file(model_path)
            return exists and file_size_gb >= 1.0
            
    except Exception as e:
        logger.error(f"Failed to test model loading: {str(e)}")
        return False 

def detect_gpu_capabilities() -> dict:
    """
    Detect GPU capabilities and return appropriate configurations for LLMs.
    
    This function detects if CUDA/GPU is available and returns appropriate
    configuration settings for model loading.
    
    Returns:
        Dictionary with configuration including:
        - 'device': 'cuda', 'mps', or 'cpu'
        - 'n_gpu_layers': Number of layers to offload to GPU
        - 'use_gpu': Boolean indicating if GPU is available
        - 'gpu_info': Information about detected GPU
    """
    result = {
        'device': 'cpu',
        'n_gpu_layers': 0,
        'use_gpu': False,
        'gpu_info': 'No GPU detected'
    }
    
    try:
        # Try importing torch to check for CUDA/MPS availability
        import torch
        
        # First check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            # Get CUDA information
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown NVIDIA GPU"
            gpu_memory = None
            
            try:
                # Try to get GPU memory info if available
                gpu_properties = torch.cuda.get_device_properties(0)
                if hasattr(gpu_properties, 'total_memory'):
                    gpu_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert to GB
            except:
                pass
            
            # Configure based on GPU capabilities
            result['device'] = 'cuda'
            result['use_gpu'] = True
            result['gpu_info'] = f"{gpu_name} ({gpu_count} available)"
            
            # Check if we're using the 70B model
            try:
                from utils.model_config import get_model_info
                model_info = get_model_info()
                is_large_model = "70b" in model_info["name"].lower()
            except:
                is_large_model = True  # Assume large model to be safer
            
            # Special case for multiple GPUs
            if gpu_count > 1:
                # With multiple GPUs, be more aggressive with layer offloading
                if is_large_model:
                    if gpu_count >= 4:
                        # With 4+ GPUs, we can offload more layers for large models
                        result['n_gpu_layers'] = 64
                    else:
                        # With 2-3 GPUs, be more conservative for 70B model
                        result['n_gpu_layers'] = 32
                else:
                    # For smaller models, we can use more layers with fewer GPUs
                    result['n_gpu_layers'] = 100  # All layers
            # Single GPU case
            elif gpu_memory is not None:
                # Decide based on GPU memory and model size
                if is_large_model:
                    # More conservative for large models
                    if gpu_memory < 8:
                        result['n_gpu_layers'] = 1   # Very limited
                    elif gpu_memory < 16:
                        result['n_gpu_layers'] = 8   # Limited
                    elif gpu_memory < 24:
                        result['n_gpu_layers'] = 16  # Moderate
                    elif gpu_memory < 32:
                        result['n_gpu_layers'] = 24  # Good
                    else:
                        result['n_gpu_layers'] = 32  # Very good
                else:
                    # More aggressive for smaller models
                    if gpu_memory < 4:
                        result['n_gpu_layers'] = 1
                    elif gpu_memory < 8:
                        result['n_gpu_layers'] = 8
                    elif gpu_memory < 12:
                        result['n_gpu_layers'] = 16
                    elif gpu_memory < 24:
                        result['n_gpu_layers'] = 32
                    else:
                        result['n_gpu_layers'] = 100  # All layers
            else:
                # Conservative default for unknown memory
                result['n_gpu_layers'] = is_large_model and 8 or 16
            
            info_msg(f"🔥 NVIDIA GPU detected: {result['gpu_info']}")
            info_msg(f"Will use CUDA with {result['n_gpu_layers']} GPU layers")
            if is_large_model:
                info_msg("Using conservative GPU settings for 70B model")
            return result
            
        # Next check for Metal Performance Shaders (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result['device'] = 'mps'
            result['use_gpu'] = True
            result['n_gpu_layers'] = 1  # Start conservative with Apple Silicon
            result['gpu_info'] = "Apple Silicon GPU (MPS)"
            
            # Determine device by platform
            import platform
            mac_model = platform.machine()
            if mac_model == "arm64":
                # This is Apple Silicon
                if "M1" in platform.processor():
                    result['gpu_info'] = "Apple M1"
                    result['n_gpu_layers'] = 4  # Conservative for M1
                elif "M2" in platform.processor():
                    result['gpu_info'] = "Apple M2"
                    result['n_gpu_layers'] = 8  # Better for M2
                elif "M3" in platform.processor():
                    result['gpu_info'] = "Apple M3"
                    result['n_gpu_layers'] = 12  # More aggressive for M3
            
            info_msg(f"🔥 Apple Silicon GPU detected: {result['gpu_info']}")
            info_msg(f"Will use Metal with {result['n_gpu_layers']} GPU layers")
            return result
    
    except ImportError:
        warning_msg("PyTorch not available, defaulting to CPU")
    except Exception as e:
        warning_msg(f"Error detecting GPU capabilities: {str(e)}")
    
    info_msg("Using CPU for inference (no GPU acceleration)")
    return result 