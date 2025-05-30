"""
Model Configuration for the AI Threat Model Map Generator.

This module centralizes all model-related configurations, including URLs,
variants, and default settings to avoid duplication across the codebase.
"""

import platform
import logging
import os
import sys
import subprocess
from typing import Dict, Tuple, Any

# Configure logger
logger = logging.getLogger(__name__)

# Model repository configurations
MODEL_REPOS = {
    "codellama-7b-instruct": {
        "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "description": "CodeLlama 7B Instruct GGUF (Default model)",
        "variants": {
            "Q4_0": {
                "filename": "codellama-7b-instruct.Q4_0.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_0.gguf?download=true",
                "recommended_for": ["arm64", "aarch64", "arm"],
                "description": "Low quantization model suitable for ARM-based systems"
            },
            "Q4_K_M": {
                "filename": "codellama-7b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf?download=true",
                "recommended_for": ["x86_64", "amd64", "x86"],
                "description": "Better quantization model suitable for x86 systems"
            }
        },
        "default_variant": "Q4_K_M",
        "min_ram_gb": 8,
        "context_length": 4096
    },
    "codellama-70b-instruct": {
        "repo_id": "TheBloke/CodeLlama-70B-Instruct-GGUF",
        "description": "CodeLlama 70B Instruct GGUF (High performance model)",
        "variants": {
            "Q4_0": {
                "filename": "codellama-70b-instruct.Q4_0.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF/resolve/main/codellama-70b-instruct.Q4_0.gguf?download=true",
                "recommended_for": ["arm64", "aarch64", "arm"],
                "description": "Low quantization model suitable for ARM-based systems with 32GB+ RAM"
            },
            "Q4_K_M": {
                "filename": "codellama-70b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF/resolve/main/codellama-70b-instruct.Q4_K_M.gguf?download=true",
                "recommended_for": ["x86_64", "amd64", "x86"],
                "description": "Better quantization model suitable for x86 systems with 32GB+ RAM"
            },
            "Q5_K_M": {
                "filename": "codellama-70b-instruct.Q5_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF/resolve/main/codellama-70b-instruct.Q5_K_M.gguf?download=true",
                "recommended_for": ["x86_64", "amd64", "x86"],
                "description": "Higher quality quantization for systems with 40GB+ RAM"
            }
        },
        "default_variant": "Q4_K_M",
        "min_ram_gb": 32,
        "context_length": 16384  # The 70B model supports larger context
    }
}

# Default model to use - can be overridden through environment variable
DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "codellama-7b-instruct")

def detect_architecture() -> Tuple[str, str]:
    """
    Detect system architecture and return appropriate model variant and URL.
    
    Returns:
        Tuple[str, str]: (model_variant, model_url)
    """
    try:
        platform_system = platform.system().lower()
        machine_arch = platform.machine().lower()
        is_arm = machine_arch in ['arm64', 'aarch64', 'arm']
        is_x86 = machine_arch in ['x86_64', 'amd64', 'x86']
        
        # Get model configuration
        model_config = MODEL_REPOS[DEFAULT_MODEL_NAME]
        
        # Force x86 path for Docker containers running on x86 hardware
        if is_x86:
            logger.info(f"Detected x86_64 architecture: {machine_arch}")
            # Use Q4_K_M for NVIDIA GPUs on x86 systems
            model_variant = "Q4_K_M"
        elif is_arm:
            logger.info(f"Detected ARM architecture: {machine_arch}")
            model_variant = "Q4_0"  # Less quantization for ARM
        else:
            # Fallback to default variant
            logger.warning(f"Unknown architecture: {machine_arch}, using default variant")
            model_variant = model_config["default_variant"]
        
        # Get URL for this variant
        model_url = model_config["variants"][model_variant]["url"]
        
        logger.info(f"Using model variant: {model_variant} for {platform_system}")
        
        return model_variant, model_url
    except Exception as e:
        logger.warning(f"Failed to detect architecture: {str(e)}")
        # Safe fallback
        model_variant = MODEL_REPOS[DEFAULT_MODEL_NAME]["default_variant"]
        model_url = MODEL_REPOS[DEFAULT_MODEL_NAME]["variants"][model_variant]["url"]
        return model_variant, model_url

def get_model_info(model_name: str = None, variant: str = None) -> Dict[str, Any]:
    """
    Get information about a specific model and variant.
    
    Args:
        model_name: Name of the model (default is DEFAULT_MODEL_NAME)
        variant: Specific variant to get info for (default is determined by architecture)
    
    Returns:
        Dictionary with model information
    """
    # Get model name from environment variable MODEL_PATH if available
    model_path_env = os.environ.get("MODEL_PATH", "")
    if model_path_env and not model_name:
        # Extract model name from model path environment variable
        if "70b" in model_path_env.lower():
            model_name = "codellama-70b-instruct"
        elif "7b" in model_path_env.lower():
            model_name = "codellama-7b-instruct"
    
    # Fall back to default model name if still not determined
    model_name = model_name or DEFAULT_MODEL_NAME
    
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_REPOS[model_name]
    
    # Extract variant from model path if available
    if model_path_env and not variant:
        if "Q4_0" in model_path_env:
            variant = "Q4_0"
        elif "Q4_K_M" in model_path_env:
            variant = "Q4_K_M"
        elif "Q5_K_M" in model_path_env:
            variant = "Q5_K_M"
    
    if not variant:
        # Detect appropriate variant based on architecture
        variant, _ = detect_architecture()
    
    if variant not in model_config["variants"]:
        variant = model_config["default_variant"]
    
    variant_info = model_config["variants"][variant]
    
    return {
        "name": model_name,
        "repo_id": model_config["repo_id"],
        "variant": variant,
        "filename": variant_info["filename"],
        "url": variant_info["url"],
        "description": variant_info["description"],
        "min_ram_gb": model_config.get("min_ram_gb", 8),
        "context_length": model_config.get("context_length", 4096)
    }

def get_default_model_path() -> str:
    """
    Get the default model path based on architecture.
    
    Returns:
        Default path to the model file
    """
    # Check if MODEL_PATH is set in environment
    model_path_env = os.environ.get("MODEL_PATH", "")
    if model_path_env:
        # Use the exact path from environment if available
        return model_path_env
        
    # Otherwise use the detected configuration
    model_info = get_model_info()
    return f"models/{model_info['filename']}"

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available models.
    
    Returns:
        Dictionary with model names as keys and basic info as values
    """
    models = {}
    for model_name, config in MODEL_REPOS.items():
        models[model_name] = {
            "description": config["description"],
            "min_ram_gb": config.get("min_ram_gb", 8),
            "variants": list(config["variants"].keys()),
            "context_length": config.get("context_length", 4096)
        }
    return models

def set_default_model(model_name: str) -> bool:
    """
    Set the default model to use.
    
    Args:
        model_name: Name of the model to use as default
        
    Returns:
        True if successful, False otherwise
    """
    global DEFAULT_MODEL_NAME
    
    if model_name not in MODEL_REPOS:
        logger.error(f"Unknown model: {model_name}")
        return False
        
    DEFAULT_MODEL_NAME = model_name
    os.environ["LLM_MODEL"] = model_name
    logger.info(f"Default model set to: {model_name}")
    return True

def download_model(model_path: str = None, model_name: str = None, variant: str = None, force: bool = False) -> str:
    """
    Download the specified model if it doesn't exist locally.
    
    Args:
        model_path: Path to save the model to
        model_name: Name of the model (default is DEFAULT_MODEL_NAME)
        variant: Specific variant to download (default is determined by architecture)
        force: Force download even if the file exists
        
    Returns:
        Path to the downloaded model file
    """
    # Check if MODEL_PATH is explicitly set
    env_model_path = os.environ.get("MODEL_PATH", "")
    
    # If model_path isn't provided, use environment or default
    if not model_path:
        model_path = env_model_path
    
    # Try to determine model name from the path if not provided
    if not model_name and model_path:
        if "70b" in model_path.lower():
            model_name = "codellama-70b-instruct"
        elif "7b" in model_path.lower():
            model_name = "codellama-7b-instruct"
    
    # Get model info - using environment model path to help determine the model
    model_info = get_model_info(model_name, variant)
    
    # Log which model is being used
    logger.info(f"Using model configuration: {model_info['name']} ({model_info['variant']})")
    
    # Build paths
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # If model_path is already set, use it, otherwise build from models dir and filename
    if not model_path:
        model_path = os.path.join(models_dir, model_info["filename"])
    
    # Check if model exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists at {model_path}")
        logger.info(f"File size: {os.path.getsize(model_path) / (1024 * 1024 * 1024):.2f} GB")
        return model_path
    
    # Download the model
    logger.info(f"Downloading model {model_info['name']} ({model_info['variant']})...")
    logger.info(f"URL: {model_info['url']}")
    logger.info(f"This may take a while for large models...")
    
    try:
        # Use wget or curl depending on availability
        if sys.platform.startswith('win'):
            # Use PowerShell on Windows
            cmd = f'powershell -Command "Invoke-WebRequest -Uri "{model_info["url"]}" -OutFile "{model_path}"'
        else:
            # Use wget on Unix-like systems if available, otherwise curl
            if subprocess.run(['which', 'wget'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                cmd = f'wget -O "{model_path}" "{model_info["url"]}"'
            else:
                cmd = f'curl -L "{model_info["url"]}" -o "{model_path}"'
        
        logger.info(f"Running download command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        # Verify file exists and has content
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            logger.info(f"Download complete. Model saved to {model_path}")
            logger.info(f"File size: {os.path.getsize(model_path) / (1024 * 1024 * 1024):.2f} GB")
            return model_path
        else:
            logger.error(f"Download failed or file is empty")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        if os.path.exists(model_path):
            logger.info(f"Removing incomplete download file")
            os.remove(model_path)
        return None
