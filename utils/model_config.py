"""
Model Configuration for the AI Threat Model Map Generator.

This module centralizes all model-related configurations, including URLs,
variants, and default settings to avoid duplication across the codebase.
"""

import platform
import logging
import os
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
        "min_ram_gb": 8
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
        "min_ram_gb": 32
    }
}

# Default model to use - can be overridden through environment variable
DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "codellama-70b-instruct")

def detect_architecture() -> Tuple[str, str]:
    """
    Detect system architecture and return appropriate model variant and URL.
    
    Returns:
        Tuple[str, str]: (model_variant, model_url)
    """
    try:
        platform_system = platform.system().lower()
        is_arm = platform.machine().lower() in ['arm64', 'aarch64', 'arm']
        
        # Get model configuration
        model_config = MODEL_REPOS[DEFAULT_MODEL_NAME]
        
        # Determine model variant based on architecture
        if is_arm:
            model_variant = "Q4_0"  # Less quantization for ARM
        else:
            model_variant = "Q4_K_M"  # Better quantization for x86
        
        # Get URL for this variant
        model_url = model_config["variants"][model_variant]["url"]
        
        logger.info(f"Detected architecture: {platform.machine()} on {platform_system}")
        logger.info(f"Using model variant: {model_variant}")
        
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
    model_name = model_name or DEFAULT_MODEL_NAME
    
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_REPOS[model_name]
    
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
        "min_ram_gb": model_config.get("min_ram_gb", 8)
    }

def get_default_model_path() -> str:
    """
    Get the default model path based on architecture.
    
    Returns:
        Default path to the model file
    """
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
            "variants": list(config["variants"].keys())
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