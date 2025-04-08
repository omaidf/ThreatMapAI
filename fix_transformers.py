#!/usr/bin/env python
"""
Fix Transformers Configuration

This script helps to fix common issues with Transformers and LLM loading,
particularly for CodeLlama models.
"""

import os
import logging
import argparse
from pathlib import Path
import shutil
import json

logger = logging.getLogger(__name__)

def get_env_variable(var_name, default_value=None):
    """Get an environment variable or return a default value."""
    return os.environ.get(var_name, default_value)

def fix_gguf_loading():
    """
    Fix GGUF model loading issues by ensuring proper configuration.
    """
    try:
        # Define the model ID and expected locations
        model_id = "TheBloke/CodeLlama-7B-Instruct-GGUF"
        huggingface_format = "models--TheBloke--CodeLlama-7B-Instruct-GGUF"
        model_dir = Path("models") / huggingface_format
        
        if not model_dir.exists():
            logger.warning(f"Model directory {model_dir} not found. This script may not be needed.")
            return
        
        # Check if there's an actual model file in the models directory
        gguf_files = list(Path("models").glob("*.gguf"))
        if not gguf_files and not list(model_dir.glob("**/*.gguf")):
            logger.warning("No .gguf files found in models directory. Download may be needed.")
            return
            
        # Create the tokenizer directory structure if it doesn't exist
        snapshot_dirs = list(Path(model_dir / "snapshots").glob("*"))
        if snapshot_dirs:
            snapshot_dir = snapshot_dirs[0]
        else:
            logger.warning("No snapshot directory found. Creating one...")
            snapshot_dir = model_dir / "snapshots" / "latest"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
        # Create a tokenizer_config.json if it doesn't exist
        tokenizer_config = snapshot_dir / "tokenizer_config.json"
        if not tokenizer_config.exists():
            logger.info(f"Creating tokenizer_config.json in {snapshot_dir}")
            config_content = {
                "model_type": "llama",
                "add_bos_token": True,
                "add_eos_token": True,
                "bos_token": {"content": "<s>", "single_word": False},
                "eos_token": {"content": "</s>", "single_word": False},
                "unk_token": {"content": "<unk>", "single_word": False},
                "legacy": False
            }
            with open(tokenizer_config, 'w') as f:
                json.dump(config_content, f, indent=2)
        
        # Create special_tokens_map.json if it doesn't exist
        special_tokens_map = snapshot_dir / "special_tokens_map.json"
        if not special_tokens_map.exists():
            logger.info(f"Creating special_tokens_map.json in {snapshot_dir}")
            tokens_content = {
                "bos_token": {"content": "<s>", "single_word": False},
                "eos_token": {"content": "</s>", "single_word": False},
                "unk_token": {"content": "<unk>", "single_word": False}
            }
            with open(special_tokens_map, 'w') as f:
                json.dump(tokens_content, f, indent=2)
        
        # Create tokenizer.json if it doesn't exist
        tokenizer_json = snapshot_dir / "tokenizer.json"
        if not tokenizer_json.exists():
            # Try to download the tokenizer.json from the CodeLlama base model
            try:
                logger.info("Attempting to download tokenizer.json from the base model")
                import requests
                
                # Try to get Hugging Face token from environment
                hf_token = get_env_variable("HF_TOKEN")
                headers = {}
                if hf_token:
                    logger.info("Using HF_TOKEN for authentication")
                    headers["Authorization"] = f"Bearer {hf_token}"
                
                base_tokenizer_url = "https://huggingface.co/codellama/CodeLlama-7b-Instruct/resolve/main/tokenizer.json"
                response = requests.get(base_tokenizer_url, headers=headers)
                
                if response.status_code == 200:
                    with open(tokenizer_json, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully downloaded tokenizer.json to {tokenizer_json}")
                else:
                    logger.warning(f"Failed to download tokenizer.json: HTTP {response.status_code}")
                    # Create a basic tokenizer.json file
                    basic_tokenizer = {
                        "version": "1.0",
                        "truncation": None,
                        "padding": None,
                        "added_tokens": [
                            {"id": 0, "content": "<unk>", "single_word": False},
                            {"id": 1, "content": "<s>", "single_word": False},
                            {"id": 2, "content": "</s>", "single_word": False}
                        ],
                        "normalizer": {
                            "type": "Sequence",
                            "normalizers": [
                                {"type": "NFD"},
                                {"type": "Lowercase"},
                                {"type": "StripAccents"}
                            ]
                        },
                        "pre_tokenizer": {
                            "type": "ByteLevel",
                            "add_prefix_space": True,
                            "trim_offsets": True
                        },
                        "model": {
                            "type": "BPE",
                            "vocab_size": 32000,
                            "dropout": None,
                            "fuse_unk": False,
                            "unk_token": "<unk>"
                        }
                    }
                    with open(tokenizer_json, 'w') as f:
                        json.dump(basic_tokenizer, f, indent=2)
            except Exception as dl_e:
                logger.warning(f"Failed to download tokenizer.json: {str(dl_e)}")
                logger.info("Creating a generic tokenizer.json file")
                # Create a more comprehensive tokenizer.json file
                basic_tokenizer = {
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [
                        {"id": 0, "content": "<unk>", "single_word": False},
                        {"id": 1, "content": "<s>", "single_word": False},
                        {"id": 2, "content": "</s>", "single_word": False}
                    ],
                    "normalizer": {
                        "type": "Sequence",
                        "normalizers": [
                            {"type": "NFD"},
                            {"type": "Lowercase"},
                            {"type": "StripAccents"}
                        ]
                    },
                    "pre_tokenizer": {
                        "type": "ByteLevel",
                        "add_prefix_space": True,
                        "trim_offsets": True
                    },
                    "model": {
                        "type": "BPE",
                        "vocab_size": 32000,
                        "dropout": None,
                        "fuse_unk": False,
                        "unk_token": "<unk>"
                    }
                }
                with open(tokenizer_json, 'w') as f:
                    json.dump(basic_tokenizer, f, indent=2)
        
        # Create a symbolic link to the actual model file if needed
        for gguf_file in gguf_files:
            target_file = snapshot_dir / gguf_file.name
            if not target_file.exists():
                logger.info(f"Creating symbolic link from {gguf_file} to {target_file}")
                try:
                    # Try creating a symbolic link first
                    os.symlink(os.path.abspath(gguf_file), target_file)
                except Exception as link_e:
                    logger.warning(f"Failed to create symbolic link: {str(link_e)}, copying file instead")
                    shutil.copy(gguf_file, target_file)
        
        logger.info("GGUF model configuration fixed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fix GGUF loading: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix common issues with Transformers and LLM loading")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(levelname)s: %(message)s")
    
    # Run fixes
    fix_gguf_loading()

if __name__ == "__main__":
    main() 