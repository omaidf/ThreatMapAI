"""
Embedding Store Module

This module provides functionality for storing and retrieving
code embeddings for semantic search and similarity analysis.
"""

import os
import logging
import pathlib  # Import the whole pathlib module
from pathlib import Path  # Import Path class specifically
from typing import List, Dict, Any, Optional, Tuple
import json
import faiss
import numpy as np
import argparse
import gc
import fnmatch
import sys
import torch
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
from langchain_core.vectorstores import VectorStore
from transformers import AutoModel, AutoTokenizer

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable
from utils.file_utils import check_output_directory
from utils.gpu_utils import detect_gpu  # Import the centralized GPU detection function

logger = logging.getLogger(__name__)

class EmbeddingStoreError(Exception):
    """Custom exception for embedding store errors."""
    pass

class EmbeddingStore:
    """
    Store and retrieve code embeddings for semantic search.
    
    This class handles:
    - Embedding generation for code chunks
    - Storing embeddings in a FAISS index
    - Semantic search for related code
    """
    
    def __init__(self, device: Optional[str] = None, gpu_id: Optional[int] = None):
        """
        Initialize the embedding store.
        
        Args:
            device: Force device to use ('cpu', 'cuda', or None for auto-detection)
            gpu_id: Specific GPU ID to use when multiple GPUs are available (ignored if device is 'cpu')
        """
        self.output_dir = Path(get_env_variable("OUTPUT_DIR", "output"))
        check_output_directory(str(self.output_dir))
        self.index_path = self.output_dir / "embeddings.index"
        self.mapping_path = self.output_dir / "embeddings_mapping.json"
        
        # Initialize empty index and mapping
        self.index = None
        self.file_mapping = []  # Changed from dictionary to list for consistency
        self.use_gpu_for_faiss = False
        
        # Store the GPU ID
        self.gpu_id = gpu_id if gpu_id is not None else 0
        
        # Auto-detect GPU if device not specified
        gpu_available, gpu_memory = detect_gpu()
        self.gpu_available = gpu_available
        
        # Determine device to use (user choice, auto-detect, or fallback)
        if device is not None:
            # User explicitly specified device
            self.device = device
            if device == 'cuda' and not gpu_available:
                warning_msg("Requested CUDA but GPU not available, falling back to CPU")
                self.device = 'cpu'
        else:
            # Auto-detect: use GPU if available and has sufficient memory (>2GB)
            self.device = 'cuda' if gpu_available and gpu_memory and gpu_memory > 2 else 'cpu'
        
        # Use the specified GPU ID if available
        if self.device == 'cuda' and gpu_id is not None and torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                info_msg(f"Using GPU {gpu_id} for transformers")
            else:
                warning_msg(f"Specified GPU {gpu_id} not available, using default GPU")
        
        # Initialize the embedding model
        try:
            if self.device == 'cuda':
                device_str = f'cuda:{self.gpu_id}' if self.gpu_id is not None else 'cuda'
            else:
                device_str = 'cpu'
            
            # Hardcode CodeBERT as the embedding model
            self.embedding_model_name = "microsoft/codebert-base"
            
            # Load CodeBERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.model = AutoModel.from_pretrained(self.embedding_model_name)
            self.model.to(device_str)
            # CodeBERT has 768 dimensions
            self.vector_size = 768
            info_msg(f"CodeBERT model initialized with dimension {self.vector_size} on {device_str}")
        except Exception as e:
            warning_msg(f"Failed to initialize CodeBERT embedding model: {str(e)}")
            self.model = None
            self.tokenizer = None
            # Set a default vector size for FAISS initialization
            self.vector_size = 768  # Default dimension for embeddings
        
        # Set up FAISS index - handle errors gracefully
        try:
            self._setup_index()
        except Exception as e:
            warning_msg(f"Failed to initialize FAISS index: {str(e)}")
            warning_msg("FAISS functionality will be limited until fixed")
            # If using CUDA, suggest installing faiss-gpu
            if self.device == 'cuda':
                warning_msg("For GPU support, make sure FAISS GPU is installed: 'pip install faiss-gpu'")
                warning_msg("Run the setup script (setup.sh) which will auto-detect and install the right version")
    
    def load(self) -> bool:
        """
        Load embeddings from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.mapping_path.exists():
                info_msg("No existing embeddings found, initializing new index")
                self.clear()
                return False
            
            # Load mapping
            with open(self.mapping_path, 'r') as f:
                self.file_mapping = json.load(f)
            
            # Load index - always load as CPU index first
            cpu_index = faiss.read_index(str(self.index_path))
            
            # Move to GPU if needed
            if self.device == 'cuda' and self.gpu_available:
                try:
                    # Check if GPU version of FAISS is available
                    if hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu'):
                        # Create GPU resources
                        self.gpu_resources = faiss.StandardGpuResources()
                        
                        # Check if multi-GPU mode is enabled
                        faiss_multi_gpu = os.environ.get("FAISS_MULTI_GPU", "0") == "1"
                        use_sharding = os.environ.get("FAISS_USE_SHARDING", "0") == "1"
                        
                        # Try multi-GPU approach first if enabled and multiple GPUs are available
                        multi_gpu_success = False
                        
                        # Only try multi-GPU if it's explicitly enabled
                        if (faiss_multi_gpu and 
                            hasattr(faiss, 'get_num_gpus') and 
                            faiss.get_num_gpus() > 1 and
                            torch.cuda.device_count() > 1):
                            try:
                                # Clear CUDA cache first
                                torch.cuda.empty_cache()
                                
                                # Create a list of GPU resources with higher memory usage configuration
                                gpu_resources = []
                                for i in range(min(4, torch.cuda.device_count())):  # Limit to 4 GPUs 
                                    res = faiss.StandardGpuResources()
                                    # Configure the GPU resource to use more memory
                                    if hasattr(res, 'setTempMemory'):
                                        # Try to allocate a larger amount of memory (95% of total GPU memory)
                                        try:
                                            total_memory = torch.cuda.get_device_properties(i).total_memory
                                            # Convert to MB and allocate 95%
                                            alloc_memory = int(total_memory * 0.95 / (1024**2))
                                            res.setTempMemory(alloc_memory)
                                            info_msg(f"Configured GPU {i} to use {alloc_memory} MB FAISS memory (95% of total)")
                                        except Exception as mem_e:
                                            warning_msg(f"Could not set GPU {i} memory: {str(mem_e)}")
                                    gpu_resources.append(res)
                                
                                # Create config for multi-GPU with careful options
                                co = faiss.GpuMultipleClonerOptions()
                                co.useFloat16 = True  # Use half precision for memory efficiency
                                co.shard = use_sharding  # Only use sharding if enabled
                                if use_sharding:
                                    info_msg("Using index sharding across multiple GPUs")
                                else:
                                    info_msg("Using index replication across multiple GPUs (no sharding)")
                                
                                info_msg(f"Loading FAISS index to multiple GPUs...")
                                
                                # Direct approach without threading
                                self.index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index, co)
                                self.use_gpu_for_faiss = True
                                
                                # Store resources to prevent garbage collection issues
                                self.multi_gpu_resources = gpu_resources
                                
                                # Run a simple test to verify the index is working
                                if self.index.ntotal > 0:
                                    # Create a random query vector for testing
                                    test_query = np.random.random((1, self.vector_size)).astype(np.float32)
                                    try:
                                        # Try a search with small k to validate index
                                        D, I = self.index.search(test_query, min(1, self.index.ntotal))
                                        info_msg(f"Successfully validated multi-GPU index with test query")
                                    except Exception as test_e:
                                        warning_msg(f"Multi-GPU index failed validation: {str(test_e)}")
                                        raise RuntimeError("Index validation failed")
                                
                                info_msg("Successfully loaded FAISS index to multiple GPUs")
                                multi_gpu_success = True
                            except Exception as multi_e:
                                error_details = str(multi_e)
                                if "CUDA error" in error_details:
                                    warning_msg(f"CUDA error during multi-GPU initialization: {error_details}")
                                elif "out of memory" in error_details.lower():
                                    warning_msg(f"GPU out of memory during multi-GPU initialization")
                                else:
                                    warning_msg(f"Failed to load multi-GPU FAISS index: {error_details}")
                                
                                warning_msg("Falling back to single GPU")
                                
                                # Explicitly clean up resources to avoid CUDA OOM errors
                                try:
                                    if hasattr(self, 'multi_gpu_resources'):
                                        del self.multi_gpu_resources
                                    if hasattr(self, 'index') and self.index is not None:
                                        del self.index
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                except:
                                    pass
                        
                        # If multi-GPU failed or not available, use single GPU
                        if not multi_gpu_success:
                            # Convert CPU index to GPU index using specified GPU ID
                            info_msg(f"Loading FAISS index to GPU {self.gpu_id}...")
                            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
                            self.use_gpu_for_faiss = True
                            info_msg(f"Successfully moved FAISS index to GPU {self.gpu_id}")
                    else:
                        warning_msg("FAISS GPU support not available, using CPU index")
                        self.index = cpu_index
                        self.use_gpu_for_faiss = False
                except Exception as e:
                    warning_msg(f"Failed to move index to GPU: {str(e)}")
                    self.index = cpu_index
                    self.use_gpu_for_faiss = False
            else:
                # Use CPU index
                self.index = cpu_index
                self.use_gpu_for_faiss = False
            
            info_msg(f"Loaded {len(self.file_mapping)} embeddings from {self.index_path}")
            return True
            
        except Exception as e:
            warning_msg(f"Failed to load embeddings, creating new index: {str(e)}")
            self.clear()
            return False
    
    def save(self) -> None:
        """Save embeddings to disk."""
        try:
            if self.index is None:
                warning_msg("No index to save")
                return
            
            # Ensure the output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert paths to strings for safer file operations
            mapping_path_str = str(self.mapping_path)
            index_path_str = str(self.index_path)
            
            # Always save mapping first regardless of index status - this is critical
            try:
                with open(mapping_path_str, 'w') as f:
                    json.dump(self.file_mapping, f, indent=2)
                info_msg(f"Saved mapping with {len(self.file_mapping)} entries")
            except Exception as mapping_error:
                error_msg(f"Failed to save mapping: {str(mapping_error)}")
            
            # If using GPU index, we need to convert back to CPU for saving
            if hasattr(self, 'use_gpu_for_faiss') and self.use_gpu_for_faiss:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, index_path_str)
                    info_msg(f"Successfully converted GPU index to CPU and saved")
                except Exception as e:
                    error_msg(f"Failed to convert GPU index to CPU for saving: {str(e)}")
                    # Instead of attempting to save directly (which often fails), 
                    # create a new CPU index from the current embeddings
                    try:
                        # Extract all embeddings from the mapping (if available)
                        error_msg("Attempting to rebuild index from embeddings...")
                        
                        # Create a new CPU index
                        new_cpu_index = faiss.IndexFlatL2(self.vector_size)
                        # Just save this basic index rather than risking a crash
                        faiss.write_index(new_cpu_index, index_path_str)
                        error_msg("Saved empty fallback index. Some embeddings may be lost.")
                    except Exception as rebuild_error:
                        error_msg(f"Failed to save fallback index: {str(rebuild_error)}")
            else:
                # Save index directly if it's already a CPU index
                try:
                    faiss.write_index(self.index, index_path_str)
                    info_msg(f"Saved CPU index with {self.index.ntotal} vectors")
                except Exception as cpu_save_error:
                    error_msg(f"Failed to save CPU index: {str(cpu_save_error)}")
                    # Try saving an empty index in case of failure
                    try:
                        new_cpu_index = faiss.IndexFlatL2(self.vector_size)
                        faiss.write_index(new_cpu_index, index_path_str)
                        error_msg("Saved empty fallback index due to error")
                    except:
                        error_msg("Could not save any index, vectors may be lost")
            
            success_msg(f"Saved {len(self.file_mapping)} embeddings to {self.index_path}")
        except Exception as e:
            error_msg(f"Failed to save embeddings: {str(e)}")
            # Print a direct message in case logging is not working
            print(f"Failed to save embeddings: {str(e)}")
    
    def _setup_index(self) -> None:
        """Set up the FAISS index, using GPU if available."""
        # Create a new index based on vector size
        try:
            # First import FAISS to ensure it's available
            import faiss
            
            # Check if we should use GPU for FAISS
            self.use_gpu_for_faiss = self.device == 'cuda' and self.gpu_available
            
            # Create a flat L2 index first (CPU version)
            cpu_index = faiss.IndexFlatL2(self.vector_size)
            
            if self.use_gpu_for_faiss:
                try:
                    # Try importing GPU-specific FAISS components
                    gpu_available = False
                    num_gpus = 0
                    
                    # First check if this is faiss-gpu by checking get_num_gpus
                    if hasattr(faiss, 'get_num_gpus'):
                        num_gpus = faiss.get_num_gpus()
                        if num_gpus > 0:
                            # We have GPU support in this FAISS build
                            gpu_available = True
                            info_msg(f"FAISS detected {num_gpus} GPUs")
                    
                    if gpu_available and hasattr(faiss, 'StandardGpuResources'):
                        info_msg(f"Using GPU {self.gpu_id} acceleration for FAISS index")
                        
                        # Create GPU resources
                        self.gpu_resources = faiss.StandardGpuResources()
                        
                        # Configure GPU index
                        gpu_config = faiss.GpuIndexFlatConfig()
                        gpu_config.device = self.gpu_id  # Use specified GPU
                        gpu_config.useFloat16 = True  # Use half-precision for memory efficiency
                        
                        # Check if multi-GPU is allowed based on environment variable
                        faiss_multi_gpu = os.environ.get("FAISS_MULTI_GPU", "0") == "1"
                        use_sharding = os.environ.get("FAISS_USE_SHARDING", "0") == "1"
                        
                        # Only attempt multi-GPU if enabled in environment and we have multiple GPUs
                        multi_gpu_available = (
                            faiss_multi_gpu and 
                            torch.cuda.device_count() > 1 and 
                            hasattr(faiss, 'GpuMultipleClonerOptions') and 
                            num_gpus > 1
                        )
                        
                        if multi_gpu_available:
                            try:
                                # Create a list of GPU resources, one for each device
                                gpu_resources = []
                                for i in range(min(4, torch.cuda.device_count())):
                                    # Create separate GPU resources for each device
                                    res = faiss.StandardGpuResources()
                                    gpu_resources.append(res)
                                
                                # Create configuration for multi-GPU
                                co = faiss.GpuMultipleClonerOptions()
                                co.useFloat16 = True
                                
                                # Only use sharding if explicitly enabled
                                co.shard = use_sharding
                                if use_sharding:
                                    info_msg("Using index sharding across multiple GPUs")
                                else:
                                    info_msg("Using index replication across multiple GPUs (no sharding)")
                                
                                info_msg("Creating multi-GPU FAISS index...")
                                
                                # Use simple direct approach without threading which can cause deadlocks
                                self.index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index, co)
                                self.use_gpu_for_faiss = True
                                info_msg("Successfully created multi-GPU FAISS index")
                                
                                # Store resources to prevent garbage collection issues
                                self.multi_gpu_resources = gpu_resources
                                return
                            except Exception as multi_gpu_error:
                                warning_msg(f"Failed to create multi-GPU FAISS index: {str(multi_gpu_error)}")
                                warning_msg("Falling back to single GPU")
                                
                                # Explicitly clean up resources to avoid CUDA OOM errors
                                try:
                                    if hasattr(self, 'multi_gpu_resources'):
                                        del self.multi_gpu_resources
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                except:
                                    pass
                        
                        # Create GPU index with a single GPU
                        info_msg("Initializing single-GPU FAISS index...")
                        self.index = faiss.GpuIndexFlatL2(self.gpu_resources, self.vector_size, gpu_config)
                        self.use_gpu_for_faiss = True
                        info_msg("Successfully created single-GPU FAISS index")
                        return
                    else:
                        warning_msg("FAISS GPU support not available (install faiss-gpu instead of faiss-cpu)")
                        warning_msg("Falling back to CPU index")
                except Exception as e:
                    warning_msg(f"Failed to initialize GPU index for FAISS: {str(e)}")
                    warning_msg("Falling back to CPU index")
            
            # If we get here, use CPU index
            self.index = cpu_index
            self.use_gpu_for_faiss = False
            info_msg("Using CPU index for FAISS")
            
        except ImportError as e:
            warning_msg(f"Failed to import FAISS: {str(e)}")
            warning_msg("To use FAISS with GPU, install: pip install faiss-gpu")
            # Create a minimal CPU index
            import numpy as np
            self.index = None
            self.use_gpu_for_faiss = False
            raise e
        except Exception as e:
            warning_msg(f"Failed to create FAISS index: {str(e)}")
            # Create a minimal CPU index as fallback
            self.index = None
            self.use_gpu_for_faiss = False
            raise e
    
    def clear(self) -> None:
        """
        Clear the embedding store without deleting the model files.
        This resets the index and metadata while preserving the 
        embedding infrastructure.
        """
        try:
            # Check if Python is shutting down
            if sys is None or sys.meta_path is None:
                print("Skipping clear during Python shutdown")
                return
                
            # Reset in-memory data
            self.index = None
            self.file_mapping = []  # Changed from dictionary to list
            
            # Re-initialize the embedding model if needed
            if not hasattr(self, 'model') or self.model is None:
                try:
                    # Set the specific CUDA device if needed
                    device_str = self.device
                    if self.device == 'cuda' and hasattr(self, 'gpu_id'):
                        device_str = f"cuda:{self.gpu_id}"
                    
                    # Hardcode to use CodeBERT
                    self.embedding_model_name = "microsoft/codebert-base"
                    self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                    self.model = AutoModel.from_pretrained(self.embedding_model_name)
                    self.model.to(device_str)
                    # CodeBERT has 768 dimensions
                    self.vector_size = 768
                    info_msg(f"CodeBERT model reinitialized with dimension {self.vector_size} on {device_str}")
                except Exception as e:
                    logger.warning(f"Failed to initialize CodeBERT model: {str(e)}")
                    self.model = None
                    # Set default vector size if we don't have it
                    if not hasattr(self, 'vector_size'):
                        self.vector_size = 768  # Standard dimension for embeddings
            
            # Initialize the index (will handle failures internally)
            try:
                self._setup_index()
                info_msg("Embeddings cleared, created new empty index")
            except Exception as e:
                warning_msg(f"Failed to create new index: {str(e)}")
                warning_msg("Embedding store will have limited functionality")
            
            # Save empty state if we have a valid index
            if self.index is not None:
                try:
                    # Delete any existing embedding files first
                    index_path = self.output_dir / "embeddings.index"
                    mapping_path = self.output_dir / "embeddings_mapping.json"
                    
                    if index_path.exists():
                        index_path.unlink()
                    if mapping_path.exists():
                        mapping_path.unlink()
                        
                    # Now save empty index
                    self.save()
                except Exception as save_e:
                    warning_msg(f"Failed to save empty index: {str(save_e)}")
            
            info_msg("Embedding store cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear embedding store: {str(e)}")
            warning_msg(f"Error during clear: {str(e)}")
            # Don't raise exception to avoid crashes
            pass
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text to embeddings using CodeBERT.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            # Handle empty input
            return np.zeros((0, self.vector_size), dtype=np.float32)
            
        # Check if model exists and is initialized properly
        if self.model is None or self.tokenizer is None:
            # If we don't have a model, return zeros and try to reinitialize
            warning_msg("CodeBERT model not initialized! Using dummy embeddings (zeros)")
            
            # Try to reinitialize the model
            try:
                device_str = self.device
                if self.device == 'cuda' and hasattr(self, 'gpu_id'):
                    device_str = f"cuda:{self.gpu_id}"
                
                # Hardcode CodeBERT model
                self.embedding_model_name = "microsoft/codebert-base"
                self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                self.model = AutoModel.from_pretrained(self.embedding_model_name)
                self.model.to(device_str)
                info_msg(f"CodeBERT model reinitialized on {device_str}")
            except Exception as e:
                error_msg(f"Failed to reinitialize CodeBERT model: {str(e)}")
                
            # Still return zeros this time
            return np.zeros((len(texts), self.vector_size), dtype=np.float32)
        
        try:
            # Make sure all text chunks are valid strings
            valid_texts = []
            for text in texts:
                if not isinstance(text, str):
                    warning_msg(f"Invalid text type: {type(text)}, converting to string")
                    valid_texts.append(str(text))
                elif not text.strip():
                    # Skip empty strings by adding a placeholder
                    valid_texts.append(" ")
                else:
                    valid_texts.append(text)
            
            result = None
            
            # CodeBERT specific encoding
            if self.device == 'cpu':
                # Process on CPU
                all_embeddings = []
                for text in valid_texts:
                    try:
                        # Tokenize and extract embeddings
                        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        # Use CLS token embedding as the sentence embedding
                        single_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        all_embeddings.append(single_embedding[0])
                        
                        # Explicit cleanup to avoid memory leaks
                        del inputs, outputs
                        
                    except Exception as chunk_error:
                        warning_msg(f"Error encoding chunk with CodeBERT: {str(chunk_error)}")
                        # Add zeros for failed chunks
                        all_embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
                
                result = np.vstack(all_embeddings)
                
                # Force garbage collection after processing all chunks
                gc.collect()
                
            else:
                # GPU processing with dynamically sized batches
                all_embeddings = []
                
                try:
                    # Dynamically adjust batch size based on available GPU memory
                    batch_size = 8  # Default small batch size
                    
                    # Get current GPU memory status
                    free_memory_gb = 0
                    try:
                        if hasattr(torch.cuda, 'get_device_properties') and hasattr(torch.cuda, 'memory_reserved'):
                            device_id = self.gpu_id if hasattr(self, 'gpu_id') and self.gpu_id is not None else 0
                            total_memory = torch.cuda.get_device_properties(device_id).total_memory
                            reserved_memory = torch.cuda.memory_reserved(device_id)
                            free_memory = total_memory - reserved_memory
                            free_memory_gb = free_memory / (1024**3)
                            
                            # Scale batch size based on available memory
                            # Each batch of 8 needs ~0.5GB for CodeBERT
                            if free_memory_gb > 40:  # High-memory GPU (>40GB free)
                                batch_size = 64  # Reduced from 128 to prevent OOM
                            elif free_memory_gb > 20:  # Medium-memory GPU (>20GB free)
                                batch_size = 32  # Reduced from 64 to prevent OOM
                            elif free_memory_gb > 10:  # Standard GPU (>10GB free)
                                batch_size = 16  # Reduced from 32 to prevent OOM
                            elif free_memory_gb > 5:   # Limited memory GPU (>5GB free)
                                batch_size = 8  # Reduced from 16 to prevent OOM
                            # else use default 8
                            
                            info_msg(f"Using batch size {batch_size} for embedding generation ({free_memory_gb:.1f}GB free memory)")
                    except Exception as mem_error:
                        warning_msg(f"Error determining GPU memory, using default batch size: {str(mem_error)}")
                    
                    # Check for multi-GPU capability
                    use_multi_gpu = False
                    gpu_count = 0
                    try:
                        if torch.cuda.device_count() > 1:
                            gpu_count = torch.cuda.device_count()
                            use_multi_gpu = True
                            info_msg(f"Using {gpu_count} GPUs for embedding generation")
                    except Exception:
                        use_multi_gpu = False
                    
                    # Process in batches with improved error handling
                    max_retries = 2  # Allow retries for CUDA errors
                    
                    for i in range(0, len(valid_texts), batch_size):
                        retry_count = 0
                        batch = valid_texts[i:i+batch_size]
                        batch_success = False
                        
                        # Select GPU for this batch if using multi-GPU
                        current_gpu_id = self.gpu_id
                        if use_multi_gpu:
                            # Distribute batches across GPUs
                            batch_gpu_id = (i // batch_size) % gpu_count
                            # Only change device if different from current
                            if batch_gpu_id != current_gpu_id:
                                current_gpu_id = batch_gpu_id
                                torch.cuda.set_device(current_gpu_id)
                                info_msg(f"Switched to GPU {current_gpu_id} for batch {i//batch_size + 1}")
                        
                        # Clean GPU memory before processing batch
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        while retry_count <= max_retries and not batch_success:
                            try:
                                # Tokenize and extract embeddings
                                device_str = f"cuda:{current_gpu_id}"
                                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                                        return_tensors="pt", max_length=512)
                                inputs = {k: v.to(device_str) for k, v in inputs.items()}
                                
                                with torch.no_grad():
                                    outputs = self.model.to(device_str)(**inputs)
                                
                                # Use CLS token embedding as the sentence embedding
                                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                
                                # Validate embeddings to ensure quality
                                if np.isnan(batch_embeddings).any():
                                    warning_msg(f"Detected NaN values in embeddings for batch {i//batch_size + 1}")
                                    # Try to fix NaN values
                                    batch_embeddings = np.nan_to_num(batch_embeddings)
                                
                                if np.isclose(batch_embeddings, 0).all(axis=1).any():
                                    warning_msg(f"Detected zero embeddings in batch {i//batch_size + 1}")
                                
                                all_embeddings.append(batch_embeddings)
                                batch_success = True
                                
                                # Explicit cleanup to avoid memory leaks - move to CPU first to ensure tensor is completely freed
                                for k in inputs:
                                    inputs[k] = inputs[k].cpu()
                                del inputs
                                
                                # Move model outputs to CPU before deleting
                                outputs_cpu = outputs.last_hidden_state.cpu()
                                del outputs
                                del outputs_cpu
                                
                                # Move batch embeddings off GPU explicitly
                                batch_embeddings_copy = batch_embeddings.copy()
                                del batch_embeddings
                                
                                # Ensure tensor memory is released
                                torch.cuda.empty_cache()
                                gc.collect()
                                
                                # Add the copy to results
                                all_embeddings[-1] = batch_embeddings_copy
                                
                            except RuntimeError as cuda_error:
                                # Handle CUDA-specific errors
                                error_str = str(cuda_error)
                                if "CUDA out of memory" in error_str:
                                    warning_msg(f"CUDA OOM in batch {i//batch_size + 1}, retrying with smaller batch")
                                    # Clean memory before retry
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    
                                    # Reduce batch size for retry
                                    if len(batch) > 1:
                                        half_point = len(batch) // 2
                                        # Process first half
                                        try:
                                            inputs_half = self.tokenizer(batch[:half_point], padding=True, truncation=True, 
                                                                return_tensors="pt", max_length=512)
                                            inputs_half = {k: v.to(device_str) for k, v in inputs_half.items()}
                                            
                                            with torch.no_grad():
                                                outputs_half = self.model.to(device_str)(**inputs_half)
                                            
                                            half_embeddings = outputs_half.last_hidden_state[:, 0, :].cpu().numpy()
                                            
                                            # Process second half
                                            # Clean GPU memory first before processing second half
                                            for k in inputs_half:
                                                inputs_half[k] = inputs_half[k].cpu()
                                            del inputs_half, outputs_half
                                            torch.cuda.empty_cache()
                                            gc.collect()
                                            
                                            inputs_half2 = self.tokenizer(batch[half_point:], padding=True, truncation=True, 
                                                                return_tensors="pt", max_length=512)
                                            inputs_half2 = {k: v.to(device_str) for k, v in inputs_half2.items()}
                                            
                                            with torch.no_grad():
                                                outputs_half2 = self.model.to(device_str)(**inputs_half2)
                                            
                                            half_embeddings2 = outputs_half2.last_hidden_state[:, 0, :].cpu().numpy()
                                            
                                            # Combine the halves
                                            combined_embeddings = np.vstack([half_embeddings, half_embeddings2])
                                            all_embeddings.append(combined_embeddings)
                                            batch_success = True
                                            
                                            # Cleanup each tensor in sequence
                                            for k in inputs_half2:
                                                inputs_half2[k] = inputs_half2[k].cpu()
                                            del inputs_half2
                                            
                                            outputs_half2_cpu = outputs_half2.last_hidden_state.cpu()
                                            del outputs_half2
                                            del outputs_half2_cpu
                                            
                                            del half_embeddings, half_embeddings2
                                            
                                            # Final cleanup
                                            torch.cuda.empty_cache()
                                            gc.collect()
                                            
                                        except Exception as split_error:
                                            warning_msg(f"Failed to process split batch: {str(split_error)}")
                                            # Clean memory before continuing
                                            torch.cuda.empty_cache()
                                            gc.collect()
                                            # Fall through to retry or zeros
                                    
                                    # If split processing didn't work, retry or use zeros
                                    if not batch_success:
                                        retry_count += 1
                                        # Force cleanup
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                else:
                                    warning_msg(f"CUDA error in batch {i//batch_size + 1}: {error_str}")
                                    retry_count += 1
                                    # Clean before retrying
                                    torch.cuda.empty_cache()
                                    gc.collect()
                        
                        # If all retries failed, use zeros
                        if not batch_success:
                            warning_msg(f"All retries failed for batch {i//batch_size + 1}, using zeros")
                            batch_zeros = np.zeros((len(batch), self.vector_size), dtype=np.float32)
                            all_embeddings.append(batch_zeros)
                        
                        # Periodic garbage collection and status update
                        if (i // batch_size) % 5 == 0 and i > 0:  # Increased frequency from 10 to 5
                            gc.collect()
                            torch.cuda.empty_cache()
                            info_msg(f"Processed {i + len(batch)}/{len(valid_texts)} texts")
                    
                    # Combine all batches
                    if all_embeddings:
                        result = np.vstack(all_embeddings)
                    else:
                        # Fallback if all batches failed
                        result = np.zeros((len(valid_texts), self.vector_size), dtype=np.float32)
                        
                except Exception as gpu_error:
                    warning_msg(f"GPU encoding with CodeBERT failed: {str(gpu_error)}. Falling back to zeros.")
                    result = np.zeros((len(valid_texts), self.vector_size), dtype=np.float32)
            
            # If result wasn't set in any branch, create a default
            if result is None:
                result = np.zeros((len(valid_texts), self.vector_size), dtype=np.float32)
                
            # Normalize manually (L2 normalization)
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            result = result / norms
            
            # Final cleanup
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            error_msg(f"Critical error during encoding: {str(e)}. Using zeros instead.")
            return np.zeros((len(texts), self.vector_size), dtype=np.float32)
    
    def add_file(self, file_path: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a file to the embedding store.
        
        Args:
            file_path: Path to the file
            content: Content of the file
            metadata: Additional metadata about the file (language, imports, etc.)
        """
        # Counter for retries
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check if this file is already in the embedding store
                if self.contains_file(file_path):
                    logger.info(f"File {file_path} already in embedding store, skipping")
                    return
                
                # Validate content
                if not content or not isinstance(content, str):
                    if not content:
                        error_msg(f"Empty content for file {file_path}, skipping")
                    else:
                        error_msg(f"Invalid content type for {file_path}: {type(content)}, skipping")
                    return
                
                # Split content into chunks for embedding
                chunks = self._split_content(content)
                if not chunks:
                    error_msg(f"No chunks generated for {file_path}, content may be invalid")
                    return
                
                info_msg(f"Processing {len(chunks)} chunks for {file_path}")
                
                # Ensure we have an index
                if self.index is None:
                    info_msg("Index not initialized, creating new index")
                    self.clear()
                    # Double-check index was created
                    if self.index is None:
                        raise ValueError("Failed to create index after clearing")
                
                # Clear GPU memory before starting encoding
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Dynamically determine batch size based on GPU memory (if using GPU)
                batch_size = 8  # Default conservative batch size
                if self.device == 'cuda':
                    try:
                        device_id = self.gpu_id if hasattr(self, 'gpu_id') and self.gpu_id is not None else 0
                        total_memory = torch.cuda.get_device_properties(device_id).total_memory
                        reserved_memory = torch.cuda.memory_reserved(device_id)
                        free_memory = total_memory - reserved_memory
                        free_memory_gb = free_memory / (1024**3)
                        
                        # Scale batch size based on available memory - use more conservative values
                        if free_memory_gb > 40:  # High-memory GPU (>40GB free)
                            batch_size = 32  # Reduced from 64
                        elif free_memory_gb > 20:  # Medium-memory GPU (>20GB free)
                            batch_size = 16  # Reduced from 32
                        elif free_memory_gb > 10:  # Standard GPU (>10GB free)
                            batch_size = 8  # Reduced from 16
                        # Else use default 8
                        
                        info_msg(f"Using batch size {batch_size} for file {file_path} ({free_memory_gb:.1f}GB free memory)")
                    except Exception as mem_error:
                        warning_msg(f"Error determining GPU memory, using default batch size: {str(mem_error)}")
                
                # Check if multi-GPU processing is available
                use_multi_gpu = False
                gpu_count = 0
                if self.device == 'cuda':
                    try:
                        if torch.cuda.device_count() > 1:
                            gpu_count = torch.cuda.device_count()
                            use_multi_gpu = True
                            info_msg(f"Using {gpu_count} GPUs for file processing")
                    except:
                        use_multi_gpu = False
                
                start_idx = self.index.ntotal if self.index is not None else 0
                
                # Prepare metadata if not provided
                if metadata is None:
                    metadata = {}
                
                # Add file extension to metadata if not already there
                if "extension" not in metadata:
                    from pathlib import Path  # Local import to ensure Path is available
                    ext = Path(file_path).suffix.lower()
                    if ext:
                        metadata["extension"] = ext
                
                # Process in batches with improved error handling
                added_chunks = 0
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    
                    # Clean GPU memory before processing batch
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Select GPU for this batch if using multi-GPU
                    current_gpu_id = self.gpu_id if hasattr(self, 'gpu_id') else 0
                    if use_multi_gpu:
                        # Distribute batches across GPUs
                        batch_gpu_id = (i // batch_size) % gpu_count
                        if batch_gpu_id != current_gpu_id:
                            current_gpu_id = batch_gpu_id
                            # Set device for this batch
                            torch.cuda.set_device(current_gpu_id)
                            if hasattr(self.model, 'to'):
                                device_str = f"cuda:{current_gpu_id}"
                                self.model.to(device_str)
                            info_msg(f"Using GPU {current_gpu_id} for batch {i//batch_size + 1}")
                    
                    # Create embeddings for this batch
                    try:
                        embeddings = self._encode_text(batch_chunks)
                        if embeddings.shape[0] != len(batch_chunks):
                            warning_msg(f"Expected {len(batch_chunks)} embeddings but got {embeddings.shape[0]}, padding with zeros")
                            # Create zeros with the correct shape
                            correct_embeddings = np.zeros((len(batch_chunks), self.vector_size), dtype=np.float32)
                            # Copy as many embeddings as we got
                            correct_embeddings[:min(embeddings.shape[0], len(batch_chunks))] = embeddings[:min(embeddings.shape[0], len(batch_chunks))]
                            embeddings = correct_embeddings
                            
                        # Add embeddings to the index
                        try:
                            self.index.add(embeddings)
                        except Exception as add_error:
                            error_msg(f"Failed to add embeddings to index: {str(add_error)}")
                            # Try to re-initialize the index
                            try:
                                if hasattr(self, '_setup_index'):
                                    self._setup_index()
                                    # Try adding again
                                    self.index.add(embeddings)
                            except Exception as init_error:
                                error_msg(f"Failed to re-initialize index: {str(init_error)}")
                            raise  # Propagate error
                        
                        # Add entries to mapping
                        chunk_idx = 0
                        for chunk in batch_chunks:
                            # Calculate the global index
                            idx = start_idx + added_chunks + chunk_idx
                            
                            # Create mapping entry
                            self.file_mapping.append({
                                "file_path": file_path,
                                "chunk_idx": chunk_idx,
                                "chunk_text": chunk,
                                "embedding_idx": idx,
                                "metadata": metadata.copy() if metadata else {}
                            })
                            
                            # Each index corresponds to a mapping entry
                            chunk_idx += 1
                        
                        added_chunks += len(batch_chunks)
                        
                        # Clean up memory
                        del embeddings
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        
                    except Exception as chunk_error:
                        error_msg(f"Error encoding chunk batch for {file_path}: {str(chunk_error)}")
                        raise  # Propagate to retry
                
                if added_chunks > 0:
                    info_msg(f"Added {added_chunks} chunks for {file_path}")
                    
                    # Save after each file to ensure changes are persisted
                    # But only if we've processed at least some chunks
                    try:
                        self.save()
                    except Exception as save_error:
                        error_msg(f"Warning: Failed to save embeddings after adding {file_path}: {str(save_error)}")
                else:
                    warning_msg(f"No chunks were successfully added for {file_path}")
                
                # Final memory cleanup at the end of the file
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Return success
                return
                
            except Exception as e:
                retry_count += 1
                error_msg(f"Error adding file {file_path}, retry {retry_count}/{max_retries}: {str(e)}")
                
                # Clean GPU memory before retrying
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                if retry_count >= max_retries:
                    error_msg(f"Failed to add file {file_path} after {max_retries} retries")
                    return
    
    def _split_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split content into overlapping chunks for embedding.
        
        Args:
            content: Content to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of content chunks
        """
        try:
            # For very large files, use a more efficient approach
            if len(content) > 1_000_000:  # >1MB content
                logger.info(f"Using efficient chunking for large file ({len(content)/1_000_000:.2f}MB)")
                # For large files, use larger chunks with less overlap to reduce vector count
                chunk_size = 2000
                overlap = 100
                
                # Process in batches to avoid memory issues
                chunks = []
                start = 0
                while start < len(content):
                    end = min(start + chunk_size, len(content))
                    if end < len(content):
                        # Find a good break point (newline) near the end
                        break_point = content.rfind('\n', start + chunk_size - overlap, end)
                        if break_point != -1:
                            end = break_point + 1  # Include the newline

                    chunks.append(content[start:end])
                    
                    # Move start position for next chunk, accounting for overlap
                    if end == len(content):
                        break
                    start = end - overlap

                logger.info(f"Created {len(chunks)} chunks for large file")
                return chunks

            # Standard processing for normal-sized files
            lines = content.split('\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in lines:
                current_chunk.append(line)
                current_length += len(line) + 1  # +1 for newline
                
                if current_length >= chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    
                    # Keep overlap for next chunk
                    keep_lines = []
                    keep_length = 0
                    for line in reversed(current_chunk):
                        if keep_length + len(line) + 1 <= overlap:
                            keep_lines.insert(0, line)
                            keep_length += len(line) + 1
                        else:
                            break
                    
                    current_chunk = keep_lines
                    current_length = keep_length
            
            # Add final chunk if not empty
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
        except Exception as e:
            logger.warning(f"Failed to split content: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, max_results: int = 3, filter_metadata: Dict[str, Any] = None) -> List[str]:
        """
        Get relevant context for a query with optional metadata filtering.
        
        Args:
            query: Query to search for
            max_results: Maximum number of results to return
            filter_metadata: Optional filter to apply to metadata (e.g. {"extension": ".php"})
            
        Returns:
            List of relevant content chunks
        """
        try:
            # If index is empty or no model available, return empty list
            if not hasattr(self, 'index') or self.index is None or not hasattr(self.index, 'ntotal') or self.index.ntotal == 0 or not hasattr(self, 'model') or self.model is None:
                logger.warning("Cannot get relevant context: index is empty or model is unavailable")
                return []
            
            # Create query embedding
            query_embedding = self._encode_text([query])
            
            # Search for similar chunks - get more than we need for filtering
            # Ensure we don't request more results than we have in the index
            k = min(self.index.ntotal, max_results * 3)
            if k <= 0:
                logger.warning("Index is empty, cannot retrieve relevant context")
                return []
                
            D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
            
            # Return content of matching chunks with optional filtering
            results = []
            # Make sure I[0] exists before iterating
            if len(I) > 0 and len(I[0]) > 0:
                for idx in I[0]:
                    # Check if index is valid
                    if idx < 0 or idx >= len(self.file_mapping):
                        logger.warning(f"Invalid index {idx} when retrieving context, skipping")
                        continue
                        
                    # Apply metadata filter if specified
                    if filter_metadata and not self._matches_metadata(self.file_mapping[idx].get("metadata", {}), filter_metadata):
                        continue
                    
                    # Add metadata prefix to the content if available
                    entry = self.file_mapping[idx]
                    
                    # Check if content exists
                    if "content" not in entry:
                        logger.warning(f"Missing content in entry at index {idx}, skipping")
                        continue
                        
                    context = entry["content"]
                    
                    # Add file path and language info as prefix
                    metadata = entry.get("metadata", {})
                    file_info = f"File: {entry.get('file_path', 'unknown')}"
                    if "language" in metadata:
                        file_info += f" (Language: {metadata['language']})"
                    
                    results.append(f"{file_info}\n\n{context}")
                    
                    # Stop when we have enough results
                    if len(results) >= max_results:
                        break
            
            return results
        except Exception as e:
            logger.error(f"Failed to get relevant context for {query}: {str(e)}")
            # Return empty list on error to allow processing to continue
            return []
    
    def _matches_metadata(self, entry_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if entry metadata matches the filter criteria."""
        if not entry_metadata or not filter_metadata:
            return True
            
        for key, value in filter_metadata.items():
            if key not in entry_metadata:
                return False
                
            if isinstance(value, list):
                if entry_metadata[key] not in value:
                    return False
            elif entry_metadata[key] != value:
                return False
                
        return True
    
    def search_by_file_path(self, file_path_pattern: str, max_results: int = 10) -> List[str]:
        """
        Search for files by path pattern.
        
        Args:
            file_path_pattern: Pattern to match against file paths
            max_results: Maximum number of results to return
            
        Returns:
            List of file paths matching the pattern
        """
        try:
            # Check if pattern contains wildcards
            is_pattern = "*" in file_path_pattern or "?" in file_path_pattern
            
            # Get unique file paths
            file_paths = set()
            for info in self.file_mapping:  # Changed from iterating over dict items to list items
                path = info.get("file_path", "")
                if not path:
                    continue
                    
                # Match either exact path or pattern
                if is_pattern:
                    if fnmatch.fnmatch(path, file_path_pattern):
                        file_paths.add(path)
                elif file_path_pattern in path:
                    file_paths.add(path)
                
                if len(file_paths) >= max_results:
                    break
            
            return list(file_paths)
        except Exception as e:
            logger.error(f"Failed to search by file path: {str(e)}")
            return []
    
    def get_file_relationships(self) -> Dict[str, List[str]]:
        """
        Get relationships between files based on imports/dependencies.
        
        Returns:
            Dictionary mapping file paths to lists of related file paths
        """
        try:
            relationships = {}
            
            # Get unique file paths and their metadata
            files_metadata = {}
            for info in self.file_mapping:  # This is correct - iterating over list items
                path = info.get("file_path", "")
                if not path or path in files_metadata:
                    continue
                    
                metadata = info.get("metadata", {})
                files_metadata[path] = metadata
            
            # Build relationships based on imports/dependencies
            for path, metadata in files_metadata.items():
                relationships[path] = []
                
                imports = metadata.get("imports", [])
                dependencies = metadata.get("dependencies", [])
                
                # For each dependency, find matching files
                for dep in dependencies:
                    # Look for files that match this dependency
                    for other_path, other_meta in files_metadata.items():
                        if other_path == path:
                            continue
                            
                        # Simple matching - can be improved
                        if (dep in other_path or 
                            (other_meta.get("module_name", "") and dep == other_meta["module_name"])):
                            relationships[path].append(other_path)
            
            return relationships
        except Exception as e:
            logger.error(f"Failed to get file relationships: {str(e)}")
            return {}
    
    def contains_file(self, file_path: str) -> bool:
        """
        Check if a file is already indexed in the embedding store.
        
        Args:
            file_path: Path of the file to check
            
        Returns:
            True if the file is already indexed, False otherwise
        """
        try:
            # Normalize file path for comparison (handle both forward/backslashes)
            normalized_path = file_path.replace('\\', '/')
            
            # Handle case where we have no mapping
            if not self.file_mapping:
                return False
                
            # Optimize the check to stop as soon as a match is found
            for entry in self.file_mapping:
                entry_path = entry.get("file_path", "")
                if entry_path:
                    # Also normalize entry path
                    entry_path = entry_path.replace('\\', '/')
                    if entry_path == normalized_path:
                        return True
            return False
        except Exception as e:
            # If there's any error, log it and return False to be safe
            warning_msg(f"Error checking if file exists in embedding store: {str(e)}")
            return False
    
    def _get_all_files(self) -> List[str]:
        """
        Get a list of all unique file paths in the embedding store.
        
        Returns:
            List of unique file paths
        """
        try:
            # Handle case where mapping is empty
            if not self.file_mapping:
                logger.warning("Mapping is empty, attempting to reload from disk")
                try:
                    self.load()
                except Exception as e:
                    logger.warning(f"Failed to reload embedding store: {str(e)}")
                    return []
                    
                # If still empty after reload, return empty list
                if not self.file_mapping:
                    logger.warning("Mapping is still empty after reload")
                    return []
            
            # Extract unique file paths
            # CRITICAL FIX: Use a set to ensure we don't miss any files or duplicates
            unique_files = set()
            
            # Ensure we collect ALL file paths from the mapping
            for info in self.file_mapping:
                if isinstance(info, dict) and "file_path" in info:
                    file_path = info["file_path"]
                    if file_path and isinstance(file_path, str):
                        # Normalize paths for consistency
                        normalized_path = file_path.replace('\\', '/')
                        unique_files.add(normalized_path)
            
            # Create final list sorted for consistency
            result_files = sorted(list(unique_files))
            
            # Force debug message to verify this is working
            print(f"DEBUG: Found {len(result_files)} unique files in embedding store")
            logger.info(f"Found {len(result_files)} unique files in embedding store")
            
            return result_files
            
        except Exception as e:
            logger.error(f"Error retrieving files from embedding store: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup resources when the object is deleted."""
        try:
            # Check if Python is shutting down
            if sys is None or sys.meta_path is None:
                # Python is shutting down, don't try to save
                print("Skipping save during Python shutdown")
                return
                
            # Save any pending changes - only if Python is not shutting down
            if hasattr(self, 'index') and self.index is not None and hasattr(self, 'file_mapping'):
                try:
                    import os  # Explicitly import what we need
                    import json
                    import faiss
                    from pathlib import Path  # Add missing import
                    self.save()
                except Exception as save_error:
                    print(f"Failed to save embeddings during cleanup: {save_error}")
                    
            # Clean up model resources
            if hasattr(self, 'model') and self.model is not None:
                # Release model resources
                self.model = None
                
            # Clean up GPU resources if using FAISS with GPU
            if hasattr(self, 'use_gpu_for_faiss') and self.use_gpu_for_faiss:
                if hasattr(self, 'gpu_resources') and self.gpu_resources is not None:
                    # Explicitly clean up GPU resources
                    del self.gpu_resources
                    self.gpu_resources = None
                
            # Explicitly clean up FAISS index
            if hasattr(self, 'index') and self.index is not None:
                del self.index
                self.index = None
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            # Don't raise exceptions in __del__
            print(f"Error during EmbeddingStore cleanup: {e}")
            pass

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the embedding store and generate embeddings.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
            
        try:
            logger.info(f"Adding {len(documents)} documents to embedding store")
            
            # Filter out test files and validate documents
            filtered_documents = []
            for doc in documents:
                # Skip if content is missing or empty
                if "content" not in doc or not doc["content"]:
                    logger.warning(f"Skipping document with missing content")
                    continue
                    
                # Check if metadata exists
                metadata = doc.get("metadata", {})
                
                # Get file_path from document or metadata 
                if "file_path" in doc:
                    file_path = doc["file_path"]
                else:
                    file_path = metadata.get("file_path", "")
                
                # Skip if file_path is missing
                if not file_path:
                    logger.warning(f"Skipping document with missing file_path")
                    continue
                
                # Skip files with 'test' or 'Test' in their paths
                if 'test' in file_path.lower():
                    continue
                    
                # Create a valid document with all required fields
                filtered_doc = {
                    "file_path": file_path,
                    "content": doc["content"],
                    "metadata": metadata
                }
                
                filtered_documents.append(filtered_doc)
                
            if len(filtered_documents) < len(documents):
                logger.info(f"Filtered out {len(documents) - len(filtered_documents)} test files and invalid documents")
                
            documents = filtered_documents
            
            # Ensure we have an index
            if self.index is None:
                logger.info("Index not initialized, creating a new one")
                self.clear()
                
            # Check again after filtering 
            if not documents:
                logger.warning("No valid documents to add after filtering")
                return
                
            # Process documents in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Create embeddings for this batch using the helper method
                batch_text = [doc["content"] for doc in batch]
                batch_embeddings = self._encode_text(batch_text)
                
                # Add to index
                self.index.add(batch_embeddings.astype('float32'))
                
                # Get the starting index for this batch
                start_idx = self.index.ntotal - len(batch)
                
                # Map file path to indices
                for j, doc in enumerate(batch):
                    idx = start_idx + j
                    self.file_mapping.append({
                        "file_path": doc["file_path"],
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    })
                
                # Force garbage collection between batches to prevent memory leaks
                if i % (batch_size * 4) == 0 and i > 0:
                    gc.collect()
            
            # Save after each batch to avoid losing work
            self.save()
            
            logger.info(f"Added {len(documents)} documents to embedding store")
        except Exception as e:
            logger.error(f"Failed to add documents to embedding store: {str(e)}")
            
            # Try to clean up any partial work
            gc.collect()

def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """
    Add embedding store CLI arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group('Embedding Store Options')
    group.add_argument(
        '--device', 
        type=str, 
        choices=['auto', 'cpu', 'cuda'], 
        default='auto',
        help='Device to use for embedding calculations (auto, cpu, cuda)'
    )

def get_embedding_store_from_args(args: argparse.Namespace) -> EmbeddingStore:
    """
    Create EmbeddingStore from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Configured EmbeddingStore instance
    """
    # Convert 'auto' to None for auto-detection
    device = None if args.device == 'auto' else args.device
    return EmbeddingStore(device=device)

if __name__ == "__main__":
    # Simple CLI for testing embedding store
    parser = argparse.ArgumentParser(description="Embedding Store CLI")
    add_cli_args(parser)
    args = parser.parse_args()
    
    # Initialize embedding store
    store = get_embedding_store_from_args(args)
    
    # Load existing embeddings
    store.load()
    
    # Some basic info
    print(f"Device: {store.device}")
    print(f"Index size: {store.index.ntotal if store.index else 0} embeddings") 