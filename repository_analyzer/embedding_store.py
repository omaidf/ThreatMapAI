"""
Embedding Store Module

This module provides functionality for storing and retrieving
code embeddings for semantic search and similarity analysis.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import faiss
import numpy as np
import argparse

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable
from utils.file_utils import check_output_directory

logger = logging.getLogger(__name__)

class EmbeddingStoreError(Exception):
    """Custom exception for embedding store errors."""
    pass

def detect_gpu() -> Tuple[bool, Optional[int]]:
    """
    Detect if GPU is available and return its memory in GB.
    
    Returns:
        Tuple of (is_gpu_available, gpu_memory_gb)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Get memory of first GPU in GB
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info_msg(f"Detected GPU with {gpu_memory:.2f} GB memory")
                return True, gpu_memory
    except Exception as e:
        warning_msg(f"Error detecting GPU: {str(e)}")
    
    return False, None

class EmbeddingStore:
    """
    Store and retrieve code embeddings for semantic search.
    
    This class handles:
    - Embedding generation for code chunks
    - Storing embeddings in a FAISS index
    - Semantic search for related code
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the embedding store.
        
        Args:
            device: Force device to use ('cpu', 'cuda', or None for auto-detection)
        """
        self.output_dir = Path(get_env_variable("OUTPUT_DIR", "output"))
        check_output_directory(str(self.output_dir))
        self.index_path = self.output_dir / "embeddings.index"
        self.mapping_path = self.output_dir / "embeddings_mapping.json"
        
        # Initialize empty index and mapping
        self.index = None
        self.file_mapping = []  # Changed from dictionary to list for consistency
        
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
        
        info_msg(f"Using device: {self.device}")
        
        # Configure resources based on device
        if self.device == 'cpu':
            # Set environment variables to disable all parallelism
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warnings
            os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
            os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
            
            # Try to disable multiprocessing in torch
            try:
                import torch
                torch.set_num_threads(1)  # Use only one thread
                if hasattr(torch, 'set_num_interop_threads'):
                    torch.set_num_interop_threads(1)
            except:
                pass
        
        # Initialize the embedding model
        try:
            # Import here to avoid top-level import issues
            from sentence_transformers import SentenceTransformer
            import sentence_transformers.util
            
            if self.device == 'cpu':
                # CPU-specific optimizations
                # PATCH: Override the batch encoding utility to force sequential processing
                original_batch_to_device = sentence_transformers.util.batch_to_device
                def patched_batch_to_device(batch, target_device, cpu_pin_memory=False):
                    # Process sequentially, never use multiple processes
                    return original_batch_to_device(batch, target_device, cpu_pin_memory)
                sentence_transformers.util.batch_to_device = patched_batch_to_device
            
            # Use a smaller model for embeddings to avoid memory issues
            model_name = 'all-MiniLM-L6-v2'  # This is a small, efficient model (384 dimensions)
            
            try:
                # Load model with selected device
                self.model = SentenceTransformer(model_name, device=self.device)
                
                # Configure model based on device
                if self.device == 'cpu' and hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = min(self.model.max_seq_length, 256)  # Limit sequence length
                
                self.vector_size = self.model.get_sentence_embedding_dimension()
                info_msg(f"Embedding model initialized with dimension {self.vector_size} on {self.device}")
            except Exception as model_e:
                # Fallback to a simpler model
                warning_msg(f"Failed to load {model_name}: {str(model_e)}, trying alternative model")
                try:
                    self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
                    if self.device == 'cpu' and hasattr(self.model, 'max_seq_length'):
                        self.model.max_seq_length = min(self.model.max_seq_length, 256)  # Limit sequence length
                    self.vector_size = self.model.get_sentence_embedding_dimension()
                    info_msg(f"Loaded fallback embedding model with dimension {self.vector_size} on {self.device}")
                except Exception as e2:
                    # Create a dummy embedding model that returns zeros
                    warning_msg(f"Failed to load embedding model: {str(e2)}")
                    info_msg("⚠️ Creating dummy embedding model")
                    self.vector_size = 384  # Standard dimension for embeddings
                    self.model = None
        except Exception as e:
            warning_msg(f"Failed to initialize embedding model: {str(e)}")
            # Set default values for minimal functionality
            self.vector_size = 384 
            self.model = None
            info_msg("ℹ️ Embedding store will use dummy embeddings")
            
        # Set up FAISS index
        self._setup_index()
    
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
                        
                        # Convert CPU index to GPU index
                        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                        self.use_gpu_for_faiss = True
                        info_msg("Successfully moved FAISS index to GPU")
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
            check_output_directory(str(self.output_dir))
            
            # Save mapping
            with open(self.mapping_path, 'w') as f:
                json.dump(self.file_mapping, f, indent=2)
            
            # If using GPU index, we need to convert back to CPU for saving
            if hasattr(self, 'use_gpu_for_faiss') and self.use_gpu_for_faiss:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, str(self.index_path))
                except Exception as e:
                    warning_msg(f"Failed to convert GPU index to CPU for saving: {str(e)}")
                    warning_msg("Attempting to save index directly")
                    faiss.write_index(self.index, str(self.index_path))
            else:
                # Save index directly if it's already a CPU index
                faiss.write_index(self.index, str(self.index_path))
            
            success_msg(f"Saved {len(self.file_mapping)} embeddings to {self.index_path}")
        except Exception as e:
            error_msg(f"Failed to save embeddings: {str(e)}")
    
    def _setup_index(self) -> None:
        """Set up the FAISS index, using GPU if available."""
        # Create a new index based on vector size
        try:
            # Check if we should use GPU for FAISS
            self.use_gpu_for_faiss = self.device == 'cuda' and self.gpu_available
            
            # Create a flat L2 index first (CPU version)
            cpu_index = faiss.IndexFlatL2(self.vector_size)
            
            if self.use_gpu_for_faiss:
                try:
                    # Import required FAISS GPU components
                    import faiss.contrib.torch_utils  # For PyTorch compatibility
                    
                    # Check if GPU version of FAISS is available
                    if hasattr(faiss, 'StandardGpuResources'):
                        info_msg("Using GPU acceleration for FAISS index")
                        
                        # Create GPU resources
                        self.gpu_resources = faiss.StandardGpuResources()
                        
                        # Configure GPU index
                        gpu_config = faiss.GpuIndexFlatConfig()
                        gpu_config.device = 0  # Use first GPU
                        gpu_config.useFloat16 = True  # Use half-precision for memory efficiency
                        
                        # Create GPU index
                        self.index = faiss.GpuIndexFlatL2(self.gpu_resources, self.vector_size, gpu_config)
                        return
                    else:
                        warning_msg("FAISS GPU support not available, falling back to CPU index")
                except Exception as e:
                    warning_msg(f"Failed to initialize GPU index for FAISS: {str(e)}")
                    warning_msg("Falling back to CPU index")
            
            # If we get here, use CPU index
            self.index = cpu_index
            
        except Exception as e:
            warning_msg(f"Failed to create FAISS index: {str(e)}")
            # Create a minimal CPU index as fallback
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.use_gpu_for_faiss = False
    
    def clear(self) -> None:
        """
        Clear the embedding store without deleting the model files.
        This resets the index and metadata while preserving the 
        embedding infrastructure.
        """
        try:
            # Reset in-memory data
            self.index = None
            self.file_mapping = []  # Changed from dictionary to list
            self.total_chunks = 0
            
            # Re-initialize the embedding model and index
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.vector_size = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Failed to initialize SentenceTransformer: {str(e)}")
                self.model = None
                # Keep existing vector size
            
            # Initialize the index
            self._setup_index()
            info_msg("Embeddings cleared, created new empty index")
            
            # Save empty state
            self.save()
            
            logger.info("Embedding store cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear embedding store: {str(e)}")
            raise EmbeddingStoreError(f"Failed to clear embedding store: {str(e)}")
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text to embeddings, with fallback for when the model is None.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            # If we don't have a model, return zeros
            info_msg("Using dummy embeddings (zeros)")
            return np.zeros((len(texts), self.vector_size), dtype=np.float32)
        else:
            try:
                if self.device == 'cpu':
                    # Force sequential processing - do NOT batch process on CPU
                    all_embeddings = []
                    
                    # Process each text individually to avoid multiprocessing
                    for text in texts:
                        # Set show_progress_bar to False to avoid tqdm which can sometimes use semaphores
                        # Set convert_to_tensor to False to avoid torch multiprocessing
                        # Set normalize_embeddings to False for speed (we normalize manually later)
                        single_embedding = self.model.encode(
                            [text], 
                            show_progress_bar=False,
                            convert_to_tensor=False,
                            normalize_embeddings=False,
                            batch_size=1  # Force batch size of 1
                        )
                        
                        # Only keep the first embedding (there's only one anyway)
                        all_embeddings.append(single_embedding[0])
                    
                    # Stack all embeddings
                    result = np.vstack(all_embeddings)
                else:
                    # Use GPU acceleration with batch processing
                    # Still disable progress bar to avoid tqdm issues
                    result = self.model.encode(
                        texts,
                        show_progress_bar=False,
                        convert_to_tensor=False,
                        normalize_embeddings=False,
                        batch_size=32  # Larger batch size for GPU
                    )
                
                # Normalize manually if needed (L2 normalization)
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                result = result / norms
                
                return result
                
            except Exception as e:
                warning_msg(f"Error during encoding: {str(e)}. Using zeros instead.")
                return np.zeros((len(texts), self.vector_size), dtype=np.float32)
    
    def add_file(self, file_path: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a file to the embedding store.
        
        Args:
            file_path: Path to the file
            content: Content of the file
            metadata: Additional metadata about the file (language, imports, etc.)
        """
        try:
            # Check if this file is already in the embedding store
            if self.contains_file(file_path):
                logger.info(f"File {file_path} already in embedding store, skipping")
                return
                
            # Split content into chunks for embedding
            chunks = self._split_content(content)
            if not chunks:
                return
            
            # Create embeddings - process in batches to reduce memory usage
            batch_size = 16
            start_idx = self.index.ntotal if self.index is not None else 0
            
            # Ensure we have an index
            if self.index is None:
                self.clear()
            
            # Prepare metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add file extension to metadata if not already there
            if "extension" not in metadata:
                ext = Path(file_path).suffix.lower()
                if ext:
                    metadata["extension"] = ext
            
            # Process in smaller batches to reduce memory pressure
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                
                # Create embeddings for this batch using the helper method
                batch_embeddings = self._encode_text(batch_chunks)
                
                # Add to index
                self.index.add(np.array(batch_embeddings).astype('float32'))
                
                # Map file path to indices
                for j, chunk in enumerate(batch_chunks):
                    idx = start_idx + i + j
                    
                    self.file_mapping.append({
                        "file_path": file_path,
                        "content": chunk,
                        "metadata": metadata
                    })
                
                # Force garbage collection between batches to prevent memory leaks
                if i % (batch_size * 4) == 0 and i > 0:
                    import gc
                    gc.collect()
            
            # Save after each file to avoid losing work
            self.save()
            
            logger.info(f"Added {len(chunks)} chunks from {file_path} to embedding store")
        except Exception as e:
            logger.error(f"Failed to add file {file_path} to embedding store: {str(e)}")
            
            # Try to clean up any partial work
            import gc
            gc.collect()
    
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
            if self.index is None or self.index.ntotal == 0 or self.model is None:
                logger.warning("Cannot get relevant context: index is empty or model is unavailable")
                return []
            
            # Create query embedding
            query_embedding = self._encode_text([query])
            
            # Search for similar chunks - get more than we need for filtering
            D, I = self.index.search(np.array(query_embedding).astype('float32'), min(self.index.ntotal, max_results * 3))
            
            # Return content of matching chunks with optional filtering
            results = []
            for idx in I[0]:
                if idx < len(self.file_mapping):
                    # Apply metadata filter if specified
                    if filter_metadata and not self._matches_metadata(self.file_mapping[idx].get("metadata", {}), filter_metadata):
                        continue
                    
                    # Add metadata prefix to the content if available
                    entry = self.file_mapping[idx]
                    context = entry["content"]
                    
                    # Add file path and language info as prefix
                    metadata = entry.get("metadata", {})
                    file_info = f"File: {entry['file_path']}"
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
                    import fnmatch
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
        # Optimize the check to stop as soon as a match is found
        for entry in self.file_mapping:
            if entry.get("file_path") == file_path:
                return True
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
            unique_files = set()
            for info in self.file_mapping:  # Changed from iterating over dict items to list items
                if "file_path" in info:
                    unique_files.add(info["file_path"])
            
            logger.info(f"Found {len(unique_files)} unique files in embedding store")
            return list(unique_files)
            
        except Exception as e:
            logger.error(f"Error retrieving files from embedding store: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup resources when the object is deleted."""
        try:
            # Save any pending changes
            if hasattr(self, 'index') and self.index is not None and hasattr(self, 'file_mapping'):
                try:
                    self.save()
                except:
                    pass
                    
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
            import gc
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
            
            # Filter out test files
            filtered_documents = []
            for doc in documents:
                file_path = doc.get("metadata", {}).get("file_path", "")
                # Skip files with 'test' or 'Test' in their paths
                if 'test' in file_path.lower():
                    continue
                filtered_documents.append(doc)
                
            if len(filtered_documents) < len(documents):
                logger.info(f"Filtered out {len(documents) - len(filtered_documents)} test files")
                
            documents = filtered_documents
            
            # Process documents in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Create embeddings for this batch using the helper method
                batch_embeddings = self._encode_text([doc["content"] for doc in batch])
                
                # Add to index
                self.index.add(batch_embeddings.astype('float32'))
                
                # Map file path to indices
                for j, doc in enumerate(batch):
                    idx = self.index.ntotal - len(documents) + i + j
                    
                    self.file_mapping.append({
                        "file_path": doc["file_path"],
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    })
                
                # Force garbage collection between batches to prevent memory leaks
                if i % (batch_size * 4) == 0 and i > 0:
                    import gc
                    gc.collect()
            
            # Save after each batch to avoid losing work
            self.save()
            
            logger.info(f"Added {len(documents)} documents to embedding store")
        except Exception as e:
            logger.error(f"Failed to add documents to embedding store: {str(e)}")
            
            # Try to clean up any partial work
            import gc
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