"""
GPU Utilities for the AI Threat Model Map Generator.

This module centralizes all GPU/CPU detection, configuration, and optimization functionality.
It provides utilities for working with GPUs (CUDA, MPS), memory management, and distributed processing.
"""

import os
import sys
import logging
import subprocess
import platform
import importlib
import importlib.util
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path
import numpy as np
import math

# Import internal utilities
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable, update_env_file

# Try to import torch and related libraries at module level
# These are handled gracefully when not available
try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Configure logger
logger = logging.getLogger(__name__)

def detect_gpu() -> Tuple[bool, Optional[float]]:
    """
    Detect if GPU is available and return its memory in GB.
    
    Returns:
        Tuple of (is_gpu_available, gpu_memory_gb)
    """
    try:
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

def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        Dictionary with detailed GPU information
    """
    result = {
        'has_gpu': False,
        'gpu_count': 0,
        'gpus': [],
        'total_memory': 0,
        'cuda_version': None,
        'platform': platform.system(),
        'architecture': platform.machine()
    }
    
    try:
        if torch.cuda.is_available():
            result['has_gpu'] = True
            result['gpu_count'] = torch.cuda.device_count()
            result['cuda_version'] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
            
            # Get info for each GPU
            for i in range(result['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                gpu_memory = props.total_memory / (1024**3)  # Convert to GB
                result['total_memory'] += gpu_memory
                
                gpu_info = {
                    'id': i,
                    'name': props.name,
                    'memory_gb': gpu_memory,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count,
                    'clock_rate_mhz': props.clock_rate / 1000
                }
                
                # Get current memory usage if available
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu_info['memory_allocated_gb'] = memory_allocated
                    gpu_info['memory_reserved_gb'] = memory_reserved
                    gpu_info['memory_available_gb'] = gpu_memory - memory_reserved
                except:
                    pass
                    
                result['gpus'].append(gpu_info)
        
        # Check for MPS (Apple Silicon)
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            result['is_apple_silicon'] = True
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    result['has_mps'] = True
                    
                    # Try to determine Apple Silicon model
                    processor = platform.processor()
                    if "M1" in processor:
                        result['apple_model'] = "M1"
                    elif "M2" in processor:
                        result['apple_model'] = "M2"
                    elif "M3" in processor:
                        result['apple_model'] = "M3"
                    else:
                        result['apple_model'] = "Unknown Apple Silicon"
            except:
                result['has_mps'] = False
    
    except ImportError:
        result['error'] = "PyTorch not installed"
    except Exception as e:
        result['error'] = str(e)
    
    return result

def detect_gpu_capabilities(force_gpu: bool = False, force_cpu: bool = False) -> Dict[str, Any]:
    """
    Detect GPU capabilities and return appropriate configurations for LLMs.
    
    This function detects if CUDA/GPU is available and returns appropriate
    configuration settings for model loading.
    
    Args:
        force_gpu: Force GPU usage even if detection fails
        force_cpu: Force CPU usage even if GPU is available
        
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
    
    # Handle forced settings
    if force_cpu:
        info_msg("Forcing CPU usage as requested")
        return result
    
    try:
        # Try importing torch to check for CUDA/MPS availability
        if torch.cuda.is_available() or force_gpu:
            # Get CUDA information
            if torch.cuda.is_available():
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
            elif force_gpu:
                # If forcing GPU but CUDA not available, use placeholder values
                gpu_count = 1
                gpu_name = "Forced NVIDIA GPU"
                gpu_memory = 8  # Assume 8GB
            
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
                # Multi-GPU setup requires aggressive layer offloading
                if is_large_model:
                    # For 70B models with multiple GPUs
                    if gpu_count >= 4:
                        # With 4+ GPUs, we can offload all layers for large models
                        result['n_gpu_layers'] = -1  # All layers
                    elif gpu_count >= 2:
                        # With 2-3 GPUs, be more aggressive for 70B model
                        result['n_gpu_layers'] = -1  # All layers
                else:
                    # For smaller models, definitely use all layers with multiple GPUs
                    result['n_gpu_layers'] = -1  # All layers
                
                # Log special multi-GPU configuration
                info_msg(f"Using multi-GPU configuration with {gpu_count} GPUs")
            # Single GPU case
            elif gpu_memory is not None:
                # Decide based on GPU memory and model size
                if is_large_model:
                    # More aggressive for large models on single GPU
                    if gpu_memory < 8:
                        result['n_gpu_layers'] = 8   # Very limited
                    elif gpu_memory < 16:
                        result['n_gpu_layers'] = 24  # Limited
                    elif gpu_memory < 24:
                        result['n_gpu_layers'] = 32  # Moderate
                    elif gpu_memory < 32:
                        result['n_gpu_layers'] = 60  # Good
                    else:
                        result['n_gpu_layers'] = -1  # All layers (for 40GB+ cards)
                else:
                    # More aggressive for smaller models
                    if gpu_memory < 4:
                        result['n_gpu_layers'] = 8
                    elif gpu_memory < 8:
                        result['n_gpu_layers'] = 24
                    elif gpu_memory < 16:
                        result['n_gpu_layers'] = -1  # All layers
                    else:
                        result['n_gpu_layers'] = -1  # All layers
            elif force_gpu:
                # If forcing GPU but don't know memory, use conservative defaults
                result['n_gpu_layers'] = is_large_model and 32 or -1
            else:
                # Conservative default for unknown memory
                result['n_gpu_layers'] = is_large_model and 24 or 32
            
            info_msg(f"🔥 NVIDIA GPU detected: {result['gpu_info']}")
            info_msg(f"Will use CUDA with {result['n_gpu_layers'] if result['n_gpu_layers'] != -1 else 'all'} GPU layers")
            if is_large_model:
                info_msg(f"Using {'aggressive' if result['n_gpu_layers'] == -1 else 'conservative'} GPU settings for 70B model")
            return result
            
        # Next check for Metal Performance Shaders (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result['device'] = 'mps'
            result['use_gpu'] = True
            result['n_gpu_layers'] = 1  # Start conservative with Apple Silicon
            result['gpu_info'] = "Apple Silicon GPU (MPS)"
            
            # Determine device by platform
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
                    result['n_gpu_layers'] = 16  # More aggressive for M3
            
            info_msg(f"🔥 Apple Silicon GPU detected: {result['gpu_info']}")
            info_msg(f"Will use Metal with {result['n_gpu_layers']} GPU layers")
            return result
    
    except ImportError:
        warning_msg("PyTorch not available, defaulting to CPU")
    except Exception as e:
        warning_msg(f"Error detecting GPU capabilities: {str(e)}")
    
    info_msg("Using CPU for inference (no GPU acceleration)")
    return result

def install_faiss_gpu() -> bool:
    """
    Check if the FAISS-GPU package is installed.
    
    Returns:
        True if FAISS-GPU is installed, False otherwise
    """
    try:
        # Try importing faiss
        import faiss
        
        # Check if FAISS has GPU support
        has_gpu = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        has_indexflatl2 = hasattr(faiss, 'IndexFlatL2')
        
        if has_gpu and has_indexflatl2:
            info_msg(f"FAISS-GPU is already installed with {faiss.get_num_gpus()} GPUs and IndexFlatL2 support")
            return True
        elif has_indexflatl2:
            info_msg("FAISS is installed but without GPU support")
            return False
        else:
            info_msg("FAISS is installed but missing IndexFlatL2")
            return False
    except ImportError:
        info_msg("FAISS is not installed")
        return False
    except Exception as e:
        error_msg(f"Error checking FAISS installation: {str(e)}")
        return False

def configure_gpu_environment(force_gpu: bool = False, 
                             force_cpu: bool = False, 
                             gpu_ids: Optional[List[int]] = None,
                             memory_limit: Optional[float] = None,
                             distributed: bool = False) -> Dict[str, Any]:
    """
    Configure GPU environment for optimal performance.
    
    This function sets up appropriate environment variables and settings
    for GPU usage, based on detected hardware and user preferences.
    
    Args:
        force_gpu: Force GPU usage even if detection fails
        force_cpu: Force CPU usage even if GPU is available
        gpu_ids: List of specific GPU IDs to use
        memory_limit: Memory limit in GB per GPU
        distributed: Enable distributed processing across GPUs
        
    Returns:
        Dictionary with configuration applied
    """
    result = {
        'success': True,
        'device': 'cpu',
        'gpu_count': 0,
        'selected_gpus': [],
        'distributed': False,
        'memory_limit': None
    }
    
    try:
        # Try to increase memory lock limits for better GPU performance
        increase_memory_lock_limit()
        
        # Can't force both CPU and GPU
        if force_gpu and force_cpu:
            error_msg("Cannot force both CPU and GPU at the same time")
            result['success'] = False
            return result
        
        # Get GPU capabilities
        gpu_config = detect_gpu_capabilities(force_gpu=force_gpu, force_cpu=force_cpu)
        result['device'] = gpu_config['device']
        
        # Skip GPU configuration if CPU is forced or no GPU available
        if force_cpu or (not gpu_config['use_gpu'] and not force_gpu):
            result['device'] = 'cpu'
            os.environ["FORCE_CPU"] = "true"
            if "FORCE_GPU" in os.environ:
                del os.environ["FORCE_GPU"]
                
            # Update the .env file
            update_env_file("FORCE_CPU", "true")
            update_env_file("FORCE_GPU", "")
            
            # Clear GPU-specific settings
            for var in ["GPU_IDS", "DISTRIBUTED", "GPU_MEMORY_LIMIT"]:
                if var in os.environ:
                    del os.environ[var]
                update_env_file(var, "")
                
            info_msg("Configured for CPU-only operation")
            return result
        
        # Now handle GPU configuration
        if torch.cuda.is_available() or force_gpu:
            result['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 1
            
            # Set environment variable to force GPU
            os.environ["FORCE_GPU"] = "true"
            if "FORCE_CPU" in os.environ:
                del os.environ["FORCE_CPU"]
            update_env_file("FORCE_GPU", "true")
            update_env_file("FORCE_CPU", "")
            
            # Process GPU IDs
            all_gpus = list(range(result['gpu_count']))
            selected_gpus = gpu_ids if gpu_ids else all_gpus
            
            # Validate selected GPUs
            if torch.cuda.is_available():
                selected_gpus = [gpu_id for gpu_id in selected_gpus if gpu_id < result['gpu_count']]
                
            if not selected_gpus:
                warning_msg("No valid GPUs selected, defaulting to all available GPUs")
                selected_gpus = all_gpus
            
            # Update environment with selected GPUs
            os.environ["GPU_IDS"] = ",".join(str(gpu_id) for gpu_id in selected_gpus)
            update_env_file("GPU_IDS", ",".join(str(gpu_id) for gpu_id in selected_gpus))
            result['selected_gpus'] = selected_gpus
            
            # Configure distributed processing
            should_distribute = distributed or (len(selected_gpus) > 1)
            if should_distribute:
                os.environ["DISTRIBUTED"] = "true"
                update_env_file("DISTRIBUTED", "true")
                result['distributed'] = True
                
                # Check if FAISS-GPU is installed for better multi-GPU performance
                has_faiss_gpu = install_faiss_gpu()
                if has_faiss_gpu:
                    info_msg("FAISS-GPU is already installed, good for multi-GPU performance")
            else:
                # Clear distributed setting if not needed
                if "DISTRIBUTED" in os.environ:
                    del os.environ["DISTRIBUTED"]
                update_env_file("DISTRIBUTED", "")
                
                # Check if FAISS-GPU is installed
                install_faiss_gpu()
            
            # Set memory limit if provided
            if memory_limit:
                os.environ["GPU_MEMORY_LIMIT"] = str(memory_limit)
                update_env_file("GPU_MEMORY_LIMIT", str(memory_limit))
                result['memory_limit'] = memory_limit
            else:
                # Clear memory limit if not specified
                if "GPU_MEMORY_LIMIT" in os.environ:
                    del os.environ["GPU_MEMORY_LIMIT"]
                update_env_file("GPU_MEMORY_LIMIT", "")
            
            # Log the configuration
            info_msg(f"Configured for GPU acceleration using {len(selected_gpus)} of {result['gpu_count']} available GPUs")
            if result['distributed']:
                info_msg("Distributed processing enabled for multi-GPU operation")
            if result['memory_limit']:
                info_msg(f"GPU memory limit set to {result['memory_limit']} GB per GPU")
        
    except ImportError:
        warning_msg("PyTorch not installed, defaulting to CPU configuration")
        result['success'] = False
        result['device'] = 'cpu'
    except Exception as e:
        warning_msg(f"Error configuring GPU environment: {str(e)}")
        result['success'] = False
        result['error'] = str(e)
    
    return result

def get_optimal_gpu_configuration() -> Dict[str, Any]:
    """
    Determine the optimal GPU configuration based on system capabilities.
    
    Returns:
        Dictionary with recommended settings
    """
    result = {
        'device': 'cpu',
        'distributed': False,
        'gpu_ids': [],
        'memory_limit': None
    }
    
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            result['device'] = 'cuda'
            
            # Get total memory across all GPUs
            total_memory = 0
            gpu_info = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory / (1024**3)  # Convert to GB
                total_memory += memory
                gpu_info.append({
                    'id': i,
                    'name': props.name,
                    'memory': memory
                })
            
            # Check model size for contextual recommendations
            try:
                from utils.model_config import get_model_info
                model_info = get_model_info()
                is_large_model = "70b" in model_info["name"].lower()
            except:
                is_large_model = True  # Assume large model
            
            # Multiple GPUs available
            if gpu_count > 1:
                result['distributed'] = True
                
                # For large models, use all available GPUs
                if is_large_model:
                    if total_memory >= 80:  # Plenty of memory for 70B model
                        result['gpu_ids'] = list(range(gpu_count))
                    elif total_memory >= 48:  # Decent amount for 70B model
                        result['gpu_ids'] = list(range(min(gpu_count, 4)))
                        result['memory_limit'] = None  # Use all available memory
                    else:  # Limited memory
                        # Sort GPUs by memory and use the ones with most memory
                        sorted_gpus = sorted(gpu_info, key=lambda x: x['memory'], reverse=True)
                        result['gpu_ids'] = [gpu['id'] for gpu in sorted_gpus[:min(3, gpu_count)]]
                        # Limit memory to leave some for the system
                        result['memory_limit'] = None
                else:  # Smaller model (7B)
                    # More conservative for smaller models - often one GPU is enough
                    if total_memory >= 40:
                        # Use multiple GPUs for parallelism
                        result['gpu_ids'] = list(range(min(gpu_count, 2)))
                    else:
                        # Just use the GPU with most memory
                        best_gpu = max(range(gpu_count), key=lambda i: gpu_info[i]['memory'])
                        result['gpu_ids'] = [best_gpu]
                        result['distributed'] = False
            else:  # Single GPU
                result['gpu_ids'] = [0]
                result['distributed'] = False
                
                # For large models with limited memory, set a memory limit
                single_gpu_memory = gpu_info[0]['memory'] if gpu_info else 0
                if is_large_model and single_gpu_memory < 24:
                    # Leave 1GB for system if memory is tight
                    result['memory_limit'] = max(1, single_gpu_memory - 1)
                    
        elif platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Apple Silicon
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    result['device'] = 'mps'
            except:
                pass
                
    except ImportError:
        # PyTorch not available, use CPU
        pass
    except Exception as e:
        warning_msg(f"Error determining optimal GPU configuration: {str(e)}")
    
    return result

def run_gpu_benchmark(gpu_id: int = 0) -> Dict[str, Any]:
    """
    Run a quick GPU benchmark to test capabilities.
    
    Args:
        gpu_id: ID of the GPU to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    result = {
        'success': False,
        'device': 'cpu',
        'gpu_id': gpu_id,
        'matrix_multiply_time_ms': None,
        'estimated_tflops': None,
        'memory_bandwidth_gb_s': None
    }
    
    try:
        if not torch.cuda.is_available():
            result['error'] = "CUDA not available"
            return result
            
        gpu_count = torch.cuda.device_count()
        if gpu_id >= gpu_count:
            result['error'] = f"GPU {gpu_id} not available (only {gpu_count} GPUs)"
            return result
            
        # Set device
        torch.cuda.set_device(gpu_id)
        result['device'] = f"cuda:{gpu_id}"
        gpu_name = torch.cuda.get_device_name(gpu_id)
        result['gpu_name'] = gpu_name
        
        # Test matrix multiplication - common operation in ML
        sizes = [1000, 2000, 4000]
        timings = {}
        
        for size in sizes:
            # Create random tensors
            a = torch.randn(size, size, device=result['device'])
            b = torch.randn(size, size, device=result['device'])
            
            # Warm-up
            torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result_tensor = torch.matmul(a, b)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            
            timings[size] = elapsed_time
            
        # Use largest size for TFLOPS calculation
        largest_size = sizes[-1]
        elapsed_time = timings[largest_size]
        result['matrix_multiply_time_ms'] = elapsed_time
        
        # Calculate theoretical FLOPS for matrix multiplication
        # For an NxN matrix multiplication: 2*N^3 operations
        flops = 2 * (largest_size ** 3)
        teraflops = (flops / elapsed_time) / 1e9  # Convert to TFLOPS
        result['estimated_tflops'] = teraflops
        
        # Memory bandwidth test
        vector_size = 100_000_000  # 100M elements
        a = torch.randn(vector_size, device=result['device'])
        b = torch.randn(vector_size, device=result['device'])
        
        # Warm-up
        _ = a + b
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = a + b
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        # Calculate memory bandwidth (read 2 vectors, write 1)
        bytes_processed = 3 * vector_size * 4  # 4 bytes per float32
        bandwidth_gb_s = (bytes_processed / (elapsed_time / 1000)) / 1e9
        result['memory_bandwidth_gb_s'] = bandwidth_gb_s
        
        result['success'] = True
        
    except ImportError:
        result['error'] = "PyTorch not installed"
    except Exception as e:
        result['error'] = str(e)
    
    return result

def increase_memory_lock_limit() -> bool:
    """
    Attempt to increase the memory lock limit for the current process.
    
    This helps prevent "failed to mlock buffer" errors with FAISS-GPU.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if platform.system() != 'Linux':
            # Memory locking is primarily an issue on Linux
            return False
            
        import resource
        
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        
        # Try to increase to maximum allowed
        if soft != hard or soft == -1:
            try:
                # Set soft limit to hard limit
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
                info_msg(f"Increased memory lock limit from {soft} to {hard}")
                return True
            except Exception as e:
                warning_msg(f"Failed to increase memory lock limit: {str(e)}")
                
                # Suggest command to fix this permanently
                info_msg("To permanently increase memory lock limits, run as root:")
                info_msg("echo '* soft memlock unlimited' >> /etc/security/limits.conf")
                info_msg("echo '* hard memlock unlimited' >> /etc/security/limits.conf")
                return False
    except ImportError:
        # Resource module not available
        pass
    except Exception as e:
        warning_msg(f"Error setting memory lock limit: {str(e)}")
        
    return False

def parse_gpu_ids(gpu_ids_str: Optional[str] = None) -> List[int]:
    """
    Parse GPU IDs from a comma-separated string.
    
    Args:
        gpu_ids_str: Comma-separated string of GPU IDs (e.g., "0,1,2")
        
    Returns:
        List of GPU IDs as integers
    """
    if not gpu_ids_str:
        # If not provided, check environment
        gpu_ids_str = os.environ.get("GPU_IDS", "")
        
    if not gpu_ids_str:
        # Return empty list if no IDs specified
        return []
        
    try:
        return [int(gpu_id.strip()) for gpu_id in gpu_ids_str.split(',') if gpu_id.strip()]
    except ValueError:
        warning_msg(f"Invalid GPU IDs format: {gpu_ids_str}")
        return []

def set_process_gpu_affinity(process_id: Optional[int] = None, gpu_id: int = 0) -> bool:
    """
    Set GPU affinity for a specific process (Linux only).
    
    Args:
        process_id: Process ID (uses current process if None)
        gpu_id: GPU ID to bind to
        
    Returns:
        True if successful, False otherwise
    """
    if process_id is None:
        import os
        process_id = os.getpid()
        
    # This only works on Linux with nvidia-smi
    if platform.system() != 'Linux':
        warning_msg("GPU process affinity only supported on Linux")
        return False
        
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(["which", "nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            warning_msg("nvidia-smi not found, cannot set GPU affinity")
            return False
            
        # Set GPU affinity using nvidia-smi
        cmd = ["nvidia-smi", "-i", str(gpu_id), "-c", str(process_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            info_msg(f"Set process {process_id} to use GPU {gpu_id}")
            return True
        else:
            warning_msg(f"Failed to set GPU affinity: {result.stderr}")
            return False
            
    except Exception as e:
        warning_msg(f"Error setting GPU affinity: {str(e)}")
        return False

def is_distributed_available() -> bool:
    """
    Check if distributed processing is available and configured.
    
    Returns:
        True if distributed processing is available
    """
    # First check environment setting
    distributed_enabled = os.environ.get("DISTRIBUTED", "").lower() in ["true", "1", "yes"]
    
    if not distributed_enabled:
        return False
        
    # Check GPU availability
    try:
        if not torch.cuda.is_available():
            return False
            
        # Need at least two GPUs for meaningful distribution
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            return False
            
        # Check if any specific GPUs are configured
        gpu_ids = parse_gpu_ids()
        if len(gpu_ids) < 2:
            return False
            
        return True
        
    except ImportError:
        return False
    except Exception:
        return False

def get_gpu_memory_limit() -> Optional[float]:
    """
    Get configured GPU memory limit in GB.
    
    Returns:
        Memory limit in GB or None if not set
    """
    try:
        limit_str = os.environ.get("GPU_MEMORY_LIMIT", "")
        if not limit_str:
            return None
            
        limit = float(limit_str)
        return limit if limit > 0 else None
        
    except ValueError:
        warning_msg(f"Invalid GPU memory limit: {limit_str}")
        return None
    except Exception:
        return None

def ensure_gpu_acceleration() -> None:
    """
    Ensure GPU acceleration is activated by allocating tensors on all GPUs.
    This helps with multi-GPU operations and ensures GPU cores are warmed up for operations
    like architecture analysis, which would otherwise be CPU-bound.
    """
    try:
        # Set PyTorch memory allocation configuration to help with fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Only proceed if we have GPU capabilities
        if not HAS_TORCH or not torch.cuda.is_available():
            info_msg("No CUDA capabilities detected - running on CPU only")
            return
            
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            info_msg("No GPUs detected - running on CPU only")
            return
            
        # Log that we're activating GPUs for processing
        info_msg(f"Activating {gpu_count} GPUs for text embedding and processing")
        
        # FIRST PHASE: Collect all GPU statistics
        gpu_stats = []
        total_system_memory = 0
        
        for i in range(gpu_count):
            try:
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                reserved_memory = torch.cuda.memory_reserved(i)
                allocated_memory = torch.cuda.memory_allocated(i)
                
                free_memory_bytes = total_memory - reserved_memory
                
                memory_gb = total_memory / (1024**3)
                reserved_gb = reserved_memory / (1024**3)
                allocated_gb = allocated_memory / (1024**3)
                free_memory_gb = free_memory_bytes / (1024**3)
                
                # Store stats for this GPU
                gpu_stats.append({
                    'index': i,
                    'device_name': device_name,
                    'total_memory': total_memory,
                    'total_memory_gb': memory_gb,
                    'reserved_memory': reserved_memory,
                    'reserved_memory_gb': reserved_gb,
                    'allocated_memory': allocated_memory,
                    'allocated_memory_gb': allocated_gb,
                    'free_memory': free_memory_bytes,
                    'free_memory_gb': free_memory_gb,
                    'target_use_percent': 0.75,  # Default target
                })
                
                total_system_memory += total_memory
                
                info_msg(f"Found GPU {i}: {device_name} with {memory_gb:.1f} GB total memory, {free_memory_gb:.1f} GB free ({reserved_gb:.1f} GB reserved, {allocated_gb:.1f} GB allocated)")
            except Exception as e:
                warning_msg(f"Error getting info for GPU {i}: {str(e)}")
                # Add fallback stats
                gpu_stats.append({
                    'index': i,
                    'device_name': 'Unknown',
                    'total_memory': 0,
                    'total_memory_gb': 0,
                    'free_memory': 0,
                    'free_memory_gb': 0,
                    'has_error': True,
                })
        
        # SECOND PHASE: Determine optimal allocation strategy based on all GPU stats
        
        # Sort GPUs by free memory (descending) for optimal allocation
        gpu_stats.sort(key=lambda x: x.get('free_memory', 0), reverse=True)
        
        # Special case: Adjust allocation for model-hosting GPU (usually GPU 0)
        model_gpu_index = 0  # Default to first GPU for model
        
        # Adjust target usage percentages
        for i, gpu in enumerate(gpu_stats):
            if gpu.get('has_error', False):
                continue
                
            if gpu['index'] == model_gpu_index:
                # Model GPU gets lower target to leave room for model
                gpu['target_use_percent'] = 0.6  # 60% for GPU hosting model
            elif gpu['free_memory_gb'] > 35:
                # High-memory GPUs can use more
                gpu['target_use_percent'] = 0.8  # Reduced from 0.85 to prevent OOM
            elif gpu['free_memory_gb'] > 20:
                # Medium-memory GPUs
                gpu['target_use_percent'] = 0.75  # Reduced from 0.8 to prevent OOM
            else:
                # Low-memory GPUs
                gpu['target_use_percent'] = 0.7  # Reduced from 0.75 to prevent OOM
            
            # Calculate additional tensors based on memory
            if gpu['index'] == model_gpu_index:
                gpu['num_additional_tensors'] = 1  # Minimum for model GPU
            else:
                # Scale additional tensors based on available memory - using fewer tensors
                mem_gb = gpu['free_memory_gb']
                if mem_gb > 40:
                    gpu['num_additional_tensors'] = 3  # Reduced from 5
                elif mem_gb > 30:
                    gpu['num_additional_tensors'] = 2  # Reduced from 4
                elif mem_gb > 20:
                    gpu['num_additional_tensors'] = 2  # Reduced from 3
                elif mem_gb > 10:
                    gpu['num_additional_tensors'] = 1  # Reduced from 2
                else:
                    gpu['num_additional_tensors'] = 1
            
            # Calculate tensor size to achieve target memory usage
            target_memory_bytes = gpu['free_memory'] * gpu['target_use_percent']
            gpu['target_memory_bytes'] = target_memory_bytes
            
            # Calculate primary tensor size (will be adjusted for additional tensors)
            # Assuming 4 bytes per float32 element and additional tensors
            # Primary tensor gets 60% of target memory, the rest is for additional tensors
            primary_tensor_bytes = target_memory_bytes * 0.6
            elements = primary_tensor_bytes / 4  # 4 bytes per float32
            tensor_side = int(math.sqrt(elements))
            # Round to nearest 1000 for cleaner reporting
            tensor_side = (tensor_side // 1000) * 1000
            # Ensure minimum and maximum sizes - REDUCED MAXIMUM FROM 45000 to 35000
            tensor_side = max(5000, min(tensor_side, 35000))
            gpu['primary_tensor_size'] = tensor_side
            
            # Calculate additional tensor sizes based on remaining target memory
            remaining_bytes = target_memory_bytes * 0.4  # 40% for additional tensors
            if gpu['num_additional_tensors'] > 0:
                bytes_per_tensor = remaining_bytes / gpu['num_additional_tensors']
                elements_per_tensor = bytes_per_tensor / 4
                additional_side = int(math.sqrt(elements_per_tensor))
                # Round to nearest 1000
                additional_side = (additional_side // 1000) * 1000
                # Ensure minimum size and maximum size - added maximum of 25000
                additional_side = max(3000, min(additional_side, 25000))
                gpu['additional_tensor_size'] = additional_side
        
        # THIRD PHASE: Allocate memory on each GPU according to the optimized plan
        for gpu in gpu_stats:
            if gpu.get('has_error', False):
                warning_msg(f"Skipping GPU {gpu['index']} due to previous errors")
                continue
            
            i = gpu['index']
            device = torch.device(f"cuda:{i}")
            
            # Clean up any existing allocations first
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Get fresh memory stats before allocation
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
            free_memory_gb = free_memory / (1024**3)
            
            # Recalculate tensor size based on current free memory to avoid OOM
            target_memory_bytes = free_memory * gpu['target_use_percent']
            primary_tensor_bytes = target_memory_bytes * 0.6
            elements = primary_tensor_bytes / 4
            tensor_side = int(math.sqrt(elements))
            tensor_side = (tensor_side // 1000) * 1000
            tensor_side = max(5000, min(tensor_side, 35000))
            
            # Use the calculated tensor sizes for optimal allocation
            info_msg(f"GPU {i} has {gpu['total_memory_gb']:.1f} GB total memory, {free_memory_gb:.1f} GB free, allocating tensor of size {tensor_side}x{tensor_side} (~{tensor_side*tensor_side*4/1024/1024:.1f} MB)")
            
            dummy_tensor = None
            additional_tensors = []
            
            try:
                # Create primary tensor
                dummy_tensor = torch.rand(tensor_side, tensor_side, device=device)
                
                # Create additional tensors if needed
                num_additional = gpu['num_additional_tensors']
                additional_size = gpu.get('additional_tensor_size', tensor_side // 2)
                
                if i != model_gpu_index:  # Skip extra tensors for model GPU
                    for j in range(num_additional):
                        info_msg(f"Creating additional tensor {j+1}/{num_additional} on GPU {i} (size: {additional_size}x{additional_size})")
                        additional_tensors.append(torch.rand(additional_size, additional_size, device=device))
                        # Force memory allocation through operations
                        _ = torch.matmul(dummy_tensor[:additional_size, :additional_size], additional_tensors[j])
                        
                        # Check memory after each additional tensor
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        if mem_reserved / gpu['total_memory_gb'] > 0.8:
                            info_msg(f"Memory utilization over 80%, stopping additional tensor creation on GPU {i}")
                            # Delete the most recent tensor
                            del additional_tensors[-1]
                            additional_tensors = additional_tensors[:-1]
                            break
                else:
                    # For model GPU, just create one smaller tensor
                    secondary_size = min(additional_size, tensor_side // 2)
                    dummy_tensor2 = torch.rand(secondary_size, secondary_size, device=device)
                    _ = torch.matmul(dummy_tensor[:secondary_size, :secondary_size], dummy_tensor2)
                    # Delete immediately after use
                    del dummy_tensor2
            
            except RuntimeError as e:
                # If we run out of memory, try with a smaller tensor
                error_msg(f"Failed to allocate {tensor_side}x{tensor_side} tensor on GPU {i}: {e}")
                tensor_side = tensor_side // 2
                info_msg(f"Retrying with tensor size {tensor_side}x{tensor_side}")
                try:
                    # Clean up previous attempt first
                    if dummy_tensor is not None:
                        del dummy_tensor
                    for tensor in additional_tensors:
                        del tensor
                    additional_tensors = []
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Try with smaller tensor
                    dummy_tensor = torch.rand(tensor_side, tensor_side, device=device)
                    dummy_tensor2 = torch.rand(tensor_side // 2, tensor_size // 2, device=device)
                    _ = torch.matmul(dummy_tensor[:tensor_side//2, :tensor_side//2], dummy_tensor2)
                    del dummy_tensor2
                except Exception as retry_e:
                    error_msg(f"Even retry with smaller tensor failed on GPU {i}: {retry_e}")
                    # Try one last time with a very small tensor
                    try:
                        # Clean up previous attempt first
                        if dummy_tensor is not None:
                            del dummy_tensor
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        info_msg(f"Final attempt with minimal tensor size on GPU {i}")
                        dummy_tensor = torch.rand(1000, 1000, device=device)
                        _ = torch.nn.functional.relu(dummy_tensor)
                    except Exception as final_e:
                        error_msg(f"Could not allocate even minimal tensor on GPU {i}: {final_e}")
                        # Skip to next GPU
                        continue
            
            # --- FINISH PROCESSING ---
            # Force sync to ensure operations are complete
            torch.cuda.synchronize(device)
            
            # --- FREE MEMORY ---
            # Log allocation stats before freeing
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            percent_used = (mem_reserved / (gpu['total_memory'] / 1024**2)) * 100
            info_msg(f"GPU {i} activated with {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved ({percent_used:.1f}% of total)")
            
            # --- DELETE TENSORS ---
            # Delete all tensors explicitly
            for j, tensor in enumerate(additional_tensors):
                del tensor
            additional_tensors = []
            
            if dummy_tensor is not None:
                del dummy_tensor
                dummy_tensor = None
            
            # Force memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # --- GET NEW STATS ---
            # Get updated memory stats after freeing
            new_mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            new_mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            new_percent_used = (new_mem_reserved / (gpu['total_memory'] / 1024**2)) * 100
            info_msg(f"GPU {i} after cleanup: {new_mem_allocated:.2f} MB allocated, {new_mem_reserved:.2f} MB reserved ({new_percent_used:.1f}% of total)")
        
        # --- APPLY NEW ALLOCATION ---
        # Final operations on model GPU to ensure layers are loaded
        try:
            # Clean up all GPUs first
            for i in range(gpu_count):
                torch.cuda.empty_cache()
            gc.collect()
            
            # Allocate small tensor on model GPU
            device = torch.device(f"cuda:{model_gpu_index}")
            info_msg(f"Final activation on GPU {model_gpu_index} with small tensor")
            large_tensor = torch.rand(3000, 3000, device=device)
            _ = torch.nn.functional.relu(large_tensor)
            
            # Force sync to ensure GPU operations complete
            torch.cuda.synchronize()
            info_msg("Successfully completed GPU activation")
            
            # Delete the tensor after use
            del large_tensor
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as final_e:
            error_msg(f"Error in final GPU operations: {str(final_e)}")
        
        # Final memory status report
        info_msg("Final GPU memory status after activation:")
        for i in range(torch.cuda.device_count()):
            try:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                percent_used = (mem_reserved / total_memory) * 100
                info_msg(f"  GPU {i}: {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved ({percent_used:.1f}% of total)")
            except Exception as e:
                pass
        
    except Exception as e:
        error_msg(f"Failed to ensure GPU acceleration: {str(e)}")
        # Print traceback for better debugging
        import traceback
        error_msg(f"Traceback: {traceback.format_exc()}")

def release_gpu_memory():
    """
    Aggressively release GPU memory, including both allocated memory and reserved cache.
    """
    try:
        # Only proceed if torch.cuda is available
        if not HAS_TORCH or not torch.cuda.is_available():
            return
        
        # Force Python garbage collection first
        import gc
        gc.collect()
        
        # Log current GPU status before cleaning
        info_msg("GPU memory status before cleanup:")
        for i in range(torch.cuda.device_count()):
            try:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                info_msg(f"  GPU {i}: {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
            except:
                pass
        
        # Empty CUDA cache to free memory
        torch.cuda.empty_cache()
        
        # Try a more aggressive approach
        for i in range(torch.cuda.device_count()):
            try:
                # Reset the current device to clear its cache
                device = torch.device(f"cuda:{i}")
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)
            except Exception as e:
                warning_msg(f"Error resetting GPU {i} memory: {str(e)}")
        
        # Try using nvidia-smi to release memory (system-level)
        try:
            import subprocess
            # This doesn't actually release memory but can help diagnose
            subprocess.run("nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
            
        # Log GPU status after cleanup
        info_msg("GPU memory status after cleanup:")
        for i in range(torch.cuda.device_count()):
            try:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                info_msg(f"  GPU {i}: {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
            except:
                pass
                
    except Exception as e:
        warning_msg(f"Error during GPU memory release: {str(e)}")

def log_gpu_status(phase: str) -> None:
    """
    Log current GPU memory allocation status.
    
    Args:
        phase: Description of the current processing phase
    """
    try:
        if not HAS_TORCH or not torch.cuda.is_available():
            return
            
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return
            
        logger.info(f"GPU status at phase: {phase}")
        
        for i in range(gpu_count):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            util_percent = 0
            
            try:
                # Try to get actual GPU utilization if available
                if hasattr(torch.cuda, 'utilization'):
                    util_percent = torch.cuda.utilization(i)
            except:
                pass
                
            logger.info(f"  GPU {i}: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {util_percent}% utilization")
            
    except Exception as e:
        warning_msg(f"Failed to log GPU status: {str(e)}")

def start_gpu_monitoring() -> None:
    """
    Start monitoring GPU utilization.
    This helps track GPU usage during various processing phases.
    """
    try:
        # Only monitor if we have GPUs
        if not HAS_TORCH or not torch.cuda.is_available():
            return
            
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return
            
        # Log initial GPU state
        for i in range(gpu_count):
            if torch.cuda.memory_allocated(i) > 0:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                info_msg(f"GPU {i} memory state: {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
                
        # Force garbage collection to get accurate reading
        import gc
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
                
    except Exception as e:
        warning_msg(f"Failed to start GPU monitoring: {str(e)}") 