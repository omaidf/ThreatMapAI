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
            elif force_gpu:
                # If forcing GPU but don't know memory, use conservative defaults
                result['n_gpu_layers'] = is_large_model and 16 or 32
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

def install_faiss_gpu() -> bool:
    """
    Install the FAISS-GPU package that matches the system's CUDA version.
    
    Returns:
        True if installation was successful, False otherwise
    """
    try:
        # First unload faiss if it's already loaded
        if 'faiss' in sys.modules:
            del sys.modules['faiss']
        
        # Now try to import and check for GPU support
        import faiss
        
        if hasattr(faiss, 'StandardGpuResources'):
            # GPU support already available
            info_msg("FAISS with GPU support is already installed")
            return True
        
        # Check CUDA version to determine which package to install
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            info_msg(f"Detected CUDA version: {cuda_version}")
        else:
            warning_msg("No CUDA available, cannot install FAISS-GPU")
            return False
        
        # Select appropriate package based on CUDA version
        if cuda_version:
            major_version = cuda_version.split('.')[0]
            if major_version == '11':
                package = "faiss-gpu"
            elif major_version == '12':
                package = "faiss-gpu>=1.7.3"  # For CUDA 12 compatibility
            else:
                package = "faiss-gpu"  # Default
        else:
            package = "faiss-gpu"  # Default if can't determine version
        
        # Install the package
        info_msg(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package])
        success_msg(f"Successfully installed {package}")
        
        # Force unload faiss completely to ensure clean import
        for module_name in list(sys.modules.keys()):
            if module_name == 'faiss' or module_name.startswith('faiss.'):
                del sys.modules[module_name]
        
        # Verify installation by importing fresh
        try:
            import faiss
            
            # Test creating a GPU resource to verify it works
            if hasattr(faiss, 'StandardGpuResources'):
                try:
                    # Try to actually create a GPU resource to fully verify
                    res = faiss.StandardGpuResources()
                    success_msg("FAISS GPU support successfully verified")
                    # Keep the resource around to ensure module stays loaded correctly
                    return True
                except Exception as e:
                    warning_msg(f"FAISS GPU support detected but initialization failed: {str(e)}")
                    return False
            else:
                warning_msg("FAISS installed but GPU support not available")
                # Sometimes the module needs to be fully reloaded from scratch
                # Try reinstalling one more time to ensure we get the GPU version
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", package])
                
                # Force reload again
                for module_name in list(sys.modules.keys()):
                    if module_name == 'faiss' or module_name.startswith('faiss.'):
                        del sys.modules[module_name]
                
                # Try one more time
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    success_msg("FAISS GPU support successfully installed after reinstall")
                    return True
                else:
                    warning_msg("FAISS GPU support still not available after reinstall")
                    return False
        except ImportError:
            warning_msg("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        warning_msg(f"Failed to install FAISS-GPU: {str(e)}")
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
                
                # Install FAISS-GPU for better multi-GPU performance
                # Try multiple times with increasing force levels if needed
                faiss_installed = install_faiss_gpu()
                if not faiss_installed and 'faiss' in sys.modules:
                    # If initial installation failed but module exists, try with --force-reinstall
                    info_msg("Retrying FAISS-GPU installation with force...")
                    faiss_installed = install_faiss_gpu()
            else:
                # Clear distributed setting if not needed
                if "DISTRIBUTED" in os.environ:
                    del os.environ["DISTRIBUTED"]
                update_env_file("DISTRIBUTED", "")
                
                # Still install FAISS-GPU for single GPU performance
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