"""
Command-line interface for AI Threat Model Map Generator.

This module provides a CLI for analyzing code repositories and generating
threat models and security reports.
"""

import os
import sys
import logging
import subprocess
import platform
import json
import pathlib
from pathlib import Path
from typing import Optional, Tuple, List
import click
import threading
import time
import re
import webbrowser
from tqdm import tqdm
import importlib
import importlib.util
import signal
import psutil
import gc
import faiss
import torch
from dotenv import load_dotenv

# Import utility modules
from utils import (
    success_msg, error_msg, warning_msg, info_msg, 
    get_env_variable, update_env_file,
    detect_architecture, get_default_model_path, validate_model_path,
    set_token_interactive, test_model_loading, check_model_file
)

# Import GPU utilities
from utils.gpu_utils import (
    detect_gpu, detect_gpu_capabilities, get_gpu_info,
    configure_gpu_environment, install_faiss_gpu,
    get_optimal_gpu_configuration, parse_gpu_ids,
    is_distributed_available, get_gpu_memory_limit,
    run_gpu_benchmark
)

# Import specialized modules
from utils.model_config import download_model, DEFAULT_MODEL_NAME, MODEL_REPOS, get_model_info, get_available_models, set_default_model
from utils.file_utils import (
    check_output_directory, clean_previous_run, check_required_files,
    check_dependencies
)
from utils.env_utils import get_env_variable, update_env_file
from utils.diagram_utils import (
    find_diagrams, start_server_and_open_diagrams, view_diagrams
)

# Import core processing modules
from repository_analyzer.embedding_store import EmbeddingStore
from repository_analyzer.analyzer import RepositoryAnalyzer
from llm_processor.processor import LLMProcessor
from visualizer.visualizer import ThreatModelVisualizer

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    load_dotenv()

# Configure logging - reduce verbosity
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Only show INFO level for our modules
logging.getLogger('repository_analyzer').setLevel(logging.INFO)
logging.getLogger('llm_processor').setLevel(logging.INFO)
logging.getLogger('visualizer').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

def validate_output_dir(ctx, param, value: str) -> str:
    """Validate and create output directory if it doesn't exist."""
    try:
        # Use pathlib.Path to ensure it's always available
        path = pathlib.Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return value
    except Exception as e:
        # Try to use a default directory if the specified one can't be created
        logger.warning(f"Couldn't create output directory {value}, using default: output")
        pathlib.Path("output").mkdir(exist_ok=True)
        return "output"

def setup_tree_sitter() -> None:
    """Set up tree-sitter grammars."""
    try:
        # Import here to avoid circular imports
        analyzer = RepositoryAnalyzer()
        
        # During initialization, we don't load any specific languages
        # They will be loaded based on repository content during analysis
        analyzer._setup_tree_sitter(specific_languages=[])
        logger.info("Tree-sitter initialized successfully. Specific language grammars will be loaded during analysis.")
    except ImportError as e:
        # Handle specific import errors
        if 'git' in str(e):
            logger.error("GitPython is not installed. Please install it with: pip install gitpython")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
                logger.info("GitPython installed, retrying setup...")
                # Try again after installing GitPython
                analyzer = RepositoryAnalyzer()
                analyzer._setup_tree_sitter(specific_languages=[])
            except Exception:
                raise CLIError("Failed to install GitPython or setup tree-sitter")
        else:
            logger.error(f"Import error during tree-sitter setup: {str(e)}")
            raise CLIError(f"Failed to setup tree-sitter: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to setup tree-sitter: {str(e)}")
        raise CLIError(f"Failed to setup tree-sitter: {str(e)}")

def create_embedding_store(try_load: bool = True, device: Optional[str] = None, gpu_id: Optional[int] = None) -> Optional[EmbeddingStore]:
    """Create and initialize the embedding store.
    
    Args:
        try_load: Whether to try loading existing embeddings
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
        gpu_id: Specific GPU ID to use when multiple GPUs are available (ignored if device is 'cpu')
        
    Returns:
        Initialized EmbeddingStore or None if initialization failed
    """
    try:
        # Handle GPU detection gracefully
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                info_msg(f"Detected {gpu_count} GPU(s) for embedding generation")
        except ImportError:
            # If torch isn't available, we'll default to CPU anyway
            pass
            
        # Only use CPU if explicitly requested or if CUDA isn't available
        use_cpu = (device == 'cpu' or not cuda_available)
        
        # Check if faiss-gpu is installed when a GPU is available
        if not use_cpu:
            try:
                # Try to import faiss and check if GPU support is available
                gpu_supported = hasattr(faiss, 'StandardGpuResources')
                
                if not gpu_supported:
                    info_msg("FAISS GPU support not detected. Installing faiss-gpu...")
                    try:
                        # Try to install the right version of faiss-gpu based on CUDA version
                        try:
                            # Check CUDA version to install the right package
                            cuda_version = torch.version.cuda
                            
                            # Get major version for package selection
                            cuda_major = cuda_version.split('.')[0] if cuda_version else None
                            
                            if cuda_major == '11':
                                package = "faiss-gpu"
                            elif cuda_major == '12':
                                package = "faiss-gpu>=1.7.3"  # For CUDA 12 compatibility
                            else:
                                package = "faiss-gpu"  # Default
                                
                            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                            success_msg(f"Successfully installed {package} for CUDA {cuda_version}")
                            
                            # Reload faiss after installation
                            importlib.reload(sys.modules['faiss'])
                            gpu_supported = hasattr(faiss, 'StandardGpuResources')
                        except Exception as cuda_error:
                            # If we can't determine the CUDA version, just try the general package
                            warning_msg(f"Could not detect CUDA version: {str(cuda_error)}")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
                            success_msg("Installed faiss-gpu (default version)")
                            
                            # Reload faiss after installation
                            importlib.reload(sys.modules['faiss'])
                            gpu_supported = hasattr(faiss, 'StandardGpuResources')
                    except Exception as install_error:
                        warning_msg(f"Failed to install faiss-gpu: {str(install_error)}")
                        warning_msg("FAISS will use CPU mode for vector similarity search")
                        warning_msg("This is not ideal for performance but embedding generation will still use GPU")
            except ImportError:
                # If faiss isn't imported at all, we'll install it below
                pass

        # Configure the embedding store
        config = {}
        if not use_cpu:
            config['device'] = 'cuda'
            if gpu_id is not None:
                config['gpu_id'] = gpu_id
                
            # Log that we're using GPU for embeddings
            success_msg(f"Using GPU{f' {gpu_id}' if gpu_id is not None else ''} for embedding generation")
        else:
            config['device'] = 'cpu'
            info_msg("Using CPU for embedding generation")

        # Initialize the store with our configuration
        embedding_store = EmbeddingStore(**config)
        
        if try_load:
            # Check if embeddings already exist before trying to load
            output_dir = os.getenv("OUTPUT_DIR", "output")
            index_path = Path(output_dir) / "embeddings.index"
            mapping_path = Path(output_dir) / "embeddings_mapping.json"
            
            if index_path.exists() and mapping_path.exists():
                info_msg("Loading existing embeddings...")
                loaded = embedding_store.load()
                if loaded:
                    success_msg(f"Successfully loaded {len(embedding_store.file_mapping)} existing embeddings")
                else:
                    warning_msg("Failed to load existing embeddings, will create new ones")
            else:
                info_msg("No existing embeddings found, will create new ones")
        else:
            info_msg("Initializing new embedding store (not loading existing data)")
            
        return embedding_store
    except ImportError as e:
        if "faiss" in str(e):
            error_msg("FAISS is not installed. Installing CPU version...")
            try:
                # First check if we should install GPU version instead
                try:
                    if torch.cuda.is_available() and not use_cpu:
                        info_msg("CUDA detected, installing GPU version of FAISS...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
                        info_msg("FAISS-GPU installed successfully. Retrying...")
                    else:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
                        info_msg("FAISS-CPU installed successfully. Retrying...")
                except ImportError:
                    # Torch not available, install CPU version
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
                    info_msg("FAISS-CPU installed successfully. Retrying...")
                return create_embedding_store(try_load, device, gpu_id)
            except Exception as install_error:
                error_msg(f"Failed to install FAISS: {str(install_error)}")
                error_msg("Please run: pip install faiss-cpu")
                return None
        else:
            error_msg(f"Failed to initialize embedding store: {str(e)}")
            return None
    except Exception as e:
        error_msg(f"Failed to initialize embedding store: {str(e)}")
        return None

@click.group()
def cli():
    """AI Threat Model Map Generator CLI."""
    pass

def install_faiss_gpu() -> bool:
    """Install the FAISS-GPU package that matches the system's CUDA version.
    
    Returns:
        True if installation was successful, False otherwise
    """
    try:
        # First check if faiss is already installed with GPU support
        try:
            if hasattr(faiss, 'StandardGpuResources'):
                # GPU support already available
                info_msg("FAISS with GPU support is already installed")
                return True
        except ImportError:
            # FAISS not installed at all
            pass
        
        # Check CUDA version to determine which package to install
        cuda_version = None
        try:
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                info_msg(f"Detected CUDA version: {cuda_version}")
            else:
                warning_msg("No CUDA available, cannot install FAISS-GPU")
                return False
        except ImportError:
            warning_msg("PyTorch not installed, cannot detect CUDA version")
            warning_msg("Installing default FAISS-GPU package")
        
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        success_msg(f"Successfully installed {package}")
        
        # Verify installation
        try:
            # Force reload of faiss module if it was already imported
            if 'faiss' in sys.modules:
                importlib.reload(sys.modules['faiss'])
            else:
                import faiss
                
            if hasattr(faiss, 'StandardGpuResources'):
                success_msg("FAISS GPU support successfully installed")
                return True
            else:
                warning_msg("FAISS installed but GPU support not available")
                return False
        except ImportError:
            warning_msg("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        warning_msg(f"Failed to install FAISS-GPU: {str(e)}")
        return False

@cli.command()
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
              help='Device to use for embeddings (auto, cpu, cuda)')
@click.option('--gpu-ids', help='Specify GPU IDs to use (comma-separated, e.g., "0,1,2")')
def init(device: str, gpu_ids: str):
    """Initialize the framework.
    
    Detects hardware capabilities and sets up components for optimal performance.
    For multi-GPU setups, you can specify which GPUs to use with --gpu-ids.
    """
    try:
        # Process GPU IDs if provided
        selected_gpus = None
        if gpu_ids:
            try:
                selected_gpus = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
                info_msg(f"Selected GPUs: {selected_gpus}")
            except Exception as e:
                warning_msg(f"Invalid GPU IDs format: {str(e)}")
                warning_msg("Using default GPU selection")
        
        # Check hardware capabilities first
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                
                # Check if the selected GPUs are valid
                if selected_gpus:
                    for gpu_id in selected_gpus:
                        if gpu_id >= gpu_count:
                            warning_msg(f"GPU {gpu_id} does not exist (only {gpu_count} GPUs available)")
                            warning_msg("Ignoring invalid GPU IDs")
                            # Reset to use all available GPUs
                            selected_gpus = list(range(gpu_count))
                            break
                
                # Use all available GPUs if none specified
                if not selected_gpus:
                    selected_gpus = list(range(gpu_count))
                
                # Show information about detected GPUs
                info_msg(f"Detected {gpu_count} GPU(s):")
                total_memory = 0
                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_name = gpu_props.name
                    gpu_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
                    total_memory += gpu_memory
                    
                    # Mark if this GPU is selected
                    is_selected = i in selected_gpus
                    marker = "→ " if is_selected else "  "
                    info_msg(f"{marker}GPU {i}: {gpu_name} with {gpu_memory:.2f} GB VRAM")
                
                # Set environment variables for selected GPUs
                os.environ["GPU_IDS"] = ",".join(str(gpu_id) for gpu_id in selected_gpus)
                from utils.env_utils import update_env_file
                update_env_file("GPU_IDS", ",".join(str(gpu_id) for gpu_id in selected_gpus))
                
                # Provide recommendations based on detected hardware
                success_msg(f"Hardware acceleration available: {gpu_count} GPU(s) with {total_memory:.2f} GB total VRAM")
                
                # Check for multi-GPU capabilities
                if gpu_count > 1:
                    if total_memory >= 24:  # Generous threshold for good multi-GPU performance
                        success_msg("Multi-GPU configuration detected with good memory capacity")
                        success_msg("Enabling distributed processing for better performance")
                        os.environ["DISTRIBUTED"] = "true"
                        update_env_file("DISTRIBUTED", "true")
                        
                        # When detecting multiple GPUs, automatically install FAISS-GPU
                        if device != 'cpu':  # Skip if user explicitly requested CPU
                            info_msg("Installing GPU-accelerated libraries for multi-GPU support...")
                            install_faiss_gpu()
                    else:
                        info_msg("Multi-GPU configuration detected but with limited memory")
                        info_msg("For optimal performance with large models, consider:")
                        info_msg("python -m cli configure-gpu --gpu-ids 0")
                elif gpu_memory and gpu_memory < 6:
                    warning_msg("GPU memory is limited. You may experience better performance with CPU mode for large models.")
                    info_msg("Consider using --device cpu for analysis of large codebases")
            else:
                info_msg("No GPU detected. Running in CPU-only mode, which is perfectly fine but slower.")
                # Set environment variable to force CPU
                if device == 'auto' or device == 'cpu':
                    os.environ["FORCE_CPU"] = "true"
                    from utils.env_utils import update_env_file
                    update_env_file("FORCE_CPU", "true")
                    update_env_file("FORCE_GPU", "")
        except ImportError:
            info_msg("PyTorch not installed. Cannot detect GPU capabilities.")
        
        # Set the selected device
        if device != 'auto':
            os.environ["FORCE_CPU"] = "true" if device == "cpu" else ""
            os.environ["FORCE_GPU"] = "true" if device == "cuda" else ""
            from utils.env_utils import update_env_file
            update_env_file("FORCE_CPU", "true" if device == "cpu" else "")
            update_env_file("FORCE_GPU", "true" if device == "cuda" else "")
        
        # Setup tree-sitter grammars
        success_msg("Initializing tree-sitter grammars...")
        try:
            setup_tree_sitter()
            success_msg("Tree-sitter grammars set up successfully")
        except Exception as e:
            warning_msg(f"Failed to set up tree-sitter grammars: {str(e)}")
            warning_msg("The framework may not function correctly")
            return
        
        # Create embedding store using specified settings
        success_msg("Initializing embedding store...")
        try:
            # Convert 'auto' to None for auto-detection
            device_param = None if device == 'auto' else device
            
            # Use first selected GPU if specified
            gpu_id_param = None
            if selected_gpus:
                gpu_id_param = selected_gpus[0]
                
            embedding_store = create_embedding_store(try_load=True, device=device_param, gpu_id=gpu_id_param)
            if embedding_store:
                if embedding_store.gpu_available and device_param != 'cpu':
                    success_msg(f"Embedding store initialized using {embedding_store.device}")
                    if hasattr(embedding_store, 'use_gpu_for_faiss') and embedding_store.use_gpu_for_faiss:
                        success_msg(f"FAISS index using GPU acceleration (GPU {embedding_store.gpu_id if embedding_store.gpu_id is not None else 0})")
                else:
                    success_msg("Embedding store initialized using CPU")
                    if device_param == 'cuda':
                        warning_msg("Requested CUDA but GPU not available. Running in CPU mode.")
            else:
                warning_msg("Embedding store initialization incomplete")
        except Exception as e:
            warning_msg(f"Failed to initialize embedding store: {str(e)}")
            warning_msg("Embedding store functionality will be limited")
            
        success_msg("Framework initialized successfully")
        
        # Provide specific recommendations based on detected hardware
        try:
            cpu_count = psutil.cpu_count(logical=False)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if cuda_available:
                if gpu_count > 1:
                    info_msg("\nMulti-GPU setup detected with automatic optimization enabled!")
                    info_msg(f"The system will automatically use all {gpu_count} GPUs with distributed processing")
                    info_msg("Simply run: python -m cli analyze")
                else:
                    info_msg("\nSingle GPU setup detected. For best performance:")
                    info_msg("python -m cli analyze")
            elif memory_gb > 32 and cpu_count >= 8:
                info_msg("\nHigh-performance CPU setup detected. For best performance:")
                info_msg("python -m cli analyze --device cpu")
            elif memory_gb < 16:
                warning_msg("\nLimited memory detected. For better stability:")
                warning_msg("python -m cli analyze --device cpu --memory-limit 4")
        except Exception:
            pass
        
    except Exception as e:
        error_msg(f"Failed to initialize framework: {str(e)}")

@cli.command()
@click.argument('repository_url', required=False)
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Directory to store analysis results')
@click.option('--model-path', default=None, callback=validate_model_path,
              help='Path to the CodeLlama model (auto-detected if not provided)')
@click.option('--local', is_flag=True, help='Analyze a local repository')
@click.option('--clean', is_flag=True, help='Clean previous run data before analysis')
@click.option('--clear-embeddings', is_flag=True, help='Clear embedding store from previous runs')
@click.option('--reuse-embeddings', is_flag=True, help='Reuse existing embeddings without asking')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
              help='Device to use for embeddings (auto, cpu, cuda)')
@click.option('--gpu-id', type=int, help='Specific GPU ID to use for embeddings (for single-GPU use)')
@click.option('--distributed', is_flag=True, 
              help='Force distributed processing for multi-GPU setups (automatically enabled when multiple GPUs are available)')
@click.option('--memory-limit', type=float, help='Limit GPU memory usage (in GB)')
@click.option('--suppress-warnings', is_flag=True, default=True, help='Suppress non-critical warnings')
def analyze(repository_url: str, output_dir: str, model_path: str, local: bool, clean: bool, 
           clear_embeddings: bool, reuse_embeddings: bool, device: str, gpu_id: int, 
           distributed: bool, memory_limit: float, suppress_warnings: bool):
    """Analyze a repository for security threats."""
    analyzer = None
    embedding_store = None
    
    try:
        # Ensure output directory exists
        check_output_directory(output_dir)
            
        # If no repository URL is provided, prompt the user for repository info
        if not repository_url:
            repo_type = click.prompt(
                "Would you like to analyze a remote GitHub repository or a local repository?",
                type=click.Choice(["github", "local"], case_sensitive=False),
                default="github"
            )
            
            if repo_type.lower() == "github":
                repository_url = click.prompt("Enter GitHub repository URL")
                local = False
            else:
                repository_url = click.prompt("Enter path to local repository")
                local = True
        
        # Clean previous run data if requested
        if clean:
            clean_previous_run(output_dir, force_clean=True, clear_embeddings=clear_embeddings)
        else:
            # Still check for previous runs but ask for confirmation
            clean_previous_run(output_dir, clear_embeddings=clear_embeddings)
            
        info_msg(f"Starting analysis of repository: {repository_url}")
        
        # Set environment variables
        os.environ["OUTPUT_DIR"] = output_dir
        
        # Configure GPU environment using the centralized utility
        force_gpu = device == 'cuda'
        force_cpu = device == 'cpu'
        
        # Parse gpu_ids if a single gpu_id is provided
        selected_gpu_ids = None
        if gpu_id is not None:
            selected_gpu_ids = [gpu_id]
        
        # Configure the GPU environment with our new utility
        gpu_config = configure_gpu_environment(
            force_gpu=force_gpu,
            force_cpu=force_cpu,
            gpu_ids=selected_gpu_ids,
            memory_limit=memory_limit,
            distributed=distributed
        )
        
        # Log the configuration that was applied
        if gpu_config['device'] == 'cpu':
            info_msg("Using CPU for analysis (no GPU acceleration)")
        else:
            success_msg(f"GPU acceleration enabled: using {len(gpu_config['selected_gpus'])} GPU(s)")
            if gpu_config['distributed']:
                success_msg(f"Multi-GPU distributed processing enabled for GPUs: {gpu_config['selected_gpus']}")
            if gpu_config['memory_limit']:
                info_msg(f"GPU memory limit set to {gpu_config['memory_limit']} GB per GPU")
        
        # Ensure model exists
        model_path = validate_model_path(model_path)
        
        # Import pathlib.Path for file existence check
        model_path_obj = pathlib.Path(model_path)
        
        if not model_path_obj.exists():
            logger.warning(f"CodeLlama model not found at {model_path}")
            if click.confirm("Model not found. Would you like to download it now?"):
                # Use the correct download_model function
                downloaded_path = download_model(model_path=model_path, force=True)
                if not downloaded_path:
                    error_msg("Failed to download model. Please try again or check your internet connection.")
                    return
            else:
                error_msg("Model must be downloaded before analysis.")
                return
        
        # Initialize components with progress indicators
        with tqdm(total=4, desc="Initializing") as progress:
            # Check for existing embeddings
            embeddings_index_path = pathlib.Path(output_dir) / "embeddings.index"
            embeddings_mapping_path = pathlib.Path(output_dir) / "embeddings_mapping.json"
            embeddings_exist = embeddings_index_path.exists() and embeddings_mapping_path.exists()
            
            if embeddings_exist:
                info_msg("Loading existing embeddings...")
                embedding_store = create_embedding_store(try_load=True)
                if embedding_store:
                    success_msg(f"Successfully loaded {len(embedding_store.file_mapping)} existing embeddings")
                else:
                    warning_msg("Failed to load existing embeddings, will create new ones")
            else:
                info_msg("No existing embeddings found, will create new ones")
            
            progress.update(1)
            
            # Import and initialize analyzer
            info_msg("Initializing repository analyzer...")
            analyzer = RepositoryAnalyzer(embedding_store=embedding_store)
            
            # Import processor for LLM-based analysis
            info_msg("Initializing LLM processor...")
            processor = LLMProcessor(
                model_path_or_embedding_store=model_path,
                distributed=gpu_config['distributed'],
                gpu_id=gpu_id,
                gpu_ids=gpu_config['selected_gpus'],
                memory_limit=gpu_config['memory_limit'],
                device=gpu_config['device']
            )
            progress.update(1)
            
            # Analyze repository
            info_msg(f"Analyzing repository: {repository_url}")
            analysis_results = analyzer.analyze_repository(repository_url)
            progress.update(1)
            
            # Generate threat model
            info_msg("Generating threat model from analysis results...")
            threat_model = processor.generate_threat_model(analysis_results)
            
            # Save threat model
            with open(os.path.join(output_dir, "threat_model.json"), "w") as f:
                json.dump(threat_model, f, indent=4)
                
            # Generate visualizations
            info_msg("Generating visualizations...")
            visualizer = ThreatModelVisualizer()
            diagrams = visualizer.generate_visualizations_from_dir(output_dir)
            progress.update(1)
            
            # Show generated files
            info_msg("Analysis completed. Generated files:")
            for file_name in os.listdir(output_dir):
                if file_name.endswith(('.json', '.mmd', '.html')):
                    info_msg(f"  - {file_name}")
            
            # Start server and open diagrams in browser
            diagram_paths = []
            for diagram_type, path in diagrams.items():
                if path and os.path.exists(path):
                    # Make sure we're using string paths
                    diagram_paths.append(str(path))
                    
            info_msg(f"Analysis completed. Results saved to {output_dir}")
            
            # Start server and open diagrams in browser if diagrams were generated
            if diagram_paths:
                info_msg("Starting diagram viewer server...")
                start_server_and_open_diagrams(diagram_paths, output_dir)
            
            # Keep the server running until user presses Ctrl+C
            try:
                click.echo("\nDiagram viewer server running. Press Ctrl+C to exit.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                info_msg("Shutting down server...")
        
    except Exception as e:
        error_msg(f"Analysis failed: {str(e)}")
        raise click.ClickException(str(e))
    finally:
        # Final cleanup of resources
        if embedding_store:
            try:
                # Check if Python is shutting down
                if sys is None or sys.meta_path is None:
                    info_msg("Skipping final save during Python shutdown")
                else:
                    info_msg("Saving final embedding store state...")
                    embedding_store.save()
                    info_msg("Final embedding store save completed")
            except Exception as save_error:
                warning_msg(f"Failed to save embedding store during cleanup: {str(save_error)}")
            
            # Set to None to release reference
            embedding_store = None
            
        if analyzer:
            analyzer = None
            
        # Force final garbage collection
        gc.collect()

@cli.command()
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Directory containing analysis results')
def visualize(output_dir: str):
    """Generate visualizations from analysis results."""
    visualizer = ThreatModelVisualizer()
    
    try:
        info_msg("Generating visualizations...")
        
        # Check for required files
        required_files = ["analysis_results.json", "threat_model.json"]
        if not check_required_files(output_dir, required_files):
            raise CLIError(f"Required analysis files not found in {output_dir}. Run 'analyze' command first.")
        
        with tqdm(total=3, desc="Creating diagrams") as progress:
            visualizer = ThreatModelVisualizer()
            progress.update(1)
            
            diagrams = visualizer.generate_visualizations_from_dir(output_dir)
            progress.update(1)
            
            info_msg("Diagrams generated:")
            diagram_paths = []
            for diagram_type, path in diagrams.items():
                success_msg(f"{diagram_type}: {path}")
                if path and os.path.exists(path):
                    # Make sure we're using string paths
                    diagram_paths.append(str(path))
            
            progress.update(1)
        
        # Display the diagrams using common method
        if diagram_paths:
            start_server_and_open_diagrams(diagram_paths, output_dir)
                
    except Exception as e:
        error_msg(f"Failed to generate visualizations: {str(e)}")

@cli.command()
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Directory containing diagrams')
@click.option('--port', default=8000, help='Port for local server')
def view(output_dir: str, port: int):
    """View generated diagrams in a browser."""
    try:
        view_diagrams(output_dir, port)
    except Exception as e:
        error_msg(f"Failed to view diagrams: {str(e)}")

@cli.command()
@click.option('--force', is_flag=True, help='Force re-download even if model exists')
def download_model_cmd(force: bool):
    """Download the CodeLlama model for local inference using huggingface_hub."""
    try:
        from utils.model_config import get_model_info
        
        info_msg("Starting model download...")
        model_path = get_default_model_path()
        
        # Get model info to display requirements
        model_name = get_env_variable("LLM_MODEL", "codellama-7b-instruct")
        model_info = get_model_info(model_name)
        
        # Show hardware requirements
        warning_msg(f"Model: {model_info['name']} - Hardware Requirements:")
        
        # RAM requirements
        min_ram = model_info.get('min_ram_gb', 16)
        info_msg(f"- Minimum RAM: {min_ram}GB")
        
        # GPU requirements
        is_large_model = "70b" in model_name.lower()
        if is_large_model:
            warning_msg("- GPU: Recommended 24GB+ VRAM for GPU acceleration")
            warning_msg("  Without sufficient GPU memory, the model will run on CPU (much slower)")
        else:
            info_msg("- GPU: Optional. 8GB+ VRAM recommended for acceleration")
            info_msg("  Model will work on CPU-only systems but will be slower")
        
        # Check if user has enough RAM
        system_ram = psutil.virtual_memory().total / (1024**3)  # in GB
        if system_ram < min_ram:
            warning_msg(f"⚠️ WARNING: Your system has {system_ram:.1f}GB RAM but model requires {min_ram}GB")
            warning_msg("The model may not load or could cause system instability")
            if not click.confirm("Continue with download anyway?", default=False):
                info_msg("Download canceled")
                return
        
        # Check if user has a GPU
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
                if is_large_model and gpu_memory < 24:
                    warning_msg(f"GPU has {gpu_memory:.1f}GB VRAM, which may be insufficient for the 70B model")
                    warning_msg("The model will likely run on CPU or with limited GPU acceleration")
                else:
                    info_msg(f"Detected GPU with {gpu_memory:.1f}GB VRAM, which should be sufficient")
            else:
                if is_large_model:
                    warning_msg("No GPU detected. 70B model will be very slow on CPU-only systems")
                    if not click.confirm("Continue with download for CPU-only usage?", default=False):
                        info_msg("Download canceled")
                        return
                else:
                    info_msg("No GPU detected. Model will run on CPU (slower but functional)")
        except ImportError:
            info_msg("PyTorch not detected. Cannot check GPU capabilities")
        
        # Download the model using huggingface_hub
        if click.confirm("Ready to download model? This may take a while depending on your connection", default=True):
            downloaded_path = download_model(model_path=model_path, force=force)
            success_msg(f"Model downloaded successfully to {downloaded_path}")
            info_msg("You can now run `python -m cli analyze` to analyze a repository")
        else:
            info_msg("Download canceled")
    except Exception as e:
        error_msg(f"Failed to download model: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('model_name', required=False)
@click.option('--list', 'list_models', is_flag=True, help='List available models')
@click.option('--variant', help='Specific model variant (e.g., Q4_0, Q4_K_M, Q5_K_M)')
@click.option('--download', is_flag=True, help='Download the model after selection')
def select_model(model_name: str, list_models: bool, variant: str, download: bool):
    """Select and optionally download a different model (e.g., codellama-70b-instruct)."""
    try:
        # Import model configuration functions
        from utils.model_config import get_available_models, set_default_model
        
        # List available models if requested or if no model name provided
        if list_models or not model_name:
            info_msg("Available models:")
            available_models = get_available_models()
            
            for name, info in available_models.items():
                click.echo(f"  - {name}: {info['description']}")
                click.echo(f"    Variants: {', '.join(info['variants'])}")
                click.echo(f"    Minimum RAM: {info['min_ram_gb']}GB")
                click.echo("")
                
            if not model_name:
                return
        
        # Set the selected model as default
        if set_default_model(model_name):
            success_msg(f"Selected model: {model_name}")
            
            # Update the .env file
            from utils.env_utils import update_env_file
            model_path = get_default_model_path()
            update_env_file("LLM_MODEL_PATH", model_path)
            success_msg(f"Updated LLM_MODEL_PATH in .env: {model_path}")
            
            # Download if requested
            if download:
                info_msg(f"Downloading {model_name}...")
                download_model(model_path=model_path, model_name=model_name)
            else:
                info_msg(f"To download this model, run: python -m cli download_model_cmd")
        else:
            error_msg(f"Invalid model name: {model_name}")
            info_msg("Use --list to see available models")
            
    except Exception as e:
        error_msg(f"Error selecting model: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--output-dir', default='output',
              help='Directory containing output files')
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
@click.option('--clear-embeddings', is_flag=True, help='Also clear embedding store')
def cleanup(output_dir: str, force: bool, clear_embeddings: bool):
    """Clean up data from previous runs."""
    try:
        info_msg(f"Checking for previous run data in {output_dir}...")
        clean_previous_run(output_dir, force_clean=force, clear_embeddings=clear_embeddings)
        info_msg("Cleanup operation completed.")
    except Exception as e:
        error_msg(f"Cleanup failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
def check_model():
    """Check if the CodeLlama model exists and is valid."""
    try:
        info_msg("Checking CodeLlama model status...")
        
        # Identify expected model path
        model_path = get_default_model_path()
        
        # Identify model type from env or config 
        from utils.model_config import DEFAULT_MODEL_NAME
        model_name = os.environ.get("LLM_MODEL", DEFAULT_MODEL_NAME)
        info_msg(f"Current configured model: {model_name}")

        # Check if the model file exists
        env_model_path = get_env_variable("MODEL_PATH")
        
        # Check for mismatched paths
        if env_model_path and env_model_path != model_path:
            warning_msg(f"Model path in .env ({env_model_path}) doesn't match expected path ({model_path})")
            if click.confirm("Update .env file with correct model path?", default=True):
                update_env_file("MODEL_PATH", model_path)
                success_msg(f"Updated .env file with model path: {model_path}")
                env_model_path = model_path
        
        # Check if the model exists and its size
        exists, file_size_gb = check_model_file(model_path)
        
        if exists:
            info_msg(f"Model found at: {model_path}")
            
            # Size expectations based on model type
            min_size_gb = 1.0
            if "70b" in model_path.lower() or "70b" in model_name.lower():
                min_size_gb = 30.0  # 70B models should be at least 30GB
                
            if file_size_gb < min_size_gb:
                warning_msg(f"Model file size is only {file_size_gb:.2f} GB, which seems too small")
                if "70b" in model_path.lower() and file_size_gb < 30.0:
                    warning_msg("This appears to be a 7B model file mistakenly named as a 70B model.")
                
                if click.confirm("Would you like to re-download the correct model?", default=True):
                    download_model(model_path=model_path, model_name=model_name, force=True)
            else:
                success_msg(f"Model file size looks good: {file_size_gb:.2f} GB")
        else:
            warning_msg(f"Model not found at: {model_path}")
            if click.confirm("Would you like to download the model now?", default=True):
                # Important to pass the model name to ensure right model
                from utils.model_config import get_model_info
                model_info = get_model_info(model_name)
                
                # Show user which model will be downloaded
                info_msg(f"Preparing to download {model_info['name']} ({model_info['variant']})")
                info_msg(f"The {model_info['name']} model is approximately {model_info['min_ram_gb']/4:.1f}GB in size.")
                info_msg(f"This will take some time to download. Please be patient.")
                
                download_model(model_path=model_path, model_name=model_name)
            else:
                error_msg("Model is required for analysis. Please download it before proceeding.")
                return
        
        # Try loading the model to verify it's working
        info_msg("Testing if model can be loaded...")
        if test_model_loading(model_path):
            success_msg("Model loaded successfully!")
            success_msg("Your environment is correctly set up for AI Threat Model generation")
        else:
            error_msg(f"Failed to load model")
            warning_msg("You may have a corrupted model file or incompatible hardware.")
            if click.confirm("Would you like to re-download the model?", default=True):
                download_model(model_path=model_path, model_name=model_name, force=True)
    
    except Exception as e:
        error_msg(f"Error checking model: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
def set_token():
    """Set your Hugging Face token for model downloads."""
    try:
        # Check if token is already set
        current_token = get_env_variable("HF_TOKEN", "")
        
        if current_token:
            if not click.confirm(f"HF_TOKEN is already set. Do you want to update it?", default=True):
                info_msg("Token unchanged.")
                return
        
        token = set_token_interactive()
        if token:
            success_msg("Hugging Face token has been saved.")
            success_msg("You can now download models that require authentication.")
        else:
            error_msg("Failed to set Hugging Face token.")
        
    except Exception as e:
        error_msg(f"Error setting token: {str(e)}")

@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed GPU information')
@click.option('--benchmark', is_flag=True, help='Run a quick benchmark to test GPU performance')
def gpu_info(detailed: bool, benchmark: bool):
    """Show GPU information for embedding and model acceleration."""
    try:
        # Check for PyTorch
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                success_msg(f"PyTorch detected {gpu_count} CUDA-capable GPU(s)")
                
                # Get current GPU configuration from environment
                current_gpu_ids = os.environ.get("GPU_IDS", "")
                if current_gpu_ids:
                    info_msg(f"Currently selected GPUs: {current_gpu_ids}")
                    
                memory_limit = os.environ.get("GPU_MEMORY_LIMIT", "")
                if memory_limit:
                    info_msg(f"GPU memory limit: {memory_limit} GB")
                    
                distributed_mode = os.environ.get("DISTRIBUTED", "").lower() in ["true", "1", "yes"]
                if distributed_mode:
                    info_msg("Multi-GPU distributed mode is enabled")
                
                for i in range(gpu_count):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_name = device_props.name
                    gpu_memory = device_props.total_memory / (1024**3)  # Convert to GB
                    
                    # Check if this GPU is currently being used
                    is_selected = str(i) in current_gpu_ids.split(",") if current_gpu_ids else False
                    selection_marker = "→ " if is_selected else "  "
                    
                    info_msg(f"{selection_marker}GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
                    
                    # Show detailed information if requested
                    if detailed:
                        compute_capability = f"{device_props.major}.{device_props.minor}"
                        multi_processor_count = device_props.multi_processor_count
                        
                        click.echo(f"     Compute capability: {compute_capability}")
                        click.echo(f"     Multi-processors: {multi_processor_count}")
                        click.echo(f"     Clock rate: {device_props.clock_rate / 1000:.0f} MHz")
                        
                        # Get current memory usage
                        try:
                            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                            click.echo(f"     Memory allocated: {memory_allocated:.2f} GB")
                            click.echo(f"     Memory reserved: {memory_reserved:.2f} GB")
                            click.echo(f"     Memory available: {gpu_memory - memory_reserved:.2f} GB")
                        except:
                            click.echo("     Memory usage information not available")
                
                # Show recommended configuration based on hardware
                if gpu_count > 1:
                    info_msg("\nRecommended multi-GPU configuration:")
                    for config_type, gpus in [
                        ("High Performance", list(range(min(3, gpu_count)))),
                        ("Balanced", list(range(min(2, gpu_count)))),
                        ("Memory Efficient", [0])
                    ]:
                        gpu_ids_str = ",".join(str(i) for i in gpus)
                        click.echo(f"  {config_type}: --gpu-ids {gpu_ids_str}" + 
                                  (" --distributed" if len(gpus) > 1 else ""))
                
                # Run a quick benchmark if requested
                if benchmark:
                    info_msg("\nRunning quick GPU benchmark...")
                    
                    # Set benchmark device
                    if current_gpu_ids:
                        benchmark_device = int(current_gpu_ids.split(",")[0])
                        info_msg(f"Using selected GPU {benchmark_device} for benchmark")
                    else:
                        benchmark_device = 0
                        info_msg(f"Using GPU {benchmark_device} for benchmark")
                    
                    # Create a simple benchmark test
                    try:
                        # Benchmark matrix multiplication as a simple test
                        torch.cuda.set_device(benchmark_device)
                        
                        # Test different sizes to see scaling
                        sizes = [1000, 2000, 4000]
                        for size in sizes:
                            # Create random tensors
                            a = torch.randn(size, size, device=f"cuda:{benchmark_device}")
                            b = torch.randn(size, size, device=f"cuda:{benchmark_device}")
                            
                            # Warm-up
                            torch.matmul(a, b)
                            torch.cuda.synchronize()
                            
                            # Benchmark
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            
                            start_event.record()
                            result = torch.matmul(a, b)
                            end_event.record()
                            
                            torch.cuda.synchronize()
                            elapsed_time = start_event.elapsed_time(end_event)
                            
                            success_msg(f"Matrix multiplication ({size}x{size}): {elapsed_time:.2f} ms")
                            
                        # Calculate theoretical FLOPS for matrix multiplication
                        # For an NxN matrix multiplication: 2*N^3 operations
                        largest_size = sizes[-1]
                        flops = 2 * (largest_size ** 3)
                        teraflops = (flops / elapsed_time) / 1e9  # Convert to TFLOPS
                        info_msg(f"Approximate performance: {teraflops:.2f} TFLOPS")
                        
                        # Memory bandwidth test
                        vector_size = 100_000_000  # 100M elements
                        a = torch.randn(vector_size, device=f"cuda:{benchmark_device}")
                        b = torch.randn(vector_size, device=f"cuda:{benchmark_device}")
                        
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
                        
                        success_msg(f"Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
                        
                    except Exception as bench_error:
                        warning_msg(f"Benchmark error: {str(bench_error)}")
                    
                # Test for sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    info_msg("\nTesting embedding model on GPU...")
                    
                    # Use the first GPU by default or the first selected one
                    test_device = 0
                    if current_gpu_ids:
                        test_device = int(current_gpu_ids.split(",")[0])
                        
                    model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=f'cuda:{test_device}')
                    success_msg(f"✅ Successfully loaded embedding model on GPU {test_device}")
                    
                    # Test embedding
                    embedding = model.encode(["This is a test sentence"])
                    success_msg(f"✅ Generated embedding vector with shape: {embedding.shape}")
                    
                    # If there are multiple GPUs, suggest distributed setup
                    if gpu_count > 1 and not distributed_mode:
                        info_msg("You have multiple GPUs. For better performance, consider using:")
                        info_msg("python -m cli configure-gpu --gpu-ids 0,1" + 
                                ("" if gpu_count <= 2 else f",2..{gpu_count-1}") + 
                                " --distributed")
                except Exception as e:
                    warning_msg(f"Failed to load embedding model on GPU: {str(e)}")
            else:
                info_msg("No CUDA-capable GPUs detected by PyTorch")
                info_msg("The system will run in CPU-only mode, which is perfectly fine but slower")
                # Check CPU information
                try:
                    cpu_count = psutil.cpu_count(logical=False)
                    logical_cpu_count = psutil.cpu_count(logical=True)
                    memory = psutil.virtual_memory()
                    memory_gb = memory.total / (1024**3)
                    
                    info_msg(f"CPU: {cpu_count} physical cores, {logical_cpu_count} logical cores")
                    info_msg(f"System memory: {memory_gb:.2f} GB")
                    
                    # Add CPU recommendation
                    if memory_gb > 32:
                        info_msg("Your system has sufficient RAM for CPU-based model execution")
                    elif memory_gb > 16:
                        info_msg("Your system has adequate RAM for smaller models in CPU mode")
                    else:
                        warning_msg("Limited RAM may affect performance with larger models")
                except:
                    pass
        except ImportError:
            warning_msg("PyTorch not installed. Cannot check GPU availability.")
            info_msg("Run the setup script to install all dependencies")
            
        # Check for FAISS
        try:
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                success_msg(f"\nFAISS with GPU acceleration available")
            else:
                info_msg("\nFAISS is using CPU version (this is normal)")
                
                # Provide clear information about GPU usage
                try:
                    if torch.cuda.is_available():
                        success_msg("Your system will still use GPU for all important operations:")
                        info_msg("✓ GPU acceleration for the LLM model")
                        info_msg("✓ GPU acceleration for embedding generation")
                        info_msg("✓ Only vector similarity search uses CPU (minimal impact)")
                except ImportError:
                    pass
        except ImportError:
            warning_msg("FAISS not installed.")
            try:
                if click.confirm("Would you like to install FAISS CPU version now?", default=True):
                    info_msg("Installing faiss-cpu...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
                    success_msg("FAISS CPU version installed successfully")
            except Exception as e:
                warning_msg(f"Failed to install FAISS: {str(e)}")
            
    except Exception as e:
        error_msg(f"Error checking GPU information: {str(e)}")

@cli.command()
@click.option('--force-gpu', is_flag=True, help='Force GPU usage for LLM (overrides detection)')
@click.option('--force-cpu', is_flag=True, help='Force CPU usage for LLM (overrides detection)')
@click.option('--gpu-ids', help='Specify GPU IDs to use (comma-separated, e.g., "0,1,2")')
@click.option('--memory-limit', type=float, help='Limit GPU memory usage (in GB)')
@click.option('--distributed', is_flag=True, help='Enable multi-GPU distributed processing (experimental)')
def configure_gpu(force_gpu: bool, force_cpu: bool, gpu_ids: str, memory_limit: float, distributed: bool):
    """Configure GPU usage for LLaMA model and embeddings.
    
    For multi-GPU setups, you can specify which GPUs to use with --gpu-ids.
    The --memory-limit option helps when sharing GPU resources with other applications.
    """
    try:
        # Use the function from gpu_utils instead of model_utils
        from utils.gpu_utils import detect_gpu_capabilities
        
        # Can't force both CPU and GPU
        if force_gpu and force_cpu:
            error_msg("Cannot force both CPU and GPU at the same time")
            return
        
        # Process GPU IDs if provided
        selected_gpus = None
        if gpu_ids:
            try:
                selected_gpus = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
                info_msg(f"Selected GPUs: {selected_gpus}")
                
                # Quick validation of GPU IDs
                if torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                    for gpu_id in selected_gpus:
                        if gpu_id >= available_gpus:
                            warning_msg(f"GPU {gpu_id} does not exist (only {available_gpus} GPUs available)")
                            if not click.confirm("Continue with other valid GPUs?", default=True):
                                info_msg("Operation canceled")
                                return
                            # Filter out invalid GPUs
                            selected_gpus = [gpu_id for gpu_id in selected_gpus if gpu_id < available_gpus]
                            if not selected_gpus:
                                error_msg("No valid GPUs selected")
                                return
                            info_msg(f"Using GPUs: {selected_gpus}")
            except Exception as e:
                error_msg(f"Invalid GPU IDs format: {str(e)}")
                info_msg("Please use comma-separated numbers, e.g., '0,1,2'")
                return
        
        if force_gpu:
            # First check if CUDA is available
            cuda_available = False
            try:
                cuda_available = torch.cuda.is_available()
                if not cuda_available:
                    warning_msg("WARNING: No CUDA-capable GPU detected, but trying to force GPU mode")
                    warning_msg("This may cause errors or falls back to CPU mode")
                    if not click.confirm("Continue with force GPU mode anyway?", default=False):
                        info_msg("Operation canceled")
                        return
            except ImportError:
                warning_msg("PyTorch not installed, cannot verify GPU availability")
                warning_msg("Run setup.sh to install all dependencies properly")
                if not click.confirm("Continue with force GPU mode anyway?", default=False):
                    info_msg("Operation canceled")
                    return
            
        # Detect current settings
        current_gpu = os.environ.get("FORCE_GPU", "").lower() in ["true", "1", "yes"]
        current_cpu = os.environ.get("FORCE_CPU", "").lower() in ["true", "1", "yes"]
        
        # Check current GPU capabilities
        gpu_config = detect_gpu_capabilities(force_gpu=force_gpu, force_cpu=force_cpu)
        
        if force_gpu:
            # Set environment variable to force GPU
            os.environ["FORCE_GPU"] = "true"
            if "FORCE_CPU" in os.environ:
                del os.environ["FORCE_CPU"]
                
            # Also update the .env file if it exists
            from utils.env_utils import update_env_file
            update_env_file("FORCE_GPU", "true")
            update_env_file("FORCE_CPU", "")
            
            # Update GPU IDs if provided
            if selected_gpus:
                update_env_file("GPU_IDS", ",".join(str(gpu_id) for gpu_id in selected_gpus))
                os.environ["GPU_IDS"] = ",".join(str(gpu_id) for gpu_id in selected_gpus)
                success_msg(f"Selected GPUs {selected_gpus} will be used for processing")
            
            # Update memory limit if provided
            if memory_limit:
                update_env_file("GPU_MEMORY_LIMIT", str(memory_limit))
                os.environ["GPU_MEMORY_LIMIT"] = str(memory_limit)
                success_msg(f"GPU memory usage limited to {memory_limit} GB")
            
            # Update distributed flag
            if distributed:
                update_env_file("DISTRIBUTED", "true")
                os.environ["DISTRIBUTED"] = "true"
                success_msg("Multi-GPU distributed processing enabled (experimental)")
                info_msg("This feature requires all selected GPUs to have sufficient memory")
            
            # When forcing GPU mode, automatically install FAISS-GPU
            if cuda_available:
                info_msg("Installing GPU-accelerated libraries for FAISS...")
                if install_faiss_gpu():
                    success_msg("GPU acceleration packages installed successfully")
                else:
                    warning_msg("Failed to install GPU acceleration packages")
                    warning_msg("System will still use GPU for model inference and embedding generation")
            
            success_msg("GPU acceleration FORCED ON for LLM")
            if not gpu_config['use_gpu']:
                warning_msg("WARNING: No GPU detected but forcing GPU acceleration. This may cause errors.")
                info_msg("If you encounter issues, use --force-cpu to switch back to CPU mode")
        elif force_cpu:
            # Set environment variable to force CPU
            os.environ["FORCE_CPU"] = "true"
            if "FORCE_GPU" in os.environ:
                del os.environ["FORCE_GPU"]
                
            # Also update the .env file if it exists
            from utils.env_utils import update_env_file
            update_env_file("FORCE_CPU", "true")
            update_env_file("FORCE_GPU", "")
            
            # Clear any multi-GPU settings
            update_env_file("GPU_IDS", "")
            update_env_file("DISTRIBUTED", "")
            update_env_file("GPU_MEMORY_LIMIT", "")
            if "GPU_IDS" in os.environ:
                del os.environ["GPU_IDS"]
            if "DISTRIBUTED" in os.environ:
                del os.environ["DISTRIBUTED"]
            if "GPU_MEMORY_LIMIT" in os.environ:
                del os.environ["GPU_MEMORY_LIMIT"]
            
            success_msg("GPU acceleration FORCED OFF for LLM (using CPU only)")
            info_msg("The system will run slower but can work on machines without GPU")
        else:
            # No change to force settings, but update other GPU options
            
            # Update GPU IDs if provided
            if selected_gpus:
                update_env_file("GPU_IDS", ",".join(str(gpu_id) for gpu_id in selected_gpus))
                os.environ["GPU_IDS"] = ",".join(str(gpu_id) for gpu_id in selected_gpus)
                success_msg(f"Selected GPUs {selected_gpus} will be used for processing")
            
            # Update memory limit if provided
            if memory_limit:
                update_env_file("GPU_MEMORY_LIMIT", str(memory_limit))
                os.environ["GPU_MEMORY_LIMIT"] = str(memory_limit)
                success_msg(f"GPU memory usage limited to {memory_limit} GB")
            
            # Update distributed flag
            if distributed:
                update_env_file("DISTRIBUTED", "true")
                os.environ["DISTRIBUTED"] = "true"
                success_msg("Multi-GPU distributed processing enabled (experimental)")
                info_msg("This feature requires all selected GPUs to have sufficient memory")
            
            # Just display current status
            if current_gpu:
                info_msg("Current setting: GPU acceleration FORCED ON")
            elif current_cpu:
                info_msg("Current setting: GPU acceleration FORCED OFF (CPU only)")
            else:
                info_msg("Current setting: AUTO-DETECT GPU")
                
            if gpu_config['use_gpu']:
                success_msg(f"GPU detected: {gpu_config['gpu_info']}")
                success_msg(f"Will use {gpu_config['n_gpu_layers']} GPU layers for LLM")
            else:
                info_msg("No GPU detected or GPU not available. Using CPU.")
                info_msg("The system will run slower but is fully functional in CPU-only mode")
    except Exception as e:
        error_msg(f"Error configuring GPU settings: {str(e)}")

@cli.command()
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Output directory for analysis results')
def report(output_dir: str):
    """Generate a comprehensive security report from analysis results."""
    visualizer = ThreatModelVisualizer()
    
    try:
        info_msg(f"Generating report from {output_dir}")
        
        with tqdm(total=2, desc="Generating report") as progress:
            visualizer = ThreatModelVisualizer()
            progress.update(1)
            
            # Check if threat model exists
            threat_model_path = Path(output_dir) / "threat_model.json"
            if not threat_model_path.exists():
                warning_msg(f"Threat model file not found at {threat_model_path}, using default model")
            
            # Generate report
            report_path = visualizer.generate_report_from_dir(output_dir)
            progress.update(1)
            
            success_msg(f"Report generated at {report_path}")
            
            # Open the report in a browser
            info_msg(f"Opening report in browser: {report_path}")
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except Exception as e:
        error_msg(f"Error: {str(e)}")

def main():
    """Main entry point for the CLI."""
    try:
        # Set up signal handlers for clean shutdown
        def signal_handler(sig, frame):
            """Handle signals to ensure clean shutdown."""
            logger.info("Received signal to terminate, cleaning up...")
            # Force cleanup of any global resources
            gc.collect()
            
            # Exit with success code
            sys.exit(0)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Check if this is the first run (no arguments) and provide helpful guidance
        if len(sys.argv) == 1:
            try:
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"
                    info_msg(f"Hardware acceleration available: {gpu_name}")
                    info_msg("For optimal performance on this GPU, run:")
                    info_msg("  python -m cli init --device cuda")
                    info_msg("  python -m cli configure-gpu --force-gpu")
                else:
                    info_msg("Running in CPU-only mode (no GPU detected)")
                    info_msg("This is perfectly fine and the tool will work normally, just slower")
                    info_msg("For optimal CPU setup, run:")
                    info_msg("  python -m cli init --device cpu")
                    info_msg("  python -m cli configure-gpu --force-cpu")
            except ImportError:
                pass
        
        # Check if using default model and show message about 70B option
        is_analyze_command = len(sys.argv) > 1 and sys.argv[1] == "analyze"
        env_model = get_env_variable("LLM_MODEL")
        model_path = get_env_variable("LLM_MODEL_PATH", "")
        
        if is_analyze_command and (not env_model or env_model == "codellama-7b-instruct") and "70b" not in model_path.lower():
            info_msg("")
            info_msg("ℹ️  TIP: This project supports CodeLlama 70B for much better results!")
            info_msg("To use it, run: python -m cli select_model codellama-70b-instruct --download")
            info_msg("Note: 70B model requires more RAM and GPU memory")
            info_msg("")
        
        # Run the CLI
        cli()
    except Exception as e:
        error_msg(f"CLI error: {str(e)}")
        # Clean up and exit with error
        gc.collect()
        sys.exit(1)

if __name__ == '__main__':
    main() 