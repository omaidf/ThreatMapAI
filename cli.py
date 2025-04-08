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
from pathlib import Path
from typing import Optional, Tuple, List
import click
import threading
import time
import re
import webbrowser
from tqdm import tqdm
import importlib.util
import signal
import psutil

# Import utility modules
from utils import (
    success_msg, error_msg, warning_msg, info_msg, 
    get_env_variable, update_env_file,
    detect_architecture, get_default_model_path, validate_model_path,
    download_model, set_token_interactive, test_model_loading, check_model_file
)
from utils.file_utils import (
    check_output_directory, clean_previous_run, check_required_files,
    check_dependencies
)
from utils.diagram_utils import (
    find_diagrams, start_server_and_open_diagrams, view_diagrams
)

# Import core modules when needed
from repository_analyzer.embedding_store import EmbeddingStore

# Delay loading of other optional/heavyweight modules into functions where needed
# from dotenv import load_dotenv
# from repository_analyzer.analyzer import RepositoryAnalyzer
# from llm_processor.processor import LLMProcessor
# from visualizer.visualizer import ThreatModelVisualizer
# from view_diagram import convert_to_html, start_server
# from tqdm import tqdm

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    from dotenv import load_dotenv
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

# Define colored output styles for click
SUCCESS_STYLE = {'fg': 'green', 'bold': True}
ERROR_STYLE = {'fg': 'red', 'bold': True}
WARNING_STYLE = {'fg': 'yellow', 'bold': False}
INFO_STYLE = {'fg': 'blue', 'bold': False}

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

def validate_output_dir(ctx, param, value: str) -> str:
    """Validate and create output directory if it doesn't exist."""
    try:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return value
    except Exception as e:
        # Try to use a default directory if the specified one can't be created
        logger.warning(f"Couldn't create output directory {value}, using default: output")
        Path("output").mkdir(exist_ok=True)
        return "output"

def check_requirements() -> None:
    """Check if all required dependencies are installed."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        with open('requirements.txt') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            missing_deps = []
            for req in requirements:
                try:
                    # Parse the requirement string
                    req_name = req.split('>=')[0].split('==')[0].strip()
                    if ';' in req_name:  # Handle environment markers
                        req_name = req_name.split(';')[0].strip()
                    
                    # Try to get the version - will raise an exception if missing
                    version(req_name)
                except PackageNotFoundError:
                    missing_deps.append(req)
            
            if missing_deps:
                logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                if not click.confirm("Install missing dependencies?", default=True):
                    logger.warning("Continuing without installing dependencies. This may cause errors.")
                    return
                
                for req in missing_deps:
                    logger.info(f"Installing missing dependency: {req}")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to install {req}. Continuing anyway.")
    except Exception as e:
        logger.error(f"Failed to check/install requirements: {str(e)}")
        logger.warning("Continuing without checking dependencies. This may cause errors.")

def setup_tree_sitter() -> None:
    """Set up tree-sitter grammars."""
    try:
        # Import here to avoid circular imports
        from repository_analyzer.analyzer import RepositoryAnalyzer
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
                from repository_analyzer.analyzer import RepositoryAnalyzer
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
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            # If torch isn't available, we'll default to CPU anyway
            pass
            
        # Only use CPU if explicitly requested or if CUDA isn't available
        use_cpu = (device == 'cpu' or not cuda_available)
        
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

def find_diagrams(output_dir: str) -> List[Path]:
    """Find Mermaid diagram files in the output directory."""
    diagram_files = []
    output_path = Path(output_dir)
    
    if output_path.exists() and output_path.is_dir():
        for file in output_path.glob("**/*.mmd"):
            diagram_files.append(file)
    
    return diagram_files

@click.group()
def cli():
    """AI Threat Model Map Generator CLI."""
    pass

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
            import torch
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
            import psutil
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

def clean_previous_run(output_dir: str, force_clean: bool = False, clear_embeddings: bool = False) -> None:
    """
    Clean up data from previous runs while preserving model files.
    
    Args:
        output_dir: Directory containing output files
        force_clean: Force cleaning even without confirmation
        clear_embeddings: Whether to also clear the embedding store
    """
    output_path = Path(output_dir)
    
    # Check if output directory exists and has previous run files
    if not output_path.exists():
        return
        
    # Files to check for previous runs
    previous_run_indicators = [
        "analysis_results.json",
        "threat_model.json",
        "class_diagram.mmd",
        "flow_diagram.mmd",
        "threat_diagram.mmd",
        "threat_analysis_report.html"
    ]
    
    # Check for embedding store files
    embedding_files = [
        "embeddings.index",
        "embeddings_mapping.json"
    ]
    
    # Check for output files
    found_files = []
    for indicator in previous_run_indicators:
        file_path = output_path / indicator
        if file_path.exists():
            found_files.append(indicator)
    
    # Check for embedding files separately
    found_embeddings = []
    for emb_file in embedding_files:
        file_path = output_path / emb_file
        if file_path.exists():
            found_embeddings.append(emb_file)
    
    # Count embeddings if they exist
    embedding_count = 0
    if found_embeddings and len(found_embeddings) == 2:  # Both embedding files need to exist
        try:
            with open(output_path / "embeddings_mapping.json", 'r') as f:
                embedding_data = json.load(f)
                embedding_count = len(embedding_data)
        except Exception:
            pass
    
    if not found_files and not found_embeddings:
        return
    
    # Ask for confirmation if not forced
    if not force_clean:
        if found_files:
            info_msg(f"Found {len(found_files)} files from previous run in {output_dir}:")
            for file in found_files:
                click.echo(f"  - {file}")
            
            # Ask about cleaning output files
            if not click.confirm("Do you want to clean up these output files before running a new analysis?", default=True):
                info_msg("Output files will be preserved and may be overwritten during analysis")
            else:
                # Clean output files
                with tqdm(total=len(found_files), desc="Cleaning previous run files") as progress:
                    for indicator in previous_run_indicators:
                        file_path = output_path / indicator
                        if file_path.exists():
                            file_path.unlink()
                            progress.update(1)
                success_msg(f"Cleaned {len(found_files)} files from previous run")
        
        # Handle embeddings separately with more options
        if found_embeddings and len(found_embeddings) == 2:
            info_msg(f"Found existing embedding store with {embedding_count} entries")
            
            # Don't ask if clear_embeddings was explicitly set
            if clear_embeddings:
                info_msg("Will clear embedding store as requested via --clear-embeddings")
            else:
                # Give more options for embeddings
                action = click.prompt(
                    "What would you like to do with the existing embeddings?",
                    type=click.Choice(["keep", "clear", "ignore"], case_sensitive=False),
                    default="keep"
                )
                
                if action.lower() == "keep":
                    info_msg("Keeping existing embeddings for reuse")
                    clear_embeddings = False
                elif action.lower() == "clear":
                    info_msg("Will clear existing embeddings before analysis")
                    clear_embeddings = True
                else:  # ignore
                    info_msg("Ignoring embeddings - you'll be asked again during analysis")
                    clear_embeddings = False
    else:
        # Force clean all output files
        if found_files:
            with tqdm(total=len(found_files), desc="Cleaning previous run files") as progress:
                for indicator in previous_run_indicators:
                    file_path = output_path / indicator
                    if file_path.exists():
                        file_path.unlink()
                        progress.update(1)
            success_msg(f"Cleaned {len(found_files)} files from previous run")
    
    # Clean embedding store if requested
    if clear_embeddings and found_embeddings:
        try:
            # Initialize and clear the embedding store
            info_msg("Clearing embedding store contents...")
            embedding_store = create_embedding_store(try_load=False)  # Don't load existing data
            if embedding_store:
                embedding_store.clear()
                success_msg("Embedding store cleared successfully")
        except Exception as e:
            warning_msg(f"Failed to clear embedding store: {str(e)}")
            warning_msg("Continuing without clearing embeddings")

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
        # Suppress warnings if requested
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            # Specifically suppress FAISS-related warnings
            warnings.filterwarnings("ignore", message=".*GPU FAISS.*")
            warnings.filterwarnings("ignore", message=".*CUDA.*")
            # Reduce logging level for known verbose modules
            logging.getLogger("faiss").setLevel(logging.ERROR)
            
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
        
        # Determine optimal GPU configuration
        selected_gpu_ids = None
        use_distributed = distributed  # Start with user preference
        
        # Force GPU usage when available (unless explicitly set to CPU)
        if device != 'cpu':
            # Check if GPU is available
            try:
                import torch
                if torch.cuda.is_available():
                    # For CUDA operations, force GPU
                    os.environ["FORCE_GPU"] = "true"
                    if "FORCE_CPU" in os.environ:
                        del os.environ["FORCE_CPU"]
                    
                    # Get number of available GPUs
                    gpu_count = torch.cuda.device_count()
                    
                    # For multi-GPU setups, automatically enable distributed mode 
                    # unless user specifically provided a single GPU ID
                    if gpu_count > 1:
                        # Auto-determine if distributed should be enabled
                        if gpu_id is None:  # If no specific GPU was requested
                            use_distributed = True  # Enable by default for multiple GPUs
                            selected_gpu_ids = list(range(gpu_count))
                            info_msg(f"Multi-GPU setup detected: automatically using all {gpu_count} GPUs with distributed processing")
                            os.environ["DISTRIBUTED"] = "true"
                            os.environ["GPU_IDS"] = ",".join(str(id) for id in selected_gpu_ids)
                        else:
                            # User specified a specific GPU, use only that one
                            if gpu_id < gpu_count:
                                selected_gpu_ids = [gpu_id]
                                info_msg(f"Using specific GPU {gpu_id} as requested (out of {gpu_count} available)")
                                os.environ["GPU_IDS"] = str(gpu_id)
                            else:
                                warning_msg(f"GPU {gpu_id} not available (only {gpu_count} GPUs detected)")
                                warning_msg(f"Using default GPU 0 instead")
                                selected_gpu_ids = [0]
                                os.environ["GPU_IDS"] = "0"
                    else:
                        # Single GPU setup
                        info_msg(f"Single GPU detected: using GPU 0")
                        selected_gpu_ids = [0]
                        os.environ["GPU_IDS"] = "0"
                    
                    # Set memory limit if provided
                    if memory_limit:
                        os.environ["GPU_MEMORY_LIMIT"] = str(memory_limit)
                        info_msg(f"GPU memory limit set to {memory_limit} GB")
                    
                    success_msg("GPU acceleration enabled for analysis")
                else:
                    warning_msg("No GPU detected, falling back to CPU")
                    os.environ["FORCE_CPU"] = "true"
                    if "FORCE_GPU" in os.environ:
                        del os.environ["FORCE_GPU"]
            except ImportError:
                warning_msg("PyTorch not installed, cannot detect GPU. Falling back to CPU")
                os.environ["FORCE_CPU"] = "true"
                if "FORCE_GPU" in os.environ:
                    del os.environ["FORCE_GPU"]
        else:
            # Explicitly requested CPU mode
            info_msg("Using CPU as requested")
            os.environ["FORCE_CPU"] = "true"
            if "FORCE_GPU" in os.environ:
                del os.environ["FORCE_GPU"]
        
        # Ensure model exists
        model_path = validate_model_path(model_path)
        
        if not Path(model_path).exists():
            logger.warning(f"CodeLlama model not found at {model_path}")
            if click.confirm("Model not found. Would you like to download it now?"):
                download_model(model_path=model_path, force=True)
            else:
                error_msg("Model must be downloaded before analysis.")
                return
        
        # Initialize components with progress indicators
        with tqdm(total=4, desc="Initializing") as progress:
            # Check for existing embeddings
            embeddings_index_path = Path(output_dir) / "embeddings.index"
            embeddings_mapping_path = Path(output_dir) / "embeddings_mapping.json"
            embeddings_exist = embeddings_index_path.exists() and embeddings_mapping_path.exists()
            
            if embeddings_exist:
                # Count existing embeddings to provide more information
                embedding_count = 0
                try:
                    with open(embeddings_mapping_path, 'r') as f:
                        embedding_data = json.load(f)
                        embedding_count = len(embedding_data)
                except Exception:
                    pass
                
                info_msg(f"Found existing embeddings with {embedding_count} entries")
                
                # Give the user options unless reuse is explicitly set
                if reuse_embeddings:
                    info_msg("Reusing existing embeddings as requested (--reuse-embeddings) - only new or changed files will be indexed")
                    use_existing = True
                    clear_embeddings = False
                else:
                    # Ask what the user wants to do
                    options = ["reuse", "recreate", "cancel"]
                    action = click.prompt(
                        "Found existing embeddings. What would you like to do?",
                        type=click.Choice(options, case_sensitive=False),
                        default="reuse"
                    )
                    
                    if action.lower() == "reuse":
                        info_msg("Reusing existing embeddings - only new or changed files will be indexed")
                        use_existing = True
                        clear_embeddings = False
                    elif action.lower() == "recreate":
                        info_msg("Recreating embeddings from scratch")
                        use_existing = False
                        clear_embeddings = True
                    else:  # cancel
                        info_msg("Analysis cancelled by user")
                        return
            else:
                use_existing = False
            
            # Initialize embedding store with specified GPU ID
            if clear_embeddings:
                # Initialize without loading existing data
                if device == 'auto':
                    device_param = None  # Auto-detect
                else:
                    device_param = device
                embedding_store = create_embedding_store(try_load=False, device=device_param, 
                                                         gpu_id=selected_gpu_ids[0] if selected_gpu_ids else None)
                # Explicitly clear any existing data
                if embedding_store:
                    embedding_store.clear()
                    info_msg("Cleared existing embeddings")
            else:
                # Try to load existing embeddings if available
                if device == 'auto':
                    device_param = None  # Auto-detect
                else:
                    device_param = device
                embedding_store = create_embedding_store(try_load=use_existing, device=device_param, 
                                                         gpu_id=selected_gpu_ids[0] if selected_gpu_ids else None)
            progress.update(1)
            
            # Initialize the repository analyzer directly, not in a subprocess
            analysis_results = {}
            
            # Skip the model loading test and subprocess approach
            progress.update(1)
            
            # Import required modules directly
            from repository_analyzer.analyzer import RepositoryAnalyzer
            
            # Initialize analyzer component with GPU configuration
            analyzer_config = {
                'repo_path': repository_url if local else None,
                'embedding_store': embedding_store,
                'distributed': use_distributed,
                'gpu_ids': selected_gpu_ids,
            }
            
            # Add memory limit if specified
            if memory_limit:
                analyzer_config['memory_limit'] = memory_limit
            
            analyzer = RepositoryAnalyzer(**analyzer_config)
            
            # Run analysis directly
            with tqdm(total=1, desc="Analyzing repository") as progress:
                try:
                    if local:
                        info_msg(f"Analyzing local repository: {repository_url}")
                        analysis_results = analyzer.analyze_code()
                    else:
                        info_msg(f"Analyzing remote repository: {repository_url}")
                        info_msg("Cloning repository - this may take some time for large repositories...")
                        analysis_results = analyzer.analyze_repository(repository_url)
                except Exception as analysis_error:
                    error_msg(f"Repository analysis failed: {str(analysis_error)}")
                    # Create minimal analysis results to allow proceeding
                    analysis_results = {
                        "components": [],
                        "data_flows": [],
                        "dependencies": [],
                        "architecture": {
                            "file_types": {},
                            "directory_structure": {},
                            "entry_points": []
                        }
                    }
                progress.update(1)
            
            # Initialize LLM processor with hardware config
            processor = None
            try:
                # Import in function scope to avoid memory issues
                from llm_processor.processor import LLMProcessor
                
                # Prepare LLM processor configuration with hardware settings
                llm_config = {'model_path_or_embedding_store': model_path}
                
                # Add device configuration
                if device != 'auto':
                    llm_config['device'] = device
                
                # Add distributed configuration and GPU IDs
                if use_distributed:
                    llm_config['distributed'] = True
                    if selected_gpu_ids:
                        llm_config['gpu_ids'] = selected_gpu_ids
                elif selected_gpu_ids and len(selected_gpu_ids) == 1:
                    # Single GPU mode
                    llm_config['gpu_id'] = selected_gpu_ids[0]
                
                # Add memory limit if specified
                if memory_limit:
                    llm_config['memory_limit'] = memory_limit
                
                processor = LLMProcessor(**llm_config)
                info_msg(f"LLM processor initialized with model: {model_path}")
                
                # Log hardware configuration
                if hasattr(processor, 'device') and processor.device == 'cuda':
                    if hasattr(processor, 'distributed') and processor.distributed:
                        info_msg(f"LLM running in distributed mode across GPUs: {selected_gpu_ids}")
                    elif hasattr(processor, 'gpu_id') and processor.gpu_id is not None:
                        info_msg(f"LLM running on GPU {processor.gpu_id}")
                    else:
                        info_msg("LLM running on GPU")
                elif hasattr(processor, 'device'):
                    info_msg(f"LLM running on {processor.device}")
                
            except Exception as e:
                warning_msg(f"Failed to initialize LLM: {str(e)}")
                warning_msg("Using fallback processor. Results will be limited.")
                # Use a minimal processor if available
                try:
                    from llm_processor.processor import LLMProcessor
                    processor = LLMProcessor(model_path_or_embedding_store="")
                except Exception as e2:
                    processor = None
            progress.update(1)
            
            # Initialize visualizer
            from visualizer.visualizer import ThreatModelVisualizer
            from view_diagram import convert_to_html, start_server
            visualizer = ThreatModelVisualizer()
            progress.update(1)
        
        # Generate threat model
        threat_model = {}
        with tqdm(total=1, desc="Generating threat model") as progress:
            try:
                if processor:
                    threat_model = processor.generate_threat_model(analysis_results)
                else:
                    # Create a minimal threat model when no processor is available
                    warning_msg("No LLM processor available. Creating minimal threat model.")
                    threat_model = {
                        "components": [],
                        "data_flows": [],
                        "overall_risk": "Unknown (No LLM available)",
                        "architecture": {"description": "No LLM available for analysis"},
                        "code_flow": [],
                        "security_boundaries": [],
                        "vulnerabilities": [],
                        "cross_boundary_flows": []
                    }
                    # Add some basic component info
                    for component in analysis_results.get("components", []):
                        threat_model["components"].append({
                            "name": component["name"],
                            "path": component["path"],
                            "type": component.get("type", "Unknown"),
                            "threats": []
                        })
            except Exception as threat_error:
                error_msg(f"Threat model generation failed: {str(threat_error)}")
                # Create minimal threat model to allow proceeding
                threat_model = {
                    "components": [],
                    "data_flows": [],
                    "overall_risk": "Unknown (Error in threat model generation)",
                    "architecture": {"description": "Error in threat model generation"},
                    "code_flow": [],
                    "security_boundaries": [],
                    "vulnerabilities": [],
                    "cross_boundary_flows": []
                }
            progress.update(1)
        
        # Generate visualizations with progress bar
        with tqdm(total=4, desc="Generating visualizations") as progress:
            try:
                class_diagram = visualizer.generate_class_diagram(analysis_results)
                progress.update(1)
                
                flow_diagram = visualizer.generate_flow_diagram(analysis_results)
                progress.update(1)
                
                threat_diagram = visualizer.generate_threat_diagram(threat_model)
                progress.update(1)
                
                # Save results
                with open(os.path.join(output_dir, "threat_model.json"), 'w') as f:
                    json.dump(threat_model, f, indent=2)
                
                with open(os.path.join(output_dir, "analysis_results.json"), 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                    
                progress.update(1)
            except Exception as e:
                error_msg(f"Failed to generate visualizations: {str(e)}")
        
        # Cleanup resources before starting server
        if embedding_store:
            # Save state before cleanup
            try:
                embedding_store.save()
            except:
                pass
            embedding_store = None
            
        # Clean up analyzer
        analyzer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Convert to HTML and start server
        info_msg(f"Analysis completed. Results saved to {output_dir}")
        
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
                embedding_store.save()
            except:
                pass
            embedding_store = None
            
        if analyzer:
            analyzer = None
            
        # Force final garbage collection
        import gc
        gc.collect()

@cli.command()
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Output directory for analysis results')
def report(output_dir: str):
    """Generate a comprehensive security report from analysis results."""
    import webbrowser
    from visualizer.visualizer import ThreatModelVisualizer
    
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

@cli.command()
@click.option('--output-dir', default='output', callback=validate_output_dir,
              help='Directory containing analysis results')
def visualize(output_dir: str):
    """Generate visualizations from analysis results."""
    from tqdm import tqdm
    from visualizer.visualizer import ThreatModelVisualizer
    
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
                if path and Path(path).exists():
                    diagram_paths.append(path)
            
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
        from utils.env_utils import get_env_variable
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
            import torch
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
        
        # Check if the model file exists
        model_path = env_model_path or model_path
        
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
            import torch
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
                    import psutil
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
            import faiss
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                success_msg(f"\nFAISS with GPU acceleration available")
            else:
                info_msg("\nFAISS is using CPU version (this is normal)")
                
                # Provide clear information about GPU usage
                try:
                    import torch
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
        from utils.model_utils import detect_gpu_capabilities
        
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
                import torch
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
                import torch
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

def main():
    """Main entry point for the CLI."""
    try:
        # Set up signal handlers for clean shutdown
        def signal_handler(sig, frame):
            """Handle signals to ensure clean shutdown."""
            logger.info("Received signal to terminate, cleaning up...")
            # Force cleanup of any global resources
            import gc
            
            # Release threading resources
            gc.collect()
            
            # Exit with success code
            sys.exit(0)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Check if this is the first run (no arguments) and provide helpful guidance
        if len(sys.argv) == 1:
            try:
                import torch
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
        from utils.model_config import DEFAULT_MODEL_NAME, MODEL_REPOS
        from utils.env_utils import get_env_variable
        
        # Check if this is an analyze command that could benefit from 70B model
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
        import gc
        gc.collect()
        sys.exit(1)

if __name__ == '__main__':
    main() 