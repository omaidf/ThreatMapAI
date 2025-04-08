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

def create_embedding_store(try_load: bool = True) -> Optional[EmbeddingStore]:
    """Create and initialize the embedding store.
    
    Args:
        try_load: Whether to try loading existing embeddings
        
    Returns:
        Initialized EmbeddingStore or None if initialization failed
    """
    try:
        # Set environment variables to prevent multiprocessing issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Initialize the embedding store
        embedding_store = EmbeddingStore()
        
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
        if 'faiss' in str(e):
            error_msg("FAISS is not installed. Continuing without semantic search capabilities.")
            try:
                if click.confirm("Would you like to install FAISS now?", default=True):
                    info_msg("Installing FAISS... This may take a while.")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
                    info_msg("FAISS installed successfully. Please restart the tool.")
            except Exception as install_e:
                error_msg(f"Failed to install FAISS: {str(install_e)}")
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
def init():
    """Initialize the framework."""
    try:
        # Setup tree-sitter grammars
        success_msg("Initializing tree-sitter grammars...")
        try:
            setup_tree_sitter()
            success_msg("Tree-sitter grammars set up successfully")
        except Exception as e:
            warning_msg(f"Failed to set up tree-sitter grammars: {str(e)}")
            warning_msg("The framework may not function correctly")
            return
        
        # Create embedding store
        success_msg("Initializing embedding store...")
        try:
            embedding_store = create_embedding_store(try_load=True)
            success_msg("Embedding store initialized")
        except Exception as e:
            warning_msg(f"Failed to initialize embedding store: {str(e)}")
            warning_msg("Embedding store functionality will be limited")
            
        success_msg("Framework initialized successfully")
        
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
def analyze(repository_url: str, output_dir: str, model_path: str, local: bool, clean: bool, 
           clear_embeddings: bool, reuse_embeddings: bool):
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
        
        # Ensure model exists
        model_path = validate_model_path(model_path)
        
        if not Path(model_path).exists():
            logger.warning(f"CodeLlama model not found at {model_path}")
            if click.confirm("Model not found. Would you like to download it now?"):
                download_model(model_path)
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
                    info_msg("Reusing existing embeddings as requested (--reuse-embeddings)")
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
                        info_msg("Reusing existing embeddings")
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
            
            # Initialize embedding store
            if clear_embeddings:
                # Initialize without loading existing data
                embedding_store = create_embedding_store(try_load=False)
                # Explicitly clear any existing data
                if embedding_store:
                    embedding_store.clear()
                    info_msg("Cleared existing embeddings")
            else:
                # Try to load existing embeddings if available
                embedding_store = create_embedding_store(try_load=use_existing)
            progress.update(1)
            
            # Initialize the repository analyzer directly, not in a subprocess
            analysis_results = {}
            
            # Skip the model loading test and subprocess approach
            progress.update(1)
            
            # Import required modules directly
            from repository_analyzer.analyzer import RepositoryAnalyzer
            
            # Initialize analyzer component
            analyzer = RepositoryAnalyzer(
                repo_path=repository_url if local else None,
                embedding_store=embedding_store
            )
            
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
            
            # Initialize LLM processor in isolation to avoid memory conflicts with analyzer
            processor = None
            try:
                # Import in function scope to avoid memory issues
                from llm_processor.processor import LLMProcessor
                processor = LLMProcessor(model_path_or_embedding_store=model_path)
                info_msg(f"LLM processor initialized with model: {model_path}")
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
        info_msg("Starting model download...")
        model_path = get_default_model_path()
        
        # Download the model using huggingface_hub
        downloaded_path = download_model(model_path, force)
        
        success_msg(f"Model downloaded successfully to {downloaded_path}")
        info_msg("You can now run `python -m cli analyze` to analyze a repository")
    except Exception as e:
        error_msg(f"Failed to download model: {str(e)}")
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
            
            if file_size_gb < 1.0:
                warning_msg(f"Model file size is only {file_size_gb:.2f} GB, which seems too small")
                if click.confirm("Would you like to re-download the model?", default=True):
                    download_model(model_path, force=True)
            else:
                success_msg(f"Model file size looks good: {file_size_gb:.2f} GB")
        else:
            warning_msg(f"Model not found at: {model_path}")
            if click.confirm("Would you like to download the model now?", default=True):
                download_model(model_path)
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
                download_model(model_path, force=True)
    
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