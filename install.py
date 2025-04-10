#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
import shutil
import time
import re
import signal
import venv
from pathlib import Path
import importlib.util

# ANSI colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Initialize default values for variables
BATCH_SIZE = 5
PARALLEL_JOBS = 2
PIP_PARALLEL_JOBS = 2
CHECKPOINT_FILE = ".setup_checkpoints"
VENV_NAME = "venv311"
PYTHON_CMD = sys.executable

def print_status(message):
    print(f"{BLUE}[INFO]{NC} {message}")

def print_success(message):
    print(f"{GREEN}[SUCCESS]{NC} {message}")

def print_error(message):
    print(f"{RED}[ERROR]{NC} {message}")

def print_warning(message):
    print(f"{YELLOW}[WARNING]{NC} {message}")

def show_progress(total, step, message):
    width = 50
    percent = int(step * 100 / total)
    completed = int(width * step / total)
    remaining = width - completed
    bar = '=' * completed
    empty = ' ' * remaining
    print(f"\r[{bar}{empty}] {percent}% - {message}", end='')
    if step == total:
        print()

def is_already_installed(component):
    """Check if a component is already installed based on checkpoint file"""
    if os.path.isfile(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            for line in f:
                if line.strip() == f"{component}=installed":
                    return True
    return False

def mark_as_installed(component):
    """Mark a component as installed in the checkpoint file"""
    components = {}
    
    # Read existing components if file exists
    if os.path.isfile(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    components[key] = value
    
    # Update or add the component
    components[component] = "installed"
    
    # Write back to file
    with open(CHECKPOINT_FILE, 'w') as f:
        for key, value in components.items():
            f.write(f"{key}={value}\n")

def is_in_our_venv():
    """Check if we're in a venv that we created previously"""
    if hasattr(sys, 'base_prefix'):
        return sys.base_prefix != sys.prefix and VENV_NAME in sys.prefix
    return False

def run_command(cmd, env=None, silent=True):
    """Run a shell command and return the output and success status"""
    try:
        env_vars = env or os.environ.copy()
        if silent:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env_vars)
        else:
            result = subprocess.run(cmd, shell=True, check=True, env=env_vars)
        return True, result.stdout.decode('utf-8') if hasattr(result, 'stdout') and result.stdout else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else str(e)

def check_command(cmd):
    """Check if a command is available in the system"""
    return shutil.which(cmd) is not None

def create_directories():
    """Create required directories for the application"""
    print_status("Creating required directories...")
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizer/templates", exist_ok=True)
    print_success("Directories created")

def detect_architecture():
    """Detect system architecture and return appropriate model variant"""
    print_status("Detecting system architecture...")
    arch = platform.machine()
    if arch in ["arm64", "aarch64"]:
        model_variant = "Q4_0"
        print_status(f"Detected ARM architecture, will use {model_variant} model variant")
    else:
        model_variant = "Q4_K_M"
        print_status(f"Detected x86 architecture, will use {model_variant} model variant")
    return model_variant

def setup_env_file(model_variant, hf_token=None, token_set=False):
    """Set up the .env file with appropriate configuration"""
    env_content = {}
    
    # If .env already exists, read its contents
    if os.path.isfile(".env"):
        print_status("Found existing .env file")
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_content[key] = value
    
    # Set or update HF_TOKEN if provided
    if token_set and hf_token:
        env_content["HF_TOKEN"] = hf_token
        print_status(f"{'Updated' if 'HF_TOKEN' in env_content else 'Added'} HF_TOKEN in .env file")
    elif "HF_TOKEN" in env_content:
        print_status("HF_TOKEN already set in .env file, preserving it")
    else:
        print_warning("No HF_TOKEN found in .env file")
        print_status("You'll need to set your Hugging Face token with 'python -m cli set_token' before downloading models")
    
    # Update or add LLM_MODEL_PATH
    env_content["LLM_MODEL_PATH"] = f"models/codellama-7b-instruct.{model_variant}.gguf"
    
    # Make sure we have default settings
    defaults = {
        "OUTPUT_DIR": "output",
        "LLM_TEMPERATURE": "0.7",
        "LLM_MAX_TOKENS": "4096",
        "LLM_CONTEXT_SIZE": "8192",
        "LLM_BATCH_SIZE": "512",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
    }
    
    # Add any missing defaults
    for key, value in defaults.items():
        if key not in env_content:
            env_content[key] = value
    
    # Write the updated .env file
    with open(".env", "w") as f:
        f.write("# Environment settings\n")
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    print_status(f"{'Updated' if os.path.isfile('.env') else 'Created'} .env file")

def prompt_for_hf_token():
    """Prompt user for Hugging Face token"""
    setup_hf_token = input("Would you like to set up your Hugging Face token now? (yes/no, default: yes): ").lower() or "yes"
    
    if setup_hf_token in ["yes", "y"]:
        print_status("To download models, you need a Hugging Face token from https://huggingface.co/settings/tokens")
        hf_token = input("Enter your Hugging Face token (press Enter to skip): ")
        
        if hf_token:
            # Set token in environment for immediate use
            os.environ["HF_TOKEN"] = hf_token
            return True, hf_token
        else:
            print_warning("No token provided. You'll need to set it later with 'python -m cli set_token'")
            return False, None
    else:
        print_status("You can set your token later with 'python -m cli set_token'")
        return False, None

def detect_system_resources():
    """Detect system resources and set appropriate optimization parameters"""
    global BATCH_SIZE, PARALLEL_JOBS, PIP_PARALLEL_JOBS
    
    # Detect RAM (in GB)
    mem_gb = 4  # Default
    if platform.system() == "Darwin":  # macOS
        try:
            mem_info = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            mem_gb = int(int(mem_info) / (1024**3))
        except:
            pass
    else:  # Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb // (1024**2)
                        break
        except:
            pass
    
    # Detect CPU count
    cpu_count = os.cpu_count() or 2
    
    # Scale parallel jobs based on available CPUs (leave 1 core free)
    PARALLEL_JOBS = max(1, cpu_count - 1)
    
    # Scale batch size based on memory
    if mem_gb >= 16:
        BATCH_SIZE = 10
    elif mem_gb >= 8:
        BATCH_SIZE = 8
    elif mem_gb >= 4:
        BATCH_SIZE = 5
    else:
        BATCH_SIZE = 3
    
    # Set pip parallelism based on CPU count
    PIP_PARALLEL_JOBS = max(1, cpu_count - 1)
    
    print_status(f"System resources detected: {mem_gb}GB RAM, {cpu_count} CPUs")
    print_status(f"Using optimization settings: batch size {BATCH_SIZE}, parallel jobs {PARALLEL_JOBS}, pip parallel {PIP_PARALLEL_JOBS}")

def setup_cache_dir():
    """Set up cache directory to speed up repeated installations"""
    if platform.system() == "Darwin":  # macOS
        cache_dir = os.path.expanduser("~/Library/Caches/aithreatmap")
    else:  # Linux
        cache_dir = os.path.expanduser("~/.cache/aithreatmap")
    
    # Create cache directories
    os.makedirs(f"{cache_dir}/pip", exist_ok=True)
    os.makedirs(f"{cache_dir}/wheels", exist_ok=True)
    
    # Set environment variables for pip
    os.environ["PIP_CACHE_DIR"] = f"{cache_dir}/pip"
    os.environ["PYTHONWHEELHOUSE"] = f"{cache_dir}/wheels"
    os.environ["PIP_WHEEL_DIR"] = f"{cache_dir}/wheels"
    
    print_status(f"Using cache directory: {cache_dir}")

def install_deps_in_batches(req_file):
    """Install dependencies in smaller batches to avoid hanging"""
    print_status("Installing dependencies in batches to prevent hanging...")
    
    # Check if file exists
    if not os.path.isfile(req_file):
        print_warning(f"Requirements file '{req_file}' not found")
        return False
    
    # Read and count non-comment, non-empty lines
    with open(req_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    
    total_lines = len(requirements)
    
    # If requirements file is empty, return
    if total_lines == 0:
        print_warning("Requirements file is empty")
        return False
    
    # Use dynamic batch size based on system resources
    batch_size = BATCH_SIZE
    batches = (total_lines + batch_size - 1) // batch_size  # Ceiling division
    
    print_status(f"Found {total_lines} packages, installing in {batches} batches with size {batch_size}")
    
    # Process each batch
    for i in range(batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_lines)
        batch_deps = requirements[start_idx:end_idx]
        
        # Skip if empty
        if not batch_deps:
            continue
        
        # Install this batch
        print_status(f"Installing batch {i+1}/{batches}...")
        cmd = f"{PYTHON_CMD} -m pip install -q --no-cache-dir -j{PIP_PARALLEL_JOBS} {' '.join(batch_deps)}"
        success, _ = run_command(cmd)
        if not success:
            print_warning(f"Some packages in batch {i+1} failed, continuing anyway")
        
        # Show progress
        show_progress(batches, i+1, f"Installing dependency batch {i+1}/{batches}")
    
    return True

def fix_sentence_transformers():
    """Apply fixes for sentence_transformers installation issues"""
    print_status("Checking sentence_transformers installation...")
    
    # First check if we can import it at all
    try:
        import sentence_transformers
        print_status("sentence_transformers seems to be installed correctly")
        return True
    except ImportError:
        print_warning("sentence_transformers issues detected, applying fixes...")
    
    # Fix 1: Try reinstalling with explicit dependencies
    print_status("Fix 1: Reinstalling with explicit dependencies...")
    run_command(f"{PYTHON_CMD} -m pip install -q --no-cache-dir --force-reinstall -U transformers torch numpy scipy")
    run_command(f"{PYTHON_CMD} -m pip install -q --no-cache-dir --force-reinstall -U sentence-transformers")
    
    # Try importing again
    try:
        import sentence_transformers
        print_success("Fix 1 succeeded!")
        return True
    except ImportError:
        pass
    
    # Fix 2: Try with a specific version
    print_status("Fix 2: Trying specific version...")
    run_command(f"{PYTHON_CMD} -m pip install -q --no-cache-dir --force-reinstall 'sentence-transformers==2.2.2'")
    
    # Try importing again
    try:
        import sentence_transformers
        print_success("Fix 2 succeeded!")
        return True
    except ImportError:
        pass
    
    # Fix 3: Try building from source
    print_status("Fix 3: Building from source...")
    run_command(f"{PYTHON_CMD} -m pip install -q --no-cache-dir git+https://github.com/UKPLab/sentence-transformers.git")
    
    # Final check
    try:
        import sentence_transformers
        print_success("Fix 3 succeeded!")
        return True
    except ImportError:
        print_error("All fixes failed for sentence_transformers")
        return False

def install_faiss():
    """Check if FAISS with GPU support is available"""
    print_status("Checking FAISS for vector search...")
    
    try:
        # Try to import faiss
        import faiss
        
        # Check if FAISS has GPU support
        cuda_available = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        has_indexflatl2 = hasattr(faiss, 'IndexFlatL2')
        
        if cuda_available:
            gpu_count = faiss.get_num_gpus()
            
            if has_indexflatl2:
                print_success(f"FAISS-GPU is installed with {gpu_count} GPUs and IndexFlatL2 support")
                return True
            else:
                print_warning("FAISS-GPU is installed but missing IndexFlatL2 attribute")
                return False
        elif has_indexflatl2:
            print_status("FAISS CPU version is installed with IndexFlatL2 support")
            return True
        else:
            print_warning("FAISS is installed but missing required functionality")
            return False
            
    except ImportError:
        print_warning("FAISS is not installed")
        print_status("To use vector search capabilities, you need to install FAISS:")
        print_status("  - For GPU support: pip install faiss-gpu==1.7.2")
        print_status("  - For CPU only: pip install faiss-cpu==1.10.0")
        return False
    except Exception as e:
        print_error(f"Error checking FAISS installation: {str(e)}")
        return False

def install_llama_cpp_python():
    """Install llama-cpp-python with appropriate CUDA support if available"""
    print_status("Installing llama-cpp-python...")
    
    # First check if we already have it
    try:
        import llama_cpp
        print_success("llama-cpp-python already installed")
        return True
    except:
        pass
    
    # Check for CUDA
    cuda_available = False
    cuda_compute = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            # Get CUDA compute capability
            device = torch.cuda.get_device_properties(0)
            cuda_compute = f"{device.major}.{device.minor}"
            print_success(f"CUDA detected with compute capability {cuda_compute}")
    except:
        # Try to detect CUDA via system
        if shutil.which("nvcc") or os.path.exists("/usr/local/cuda"):
            cuda_available = True
            print_success("CUDA detected via system!")
    
    # Check for Apple Silicon MPS
    mps_available = False
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
            print_success("Apple Silicon MPS support detected!")
    except:
        pass
    
    # Set up environment variables for llama-cpp compilation
    env_vars = os.environ.copy()
    
    if cuda_available:
        print_status("Configuring for CUDA compilation...")
        env_vars["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
        env_vars["FORCE_CMAKE"] = "1"
        
        if cuda_compute:
            env_vars["CUDA_COMPUTE"] = cuda_compute
    
    elif mps_available:
        print_status("Configuring for Apple Silicon Metal acceleration...")
        env_vars["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        env_vars["FORCE_CMAKE"] = "1"
    
    # Try installing with CUDA support
    print_status("Installing llama-cpp-python (this may take several minutes)...")
    
    if cuda_available or mps_available:
        cmd = f"{PYTHON_CMD} -m pip install llama-cpp-python --no-cache-dir --verbose"
        success, output = run_command(cmd, env=env_vars, silent=False)
        
        if not success:
            print_warning("Failed to install with GPU acceleration, falling back to CPU version")
            cmd = f"{PYTHON_CMD} -m pip install llama-cpp-python --no-cache-dir"
            success, _ = run_command(cmd)
    else:
        # CPU-only installation
        cmd = f"{PYTHON_CMD} -m pip install llama-cpp-python --no-cache-dir"
        success, _ = run_command(cmd)
    
    # Verify installation
    try:
        import llama_cpp
        print_success("llama-cpp-python installation successful!")
        return True
    except ImportError as e:
        print_error(f"llama-cpp-python installation failed: {str(e)}")
        return False

def verify_installation():
    """Verify that all critical packages are installed and working"""
    print_status("Verifying dependencies...")
    
    # Define verification script
    verify_script = """
import sys
import importlib

all_passed = True
critical_packages = [
    'click', 'langchain', 'langchain_core', 'langchain_community', 
    'tree_sitter', 'requests', 'tqdm', 'xmltodict', 'importlib.metadata',
    'huggingface_hub', 'joblib'
]

for pkg in critical_packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg} - {str(e)}')
        all_passed = False

# Special check for sentence_transformers - critical for embeddings
try:
    import sentence_transformers
    print(f'✓ sentence_transformers {sentence_transformers.__version__}')
except ImportError as e:
    print(f'✗ sentence_transformers - {str(e)}')
    print('  → This is critical for embeddings functionality')
    all_passed = False
except Exception as e:
    print(f'⚠ sentence_transformers - imports but has issues: {str(e)}')
    print('  → May have limited functionality')

# Special check for faiss - critical for vector search
try:
    import faiss
    has_gpu = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
    print(f'✓ faiss-{"gpu" if has_gpu else "cpu"}')
    if has_gpu:
        gpu_count = faiss.get_num_gpus()
        print(f'  → {gpu_count} GPUs available for FAISS')
except ImportError as e:
    print(f'✗ faiss-cpu - {str(e)}')
    print('  → This is critical for vector search functionality')
    all_passed = False
except Exception as e:
    print(f'⚠ faiss-cpu - imports but has issues: {str(e)}')
    print('  → May have limited functionality')

# Check llama-cpp-python
try:
    import llama_cpp
    print(f'✓ llama_cpp (Python binding for llama.cpp)')
except ImportError:
    print(f'○ llama_cpp - not installed or not found')
    print('  → LLM functionality may be limited')

# Check optional packages
optional_packages = [
    ('git', 'dulwich', 'Git support for remote repositories')
]

print('\\nOptional packages:')
for pkg_name, import_name, description in optional_packages:
    try:
        __import__(import_name)
        print(f'✓ {pkg_name} ({description})')
    except ImportError:
        print(f'○ {pkg_name} - not installed ({description})')

# Special check for torch with GPU support
try:
    import torch
    gpu_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if gpu_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else 'Unknown'
        print(f'✓ torch with CUDA support - {device_name} ({device_count} devices)')
    elif mps_available:
        print(f'✓ torch with MPS support (Apple Silicon)')
    else:
        print(f'✓ torch (CPU only)')
except ImportError:
    print('○ torch - not installed')
except Exception as e:
    print(f'⚠ torch - imports but has issues: {str(e)}')

sys.exit(0 if all_passed else 1)
"""
    
    # Run verification script
    with open("verify_install.py", "w") as f:
        f.write(verify_script)
    
    result = subprocess.run([PYTHON_CMD, "verify_install.py"], capture_output=True, text=True)
    print(result.stdout)
    
    # Clean up
    try:
        os.remove("verify_install.py")
    except:
        pass
    
    return result.returncode == 0

def install_dependencies():
    """Install all required dependencies"""
    print_status("Checking Python dependencies...")
    
    # If all critical components are already installed, we can skip
    if (is_already_installed("core_packages") and 
        is_already_installed("sentence_transformers") and 
        is_already_installed("tree_sitter") and 
        (is_already_installed("faiss_cpu") or is_already_installed("faiss_gpu")) and
        is_already_installed("llama_cpp_python")):
        print_success("All critical dependencies are already installed. Skipping installation.")
        return True
    
    print_status("Installing Python dependencies...")
    
    # Set up cache directories
    setup_cache_dir()
    
    # Detect system resources
    detect_system_resources()
    
    # Define total steps for progress tracking
    total_steps = 10  # Increased to account for new steps
    current_step = 0
    
    # Upgrade pip first
    current_step += 1
    show_progress(total_steps, current_step, "Upgrading pip")
    success, _ = run_command(f"{PYTHON_CMD} -m pip install --upgrade pip")
    if not success:
        print_error("Failed to upgrade pip")
        return False
    
    # Install build tools
    current_step += 1
    show_progress(total_steps, current_step, "Installing build tools")
    success, _ = run_command(f"{PYTHON_CMD} -m pip install -q wheel setuptools build")
    if not success:
        print_error("Failed to install build tools")
        return False
    
    # Try to install all requirements at once first
    current_step += 1
    show_progress(total_steps, current_step, "Installing dependencies with optimized settings")
    
    if os.path.isfile("requirements.txt"):
        print_status("Installing all packages with optimized settings...")
        cmd = f"{PYTHON_CMD} -m pip install -q --upgrade-strategy only-if-needed -j{PIP_PARALLEL_JOBS} -r requirements.txt"
        success, _ = run_command(cmd)
        if not success:
            print_warning("Optimized installation failed, falling back to batch installation")
            install_deps_in_batches("requirements.txt")
    else:
        print_warning("requirements.txt not found, skipping batch installation")
    
    # Install core packages (split into parallel operations in Python)
    if not is_already_installed("core_packages"):
        current_step += 1
        show_progress(total_steps, current_step, "Installing critical packages")
        
        # Install critical packages
        import threading
        threads = []
        
        cmd1 = f"{PYTHON_CMD} -m pip install -q click python-dotenv colorama tqdm"
        cmd2 = f"{PYTHON_CMD} -m pip install -q pydantic huggingface_hub joblib"
        cmd3 = f"{PYTHON_CMD} -m pip install -q requests"
        
        # Run in parallel using threading
        for cmd in [cmd1, cmd2, cmd3]:
            thread = threading.Thread(target=run_command, args=(cmd,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Install LangChain separately
        cmd_langchain = f"{PYTHON_CMD} -m pip install -q langchain langchain-core langchain-community"
        success, _ = run_command(cmd_langchain)
        if not success:
            print_error("Failed to install critical packages")
            return False
        
        mark_as_installed("core_packages")
    else:
        print_status("Core packages already installed, skipping...")
        current_step += 1
        show_progress(total_steps, current_step, "Critical packages already installed")
    
    # Install sentence-transformers prerequisites in parallel if needed
    if not is_already_installed("sentence_transformers"):
        current_step += 1
        show_progress(total_steps, current_step, "Installing transformers & torch (for sentence-transformers)")
        
        # Install prerequisites in parallel
        import threading
        threads = []
        
        cmd1 = f"{PYTHON_CMD} -m pip install -q transformers"
        cmd2 = f"{PYTHON_CMD} -m pip install -q torch torchvision"
        
        for cmd in [cmd1, cmd2]:
            thread = threading.Thread(target=run_command, args=(cmd,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Install sentence-transformers
        current_step += 1
        show_progress(total_steps, current_step, "Installing sentence-transformers")
        success, _ = run_command(f"{PYTHON_CMD} -m pip install -q sentence-transformers")
        
        # Apply fixes if needed
        if not success or not verify_sentence_transformers():
            fix_sentence_transformers()
        
        mark_as_installed("sentence_transformers")
    else:
        print_status("Sentence-transformers already installed, skipping...")
        # Skip two steps
        current_step += 2
        show_progress(total_steps, current_step, "Sentence transformers already installed")
    
    # Install tree-sitter versions specifically if needed
    if not is_already_installed("tree_sitter"):
        current_step += 1
        show_progress(total_steps, current_step, "Installing tree-sitter")
        success, _ = run_command(f"{PYTHON_CMD} -m pip install -q --force-reinstall tree-sitter==0.20.1 tree-sitter-languages==1.8.0")
        if not success:
            print_error("Failed to install tree-sitter")
            return False
        mark_as_installed("tree_sitter")
    else:
        current_step += 1
        show_progress(total_steps, current_step, "Tree-sitter already installed")
    
    # Install faiss for vector search if needed
    if not is_already_installed("faiss_cpu") and not is_already_installed("faiss_gpu"):
        current_step += 1
        show_progress(total_steps, current_step, "Checking FAISS for vector search")
        
        # Check FAISS installation
        if install_faiss():
            print_success("FAISS is already installed.")
        else:
            print_warning("FAISS is not installed. Vector search functionality will be limited.")
            print_warning("To install FAISS, run one of the following commands:")
            print_warning("  - For GPU support: pip install faiss-gpu==1.7.2")
            print_warning("  - For CPU only: pip install faiss-cpu==1.10.0")
    else:
        current_step += 1
        show_progress(total_steps, current_step, "FAISS already installed")
        if is_already_installed("faiss_gpu"):
            print_status("FAISS with GPU support is already installed")
        else:
            print_status("FAISS CPU version is already installed")
    
    # Install llama-cpp-python with acceleration if needed
    if not is_already_installed("llama_cpp_python"):
        current_step += 1
        show_progress(total_steps, current_step, "Installing llama-cpp-python with hardware acceleration")
        if install_llama_cpp_python():
            mark_as_installed("llama_cpp_python")
    else:
        current_step += 1
        show_progress(total_steps, current_step, "llama-cpp-python already installed")
    
    # Complete the progress bar
    current_step = total_steps
    show_progress(total_steps, current_step, "Dependencies installation completed")
    
    print_success("All dependencies installed!")
    return True

def verify_sentence_transformers():
    """Verify that sentence_transformers is installed correctly"""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False

def setup_venv():
    """Set up and activate a Python virtual environment"""
    global PYTHON_CMD
    
    # If we're already in the venv, no need to reactivate
    if is_in_our_venv():
        print_success(f"Already in virtual environment {VENV_NAME}")
        return True
    
    venv_path = os.path.join(os.getcwd(), VENV_NAME)
    
    if not os.path.exists(venv_path):
        print_status(f"Creating Python virtual environment in ./{VENV_NAME}...")
        try:
            venv.create(venv_path, with_pip=True)
            print_success("Virtual environment created!")
        except Exception as e:
            print_error(f"Failed to create virtual environment: {str(e)}")
            print_status("Try installing the venv module with: pip3 install virtualenv")
            return False
    else:
        print_status(f"Using existing virtual environment at ./{VENV_NAME}")
        
        # Check if marker file exists inside venv
        if os.path.isfile(f"{VENV_NAME}/.setup_complete"):
            print_status("This venv was previously set up successfully")
    
    # Create a marker file inside venv
    with open(f"{VENV_NAME}/.setup_complete", "w") as f:
        f.write("")
    
    # Update PYTHON_CMD to use the virtual environment's Python
    if platform.system() == "Windows":
        PYTHON_CMD = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        PYTHON_CMD = os.path.join(venv_path, "bin", "python")
    
    print_success("Virtual environment ready!")
    print_status(f"To activate this environment later, run: source {VENV_NAME}/bin/activate")
    
    return True

def handle_git_support():
    """Handle Git support installation"""
    if is_already_installed("git_support"):
        print_success("Git support already installed.")
        return True
    
    print_status("Git support is optional and only required for analyzing remote repositories.")
    if check_command("git"):
        print_status("Git is already installed on your system.")
        install_git = "yes"
    else:
        install_git = input("Would you like to install Git support for analyzing remote repositories? (yes/no, default: yes): ").lower() or "yes"
    
    if install_git in ["yes", "y"]:
        print_status("Installing Git support...")
        success, _ = run_command(f"{PYTHON_CMD} -m pip install -q dulwich==0.21.6")
        if success:
            print_success("Git support installed!")
            mark_as_installed("git_support")
        else:
            print_warning("Failed to install Git support")
    else:
        print_status("Skipping Git support. You'll only be able to analyze local repositories.")
    
    return True

def setup_multi_gpu():
    """Configure the environment for multi-GPU setups"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0
        
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            return False, gpu_count
        
        print_success(f"Detected {gpu_count} GPUs for multi-GPU setup")
        
        # Get GPU information
        gpu_info = []
        total_memory_gb = 0
        gpu_memory_allocation = []
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_memory_gb += memory_gb
            
            # Calculate memory allocation - use 80% of available memory by default
            allocation_gb = int(memory_gb * 0.8)
            gpu_memory_allocation.append(allocation_gb)
            
            gpu_info.append(f"GPU {i}: {name} ({memory_gb:.1f} GB, allocating {allocation_gb} GB)")
        
        print_status("\n".join(gpu_info))
        print_status(f"Total GPU memory: {total_memory_gb:.1f} GB")
        
        # Add multi-GPU configuration to .env
        with open(".env", "a") as f:
            f.write("\n# Multi-GPU Configuration\n")
            f.write(f"GPU_COUNT={gpu_count}\n")
            f.write("DISTRIBUTED=1\n")
            f.write(f"GPU_IDS={','.join([str(i) for i in range(gpu_count)])}\n")
            
            # Add GPU memory allocation settings
            f.write(f"GPU_MEMORY_ALLOCATION={','.join([str(mem) for mem in gpu_memory_allocation])}\n")
            f.write("TENSOR_PARALLEL=1\n")
            f.write("MAX_BATCH_SIZE=32\n")
            f.write("GPU_MEMORY_UTILIZATION=0.8\n")
            
            # Add tensor dimensions for large models
            f.write("EMBEDDING_DIMENSION=4096\n")
            f.write("CONTEXT_SIZE=32768\n")
            
            # If we have more than 32GB total, suggest 70B model
            if total_memory_gb > 32:
                print_status("Detected sufficient GPU memory for CodeLlama-70B model")
                f.write("# Uncomment to use 70B model\n")
                f.write("LLM_MODEL=codellama-70b-instruct\n")
            
            # For 50GB+ GPUs, enable more aggressive memory allocation
            if any(torch.cuda.get_device_properties(i).total_memory / (1024**3) >= 50 for i in range(gpu_count)):
                print_status("Detected high-memory GPUs (50GB+), enabling optimized memory settings")
                f.write("\n# High-memory GPU optimization\n")
                f.write("HIGH_MEM_GPUS=1\n")
                f.write("TENSOR_SIZE_MULTIPLIER=10\n")  # 10x larger tensors as requested
                f.write("MODEL_PARALLEL=1\n")
                f.write("KV_CACHE_ENABLED=1\n")
                f.write("BATCH_SIZE=64\n")  # Increase batch size for processing
        
        print_success("Multi-GPU configuration set up successfully")
        return True, gpu_count
    
    except Exception as e:
        print_warning(f"Failed to set up multi-GPU configuration: {str(e)}")
        return False, 0

def main():
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print_warning("\nInstallation interrupted. You can resume by running the script again.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print_status(f"AI Threat Model Map Generator Installer v1.2.0")
    print_status(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    
    # Detect system resources for optimization
    detect_system_resources()
    
    # Set up cache directories
    setup_cache_dir()
    
    # Create required directories
    create_directories()
    
    # Set up Python environment
    if not is_in_our_venv():
        print_status("Setting up virtual environment...")
        setup_venv()
        print_success("Virtual environment created. Please run the installer again from within the virtual environment.")
        print_status(f"On Linux/macOS: source {VENV_NAME}/bin/activate")
        print_status(f"On Windows: .\\{VENV_NAME}\\Scripts\\activate")
        print_status(f"Then run: python install.py")
        sys.exit(0)
    
    # Detect architecture for model selection
    model_variant = detect_architecture()
    
    # Prompt for Hugging Face token
    token_set, hf_token = prompt_for_hf_token()
    
    # Install dependencies
    print_status("Installing dependencies...")
    install_dependencies()
    
    # Specific installations and fixes
    install_faiss()
    install_llama_cpp_python()
    fix_sentence_transformers()
    
    # Set up multi-GPU if available
    multi_gpu, gpu_count = setup_multi_gpu()
    
    # Set up .env file with configuration
    setup_env_file(model_variant, hf_token, token_set)
    
    # Final verification
    if verify_installation():
        print_success("Installation completed successfully!")
        
        print_status("\nNext steps:")
        print_status("1. Download models: python -m cli init")
        if multi_gpu:
            print_status("2. For multi-GPU analysis: python -m cli analyze path/to/repo --distributed")
        else:
            print_status("2. Analyze a repository: python -m cli analyze path/to/repo")
    else:
        print_warning("Installation completed with some warnings. Some components may not work as expected.")
        print_status("You can try manually fixing issues or re-running the installer.")
    
    return 0

if __name__ == "__main__":
    main() 