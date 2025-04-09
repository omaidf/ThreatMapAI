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

def run_command(cmd, silent=True):
    """Run a shell command and return the output and success status"""
    try:
        if silent:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            result = subprocess.run(cmd, shell=True, check=True)
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
    """Install FAISS with GPU support if available"""
    print_status("Installing FAISS for vector search...")
    
    # First ensure we have a compatible NumPy version (< 2.0)
    print_status("Installing compatible NumPy for FAISS...")
    run_command(f"{PYTHON_CMD} -m pip install 'numpy<2.0.0'")
    
    # Check if PyTorch with CUDA is available for GPU version
    cuda_available = False
    print_status("Checking for CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            print_success("CUDA detected via PyTorch! Will install GPU-enabled FAISS.")
    except ImportError:
        # If torch isn't installed, check for CUDA via system
        if shutil.which("nvcc") or os.path.exists("/usr/local/cuda"):
            cuda_available = True
            print_success("CUDA detected via nvcc/cuda directory! Will install GPU-enabled FAISS.")
        else:
            print_status("No CUDA detected, using CPU version of FAISS.")
    
    # Install appropriate FAISS version
    if cuda_available:
        # Install PyTorch with CUDA first if needed
        try:
            import torch
        except ImportError:
            print_status("Installing PyTorch with CUDA support first...")
            run_command(f"{PYTHON_CMD} -m pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        # Install faiss-gpu
        print_status("Installing FAISS with GPU support...")
        success, _ = run_command(f"{PYTHON_CMD} -m pip install -q faiss-gpu")
        
        if not success:
            print_warning("Failed to install faiss-gpu, trying with specific CUDA version...")
            # Try with specific pinned version
            success, _ = run_command(f"{PYTHON_CMD} -m pip install -q 'faiss-gpu>=1.7.0'")
            
            if not success:
                print_warning("Failed to install faiss-gpu. Falling back to CPU version.")
                run_command(f"{PYTHON_CMD} -m pip install -q faiss-cpu")
        
        # Verify GPU FAISS installation
        try:
            import faiss
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                print_success("Successfully installed FAISS with GPU support!")
                detected_gpus = faiss.get_num_gpus()
                print_status(f"Detected {detected_gpus} GPUs for FAISS acceleration")
                mark_as_installed("faiss_gpu")
                
                # Set environment variable for the application
                os.environ["FAISS_MULTI_GPU"] = "1" if detected_gpus > 1 else "0"
                
                return True
            else:
                print_warning("FAISS installed but GPU support not detected. Using CPU version.")
                mark_as_installed("faiss_cpu")
        except ImportError:
            print_error("Failed to import FAISS after installation. Vector search will be limited.")
    else:
        # Now install faiss-cpu
        print_status("Installing FAISS CPU version...")
        success, _ = run_command(f"{PYTHON_CMD} -m pip install -q faiss-cpu")
        
        if not success:
            print_warning("Failed to install faiss-cpu, trying alternative method...")
            # Try with specific pinned version
            success, _ = run_command(f"{PYTHON_CMD} -m pip install -q 'faiss-cpu==1.7.4'")
            
            if not success:
                # Try with known-compatible versions
                run_command(f"{PYTHON_CMD} -m pip install -q 'numpy==1.24.3'")
                run_command(f"{PYTHON_CMD} -m pip install -q 'faiss-cpu==1.7.4'")
        
        # Verify faiss installation
        try:
            import faiss
            print_success("Successfully installed FAISS CPU version")
            mark_as_installed("faiss_cpu")
            return True
        except ImportError:
            print_error("Failed to import FAISS after installation. Vector search will be limited.")
            return False

def install_llama_cpp_python():
    """Install llama-cpp-python with hardware acceleration if available"""
    print_status("Installing llama-cpp-python...")
    
    # Check if we're on a system with CUDA
    if shutil.which("nvcc") or os.path.exists("/usr/local/cuda"):
        print_status("CUDA detected, installing llama-cpp-python with CUDA support...")
        os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
        success, _ = run_command(f"{PYTHON_CMD} -m pip install --force-reinstall llama-cpp-python")
        
        if success:
            print_success("Installed llama-cpp-python with CUDA support!")
            return True
        else:
            print_warning("Failed to install with CUDA. Installing standard version...")
    # Check if we're on a Mac with Metal (Apple Silicon)
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        print_status("Apple Silicon detected, installing llama-cpp-python with Metal support...")
        os.environ["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        success, _ = run_command(f"{PYTHON_CMD} -m pip install --force-reinstall llama-cpp-python")
        
        if success:
            print_success("Installed llama-cpp-python with Metal support!")
            return True
        else:
            print_warning("Failed to install with Metal. Installing standard version...")
    
    # Fall back to standard version
    print_status("Installing standard llama-cpp-python...")
    success, _ = run_command(f"{PYTHON_CMD} -m pip install --force-reinstall llama-cpp-python")
    
    if success:
        print_success("Installed standard llama-cpp-python")
        return True
    else:
        print_warning("Failed to install llama-cpp-python.")
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
    
    # Install FAISS for vector search if needed
    if not is_already_installed("faiss_cpu") and not is_already_installed("faiss_gpu"):
        current_step += 1
        show_progress(total_steps, current_step, "Installing FAISS for vector search")
        install_faiss()
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

def main():
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print_warning("\nInterrupted! Cleaning up...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the setup
    print_status("Starting setup...")
    print("=================================================================")
    print(f"          {GREEN}AI Threat Map{NC} - {BLUE}Installation{NC}")
    print("=================================================================")
    print("")
    
    # Setup steps tracker
    total_setup_steps = 8
    current_step = 0
    
    # Check for Python
    current_step += 1
    show_progress(total_setup_steps, current_step, "Checking Python installation")
    
    if platform.python_version().startswith("3.11"):
        print_success("Python 3.11 is installed!")
    else:
        py_version = platform.python_version()
        print_status(f"Python version: {py_version}")
        print_warning(f"You're using Python {py_version}, but Python 3.11 is recommended for best compatibility.")
        print_status("The setup will continue, but you may encounter dependency issues.")
        time.sleep(1)
    
    # Check for pip
    current_step += 1
    show_progress(total_setup_steps, current_step, "Checking pip installation")
    
    if check_command("pip3") or check_command("pip"):
        print_success("pip is installed!")
    else:
        print_error("pip is not installed. Please install pip and try again.")
        sys.exit(1)
    
    # Setup virtual environment
    current_step += 1
    show_progress(total_setup_steps, current_step, "Setting up virtual environment")
    if not setup_venv():
        sys.exit(1)
    
    # Install dependencies
    current_step += 1
    show_progress(total_setup_steps, current_step, "Installing dependencies")
    if not install_dependencies():
        print_error("Failed to install critical dependencies. Exiting.")
        sys.exit(1)
    
    # Handle Git support
    current_step += 1
    show_progress(total_setup_steps, current_step, "Configuring Git support")
    handle_git_support()
    
    # Create required directories
    current_step += 1
    show_progress(total_setup_steps, current_step, "Creating directories")
    create_directories()
    
    # Detect architecture and setup environment
    current_step += 1
    show_progress(total_setup_steps, current_step, "Detecting architecture & setting up environment")
    model_variant = detect_architecture()
    token_set, hf_token = prompt_for_hf_token()
    setup_env_file(model_variant, hf_token, token_set)
    
    # Initialize the framework
    current_step += 1
    show_progress(total_setup_steps, current_step, "Initializing framework")
    print_status("Initializing the framework...")
    success, _ = run_command(f"{PYTHON_CMD} -m cli init")
    if not success:
        print_warning("Framework initialization encountered issues, but we can continue")
    
    # Verify installation
    if not verify_installation():
        print_error("Some critical dependencies are missing. There may have been installation errors.")
        print_status("You can try installing them manually:")
        print_status("pip install click langchain langchain-core langchain-community python-dotenv tree-sitter==0.20.1 tree-sitter-languages==1.8.0 requests tqdm xmltodict==0.13.0 click-plugins XlsxWriter faiss-cpu==1.7.4")
        sys.exit(1)
    
    # Final success message and instructions
    print("")
    print("=================================================================")
    print(f"      {GREEN}✓ Setup completed successfully!{NC}")
    print("=================================================================")
    print("")
    print_status("You can now run the tool with: python -m cli analyze <repository_url>")
    print_status("")
    print_status(f"IMPORTANT: To use AI Threat Map in the future, you must first activate the virtual environment:")
    print_status(f"  source {VENV_NAME}/bin/activate")
    print_status("")
    if not token_set:
        print_status("If you haven't set up your Hugging Face token yet, do that first:")
        print_status("  python -m cli set_token")
        print_status("")
    print_status("Models are downloaded automatically by the Python code using huggingface_hub.")
    print_status("You can also manually download models with:")
    print_status("  python example_hf_download.py")
    print_status("")
    print_status("After activation, you can run commands like:")
    print_status("  python -m cli analyze <repository_url>")
    print_status("")
    print_status("You can create a simple alias in your shell config (~/.bashrc or ~/.zshrc):")
    print_status(f"  alias aithreatmap='cd {os.getcwd()} && source {VENV_NAME}/bin/activate && python -m cli'")
    print_status("")
    print_status("Then use it like: aithreatmap analyze <repository_url>")

if __name__ == "__main__":
    main() 