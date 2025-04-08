#!/bin/bash

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Progress bar function
show_progress() {
    local total=$1
    local step=$2
    local width=50
    local percent=$((step * 100 / total))
    local completed=$((width * step / total))
    local remaining=$((width - completed))
    local bar=$(printf "%${completed}s" | tr ' ' '=')
    local empty=$(printf "%${remaining}s" | tr ' ' ' ')
    printf "\r[%s%s] %d%% - %s" "$bar" "$empty" "$percent" "$3"
    if [ $step -eq $total ]; then
        printf "\n"
    fi
}

# Define output functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Define helper functions
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it and try again."
        return 1
    fi
    return 0
}

create_directories() {
    print_status "Creating required directories..."
    mkdir -p output models visualizer/templates
    print_success "Directories created"
}

detect_architecture() {
    print_status "Detecting system architecture..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        MODEL_VARIANT="Q4_0"
        print_status "Detected ARM architecture, will use Q4_0 model variant"
    else
        MODEL_VARIANT="Q4_K_M"
        print_status "Detected x86 architecture, will use Q4_K_M model variant"
    fi
    echo "$MODEL_VARIANT"
}

setup_env_file() {
    local model_variant=$1
    local hf_token=$2
    local token_set=$3

    if [ -f ".env" ]; then
        print_status "Found existing .env file"
        
        # Update or add HF_TOKEN if provided
        if [ "$token_set" = true ]; then
            if grep -q "HF_TOKEN=" .env; then
                sed -i'' -e "s|HF_TOKEN=.*|HF_TOKEN=$hf_token|" .env
                print_status "Updated HF_TOKEN in .env file"
            else
                echo "HF_TOKEN=$hf_token" >> .env
                print_status "Added HF_TOKEN to .env file"
            fi
        else
            # Check if token already exists in file
            if grep -q "HF_TOKEN=" .env; then
                print_status "HF_TOKEN already set in .env file, preserving it"
            else
                print_warning "No HF_TOKEN found in .env file"
                print_status "You'll need to set your Hugging Face token with 'python -m cli set_token' before downloading models"
            fi
        fi
        
        # Update model path based on detected architecture
        if grep -q "LLM_MODEL_PATH=" .env; then
            sed -i'' -e "s|LLM_MODEL_PATH=.*|LLM_MODEL_PATH=models/codellama-7b-instruct.$model_variant.gguf|" .env
            print_status "Updated LLM_MODEL_PATH in .env file to use $model_variant model"
        else
            # Add model path if it doesn't exist
            echo "LLM_MODEL_PATH=models/codellama-7b-instruct.$model_variant.gguf" >> .env
            print_status "Added LLM_MODEL_PATH to .env file"
        fi
    else
        # Create new .env file with default settings
        print_status "Creating .env file with default settings..."
        cat > .env << EOL
# Environment settings
OUTPUT_DIR=output
LLM_MODEL_PATH=models/codellama-7b-instruct.$model_variant.gguf
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
LLM_CONTEXT_SIZE=8192
LLM_BATCH_SIZE=512
EMBEDDING_MODEL=all-MiniLM-L6-v2
EOL

        # Add token if provided
        if [ "$token_set" = true ]; then
            echo "HF_TOKEN=$hf_token" >> .env
            print_status "Added HF_TOKEN to .env file"
        else
            print_warning "No HF_TOKEN set. You'll need to set your Hugging Face token with 'python -m cli set_token' before downloading models"
        fi
        
        print_status "Created default .env file"
    fi
}

prompt_for_hf_token() {
    read -p "Would you like to set up your Hugging Face token now? (yes/no, default: yes): " SETUP_HF_TOKEN
    SETUP_HF_TOKEN=${SETUP_HF_TOKEN:-yes}

    if [[ "$SETUP_HF_TOKEN" =~ ^[Yy][Ee][Ss]$ ]]; then
        print_status "To download models, you need a Hugging Face token from https://huggingface.co/settings/tokens"
        read -p "Enter your Hugging Face token (press Enter to skip): " HF_TOKEN
        
        if [ -n "$HF_TOKEN" ]; then
            # Set token in environment for immediate use
            export HF_TOKEN="$HF_TOKEN"
            echo "true $HF_TOKEN"
        else
            print_warning "No token provided. You'll need to set it later with 'python -m cli set_token'"
            echo "false"
        fi
    else
        print_status "You can set your token later with 'python -m cli set_token'"
        echo "false"
    fi
}

verify_installation() {
    print_status "Verifying dependencies..."
    
    # Simpler verification that just tests imports of core packages
    python -c "
import sys
import importlib
import traceback

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
    print(f'✓ faiss-cpu')
except ImportError as e:
    print(f'✗ faiss-cpu - {str(e)}')
    print('  → This is critical for vector search functionality')
    all_passed = False
except Exception as e:
    print(f'⚠ faiss-cpu - imports but has issues: {str(e)}')
    print('  → May have limited functionality')

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
"
    return $?
}

install_deps_in_batches() {
    # This function installs dependencies in smaller batches to avoid hanging
    local req_file=$1
    
    print_status "Installing dependencies in batches to prevent hanging..."
    
    # Count lines in requirements file
    local total_lines=$(grep -v "^#" "$req_file" | grep -v "^$" | wc -l | tr -d ' ')
    
    # If requirements file is empty or missing, return
    if [ "$total_lines" -eq 0 ]; then
        print_warning "Requirements file is empty or missing"
        return 1
    fi
    
    # Split into smaller batches (about 5 packages per batch)
    local batch_size=5
    local batches=$((total_lines / batch_size + 1))
    
    print_status "Found $total_lines packages, installing in $batches batches"
    
    # Process each batch
    for ((i=1; i<=batches; i++)); do
        local start_line=$(( (i-1) * batch_size + 1 ))
        
        # Extract batch of dependencies
        local deps=$(grep -v "^#" "$req_file" | grep -v "^$" | sed -n "${start_line},+$((batch_size-1))p")
        
        # Skip if empty
        if [ -z "$deps" ]; then
            continue
        fi
        
        # Install this batch
        print_status "Installing batch $i/$batches..."
        echo "$deps" | xargs pip install -q --no-cache-dir >/dev/null 2>&1 || {
            print_warning "Some packages in batch $i failed, continuing anyway"
        }
        
        # Show progress
        show_progress $batches $i "Installing dependency batch $i/$batches"
    done
    
    return 0
}

fix_sentence_transformers() {
    print_status "Applying fix for sentence_transformers..."
    
    # First check if we can import it at all
    if python -c "import sentence_transformers" 2>/dev/null; then
        print_status "sentence_transformers seems to be installed correctly"
        return 0
    fi
    
    print_warning "sentence_transformers issues detected, applying fixes..."
    
    # Fix 1: Try reinstalling with explicit dependencies
    print_status "Fix 1: Reinstalling with explicit dependencies..."
    pip install -q --no-cache-dir --force-reinstall -U transformers torch numpy scipy >/dev/null 2>&1
    pip install -q --no-cache-dir --force-reinstall -U sentence-transformers >/dev/null 2>&1
    
    # Try importing again
    if python -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 1 succeeded!"
        return 0
    fi
    
    # Fix 2: Try with a specific version
    print_status "Fix 2: Trying specific version..."
    pip install -q --no-cache-dir --force-reinstall "sentence-transformers==2.2.2" >/dev/null 2>&1
    
    # Try importing again
    if python -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 2 succeeded!"
        return 0
    fi
    
    # Fix 3: Try building from source
    print_status "Fix 3: Building from source..."
    pip install -q --no-cache-dir git+https://github.com/UKPLab/sentence-transformers.git >/dev/null 2>&1
    
    # Final check
    if python -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 3 succeeded!"
        return 0
    fi
    
    print_error "All fixes failed for sentence_transformers"
    return 1
}

install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Define total steps for progress tracking
    local total_steps=9  # Increased to 9 to account for transformer fixes
    local current_step=0
    
    # Upgrade pip first
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Upgrading pip"
    python -m pip install --upgrade pip >/dev/null 2>&1 || {
        print_error "Failed to upgrade pip"
        return 1
    }
    
    # Install build tools
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing build tools"
    pip install -q wheel setuptools build >/dev/null 2>&1 || {
        print_error "Failed to install build tools"
        return 1
    }
    
    # Use optimized installation for dependencies
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing dependencies with optimized settings"
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing all packages with optimized settings..."
        pip install -q --no-cache-dir --upgrade-strategy only-if-needed -j8 -r requirements.txt >/dev/null 2>&1 || {
            print_warning "Optimized installation failed, falling back to batch installation"
            install_deps_in_batches "requirements.txt"
        }
    else
        print_warning "requirements.txt not found, skipping batch installation"
    fi
    
    # Install critical packages separately to ensure they're installed
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing critical packages"
    pip install -q --no-cache-dir click python-dotenv requests colorama tqdm langchain langchain-core langchain-community pydantic huggingface_hub joblib >/dev/null 2>&1 || {
        print_error "Failed to install critical packages"
        return 1
    }
    
    # Install sentence-transformers prerequisites
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing transformers & torch (for sentence-transformers)"
    pip install -q --no-cache-dir transformers torch torchvision >/dev/null 2>&1 || {
        print_warning "Failed to install transformers prerequisites"
    }
    
    # Install sentence-transformers specifically - this is critical for embeddings
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing sentence-transformers"
    pip install -q --no-cache-dir sentence-transformers >/dev/null 2>&1 || {
        print_warning "Failed to install sentence-transformers. Trying alternative method..."
        pip install -q --no-cache-dir "sentence-transformers>=2.2.2" >/dev/null 2>&1 || {
            print_error "Failed to install sentence-transformers. Embedding functionality will be limited."
        }
    }
    
    # Apply fixes for sentence-transformers if needed
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Fixing sentence-transformers issues"
    fix_sentence_transformers || {
        print_warning "Could not fully resolve sentence-transformers issues. Some functionality may be limited."
    }
    
    # Install accelerate for model loading
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing accelerate for model loading"
    pip install -q --no-cache-dir accelerate >/dev/null 2>&1 || {
        print_warning "Failed to install accelerate. Model loading might encounter issues."
    }
    
    # Install tree-sitter versions specifically
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing tree-sitter"
    pip install -q --no-cache-dir --force-reinstall tree-sitter==0.20.1 tree-sitter-languages==1.8.0 >/dev/null 2>&1 || {
        print_error "Failed to install tree-sitter"
        return 1
    }
    
    # Install faiss for vector search
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing faiss for vector search"
    pip install -q --no-cache-dir faiss-cpu >/dev/null 2>&1 || {
        print_warning "Failed to install faiss-cpu, trying alternative method..."
        pip install -q --no-cache-dir -U "faiss-cpu>=1.7.3" >/dev/null 2>&1 || {
            print_error "Failed to install faiss-cpu. Vector search functionality will be limited."
        }
    }
    
    # Install llama-cpp-python with CUDA support if available
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing llama-cpp-python with hardware acceleration"
    
    # Check if we're on a system with CUDA
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        print_status "CUDA detected, installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --no-cache-dir --force-reinstall llama-cpp-python >/dev/null 2>&1 && \
            print_success "Installed llama-cpp-python with CUDA support!" || \
            print_warning "Failed to install with CUDA. Installing standard version..."
    # Check if we're on a Mac with Metal (Apple Silicon)
    elif [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "arm64" ]; then
        print_status "Apple Silicon detected, installing llama-cpp-python with Metal support..."
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python >/dev/null 2>&1 && \
            print_success "Installed llama-cpp-python with Metal support!" || \
            print_warning "Failed to install with Metal. Installing standard version..."
    else
        print_status "Installing standard llama-cpp-python..."
        pip install --no-cache-dir --force-reinstall llama-cpp-python >/dev/null 2>&1 || \
            print_warning "Failed to install llama-cpp-python."
    fi
    
    print_success "All dependencies installed!"
    return 0
}

handle_git_support() {
    print_status "Git support is optional and only required for analyzing remote repositories."
    if command -v git &> /dev/null; then
        print_status "Git is already installed on your system."
        INSTALL_GIT="yes"
    else
        read -p "Would you like to install Git support for analyzing remote repositories? (yes/no, default: yes): " INSTALL_GIT
        INSTALL_GIT=${INSTALL_GIT:-yes}
    fi

    if [[ "$INSTALL_GIT" =~ ^[Yy][Ee][Ss]$ ]]; then
        print_status "Installing Git support..."
        pip install -q --no-cache-dir dulwich==0.21.6 >/dev/null 2>&1 && print_success "Git support installed!" || print_warning "Failed to install Git support"
    else
        print_status "Skipping Git support. You'll only be able to analyze local repositories."
    fi
}

setup_venv() {
    VENV_NAME="venv311"
    
    if [ ! -d "$VENV_NAME" ]; then
        print_status "Creating Python virtual environment in ./$VENV_NAME..."
        $PYTHON_CMD -m venv $VENV_NAME || {
            print_error "Failed to create virtual environment."
            print_status "Try installing the venv module with: pip3 install virtualenv"
            return 1
        }
        print_success "Virtual environment created!"
    else
        print_status "Using existing virtual environment at ./$VENV_NAME"
    fi
    
    # Activate the virtual environment
    print_status "Activating virtual environment..."
    source $VENV_NAME/bin/activate || {
        print_error "Failed to activate virtual environment."
        return 1
    }
    
    print_success "Virtual environment activated!"
    return 0
}

# Main execution starts here
print_status "Starting setup..."
echo "================================================================="
printf "          ${GREEN}AI Threat Map${NC} - ${BLUE}Installation${NC}\n"
echo "================================================================="
echo ""

# Setup steps tracker
TOTAL_SETUP_STEPS=8
CURRENT_STEP=0

# Check for Python
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Checking Python installation"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    print_success "Python 3.11 is installed!"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Python version: $PY_VERSION"
    
    if [[ "$PY_VERSION" != "3.11" ]]; then
        print_warning "You're using Python $PY_VERSION, but Python 3.11 is recommended for best compatibility."
        print_status "The setup will continue, but you may encounter dependency issues."
        sleep 1
    else
        print_success "Python 3.11 is installed!"
    fi
else
    print_error "Python 3 is not installed. Please install Python 3.11 and try again."
    exit 1
fi

# Check for pip
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Checking pip installation"
check_command pip3 || exit 1
print_success "pip3 is installed!"

# Setup virtual environment
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Setting up virtual environment"
setup_venv || exit 1

# Install dependencies
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Installing dependencies"
install_dependencies || {
    print_error "Failed to install critical dependencies. Exiting."
    exit 1
}

# Handle Git support
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Configuring Git support"
handle_git_support

# Create required directories
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Creating directories"
create_directories

# Detect architecture and setup environment
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Detecting architecture & setting up environment"
MODEL_VARIANT=$(detect_architecture)
TOKEN_INFO=$(prompt_for_hf_token)
read HF_TOKEN_SET HF_TOKEN <<< "$TOKEN_INFO"
setup_env_file "$MODEL_VARIANT" "$HF_TOKEN" "$HF_TOKEN_SET"

# Initialize the framework
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Initializing framework"
print_status "Initializing the framework..."
python -m cli init 2>/dev/null || print_warning "Framework initialization encountered issues, but we can continue"

# Verify installation
verify_installation
if [ $? -ne 0 ]; then
    print_error "Some critical dependencies are missing. There may have been installation errors."
    print_status "You can try installing them manually:"
    print_status "pip install click langchain langchain-core langchain-community python-dotenv tree-sitter==0.20.1 tree-sitter-languages==1.8.0 requests tqdm xmltodict==0.13.0 click-plugins XlsxWriter faiss-cpu==1.7.4"
    exit 1
fi

# Final success message and instructions
echo ""
echo "================================================================="
printf "      ${GREEN}✓ Setup completed successfully!${NC}\n"
echo "================================================================="
echo ""
print_status "You can now run the tool with: python -m cli analyze <repository_url>"
print_status ""
print_status "IMPORTANT: To use AI Threat Map in the future, you must first activate the virtual environment:"
print_status "  source $VENV_NAME/bin/activate"
print_status ""
if [ "$HF_TOKEN_SET" != "true" ]; then
    print_status "If you haven't set up your Hugging Face token yet, do that first:"
    print_status "  python -m cli set_token"
    print_status ""
fi
print_status "Models are downloaded automatically by the Python code using huggingface_hub."
print_status "You can also manually download models with:"
print_status "  python example_hf_download.py"
print_status ""
print_status "After activation, you can run commands like:"
print_status "  python -m cli analyze <repository_url>"
print_status ""
print_status "You can create a simple alias in your shell config (~/.bashrc or ~/.zshrc):"
print_status "  alias aithreatmap='cd $(pwd) && source $VENV_NAME/bin/activate && python -m cli'"
print_status ""
print_status "Then use it like: aithreatmap analyze <repository_url>" 