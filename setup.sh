#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize default values for variables (will be adjusted based on system detection)
BATCH_SIZE=5
PARALLEL_JOBS=2
PIP_PARALLEL_JOBS=2

# Create a checkpoint file to track installation progress
CHECKPOINT_FILE=".setup_checkpoints"
VENV_NAME="venv311"

# Check if a component is already installed
is_already_installed() {
    local component=$1
    
    # Check if checkpoint file exists and contains the component
    if [ -f "$CHECKPOINT_FILE" ] && grep -q "^${component}=installed$" "$CHECKPOINT_FILE"; then
        return 0  # Already installed
    fi
    return 1  # Not installed
}

# Mark a component as installed
mark_as_installed() {
    local component=$1
    
    # Create checkpoint file if it doesn't exist
    touch "$CHECKPOINT_FILE"
    
    # Add or update the component in the checkpoint file
    if grep -q "^${component}=" "$CHECKPOINT_FILE"; then
        # Update existing entry
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS requires different sed syntax
            sed -i '' -e "s/^${component}=.*/${component}=installed/" "$CHECKPOINT_FILE"
        else
            # Linux
            sed -i -e "s/^${component}=.*/${component}=installed/" "$CHECKPOINT_FILE"
        fi
    else
        # Add new entry
        echo "${component}=installed" >> "$CHECKPOINT_FILE"
    fi
}

# Check if we're in a venv that we created previously
is_in_our_venv() {
    if [ -n "$VIRTUAL_ENV" ] && [[ "$VIRTUAL_ENV" == *"$VENV_NAME"* ]]; then
        return 0  # Yes, we're in our venv
    fi
    return 1  # No, we're not in our venv
}

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
    if ! command -v "$1" &> /dev/null; then
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
    local model_variant="$1"
    local hf_token="$2"
    local token_set="$3"

    if [ -f ".env" ]; then
        print_status "Found existing .env file"
        
        # Update or add HF_TOKEN if provided
        if [ "$token_set" = true ]; then
            if grep -q "HF_TOKEN=" .env; then
                # Use different sed syntax for macOS vs Linux
                if [[ "$(uname)" == "Darwin" ]]; then
                    sed -i '' -e "s|HF_TOKEN=.*|HF_TOKEN=$hf_token|" .env
                else
                    sed -i -e "s|HF_TOKEN=.*|HF_TOKEN=$hf_token|" .env
                fi
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
            # Use different sed syntax for macOS vs Linux
            if [[ "$(uname)" == "Darwin" ]]; then
                sed -i '' -e "s|LLM_MODEL_PATH=.*|LLM_MODEL_PATH=models/codellama-7b-instruct.$model_variant.gguf|" .env
            else
                sed -i -e "s|LLM_MODEL_PATH=.*|LLM_MODEL_PATH=models/codellama-7b-instruct.$model_variant.gguf|" .env
            fi
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
    print_status "Verifying installation..."
    
    MISSING_DEPS=()
    
    # Check for critical dependencies
    if ! "$PYTHON_CMD" -c "import click" 2>/dev/null; then
        MISSING_DEPS+=("click")
    fi
    
    if ! "$PYTHON_CMD" -c "import langchain" 2>/dev/null; then
        MISSING_DEPS+=("langchain and langchain-core")
    fi
    
    if ! "$PYTHON_CMD" -c "import dotenv" 2>/dev/null; then
        MISSING_DEPS+=("python-dotenv")
    fi
    
    if ! "$PYTHON_CMD" -c "import tree_sitter" 2>/dev/null; then
        MISSING_DEPS+=("tree-sitter")
    fi
    
    # Check for sentence_transformers
    if ! "$PYTHON_CMD" -c "import sentence_transformers" 2>/dev/null; then
        MISSING_DEPS+=("sentence_transformers")
    fi
    
    # Check for FAISS (either CPU or GPU version is fine)
    if ! "$PYTHON_CMD" -c "import faiss" 2>/dev/null; then
        MISSING_DEPS+=("faiss-cpu or faiss-gpu")
    else
        # Check if FAISS has GPU support
        if "$PYTHON_CMD" -c "import faiss; print(hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0)" 2>/dev/null | grep -q "True"; then
            print_success "FAISS with GPU support verified!"
            mark_as_installed "faiss_gpu"
            # Make sure we don't have both markers
            rm -f "$INSTALL_MARKERS_DIR/faiss_cpu"
        else
            print_status "FAISS CPU version verified."
            mark_as_installed "faiss_cpu"
            # Make sure we don't have both markers
            rm -f "$INSTALL_MARKERS_DIR/faiss_gpu"
        fi
    fi
    
    # Missing dependencies check
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_error "The following critical dependencies are missing:"
        for dep in "${MISSING_DEPS[@]}"; do
            print_error "  - $dep"
        done
        return 1
    fi
    
    print_success "All critical dependencies are installed!"
    return 0
}

install_deps_in_batches() {
    # This function installs dependencies in smaller batches to avoid hanging
    local req_file="$1"
    
    print_status "Installing dependencies in batches to prevent hanging..."
    
    # Check if file exists
    if [ ! -f "$req_file" ]; then
        print_warning "Requirements file '$req_file' not found"
        return 1
    fi
    
    # Count lines in requirements file
    local total_lines=$(grep -v "^#" "$req_file" | grep -v "^$" | wc -l | tr -d ' ')
    
    # If requirements file is empty or missing, return
    if [ "$total_lines" -eq 0 ]; then
        print_warning "Requirements file is empty"
        return 1
    fi
    
    # Use dynamic batch size based on system resources
    local batch_size=$BATCH_SIZE
    local batches=$((total_lines / batch_size + 1))
    
    print_status "Found $total_lines packages, installing in $batches batches with size $batch_size"
    
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
        echo "$deps" | xargs "$PYTHON_CMD" -m pip install -q --no-cache-dir -j"$PIP_PARALLEL_JOBS" >/dev/null 2>&1 || {
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
    if "$PYTHON_CMD" -c "import sentence_transformers" 2>/dev/null; then
        print_status "sentence_transformers seems to be installed correctly"
        return 0
    fi
    
    print_warning "sentence_transformers issues detected, applying fixes..."
    
    # Fix 1: Try reinstalling with explicit dependencies
    print_status "Fix 1: Reinstalling with explicit dependencies..."
    "$PYTHON_CMD" -m pip install -q --no-cache-dir --force-reinstall -U transformers torch numpy scipy >/dev/null 2>&1
    "$PYTHON_CMD" -m pip install -q --no-cache-dir --force-reinstall -U sentence-transformers >/dev/null 2>&1
    
    # Try importing again
    if "$PYTHON_CMD" -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 1 succeeded!"
        return 0
    fi
    
    # Fix 2: Try with a specific version
    print_status "Fix 2: Trying specific version..."
    "$PYTHON_CMD" -m pip install -q --no-cache-dir --force-reinstall "sentence-transformers==2.2.2" >/dev/null 2>&1
    
    # Try importing again
    if "$PYTHON_CMD" -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 2 succeeded!"
        return 0
    fi
    
    # Fix 3: Try building from source
    print_status "Fix 3: Building from source..."
    "$PYTHON_CMD" -m pip install -q --no-cache-dir git+https://github.com/UKPLab/sentence-transformers.git >/dev/null 2>&1
    
    # Final check
    if "$PYTHON_CMD" -c "import sentence_transformers" 2>/dev/null; then
        print_success "Fix 3 succeeded!"
        return 0
    fi
    
    print_error "All fixes failed for sentence_transformers"
    return 1
}

detect_system_resources() {
    # Detect RAM (in GB)
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        MEM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        # Linux
        MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    fi
    MEM_GB=${MEM_GB:-4}  # Default to 4GB if detection fails
    
    # Detect CPU count
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        CPU_COUNT=$(sysctl -n hw.ncpu)
    else
        # Linux
        CPU_COUNT=$(nproc 2>/dev/null || grep -c processor /proc/cpuinfo)
    fi
    CPU_COUNT=${CPU_COUNT:-2}  # Default to 2 CPUs if detection fails
    
    # Scale parallel jobs based on available CPUs (leave 1 core free)
    PARALLEL_JOBS=$((CPU_COUNT > 2 ? CPU_COUNT - 1 : 1))
    
    # Scale batch size based on memory
    if [ "$MEM_GB" -ge 16 ]; then
        BATCH_SIZE=10
    elif [ "$MEM_GB" -ge 8 ]; then
        BATCH_SIZE=8
    elif [ "$MEM_GB" -ge 4 ]; then
        BATCH_SIZE=5
    else
        BATCH_SIZE=3
    fi
    
    # Set pip parallelism based on CPU count
    PIP_PARALLEL_JOBS=$((CPU_COUNT > 1 ? CPU_COUNT - 1 : 1))
    
    print_status "System resources detected: ${MEM_GB}GB RAM, ${CPU_COUNT} CPUs"
    print_status "Using optimization settings: batch size ${BATCH_SIZE}, parallel jobs ${PARALLEL_JOBS}, pip parallel ${PIP_PARALLEL_JOBS}"
}

setup_cache_dir() {
    # Set up cache directory to speed up repeated installations
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        CACHE_DIR="$HOME/Library/Caches/aithreatmap"
    else
        # Linux
        CACHE_DIR="$HOME/.cache/aithreatmap"
    fi
    
    # Create cache directory if it doesn't exist
    mkdir -p "$CACHE_DIR"
    mkdir -p "$CACHE_DIR/pip"
    mkdir -p "$CACHE_DIR/wheels"
    
    # Export pip cache and wheel directory
    export PIP_CACHE_DIR="$CACHE_DIR/pip"
    export PYTHONWHEELHOUSE="$CACHE_DIR/wheels"
    export PIP_WHEEL_DIR="$CACHE_DIR/wheels"
    
    print_status "Using cache directory: $CACHE_DIR"
}

install_dependencies() {
    print_status "Checking Python dependencies..."
    
    # If all critical components are already installed, we can skip
    if is_already_installed "core_packages" && \
       is_already_installed "sentence_transformers" && \
       is_already_installed "tree_sitter" && \
       is_already_installed "faiss_cpu" && \
       is_already_installed "faiss_gpu"; then
        print_success "All critical dependencies are already installed. Skipping installation."
        return 0
    fi
    
    print_status "Installing Python dependencies..."
    
    # Set up cache directories
    setup_cache_dir
    
    # Detect system resources
    detect_system_resources
    
    # Define total steps for progress tracking
    local total_steps=9  # Increased to 9 to account for transformer fixes
    local current_step=0
    
    # Upgrade pip first
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Upgrading pip"
    "$PYTHON_CMD" -m pip install --upgrade pip >/dev/null 2>&1 || {
        print_error "Failed to upgrade pip"
        return 1
    }
    
    # Install build tools
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing build tools"
    "$PYTHON_CMD" -m pip install -q wheel setuptools build >/dev/null 2>&1 || {
        print_error "Failed to install build tools"
        return 1
    }
    
    # Use optimized installation for dependencies
    current_step=$((current_step + 1))
    show_progress $total_steps $current_step "Installing dependencies with optimized settings"
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing all packages with optimized settings..."
        "$PYTHON_CMD" -m pip install -q --upgrade-strategy only-if-needed -j"$PIP_PARALLEL_JOBS" -r requirements.txt >/dev/null 2>&1 || {
            print_warning "Optimized installation failed, falling back to batch installation"
            install_deps_in_batches "requirements.txt"
        }
    else
        print_warning "requirements.txt not found, skipping batch installation"
    fi
    
    # Install critical packages separately to ensure they're installed
    if ! is_already_installed "core_packages"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing critical packages"
        
        # Split critical packages into groups for parallel installation
        print_status "Installing critical packages in parallel..."
        {
            "$PYTHON_CMD" -m pip install -q click python-dotenv colorama tqdm >/dev/null 2>&1 &
            PID1=$!
            
            "$PYTHON_CMD" -m pip install -q pydantic huggingface_hub joblib >/dev/null 2>&1 &
            PID2=$!
            
            "$PYTHON_CMD" -m pip install -q requests >/dev/null 2>&1 &
            PID3=$!
            
            # Wait for all processes to complete
            wait $PID1 $PID2 $PID3
        }
        
        # Install LangChain separately (it has interdependencies)
        "$PYTHON_CMD" -m pip install -q langchain langchain-core langchain-community >/dev/null 2>&1 || {
            print_error "Failed to install critical packages"
            return 1
        }
        
        mark_as_installed "core_packages"
    else
        print_status "Core packages already installed, skipping..."
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Critical packages already installed"
    fi
    
    # Install sentence-transformers prerequisites in parallel if needed
    if ! is_already_installed "sentence_transformers"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing transformers & torch (for sentence-transformers)"
        {
            "$PYTHON_CMD" -m pip install -q transformers >/dev/null 2>&1 &
            PID1=$!
            
            "$PYTHON_CMD" -m pip install -q torch torchvision >/dev/null 2>&1 &
            PID2=$!
            
            # Wait for all processes to complete
            wait $PID1 $PID2
        } || {
            print_warning "Failed to install transformers prerequisites"
        }
        
        # Install sentence-transformers specifically - this is critical for embeddings
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing sentence-transformers"
        "$PYTHON_CMD" -m pip install -q sentence-transformers >/dev/null 2>&1 || {
            print_warning "Failed to install sentence-transformers. Trying alternative method..."
            "$PYTHON_CMD" -m pip install -q "sentence-transformers>=2.2.2" >/dev/null 2>&1 || {
                print_error "Failed to install sentence-transformers. Embedding functionality will be limited."
            }
        }
        
        # Apply fixes for sentence-transformers if needed
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Fixing sentence-transformers issues"
        fix_sentence_transformers || {
            print_warning "Could not fully resolve sentence-transformers issues. Some functionality may be limited."
        }
        
        mark_as_installed "sentence_transformers"
    else
        print_status "Sentence-transformers already installed, skipping..."
        # Skip 3 steps
        current_step=$((current_step + 3))
        show_progress $total_steps $current_step "Sentence transformers already installed"
    fi
    
    # Install accelerate for model loading if needed
    if ! is_already_installed "accelerate"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing accelerate for model loading"
        "$PYTHON_CMD" -m pip install -q accelerate >/dev/null 2>&1 || {
            print_warning "Failed to install accelerate. Model loading might encounter issues."
        }
        mark_as_installed "accelerate"
    else
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Accelerate already installed"
    fi
    
    # Install tree-sitter versions specifically if needed
    if ! is_already_installed "tree_sitter"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing tree-sitter"
        "$PYTHON_CMD" -m pip install -q --force-reinstall tree-sitter==0.20.1 tree-sitter-languages==1.8.0 >/dev/null 2>&1 || {
            print_error "Failed to install tree-sitter"
            return 1
        }
        mark_as_installed "tree_sitter"
    else
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Tree-sitter already installed"
    fi
    
    # Install faiss for vector search if needed
    if ! is_already_installed "faiss_cpu" && ! is_already_installed "faiss_gpu"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing FAISS for vector search"
        
        # First ensure we have a compatible NumPy version (< 2.0)
        print_status "Installing compatible NumPy for FAISS..."
        "$PYTHON_CMD" -m pip install -q "numpy<2.0.0" --force-reinstall >/dev/null 2>&1 || {
            print_warning "Failed to install compatible NumPy version"
        }
        
        # Check if PyTorch with CUDA is available for GPU version
        CUDA_AVAILABLE=false
        print_status "Checking for CUDA availability..."
        if "$PYTHON_CMD" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            CUDA_AVAILABLE=true
            print_success "CUDA detected! Will install GPU-enabled FAISS."
        elif command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
            # If torch isn't installed, check for CUDA via system
            CUDA_AVAILABLE=true
            print_success "CUDA detected via nvcc/cuda directory! Will install GPU-enabled FAISS."
        else
            print_status "No CUDA detected, using CPU version of FAISS."
        fi
        
        # Install appropriate FAISS version
        if [ "$CUDA_AVAILABLE" = true ]; then
            # Install PyTorch with CUDA first if needed
            if ! "$PYTHON_CMD" -c "import torch" 2>/dev/null; then
                print_status "Installing PyTorch with CUDA support first..."
                "$PYTHON_CMD" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu118 >/dev/null 2>&1 || {
                    print_warning "Failed to install PyTorch with CUDA. Will try FAISS-GPU directly."
                }
            fi
            
            # Install faiss-gpu with exact version 1.7.2
            print_status "Installing FAISS with GPU support (version 1.7.2)..."
            "$PYTHON_CMD" -m pip install -q faiss-gpu==1.7.2 --force-reinstall >/dev/null 2>&1 || {
                print_warning "Failed to install faiss-gpu version 1.7.2, trying alternative installation method..."
                "$PYTHON_CMD" -m pip install -q faiss-gpu==1.7.2 --force-reinstall --no-deps >/dev/null 2>&1 || {
                    print_warning "Failed to install faiss-gpu. Falling back to CPU version."
                    "$PYTHON_CMD" -m pip install -q faiss-cpu==1.10.0 --force-reinstall >/dev/null 2>&1
                    mark_as_installed "faiss_cpu"
                }
            }
            
            # Verify GPU FAISS installation
            if "$PYTHON_CMD" -c "import faiss; print(hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0)" 2>/dev/null | grep -q "True"; then
                # Also verify IndexFlatL2 is present
                if "$PYTHON_CMD" -c "import faiss; print(hasattr(faiss, 'IndexFlatL2'))" 2>/dev/null | grep -q "True"; then
                    print_success "Successfully installed FAISS-GPU 1.7.2 with IndexFlatL2 support!"
                    DETECTED_GPUS=$("$PYTHON_CMD" -c "import faiss; print(faiss.get_num_gpus())" 2>/dev/null)
                    print_status "Detected $DETECTED_GPUS GPUs for FAISS acceleration"
                    mark_as_installed "faiss_gpu"
                else
                    print_warning "FAISS-GPU installed but missing IndexFlatL2. Falling back to CPU version."
                    "$PYTHON_CMD" -m pip install -q faiss-cpu==1.10.0 --force-reinstall >/dev/null 2>&1
                    mark_as_installed "faiss_cpu"
                fi
            else
                print_warning "FAISS installed but GPU support not detected. Using CPU version."
                "$PYTHON_CMD" -m pip install -q faiss-cpu==1.10.0 --force-reinstall >/dev/null 2>&1
                mark_as_installed "faiss_cpu"
            fi
        else
            # Install CPU version with exact version 1.10.0
            print_status "Installing FAISS CPU version 1.10.0..."
            "$PYTHON_CMD" -m pip install -q faiss-cpu==1.10.0 --force-reinstall >/dev/null 2>&1 || {
                print_warning "Failed to install faiss-cpu 1.10.0, trying alternative method..."
                # Try without dependencies first
                "$PYTHON_CMD" -m pip install -q faiss-cpu==1.10.0 --force-reinstall --no-deps >/dev/null 2>&1 && \
                "$PYTHON_CMD" -m pip install -q "numpy<2.0.0" >/dev/null 2>&1 || {
                    print_error "Failed to install faiss-cpu. Vector search functionality will be limited."
                }
            }
            
            # Verify faiss installation
            if "$PYTHON_CMD" -c "import faiss; print(hasattr(faiss, 'IndexFlatL2'))" 2>/dev/null | grep -q "True"; then
                print_success "Successfully installed FAISS-CPU 1.10.0 with IndexFlatL2 support"
                mark_as_installed "faiss_cpu"
            else
                print_error "FAISS installed but missing IndexFlatL2 attribute. Vector search will be limited."
            fi
        fi
    else
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "FAISS already installed"
        
        # Check if it's the GPU version
        if is_already_installed "faiss_gpu"; then
            print_status "FAISS with GPU support is already installed"
        else
            print_status "FAISS CPU version is already installed"
        fi
    fi
    
    # Install llama-cpp-python with CUDA support if available
    if ! is_already_installed "llama_cpp_python"; then
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Installing llama-cpp-python with hardware acceleration"
        
        # Check if we're on a system with CUDA
        if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
            print_status "CUDA detected, installing llama-cpp-python with CUDA support..."
            CMAKE_ARGS="-DLLAMA_CUBLAS=on" "$PYTHON_CMD" -m pip install --force-reinstall llama-cpp-python >/dev/null 2>&1 && \
                print_success "Installed llama-cpp-python with CUDA support!" || \
                print_warning "Failed to install with CUDA. Installing standard version..."
        # Check if we're on a Mac with Metal (Apple Silicon)
        elif [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "arm64" ]; then
            print_status "Apple Silicon detected, installing llama-cpp-python with Metal support..."
            CMAKE_ARGS="-DLLAMA_METAL=on" "$PYTHON_CMD" -m pip install --force-reinstall llama-cpp-python >/dev/null 2>&1 && \
                print_success "Installed llama-cpp-python with Metal support!" || \
                print_warning "Failed to install with Metal. Installing standard version..."
        else
            print_status "Installing standard llama-cpp-python..."
            "$PYTHON_CMD" -m pip install --force-reinstall llama-cpp-python >/dev/null 2>&1 || \
                print_warning "Failed to install llama-cpp-python."
        fi
        mark_as_installed "llama_cpp_python"
    else
        current_step=$((current_step + 1))
        show_progress $total_steps $current_step "Llama-cpp-python already installed"
    fi
    
    print_success "All dependencies installed!"
    return 0
}

handle_git_support() {
    if is_already_installed "git_support"; then
        print_success "Git support already installed."
        return 0
    fi

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
        "$PYTHON_CMD" -m pip install -q dulwich==0.21.6 >/dev/null 2>&1 && {
            print_success "Git support installed!"
            mark_as_installed "git_support"
        } || print_warning "Failed to install Git support"
    else
        print_status "Skipping Git support. You'll only be able to analyze local repositories."
    fi
}

setup_venv() {
    VENV_NAME="venv311"
    
    # If we're already in the venv, no need to reactivate
    if is_in_our_venv; then
        print_success "Already in virtual environment $VENV_NAME"
        return 0
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        print_status "Creating Python virtual environment in ./$VENV_NAME..."
        "$PYTHON_CMD" -m venv "$VENV_NAME" || {
            print_error "Failed to create virtual environment."
            print_status "Try installing the venv module with: pip3 install virtualenv"
            return 1
        }
        print_success "Virtual environment created!"
    else
        print_status "Using existing virtual environment at ./$VENV_NAME"
        
        # Check if marker file exists inside venv
        if [ -f "$VENV_NAME/.setup_complete" ]; then
            print_status "This venv was previously set up successfully"
        fi
    fi
    
    # Activate the virtual environment
    print_status "Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$VENV_NAME/bin/activate" || {
        print_error "Failed to activate virtual environment."
        return 1
    }
    
    # Create a marker file inside venv
    touch "$VENV_NAME/.setup_complete"
    
    print_success "Virtual environment activated!"
    return 0
}

# Handle interruption - cleanup
cleanup() {
    print_warning "Interrupted! Cleaning up..."
    exit 1
}

# Set up trap for SIGINT
trap cleanup INT

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
    PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
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

# Turn off exit on error for the remainder so we can handle errors gracefully
set +e

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
read -r HF_TOKEN_SET HF_TOKEN <<< "$TOKEN_INFO"
setup_env_file "$MODEL_VARIANT" "$HF_TOKEN" "$HF_TOKEN_SET"

# Initialize the framework
CURRENT_STEP=$((CURRENT_STEP + 1))
show_progress $TOTAL_SETUP_STEPS $CURRENT_STEP "Initializing framework"
print_status "Initializing the framework..."
"$PYTHON_CMD" -m cli init 2>/dev/null || print_warning "Framework initialization encountered issues, but we can continue"

# Verify installation
verify_installation
VERIFY_STATUS=$?
if [ $VERIFY_STATUS -ne 0 ]; then
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

# End of script 