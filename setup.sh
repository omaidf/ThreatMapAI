#!/bin/bash

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
all_passed = True
critical_packages = [
    'click', 'langchain', 'langchain_core', 'langchain_community', 
    'tree_sitter', 'requests', 'tqdm', 'xmltodict', 'importlib.metadata', 'faiss',
    'huggingface_hub', 'joblib'
]

for pkg in critical_packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg}')
        all_passed = False

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

sys.exit(0 if all_passed else 1)
"
    return $?
}

install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    python -m pip install --upgrade pip || {
        print_error "Failed to upgrade pip"
        return 1
    }
    
    # Install build tools
    print_status "Installing basic build tools..."
    pip install -q wheel setuptools build || {
        print_error "Failed to install build tools"
        return 1
    }
    
    # Install all dependencies in one go to avoid redundancy
    print_status "Installing all dependencies..."
    pip install -q --no-cache-dir -r requirements.txt || {
        print_warning "Some requirements failed to install. Continuing with critical packages..."
    }
    
    # Install critical packages separately to ensure they're installed
    print_status "Installing critical packages..."
    pip install -q --no-cache-dir click python-dotenv requests colorama tqdm langchain langchain-core langchain-community pydantic huggingface_hub joblib || {
        print_error "Failed to install critical packages"
        return 1
    }
    
    # Install accelerate for model loading
    print_status "Installing accelerate for model loading..."
    pip install -q --no-cache-dir accelerate || {
        print_warning "Failed to install accelerate. Model loading might encounter issues."
    }
    
    # Install bitsandbytes for 8-bit quantization
    print_status "Installing bitsandbytes for model quantization..."
    pip install -q --no-cache-dir bitsandbytes || {
        print_warning "Failed to install bitsandbytes. Model loading will use more memory."
    }
    
    # Install tree-sitter versions specifically
    print_status "Installing tree-sitter with compatible versions..."
    pip install -q --no-cache-dir --force-reinstall tree-sitter==0.20.1 tree-sitter-languages==1.8.0 || {
        print_error "Failed to install tree-sitter"
        return 1
    }
    
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
        pip install -q --no-cache-dir dulwich==0.21.6 && print_success "Git support installed!" || print_warning "Failed to install Git support"
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

# Check for Python
print_status "Checking for Python..."
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
print_status "Checking for pip..."
check_command pip3 || exit 1
print_success "pip3 is installed!"

# Setup virtual environment
setup_venv || exit 1

# Install dependencies
install_dependencies || {
    print_error "Failed to install critical dependencies. Exiting."
    exit 1
}

# Handle Git support
handle_git_support

# Create required directories
create_directories

# Detect architecture
MODEL_VARIANT=$(detect_architecture)

# Prompt for Hugging Face token
TOKEN_INFO=$(prompt_for_hf_token)
read HF_TOKEN_SET HF_TOKEN <<< "$TOKEN_INFO"

# Setup .env file
setup_env_file "$MODEL_VARIANT" "$HF_TOKEN" "$HF_TOKEN_SET"

# Information about model download
print_status "Model download is now handled directly by the Python code using huggingface_hub"
print_status "No models will be downloaded during setup - they'll be downloaded as needed during runtime"
if [ "$HF_TOKEN_SET" != "true" ]; then
    print_warning "You need to set your Hugging Face token before downloading models:"
    print_status "  python -m cli set_token"
fi

# Initialize the framework only if we have basic dependencies installed
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
print_success "Setup completed successfully!"
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