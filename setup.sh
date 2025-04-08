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
    }
    
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