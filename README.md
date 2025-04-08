# AI Threat Model Map Generator

**Version: 1.2.0**

Automatically analyze code repositories to generate comprehensive threat models with attack vectors, data flow diagrams, and security recommendations.

### Updated Features (v1.2.0)

* **Latest Dependency Updates**: Using LangChain 0.2.x series with simplified API
* **Improved Language Parsing**: Integrated with tree-sitter-languages for better code analysis
* **Enhanced Performance**: Updated to use the newest versions of all dependencies

### Key Features

- **Code Repository Analysis**: Analyze repositories from GitHub URLs or local paths
- **Language Support**: Python, JavaScript, TypeScript, Java, Go, PHP
- **RAG-Based Analysis**: Unlimited repository size analysis through retrieval augmented generation
- **Security Boundary Detection**: Identifies security domains and boundaries 
- **Dynamic RAG Exploration**: Performs targeted queries to understand security-critical components
- **Cross-Boundary Data Flows**: Identifies and analyzes data flows crossing security boundaries
- **Mermaid Diagrams**: Generates class structure, data flow, and threat relationship diagrams
- **Detailed HTML Reports**: Comprehensive security findings in an organized HTML report

### Requirements

- Python 3.8+ 
- 16GB+ RAM recommended for larger repositories
- 10GB disk space (for model storage)
- GraphViz (for diagram generation)

### Installation

#### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/AIThreatMap.git
cd AIThreatMap

# Run the installer (automatically sets up a virtual environment)
chmod +x install.sh
./install.sh
```

This will automatically:
1. Create a virtual environment in the `venv` directory
2. Install all dependencies within the virtual environment
3. Download the required model files
4. Set up the configuration

#### Hugging Face Authentication

The CodeLlama models sometimes require authentication with a Hugging Face token:

1. Create a free account at [Hugging Face](https://huggingface.co/join)
2. Generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set your token using one of these methods:
   ```bash
   # Option 1: Set in your environment
   export HF_TOKEN=your_token_here
   
   # Option 2: Use the CLI tool to save it permanently
   python -m cli set_token
   ```

The installation and download scripts will automatically detect authentication failures and prompt you to provide a token when needed.

#### Manual Installation

If you prefer to set up manually:

1. Create and activate a virtual environment (required with newer Python versions):

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# OR on Windows
# venv\Scripts\activate
```

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
pip install tree-sitter-languages  # Ensure tree-sitter-languages is installed
```

2. Download the CodeLlama model:

```bash
# Create model directory
mkdir -p models

# For ARM/Apple Silicon with HF token
curl -L -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/TheBloke/CodeLlama-2-7b-Instruct-GGUF/resolve/main/codellama-2-7b-instruct.Q4_0.gguf -o models/codellama-2-7b-instruct.Q4_0.gguf

# For x86 architectures with HF token
curl -L -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/TheBloke/CodeLlama-2-7b-Instruct-GGUF/resolve/main/codellama-2-7b-instruct.Q4_K_M.gguf -o models/codellama-2-7b-instruct.Q4_K_M.gguf
```

3. Create the necessary directories:

```bash
mkdir -p output
mkdir -p visualizer/templates
```

### Usage

#### CLI

```bash
# Initialize environment and download required models
python -m cli init

# Analyze a GitHub repository
python -m cli analyze https://github.com/username/repo

# Analyze a local repository
python -m cli analyze /path/to/local/repo --local

# Generate visualizations only
python -m cli visualize --output-dir output

# View diagrams in browser
python -m cli view
```

#### API

```python
from repository_analyzer.analyzer import RepositoryAnalyzer
from repository_analyzer.embedding_store import EmbeddingStore
from llm_processor.processor import LLMProcessor
from visualizer.visualizer import ThreatModelVisualizer

# Initialize components
embedding_store = EmbeddingStore()
analyzer = RepositoryAnalyzer(repo_path="temp_repo", embedding_store=embedding_store)
llm_processor = LLMProcessor(embedding_store)
visualizer = ThreatModelVisualizer()

# Clone and analyze repository
analyzer.clone_repository("https://github.com/username/repo")
analysis_results = analyzer.analyze_code()

# Generate threat model
threat_model = llm_processor.generate_threat_model(analysis_results)

# Generate visualizations and report
diagrams = visualizer.generate_visualizations_from_dir("output")
report_path = visualizer.generate_report(threat_model)

print(f"Report generated at: {report_path}")
print(f"Diagrams generated at: {', '.join(diagrams.values())}")
```

### Viewing Diagrams

The tool generates Mermaid diagrams in the output directory:

1. **Using Mermaid Live Editor**:
   - Copy diagram content from .mmd files
   - Paste into [Mermaid Live Editor](https://mermaid.live/)

2. **Using GitHub**:
   - GitHub natively supports Mermaid in Markdown files
   - Create a Markdown file with content like:
     ```
     ```mermaid
     (content of your .mmd file)
     ```
     ```

3. **Using VS Code**:
   - Install [Mermaid Preview Extension](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)

## Output Files

All output is saved to the `output` directory (configurable):

- `analysis_results.json`: Raw analysis data
- `threat_model.json`: Generated threat model
- `class_diagram.mmd`: Class structure diagram
- `flow_diagram.mmd`: Data flow diagram
- `threat_diagram.mmd`: Threat relationship diagram
- `threat_analysis_report.html`: Comprehensive HTML report

## Troubleshooting

### Installation Issues

1. **Dependency conflicts**:
   - Try creating a fresh virtual environment
   - Install dependencies one by one: `pip install -r requirements.txt --no-deps`
   - Then resolve missing dependencies: `pip install -r requirements.txt`

2. **tree-sitter installation fails**:
   - Install tree-sitter-languages package explicitly: `pip install tree-sitter-languages`
   - Make sure you have a C compiler installed on your system

3. **llama-cpp-python installation fails**:
   - Ensure you have a C++ compiler installed
   - On macOS: Install Xcode Command Line Tools
   - On Ubuntu/Debian: `sudo apt-get install build-essential`
   - Try: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python` (on macOS)

4. **Model download fails**:
   - Hugging Face may require authentication - use `python -m cli set_token` to set up your token
   - Get a free Hugging Face token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - You can also download models manually from [Hugging Face](https://huggingface.co/TheBloke)
   - Check network connection and proxy settings
   - Ensure enough disk space (10GB+ free)

### Runtime Issues

1. **Out of memory**:
   - The tool uses a RAG approach to efficiently analyze large repositories
   - Close other memory-intensive applications
   - Ensure you have at least 16GB of RAM for large codebases

### Model Loading Issues

If you encounter the following error:
```
Failed to initialize embedding model: name 'init_empty_weights' is not defined
```

This is caused by a compatibility issue with newer versions of the transformers library. You can fix it by:

1. Making sure you have the correct transformers version by running:
   ```
   pip install 'transformers>=4.36.0,<4.52.0'
   ```
   
2. If you still encounter issues, try installing an older version of transformers:
   ```
   pip install transformers==4.35.0
   ```

### Bitsandbytes and Quantization Issues

If you see errors related to bitsandbytes or quantization such as:
```
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
The installed version of bitsandbytes was compiled without GPU support
```

You can resolve these by:

1. Disabling quantization by adding the following to your `.env` file:
   ```
   LLM_USE_QUANTIZATION=false
   ```

2. Trying to upgrade bitsandbytes to the latest version:
   ```
   pip install -U bitsandbytes
   ```

3. If you still have issues, you may need to compile bitsandbytes for your specific system:
   ```
   pip uninstall bitsandbytes -y
   pip install bitsandbytes --no-binary bitsandbytes
   ```

## License

MIT

## Author

- **Omaid F** ([@omaidf](https://github.com/omaidf))

## Acknowledgments

- [CodeLlama](https://github.com/facebookresearch/codellama) for code analysis
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing
- [Mermaid](https://mermaid.js.org/) for diagram generation 

## 🔑 Setting Up Your Hugging Face Token (REQUIRED)

This project requires a Hugging Face token to download the LLM models. Here's how to set it up:

1. **Create a Hugging Face account** if you don't have one at [https://huggingface.co/join](https://huggingface.co/join)

2. **Generate a token** at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it (e.g., "AIThreatMap")
   - Role can be "Read" (no need for write permissions)
   - Click "Generate token" and copy it

3. **Set your token** using one of these methods:

   **Method 1: Using the CLI tool (Recommended)**
   ```bash
   # Run the token setup command
   python -m cli set_token
   
   # Follow the prompts to enter your token
   # The token will be saved to your .env file
   ```

   **Method 2: Set in your environment**
   ```bash
   # Add to your shell profile (.bashrc, .zshrc, etc.)
   export HF_TOKEN=your_token_here
   
   # OR set just for the current session
   export HF_TOKEN=your_token_here
   ```

   **Method 3: Manually add to .env file**
   ```bash
   # Create or edit the .env file in your project directory
   echo "HF_TOKEN=your_token_here" >> .env
   ```

4. **Verify your token is set**
   ```bash
   # Run the initialization which will verify token access
   python -m cli init
   ```

If you see authentication errors even after setting your token, try:
- Ensure you copied the full token without extra spaces
- Restart your terminal if you set it in your environment
- Run `python -m cli set_token` to save it in your .env file 