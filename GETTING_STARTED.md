# Getting Started with ThreatMapAI

This guide will help you quickly get up and running with ThreatMapAI.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 16GB RAM recommended
- GPU recommended for large repositories (but not required)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/omaid/ThreatMapAI.git
cd ThreatMapAI
```

2. **Run the setup script**

```bash
# For Unix/Linux/Mac
chmod +x setup.sh
./setup.sh

# For Windows
# setup.bat
```

3. **Set your Hugging Face token** (required for model downloads)

```bash
python -m cli set_token
```

You can obtain a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Quick Start

### Analyze a GitHub Repository

```bash
# Replace with the GitHub repository URL you want to analyze
python -m cli analyze https://github.com/username/repository
```

### Analyze a Local Repository

```bash
python -m cli analyze /path/to/your/local/repository --local
```

### View Generated Diagrams

```bash
python -m cli view
```

This will open a browser window with the generated diagrams.

## Understanding the Output

ThreatMapAI generates several files in the `output` directory:

- `analysis_results.json`: Raw analysis data
- `threat_model.json`: Generated threat model
- `class_diagram.mmd`: Class structure diagram
- `flow_diagram.mmd`: Data flow diagram
- `threat_diagram.mmd`: Threat relationship diagram
- `threat_analysis_report.html`: Comprehensive HTML report

The HTML report provides a user-friendly view of the analysis results, including:

- Project Overview
- Security Boundaries
- Data Flows
- Identified Threats
- Recommended Mitigations
- Risk Assessment

## Next Steps

- Explore advanced configuration options in the README
- Try different model configurations for improved analysis
- Contribute to the project on GitHub

For more detailed information, see the full [README.md](README.md). 