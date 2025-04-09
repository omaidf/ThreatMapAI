"""
Repository Analyzer Module

This module provides functionality for analyzing code repositories,
including cloning, parsing, and extracting structural information.
"""

import os
import logging
import shutil
import tempfile
import platform
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import tree_sitter
from tree_sitter import Language, Parser
# Add import for tree-sitter-languages which provides pre-built languages
try:
    import tree_sitter_languages
except ImportError:
    tree_sitter_languages = None

# Optional Git dependencies - imported at runtime when needed
# from dulwich import porcelain

from .embedding_store import EmbeddingStore
from utils import info_msg
from utils.env_utils import update_env_file
import re
import json
from tqdm import tqdm
import time

# Add import for Counter for file extension counting
from collections import Counter
import fnmatch

logger = logging.getLogger(__name__)

class RepositoryAnalyzerError(Exception):
    """Custom exception for repository analyzer errors."""
    pass

class RepositoryAnalyzer:
    """
    Analyzes code repositories for structure and content.
    
    This class handles repository cloning, code parsing, and analysis
    using tree-sitter for accurate AST parsing.
    """
    
    def __init__(self, repo_path: Optional[str] = None, embedding_store: Optional[EmbeddingStore] = None,
                 distributed: bool = None, gpu_ids: Optional[List[int]] = None, memory_limit: Optional[float] = None):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_path: Path to the repository (local or temporary)
            embedding_store: EmbeddingStore for storing file embeddings (optional)
            distributed: Whether to use distributed processing (multi-GPU) when available
            gpu_ids: List of specific GPU IDs to use if multiple GPUs are available
            memory_limit: Memory limit per GPU in GB
        """
        self.repo_path = Path(repo_path) if repo_path else Path(tempfile.mkdtemp())
        self.parser = None
        self.languages = {}
        self.embedding_store = embedding_store
        
        # GPU configuration
        self.gpu_ids = gpu_ids
        self.distributed = distributed
        self.memory_limit = memory_limit
        
        # Set a very high max file size limit (100MB)
        self.max_file_size = 100 * 1024 * 1024  # 100MB in bytes
        
        # Check for available GPUs and configure distributed processing if not explicitly set
        if self.distributed is None:
            # Auto-detect if distributed processing should be enabled
            try:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    info_msg(f"Detected {gpu_count} GPUs for repository analysis")
                    if gpu_count > 1:
                        # Automatically enable distributed processing for multi-GPU setups
                        self.distributed = True
                        info_msg(f"Auto-enabling distributed processing for {gpu_count} GPUs")
                        
                        # If no specific GPUs were provided, use all available GPUs
                        if self.gpu_ids is None:
                            self.gpu_ids = list(range(gpu_count))
                            info_msg(f"Using all available GPUs: {self.gpu_ids}")
                            
                            # Set environment variables for other components
                            os.environ["DISTRIBUTED"] = "true"
                            os.environ["GPU_IDS"] = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)
                            
                            # Also set environment variable for distributed to ensure other components use it 
                            update_env_file("DISTRIBUTED", "true")
                            update_env_file("GPU_IDS", ",".join(str(gpu_id) for gpu_id in self.gpu_ids))
                            
                    # Set environment variable to enable multi-GPU for FAISS if available
                    if gpu_count > 1:
                        os.environ["FAISS_MULTI_GPU"] = "1"
                        info_msg("Enabled multi-GPU support for FAISS indexing")
                    else:
                        os.environ["FAISS_MULTI_GPU"] = "0"
            except ImportError:
                # PyTorch not available, cannot use distributed processing
                self.distributed = False
        
        # Initialize the parser but defer language loading until after repo clone
        self.parser = tree_sitter.Parser()
        # Note: We don't call _setup_tree_sitter() here anymore, it will be called
        # after the repository is cloned or otherwise available
        
    def _setup_tree_sitter(self, specific_languages=None) -> None:
        """
        Set up tree-sitter parser with ONLY the specified languages.
        
        Args:
            specific_languages: List of language names to load (e.g. ["php", "python"])
                               This method will load ONLY these languages, with no fallbacks
        """
        try:
            # Initialize parser if not already done
            if self.parser is None:
                self.parser = tree_sitter.Parser()
            
            # Try to use tree-sitter-languages package
            try:
                import tree_sitter_languages
                logger.info("Using tree-sitter-languages package for grammar support")
                
                # Initialize languages dictionary if it doesn't exist
                if not hasattr(self, 'languages') or self.languages is None:
                    self.languages = {}
                
                # Language configuration with extensions
                language_configs = {
                    "python": [".py", ".pyw", ".pyi"],
                    "javascript": [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"],
                    "java": [".java"],
                    "go": [".go"],
                    "php": [".php"],
                    "c": [".c", ".h"],
                    "cpp": [".cpp", ".hpp", ".cc"],
                    "c_sharp": [".cs"],
                    "ruby": [".rb"],
                    "rust": [".rs"],
                    "html": [".html", ".htm"],
                    "css": [".css"],
                    "json": [".json"],
                    "yaml": [".yaml", ".yml"],
                    "bash": [".sh", ".bash"],
                    "markdown": [".md", ".markdown"],
                    "swift": [".swift"],
                    "kotlin": [".kt", ".kts"]
                }
                
                # If no languages specified, use Python as default
                languages_to_load = specific_languages
                if not languages_to_load:
                    languages_to_load = ["python"]
                    logger.info("No specific languages provided, defaulting to Python")
                
                # Clear existing languages before loading new ones
                self.languages = {}
                
                # Track which languages were successfully loaded
                available_languages = []
                
                # Try to load each required language
                for lang_name in languages_to_load:
                    if lang_name not in language_configs:
                        logger.warning(f"Skipping unsupported language: {lang_name}")
                        continue
                        
                    try:
                        # Get language from tree-sitter-languages using the correct API for version 1.8.0
                        lang_obj = tree_sitter_languages.get_language(lang_name)
                        if lang_obj:
                            self.languages[lang_name] = {
                                "parser": lang_obj,
                                "extensions": language_configs[lang_name]
                            }
                            available_languages.append(lang_name)
                            logger.info(f"Loaded {lang_name} grammar")
                    except Exception as e:
                        logger.error(f"Failed to load {lang_name} grammar: {str(e)}")
                
                # Check if any languages were loaded successfully
                if available_languages:
                    logger.info(f"Successfully loaded grammar for: {', '.join(available_languages)}")
                else:
                    # If no languages could be loaded, try a default as fallback
                    logger.warning("None of the specified languages could be loaded, trying Python as fallback")
                    try:
                        lang_obj = tree_sitter_languages.get_language("python")
                        if lang_obj:
                            self.languages["python"] = {
                                "parser": lang_obj,
                                "extensions": language_configs["python"]
                            }
                            logger.info("Loaded Python grammar as fallback")
                    except Exception as e:
                        logger.error(f"Failed to load Python fallback grammar: {str(e)}")
                
                # If still no languages could be loaded, raise an exception
                if not self.languages:
                    raise RepositoryAnalyzerError("No language grammars could be loaded")
                
            except ImportError:
                # Fall back to manual compilation approach
                logger.warning("tree-sitter-languages package not found, falling back to manual grammar compilation")
                raise RepositoryAnalyzerError("tree-sitter-languages package is required. Please install it with 'pip install tree-sitter-languages==1.8.0'")
                
        except Exception as e:
            logger.error(f"Failed to setup tree-sitter: {str(e)}")
            raise RepositoryAnalyzerError(f"Tree-sitter setup failed: {str(e)}")
    
    def _detect_languages_in_repo(self) -> List[str]:
        """
        Detect programming languages used in the repository based on file extensions.
        Returns only the single most common recognized language.
        
        Returns:
            List containing only the most common language name that has a supported parser
        """
        # Extension to language mapping - expanded to cover more languages
        ext_to_lang = {
            # Python
            ".py": "python",
            ".pyw": "python",
            ".pyi": "python",
            # JavaScript/TypeScript
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "javascript",  # typescript is parsed by javascript grammar
            ".tsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            # Java
            ".java": "java",
            # Go
            ".go": "go",
            # PHP
            ".php": "php",
            ".inc": "php",  # Treat .inc as PHP, which is common
            # C/C++
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cc": "cpp",
            # C#
            ".cs": "c_sharp",
            # Ruby
            ".rb": "ruby",
            # Rust
            ".rs": "rust",
            # Swift
            ".swift": "swift",
            # Kotlin
            ".kt": "kotlin",
            ".kts": "kotlin",
            # HTML/CSS
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            # Other common types
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".md": "markdown",
            ".sql": "sql"
        }
        
        # Use Counter to count file extensions in repository
        extension_counter = Counter()
        file_count = 0
        
        # First check that the repository path exists
        if not os.path.exists(self.repo_path):
            logger.warning(f"Repository path {self.repo_path} does not exist")
            # Default to a sensible fallback
            return ["python"]
            
        # Go through all files in the repository
        for root, _, files in os.walk(self.repo_path):
            # Skip common directories to ignore
            if any(i in root for i in ['.git', 'node_modules', '__pycache__', 'venv', '.idea', 'build', 'dist']):
                continue
                
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                    
                file_count += 1
                ext = Path(file).suffix.lower()
                if ext:  # Count all extensions, not just ones we recognize
                    extension_counter[ext] += 1
        
        # If we found no files at all, return a default language
        if file_count == 0:
            logger.warning(f"No files found in repository {self.repo_path}")
            return ["python"]
            
        # Log the counts for all extensions
        if extension_counter and file_count > 0:
            logger.info(f"Found {file_count} files in repository")
            logger.info("File extension counts in repository:")
            
            # Display all extensions, even those we don't have parsers for
            for ext, count in extension_counter.most_common(20):
                percentage = (count / file_count) * 100
                lang = ext_to_lang.get(ext, "unknown")
                logger.info(f"  - {ext}: {count} files ({percentage:.1f}%) ({lang})")
                
            # Create a list of extensions sorted by frequency, but only those with known parsers
            recognized_extensions = []
            for ext, count in extension_counter.most_common():
                if ext in ext_to_lang:
                    recognized_extensions.append((ext, count, ext_to_lang[ext]))
            
            if recognized_extensions:
                # Go through sorted list until we find a supported language
                for ext, count, lang in recognized_extensions:
                    percentage = (count / file_count) * 100
                    logger.info(f"Selected language: {lang} from {ext} ({count} files, {percentage:.1f}% of repository)")
                    logger.info(f"Loading ONLY the grammar for: {lang}")
                    return [lang]
            
            # If no languages with supported parsers were found, use a reasonable default
            logger.warning("No supported languages detected in the repository")
            return ["python"]
        else:
            logger.warning("No recognized code files found in the repository")
            # Return a sensible default
            return ["python"]
    
    def _get_language(self, file_path: str) -> Optional[Any]:
        """
        Get the appropriate language parser for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language parser if supported, None otherwise
        """
        try:
            ext = Path(file_path).suffix.lower()
            
            # Direct mapping for common extensions
            for lang, config in self.languages.items():
                if ext in config["extensions"]:
                    return config["parser"]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get language for {file_path}: {str(e)}")
            return None
    
    def _parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a file using tree-sitter and extract relevant information.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Dictionary containing parsed information or None if parsing fails
        """
        try:
            language = self._get_language(file_path)
            if not language:
                return None
            
            # Read file with explicit resource management
            content = None
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                self.parser.set_language(language)
                tree = self.parser.parse(content)
                
                # Extract information and return result
                result = self._extract_ast_info(tree.root_node, content)
                
                # Explicitly delete large objects to help garbage collection
                del tree
                content = None
                
                return result
            except (IOError, OSError) as e:
                self._safe_exception_handler("_parse_file:file_io", e, "warning")
                return None
                
        except Exception as e:
            self._safe_exception_handler("_parse_file", e, "warning")
            return None
    
    def _extract_ast_info(self, node: tree_sitter.Node, content: bytes) -> Dict[str, Any]:
        """
        Extract relevant information from the AST.
        
        Args:
            node: Root node of the AST
            content: Raw file content
            
        Returns:
            Dictionary containing extracted information
        """
        info = {
            "classes": [],
            "functions": [],
            "imports": [],
            "dependencies": []
        }
        
        try:
            # Extract class definitions - handle different node types based on language
            for class_node in node.children:
                # Python class definition
                if class_node.type == "class_definition":
                    class_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "methods": []
                    }
                    
                    # Extract methods
                    for method_node in class_node.children:
                        if method_node.type == "function_definition":
                            method_info = {
                                "name": self._safe_decode(method_node.child_by_field_name("name").text if method_node.child_by_field_name("name") else b"Unknown"),
                                "parameters": []
                            }
                            
                            # Extract parameters
                            params_node = method_node.child_by_field_name("parameters")
                            if params_node:
                                for param_node in params_node.children:
                                    if param_node.type == "identifier":
                                        method_info["parameters"].append(self._safe_decode(param_node.text))
                            
                            class_info["methods"].append(method_info)
                    
                    info["classes"].append(class_info)
                
                # PHP class declaration
                elif class_node.type == "class_declaration":
                    class_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "methods": []
                    }
                    
                    # Extract PHP class methods
                    for method_node in class_node.children:
                        # PHP method definition
                        if method_node.type == "method_declaration":
                            method_info = {
                                "name": self._safe_decode(method_node.child_by_field_name("name").text if method_node.child_by_field_name("name") else b"Unknown"),
                                "parameters": []
                            }
                            
                            # Extract parameters from PHP method
                            params_node = method_node.child_by_field_name("parameters")
                            if params_node:
                                for param_node in params_node.children:
                                    if param_node.type == "parameter_declaration":
                                        name_node = param_node.child_by_field_name("name")
                                        if name_node:
                                            method_info["parameters"].append(self._safe_decode(name_node.text))
                            
                            class_info["methods"].append(method_info)
                    
                    info["classes"].append(class_info)
                
                # JavaScript/TypeScript class declarations
                elif class_node.type == "class_declaration" or class_node.type == "class":
                    class_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "methods": []
                    }
                    
                    # Extract class methods
                    for method_node in class_node.children:
                        if method_node.type == "method_definition" or method_node.type == "method":
                            method_info = {
                                "name": self._safe_decode(method_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                                "parameters": []
                            }
                            
                            # Extract parameters
                            params_node = method_node.child_by_field_name("parameters")
                            if params_node:
                                for param_node in params_node.children:
                                    if param_node.type == "identifier" or param_node.type == "formal_parameter":
                                        method_info["parameters"].append(self._safe_decode(param_node.text))
                            
                            class_info["methods"].append(method_info)
                    
                    info["classes"].append(class_info)
                
                # Extract imports
                elif class_node.type == "import_statement":
                    import_text = self._safe_decode(class_node.text)
                    info["imports"].append(import_text)
                    
                    # Extract dependency from import (simplified approach)
                    try:
                        # For Python-style imports
                        if import_text.startswith("import "):
                            dep = import_text.split(" ")[1].split(".")[0]
                            info["dependencies"].append(dep)
                        elif import_text.startswith("from "):
                            dep = import_text.split(" ")[1].split(".")[0]
                            info["dependencies"].append(dep)
                    except:
                        pass
                
                # PHP include/require statements
                elif class_node.type == "include_expression" or class_node.type == "require_expression":
                    try:
                        path_node = class_node.child_by_field_name("path")
                        if path_node:
                            include_path = self._safe_decode(path_node.text)
                            info["imports"].append(f"include {include_path}")
                            info["dependencies"].append(include_path.strip('"\''))
                    except:
                        pass
                
                # JavaScript/TypeScript require or import
                elif class_node.type == "import_statement" or class_node.type == "lexical_declaration":
                    if "require(" in self._safe_decode(class_node.text) or "import " in self._safe_decode(class_node.text):
                        import_text = self._safe_decode(class_node.text)
                        info["imports"].append(import_text)
                        
                        # Extract dependency 
                        try:
                            import re
                            matches = re.findall(r'[\'"]([^\'"]+)[\'"]', import_text)
                            for match in matches:
                                info["dependencies"].append(match)
                        except:
                            pass
                
                # Extract function definitions - Python
                elif class_node.type == "function_definition":
                    func_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "parameters": []
                    }
                    
                    # Extract parameters
                    params_node = class_node.child_by_field_name("parameters")
                    if params_node:
                        for param_node in params_node.children:
                            if param_node.type == "identifier":
                                func_info["parameters"].append(self._safe_decode(param_node.text))
                    
                    info["functions"].append(func_info)
                
                # Extract function definitions - PHP
                elif class_node.type == "function_definition" or class_node.type == "function_declaration":
                    func_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "parameters": []
                    }
                    
                    # Extract parameters from PHP function
                    params_node = class_node.child_by_field_name("parameters")
                    if params_node:
                        for param_node in params_node.children:
                            if param_node.type == "parameter_declaration":
                                name_node = param_node.child_by_field_name("name")
                                if name_node:
                                    func_info["parameters"].append(self._safe_decode(name_node.text))
                    
                    info["functions"].append(func_info)
                
                # JavaScript/TypeScript function declarations
                elif class_node.type == "function_declaration" or class_node.type == "function":
                    func_info = {
                        "name": self._safe_decode(class_node.child_by_field_name("name").text if class_node.child_by_field_name("name") else b"Unknown"),
                        "parameters": []
                    }
                    
                    # Extract parameters
                    params_node = class_node.child_by_field_name("parameters")
                    if params_node:
                        for param_node in params_node.children:
                            if param_node.type == "identifier" or param_node.type == "formal_parameter":
                                func_info["parameters"].append(self._safe_decode(param_node.text))
                    
                    info["functions"].append(func_info)
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to extract AST info: {str(e)}")
            return info

    def _safe_decode(self, byte_text: bytes) -> str:
        """Safely decode byte string to UTF-8 string."""
        try:
            if isinstance(byte_text, bytes):
                return byte_text.decode('utf-8', errors='replace')
            elif isinstance(byte_text, str):
                return byte_text
            else:
                return str(byte_text)
        except Exception as e:
            logger.warning(f"Failed to decode text: {str(e)}")
            return "Unknown"
            
    def cleanup(self, force: bool = False) -> None:
        """
        Clean up temporary repository.
        
        Args:
            force: Force cleanup even if not a temporary repository
        """
        # Only clean up temporary repositories unless forced
        if str(self.repo_path).startswith(tempfile.gettempdir()) or force:
            if self.repo_path.exists():
                try:
                    logger.info(f"Cleaning up repository at {self.repo_path}")
                    shutil.rmtree(self.repo_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up repository: {str(e)}")
        
    def verify_repository(self) -> bool:
        """
        Verify that the repository exists and contains code files.
        
        Returns:
            True if the repository is valid, False otherwise
        """
        try:
            # Check if repository path exists
            if not os.path.exists(self.repo_path):
                logger.error(f"Repository path {self.repo_path} does not exist")
                return False
            
            # Check if repository contains any files
            files = []
            dirs = []
            
            for item in os.listdir(self.repo_path):
                item_path = os.path.join(self.repo_path, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path) and item != '.git':
                    dirs.append(item)
            
            # Check if the repository has any files or directories
            if not files and not dirs:
                logger.error(f"Repository {self.repo_path} is empty")
                return False
            
            # Check for common repository indicators
            common_indicators = [
                '.git',            # Git
                '.gitignore',      # Git
                'package.json',    # Node.js
                'requirements.txt', # Python
                'composer.json',   # PHP
                'Gemfile',         # Ruby
                'pom.xml',         # Java Maven
                'build.gradle',    # Java/Kotlin Gradle
                'Cargo.toml',      # Rust
                'go.mod',          # Go
                'CMakeLists.txt',  # C/C++
                '.sln',            # C# Solution
                'Makefile',        # General
                'README.md',       # General
                'setup.py',        # Python
                'index.php',       # PHP
                'index.html',      # Web
                'main.py',         # Python
                'app.py',          # Python
                'src',             # Source directory
                'lib',             # Library directory
                'app',             # Application directory
                'test',            # Test directory
                'tests',           # Test directory
                'docs',            # Documentation
            ]
            
            # Check if any common repository indicators are present
            for indicator in common_indicators:
                if os.path.exists(os.path.join(self.repo_path, indicator)):
                    logger.info(f"Found repository indicator: {indicator}")
                    return True
            
            # If no common indicators found but we have files/directories, it's likely still valid
            if files or dirs:
                logger.warning(f"Repository {self.repo_path} does not contain common repository indicators")
                logger.info(f"Found {len(files)} files and {len(dirs)} directories")
                # List some of the files and directories to help identify what was cloned
                if files:
                    logger.info(f"Sample files: {', '.join(files[:5])}")
                if dirs:
                    logger.info(f"Sample directories: {', '.join(dirs[:5])}")
                return True
            
            # If we got here, we couldn't find any clear indicators of a valid repository
            logger.error(f"Repository {self.repo_path} does not appear to be a valid code repository")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying repository: {str(e)}")
            return False
    
    def clone_repository(self, repo_url: str) -> None:
        """
        Clone a repository from a URL.
        
        Args:
            repo_url: URL of the repository to clone
        """
        try:
            # Clean up any existing repository
            if self.repo_path.exists():
                shutil.rmtree(self.repo_path)
            
            # Create repository directory
            self.repo_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Cloning repository from {repo_url} to {self.repo_path}")
            
            # Try to use dulwich if available
            try:
                import dulwich
                from dulwich import porcelain
                
                # Clone the repository using dulwich
                porcelain.clone(repo_url, str(self.repo_path))
                logger.info(f"Repository cloned successfully using dulwich")
                
                # Verify clone succeeded by checking for common directories or files
                if not any(os.listdir(self.repo_path)):
                    raise RepositoryAnalyzerError("Repository cloning resulted in empty directory")
                
            except ImportError:
                # If dulwich is not available, try to use system git
                logger.warning("Dulwich not available, trying system Git...")
                
                try:
                    # Try using system git via subprocess
                    import subprocess
                    logger.info(f"Cloning with system git: git clone {repo_url} {str(self.repo_path)}")
                    
                    # Run git clone with progress indication
                    process = subprocess.Popen(
                        ["git", "clone", repo_url, str(self.repo_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Monitor the clone process
                    while True:
                        if process.poll() is not None:
                            break
                        time.sleep(1)  # Check status every second
                    
                    # Check if the process completed successfully
                    if process.returncode != 0:
                        stderr = process.stderr.read()
                        raise subprocess.SubprocessError(f"Git clone failed: {stderr}")
                    
                    # Verify clone succeeded by checking for common directories or files
                    if not any(os.listdir(self.repo_path)):
                        raise RepositoryAnalyzerError("Git clone resulted in empty directory")
                    
                    logger.info(f"Repository cloned successfully using system Git")
                    
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    # Neither dulwich nor git command is available
                    raise RepositoryAnalyzerError(
                        "Git support is not available. Please install Git or dulwich:\n"
                        "  pip install dulwich\n"
                        "Alternatively, you can analyze a local repository by providing a local path."
                    )
                    
            logger.info(f"Successfully cloned repository from {repo_url} to {self.repo_path}")
            
        except Exception as e:
            self._safe_exception_handler("clone_repository", e, "error", True, RepositoryAnalyzerError)
    
    def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Analyze a repository from URL or local path.
        
        Args:
            repo_url: URL of the repository to analyze or path to local repository
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            start_time = time.time()
            
            # Reset languages
            self.languages = {}
            
            # Verify and clone repository if necessary
            is_local = os.path.exists(repo_url) and (os.path.isdir(repo_url) or repo_url.endswith(".zip"))
            
            if is_local:
                logger.info(f"Analyzing local repository: {repo_url}")
                
                if repo_url.endswith(".zip"):
                    # Handle ZIP file
                    logger.info(f"Extracting ZIP file: {repo_url}")
                    import zipfile
                    
                    # Create a subdirectory inside the repo_path
                    extract_dir = Path(self.repo_path) / "extracted"
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    # Extract the ZIP file
                    with zipfile.ZipFile(repo_url, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Look for the actual repository inside the extracted directory
                    candidates = [d for d in extract_dir.iterdir() if d.is_dir()]
                    if candidates:
                        # Assume the first directory is the main repository
                        repo_contents = candidates[0]
                        
                        # Copy all files from the main directory to the repo_path
                        for item in repo_contents.iterdir():
                            # Get the destination path
                            dst = Path(self.repo_path) / item.name
                            
                            # Copy the item
                            if item.is_dir():
                                if dst.exists():
                                    shutil.rmtree(dst)
                                shutil.copytree(item, dst)
                            else:
                                shutil.copy2(item, dst)
                    
                else:
                    # Copy local directory contents to the repo_path
                    for item in Path(repo_url).iterdir():
                        try:
                            # Get the destination path
                            dst = Path(self.repo_path) / item.name
                            
                            # Skip if it's the same as the destination
                            if item == dst:
                                continue
                                
                            # Copy the item
                            if item.is_dir():
                                if dst.exists():
                                    shutil.rmtree(dst)
                                shutil.copytree(item, dst)
                            else:
                                shutil.copy2(item, dst)
                        except Exception as e:
                            logger.warning(f"Error copying {item}: {str(e)}")
            else:
                # Clone remote repository
                logger.info(f"Cloning repository from URL: {repo_url}")
                self.clone_repository(repo_url)
                
                logger.info(f"Repository cloned, verifying contents...")
                
                # Verify the repository was cloned successfully and contains valid code
                if not self.verify_repository():
                    raise RepositoryAnalyzerError(f"Repository verification failed. The cloned repository appears invalid or empty.")
                    
                logger.info(f"Repository verified successfully")
            
            # Now detect ONLY the single most common language in the repository
            logger.info("Analyzing repository structure to determine the primary language...")
            primary_language = self._detect_languages_in_repo()
            
            if primary_language:
                logger.info(f"Primary language detected: {primary_language[0]}")
                logger.info(f"Will ONLY load grammar for: {primary_language[0]}")
            else:
                logger.warning("No primary language detected, defaulting to Python")
                primary_language = ["python"]
            
            # Load grammar ONLY for the primary language
            self._setup_tree_sitter(specific_languages=primary_language)
            
            # Run code analysis
            logger.info("Running code analysis...")
            analysis_results = self.analyze_code()
            
            # Add repository URL to results
            analysis_results["repository_url"] = repo_url
            analysis_results["primary_language"] = primary_language[0] if primary_language else "unknown"
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            analysis_results["analysis_time"] = elapsed_time
            logger.info(f"Repository analysis completed in {elapsed_time:.2f} seconds")
            
            return analysis_results
        except Exception as e:
            logger.error(f"Failed to analyze repository: {str(e)}")
            raise RepositoryAnalyzerError(f"Repository analysis failed: {str(e)}")
    
    def analyze_code(self) -> Dict[str, Any]:
        """
        Analyze the code in the repository.
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Force print for debugging
            print("CRITICAL DEBUG: Analyzer running with embedding_store:", self.embedding_store is not None)
            if self.embedding_store:
                # Verify embedding store is working properly
                try:
                    initial_files = self.embedding_store._get_all_files()
                    print(f"CRITICAL DEBUG: Embedding store contains {len(initial_files)} files at start of analysis")
                except Exception as e:
                    print(f"CRITICAL DEBUG: Error checking embedding store: {str(e)}")
            
            results = {
                "components": [],
                "data_flows": [],
                "dependencies": [],
                "architecture": {
                    "file_types": {},
                    "directory_structure": {},
                    "entry_points": []
                }
            }
            
            # Track total files and files parsed
            total_files = 0
            indexed_files = 0
            parsed_files = 0
            skipped_files = 0
            primary_language_files = 0  # Add counter for primary language files
            
            # Include only the languages we've loaded grammars for
            supported_extensions = []
            for lang_config in self.languages.values():
                supported_extensions.extend(lang_config["extensions"])
                
            logger.info(f"Analyzing files with extensions: {', '.join(supported_extensions)}")
            
            # Special info for PHP projects with .inc files
            if ".php" in supported_extensions and ".inc" in supported_extensions:
                logger.info("Detected PHP project with .inc files - both file types will be analyzed and indexed")
            
            # First pass: Count files to determine if repository is too large
            total_files_approx = 0
            is_large_repo = False
            for root, dirs, files in os.walk(self.repo_path):
                # Skip common directories to ignore
                if any(i in root for i in ['.git', 'node_modules', '__pycache__', 'venv', '.idea']):
                    continue
                total_files_approx += len(files)
            
            # For large repositories, we'll use a more selective approach
            # With 100K context window, we can set these limits much higher
            max_files_to_process = 100000  # Removed conditional for large repos, just use the higher limit
            max_files_per_extension = 20000  # Removed conditional for large repos, just use the higher limit
            
            logger.info(f"Repository size: approx. {total_files_approx} files. Using 100K context window to process up to {max_files_to_process} files")
            
            # Track directories and their files
            dir_structure = {}
            
            # First collect all file paths that match our criteria
            all_files = []
            for root, dirs, files in os.walk(self.repo_path):
                # Skip common directories to ignore
                if any(i in root for i in ['.git', 'node_modules', '__pycache__', 'venv', '.idea']):
                    continue
                    
                # Skip test directories and paths containing 'test'
                if 'test' in root.lower() or 'tests' in root.lower():
                    continue
                
                # Update directory structure
                rel_root = os.path.relpath(root, self.repo_path)
                if rel_root == '.':
                    rel_root = ''
                
                # Build directory structure incrementally
                dir_path = rel_root.split(os.sep)
                current_dir = dir_structure
                for part in dir_path:
                    if part:
                        if part not in current_dir:
                            current_dir[part] = {"files": [], "dirs": {}}
                        current_dir = current_dir[part]["dirs"]
                
                # Process files
                for file in files:
                    # Skip test files
                    if 'test' in file.lower():
                        continue
                        
                    total_files += 1
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Get file extension and check if it's supported for detailed parsing
                    ext = Path(file).suffix.lower()
                    is_supported = ext in supported_extensions
                    
                    # Count files matching primary language
                    if is_supported:
                        primary_language_files += 1
                    
                    # Add to directory structure
                    if rel_root:
                        parent_dir = dir_structure
                        for part in rel_root.split(os.sep):
                            if part:
                                parent_dir = parent_dir[part]["dirs"]
                        if "files" in parent_dir:
                            parent_dir["files"].append({"name": file, "path": rel_path, "supported": is_supported})
                    else:
                        if "files" not in dir_structure:
                            dir_structure["files"] = []
                        dir_structure["files"].append({"name": file, "path": rel_path, "supported": is_supported})
                    
                    # Track file types
                    if ext:
                        if ext not in results["architecture"]["file_types"]:
                            results["architecture"]["file_types"][ext] = 0
                        results["architecture"]["file_types"][ext] += 1
                    
                    # Add ALL files to our list for processing - not just supported ones
                    # This ensures all code files get indexed for RAG
                    all_files.append((rel_path, file_path, ext))
            
            # Save directory structure
            results["architecture"]["directory_structure"] = dir_structure
            
            # Identify potential entry points before filtering files
            logger.info("Identifying potential entry points...")
            potential_entry_points = []
            
            # Common entry point patterns
            entry_point_patterns = [
                "main", "app", "index", "server", "start", "init", "application",
                "api", "controller", "service", "router", "routes", "config", "settings"
            ]
            
            # First identify all potential entry points
            for rel_path, file_path, ext in all_files:
                filename = os.path.basename(file_path)
                name_stem = Path(filename).stem
                
                # Check if the filename indicates an entry point
                is_entry_point = False
                for pattern in entry_point_patterns:
                    if pattern in name_stem.lower():
                        is_entry_point = True
                        break
                
                # If it looks like an entry point, check content to confirm
                if is_entry_point or os.path.basename(os.path.dirname(file_path)) in entry_point_patterns:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            # Read just the first 2KB to check for entry point patterns
                            content = f.read(2048)
                            
                            # Check for language-specific entry point patterns
                            if ext == ".py" and ("if __name__ == '__main__'" in content or "if __name__ == \"__main__\"" in content):
                                potential_entry_points.append((rel_path, file_path, ext))
                                results["architecture"]["entry_points"].append(rel_path)
                            elif ext in [".js", ".ts"] and any(pattern in content for pattern in ["exports.", "module.exports", "export default", "export const", "app.listen(", "createServer", "addEventListener('load'"]):
                                potential_entry_points.append((rel_path, file_path, ext))
                                results["architecture"]["entry_points"].append(rel_path)
                            elif ext == ".java" and "public static void main" in content:
                                potential_entry_points.append((rel_path, file_path, ext))
                                results["architecture"]["entry_points"].append(rel_path)
                            elif ext == ".go" and "func main()" in content:
                                potential_entry_points.append((rel_path, file_path, ext))
                                results["architecture"]["entry_points"].append(rel_path)
                            elif ext == ".php" and "<?php" in content[:100]:
                                potential_entry_points.append((rel_path, file_path, ext))
                                results["architecture"]["entry_points"].append(rel_path)
                    except Exception as e:
                        logger.warning(f"Error checking for entry point in {file_path}: {str(e)}")
            
            logger.info(f"Found {len(potential_entry_points)} potential entry points")
            
            # Add GPU info
            if self.embedding_store:
                if hasattr(self.embedding_store, 'use_gpu_for_faiss') and self.embedding_store.use_gpu_for_faiss:
                    gpu_id = getattr(self.embedding_store, 'gpu_id', 0)
                    logger.info(f"Using GPU {gpu_id} acceleration for RAG indexing")
                    if hasattr(self.embedding_store, 'model') and self.embedding_store.model is not None:
                        device = self.embedding_store.model.device
                        logger.info(f"Embedding model running on device: {device}")
                else:
                    logger.warning("GPU acceleration is NOT enabled for RAG indexing - this will be slower")
                
            # Log that all files will be indexed
            logger.info(f"Indexing all files for RAG regardless of extension type or repository size")
            
            # For large repositories, prioritize and select a subset of files
            if len(all_files) > max_files_to_process:
                logger.info(f"Repository has {len(all_files)} supported files. Prioritizing important files...")
                
                # Group files by extension
                files_by_ext = {}
                for rel_path, file_path, ext in all_files:
                    if ext not in files_by_ext:
                        files_by_ext[ext] = []
                    files_by_ext[ext].append((rel_path, file_path, ext))
                
                # Limit files per extension
                selected_files = []
                
                # Always include entry points
                selected_files.extend(potential_entry_points)
                
                # Add a limited number of files from each extension
                for ext, files in files_by_ext.items():
                    # Skip if we already have enough from this extension via entry points
                    existing_count = sum(1 for f in selected_files if f[2] == ext)
                    if existing_count >= max_files_per_extension:
                        continue
                        
                    # Select remaining files up to the limit
                    remaining = max_files_per_extension - existing_count
                    # Exclude entry points which are already included
                    remaining_files = [f for f in files if f not in potential_entry_points]
                    # Prioritize files in src/, lib/, app/ directories
                    prioritized = [f for f in remaining_files if any(dir_name in f[0] for dir_name in ["src/", "lib/", "app/", "core/", "main/"])]
                    # Add prioritized files first
                    selected_files.extend(prioritized[:remaining])
                    # If we still have room, add other files
                    if len(prioritized) < remaining:
                        non_prioritized = [f for f in remaining_files if f not in prioritized]
                        selected_files.extend(non_prioritized[:remaining - len(prioritized)])
                
                # Make sure we don't exceed overall file limit
                if len(selected_files) > max_files_to_process:
                    logger.warning(f"Limiting analysis to {max_files_to_process} files from {len(selected_files)} selected files")
                    # Preserve entry points
                    entry_point_count = len(potential_entry_points)
                    remaining_slots = max_files_to_process - entry_point_count
                    if remaining_slots > 0:
                        other_files = [f for f in selected_files if f not in potential_entry_points]
                        selected_files = potential_entry_points + other_files[:remaining_slots]
                    else:
                        selected_files = potential_entry_points[:max_files_to_process]
                
                logger.info(f"Selected {len(selected_files)} files for analysis")
                # Replace all_files with our selected subset
                all_files = selected_files
            
            # Index the selected files for RAG
            reused_files = 0
            new_files = 0
            skipped_test_files = 0
            logger.info(f"Indexing {len(all_files)} files for RAG...")
            for rel_path, file_path, ext in all_files:
                # Skip test files
                if 'test' in rel_path.lower():
                    skipped_test_files += 1
                    continue
                    
                if self.embedding_store:
                    try:
                        # Check if this file is already in the embedding store to avoid re-indexing
                        # when reusing embeddings
                        if self.embedding_store.contains_file(rel_path):
                            # Skip files that are already indexed
                            reused_files += 1
                            indexed_files += 1
                            continue
                            
                        # Skip very large binary files and common binary formats to avoid wasting storage
                        file_size = os.path.getsize(file_path)
                        if file_size > 10 * 1024 * 1024:  # 10MB is too large for effective RAG
                            logger.info(f"Skipping very large file for indexing: {rel_path} ({file_size / 1024 / 1024:.2f} MB)")
                            continue
                            
                        # Skip common binary formats that don't provide useful text
                        binary_exts = ['.exe', '.dll', '.bin', '.zip', '.jar', '.war', '.class', '.pyc', '.o', '.so', '.dylib', 
                                      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff', '.pdf', '.mp3', '.mp4', '.wav']
                        if ext in binary_exts:
                            logger.info(f"Skipping binary file: {rel_path}")
                            continue
                        
                        # Improved file reading that avoids loading entire files into memory
                        with open(file_path, 'rb') as f:
                            # Read first chunk to determine if it's text
                            first_chunk = f.read(4096)  # Read 4KB to check
                            
                            # Skip file if it appears to be binary (check for null bytes)
                            if b'\x00' in first_chunk:
                                logger.info(f"Skipping likely binary file: {rel_path}")
                                continue
                            
                            # Try to decode with utf-8, fallback to latin-1
                            try:
                                first_chunk_decoded = first_chunk.decode('utf-8', errors='replace')
                                # Seek back to start
                                f.seek(0)
                                # Read in chunks to avoid memory issues
                                decoded_content = ""
                                chunk_size = 1024 * 1024  # 1MB chunks
                                while True:
                                    chunk = f.read(chunk_size)
                                    if not chunk:
                                        break
                                    decoded_content += chunk.decode('utf-8', errors='replace')
                            except UnicodeDecodeError:
                                # Try latin-1 as fallback
                                f.seek(0)
                                decoded_content = ""
                                chunk_size = 1024 * 1024  # 1MB chunks
                                while True:
                                    chunk = f.read(chunk_size)
                                    if not chunk:
                                        break
                                    decoded_content += chunk.decode('latin-1', errors='replace')
                        
                        # Prepare metadata
                        metadata = {
                            "extension": ext,
                            "language": self._get_language_name(ext),
                            "file_size": os.path.getsize(file_path),
                            "is_supported": ext in supported_extensions,
                            "path_components": rel_path.split(os.sep)
                        }
                        
                        # Check if this is an entry point
                        is_entry_point = rel_path in results["architecture"]["entry_points"]
                        if is_entry_point:
                            metadata["is_entry_point"] = True
                        
                        # Add file to embedding store with metadata
                        self.embedding_store.add_file(rel_path, decoded_content, metadata)
                        new_files += 1
                        indexed_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to index {rel_path}: {str(e)}")
                        skipped_files += 1
                        
            if reused_files > 0:
                logger.info(f"Reused {reused_files} files already in embedding store, indexed {new_files} new files")
            
            if skipped_test_files > 0:
                logger.info(f"Skipped {skipped_test_files} test files from RAG indexing")
                
            # FORCE SAVE: Make sure all indexed files are saved immediately
            if self.embedding_store and new_files > 0:
                try:
                    logger.info("Forcing save of embedding store after indexing")
                    self.embedding_store.save()
                    
                    # Verify the save was successful by checking file count
                    pre_save_files = indexed_files
                    all_files_after_save = self.embedding_store._get_all_files()
                    if len(all_files_after_save) < pre_save_files:
                        logger.error(f"SAVE ERROR: Embedding store has fewer files after save ({len(all_files_after_save)}) than expected ({pre_save_files})")
                        print(f"CRITICAL DEBUG: Embedding store has fewer files after save: {len(all_files_after_save)} vs {pre_save_files}")
                    else:
                        logger.info(f"Verified save successful: {len(all_files_after_save)} files in embedding store")
                        print(f"CRITICAL DEBUG: Save successful with {len(all_files_after_save)} files")
                except Exception as e:
                    logger.error(f"Error forcing save of embedding store: {str(e)}")
                    print(f"CRITICAL DEBUG: Error saving embedding store: {str(e)}")

            # Process files in chunks to avoid memory issues
            chunk_size = 100  # Process 100 files at a time
            logger.info(f"Processing {len(all_files)} files in chunks of {chunk_size}...")
            
            # Calculate total chunks for progress reporting
            total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
            
            # Process each chunk
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(all_files))
                chunk = all_files[start_idx:end_idx]
                
                logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks} ({start_idx+1}-{end_idx} of {len(all_files)} files)")
                
                # Process files in this chunk
                for rel_path, file_path, ext in chunk:
                    # Skip test files
                    if 'test' in rel_path.lower():
                        skipped_files += 1
                        continue
                    
                    # If file isn't in a supported extension, we still want to index it for RAG
                    # but we'll skip the parsing/AST extraction
                    if ext not in supported_extensions:
                        # Try to add the file to the embedding store for RAG
                        if self.embedding_store:
                            try:
                                if not self.embedding_store.contains_file(rel_path):
                                    with open(file_path, 'rb') as f:
                                        content = f.read()
                                    
                                    # Try to decode with utf-8, fallback to latin-1
                                    try:
                                        decoded_content = content.decode('utf-8', errors='replace')
                                    except UnicodeDecodeError:
                                        decoded_content = content.decode('latin-1', errors='replace')
                                    
                                    # Add to embedding store
                                    metadata = {
                                        "extension": ext,
                                        "language": self._get_language_name(ext),
                                        "file_size": len(content),
                                        "is_supported": False,
                                        "path_components": rel_path.split(os.sep)
                                    }
                                    self.embedding_store.add_file(rel_path, decoded_content, metadata)
                                    indexed_files += 1
                            except Exception as e:
                                logger.warning(f"Failed to index unsupported file {rel_path}: {str(e)}")
                        continue
                        
                    # Only try to parse files with supported extensions
                    if ext in supported_extensions:
                        logger.info(f"Parsing: {rel_path}")
                        
                        # Parse file
                        ast_info = self._parse_file(file_path)
                        if not ast_info:
                            skipped_files += 1
                            continue
                        
                        parsed_files += 1
                        
                        # Get file metadata
                        metadata = {
                            "is_entry_point": rel_path in results["architecture"]["entry_points"]
                        }
                        
                        # Add AST info to metadata for cross-referencing
                        if self.embedding_store:
                            # Update metadata in embedding store
                            self._update_file_metadata(rel_path, ast_info)
                        
                        # Add component
                        component = {
                            "name": Path(file_path).stem,
                            "type": "file",
                            "path": rel_path,
                            "classes": ast_info["classes"],
                            "functions": ast_info["functions"],
                            "imports": ast_info["imports"],
                            "dependencies": ast_info["dependencies"],
                            "metadata": metadata
                        }
                        results["components"].append(component)
                        
                        # Add dependencies
                        results["dependencies"].extend(ast_info["dependencies"])
                        
                        # Analyze data flows
                        self._extract_data_flows(component, ast_info, results)
                
                # Clean up after each chunk to prevent memory buildup
                import gc
                gc.collect()
            
            # Build file relationship graph
            if self.embedding_store:
                try:
                    relationships = self.embedding_store.get_file_relationships()
                    results["file_relationships"] = relationships
                    logger.info(f"Built relationship graph for {len(relationships)} files")
                except Exception as e:
                    logger.warning(f"Failed to build relationship graph: {str(e)}")
            
            # Double-check the actual count of indexed files
            actual_indexed_files = 0
            if self.embedding_store:
                try:
                    all_files = self.embedding_store._get_all_files()
                    actual_indexed_files = len(all_files)
                    logger.info(f"Verified {actual_indexed_files} unique files in embedding store")
                    
                    # CRITICAL DEBUG OUTPUT
                    print(f"CRITICAL DEBUG: Final embedding store contains {actual_indexed_files} unique files")
                    print(f"CRITICAL DEBUG: First 10 files in embedding store: {all_files[:10]}")
                    
                    # If there's a mismatch, update the counter
                    if actual_indexed_files != indexed_files:
                        logger.warning(f"Indexed files counter ({indexed_files}) doesn't match actual count in embedding store ({actual_indexed_files})")
                        indexed_files = actual_indexed_files
                except Exception as e:
                    logger.warning(f"Failed to verify indexed files count: {str(e)}")
                    print(f"CRITICAL DEBUG: Failed to verify indexed files count: {str(e)}")
            
            logger.info(f"Code analysis completed: {parsed_files} files parsed, {indexed_files} files indexed for RAG, {skipped_files} files skipped out of {total_files} total files")
            logger.info(f"Files matching primary language: {primary_language_files} ({(primary_language_files/total_files)*100:.1f}% of repository)")
            
            # Clean up results
            results["dependencies"] = list(set(results["dependencies"]))  # Remove duplicates
            
            # Check if we found anything useful
            if not results["components"]:
                logger.warning("No code components found in the repository")
            if not results["data_flows"]:
                logger.warning("No data flows found in the repository")
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze code: {str(e)}")
            raise RepositoryAnalyzerError(f"Code analysis failed: {str(e)}")
    
    def _get_language_name(self, ext: str) -> str:
        """Get language name from file extension."""
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".jsx": "JavaScript (React)",
            ".ts": "TypeScript",
            ".tsx": "TypeScript (React)",
            ".java": "Java",
            ".go": "Go",
            ".php": "PHP",
            ".inc": "PHP",
            ".rb": "Ruby",
            ".c": "C",
            ".cpp": "C++",
            ".h": "C/C++ Header",
            ".cs": "C#",
            ".html": "HTML",
            ".css": "CSS",
            ".json": "JSON",
            ".xml": "XML",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".md": "Markdown",
            ".sql": "SQL"
        }
        return ext_to_lang.get(ext.lower(), "Unknown")
    
    def _detect_entry_point(self, filename: str, content: str, ext: str) -> bool:
        """Detect if a file is likely an entry point."""
        # Common entry point filenames
        entry_filenames = [
            "index", "main", "app", "server", "application", "program", 
            "start", "init", "bootstrap"
        ]
        
        # Check filename
        name = Path(filename).stem.lower()
        if name in entry_filenames:
            return True
        
        # Language-specific patterns
        if ext == ".py" and ("if __name__ == '__main__'" in content or "if __name__ == \"__main__\"" in content):
            return True
        elif ext == ".js" and ("exports.handler" in content or "addEventListener('load'" in content):
            return True
        elif ext == ".php" and ("<?php" in content[:100] or "function rcmail" in content):
            return True
        elif ext == ".java" and "public static void main" in content:
            return True
        elif ext == ".go" and "func main()" in content:
            return True
        
        return False
    
    def _identify_critical_files(self, architecture: Dict[str, Any]) -> List[str]:
        """Identify critical files based on architecture analysis."""
        critical_files = []
        
        # Use entry points as a starting point
        critical_files.extend(architecture.get("entry_points", []))
        
        # Look for security-related files
        security_patterns = [
            "auth", "security", "login", "password", "crypt", "hash", "permission", "access",
            "token", "jwt", "oauth", "session", "cookie", "credential", "secret", "key",
            "cert", "config", "env", "setting", "admin", "root", "sudo", "superuser"
        ]
        
        if self.embedding_store:
            # The embedding store now only contains files matching the repository's primary language
            for pattern in security_patterns:
                matching_files = self.embedding_store.search_by_file_path(f"*{pattern}*")
                critical_files.extend(matching_files)
        
        # Look for database-related files
        db_patterns = ["db", "database", "model", "schema", "migration", "sql", "query", "repository"]
        
        if self.embedding_store:
            # The embedding store now only contains files matching the repository's primary language
            for pattern in db_patterns:
                matching_files = self.embedding_store.search_by_file_path(f"*{pattern}*")
                critical_files.extend(matching_files)
        
        # Remove duplicates
        return list(set(critical_files))
    
    def _update_file_metadata(self, file_path: str, ast_info: Dict[str, Any]) -> None:
        """Update metadata for a file in the embedding store based on AST info."""
        if not self.embedding_store:
            return
            
        try:
            # Create a metadata update
            metadata_update = {
                "imports": ast_info.get("imports", []),
                "dependencies": ast_info.get("dependencies", []),
                "classes": [cls.get("name") for cls in ast_info.get("classes", [])],
                "functions": [func.get("name") for func in ast_info.get("functions", [])]
            }
            
            # Find the file in the embedding store and update its metadata
            # This is a simplified approach as our current API doesn't directly support metadata updates
            # In a real implementation, we'd want a proper method to update metadata
            found = False
            for i, info in enumerate(self.embedding_store.file_mapping):
                if info.get("file_path") == file_path:
                    found = True
                    # Update the metadata
                    current_metadata = info.get("metadata", {})
                    current_metadata.update(metadata_update)
                    self.embedding_store.file_mapping[i]["metadata"] = current_metadata
            
            if not found:
                logger.warning(f"Could not find {file_path} in embedding store to update metadata")
                
        except Exception as e:
            logger.warning(f"Failed to update metadata for {file_path}: {str(e)}")
    
    def _extract_data_flows(self, component: Dict[str, Any], ast_info: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Extract data flows from a component and add to results."""
        # Analyze data flows from functions
        for func in ast_info["functions"]:
            data_flow = {
                "source": component["name"],
                "function": func["name"],
                "parameters": func["parameters"]
            }
            results["data_flows"].append(data_flow)
            
        # Also add class methods as data flows
        for cls in ast_info["classes"]:
            for method in cls["methods"]:
                data_flow = {
                    "source": f"{component['name']}.{cls['name']}",
                    "function": method["name"],
                    "parameters": method["parameters"]
                }
                results["data_flows"].append(data_flow)

    def _safe_exception_handler(self, func_name: str, error: Exception, severity: str = "warning", reraise: bool = False, reraise_as = None) -> None:
        """
        Handle exceptions consistently.
        
        Args:
            func_name: Name of the function where exception occurred
            error: The exception object
            severity: Log severity (warning, error)
            reraise: Whether to re-raise the exception
            reraise_as: Exception class to re-raise as (if None, use original)
        """
        message = f"Failed in {func_name}: {str(error)}"
        
        if severity == "error":
            logger.error(message)
        else:
            logger.warning(message)
            
        if reraise:
            if reraise_as:
                raise reraise_as(message) from error
            else:
                raise error 

    def _get_all_files(self, repo_path: str, ignore_patterns: list = None) -> List[str]:
        """
        Get all files in the repository, filtering out unwanted files and directories.
        
        Args:
            repo_path: Path to the repository
            ignore_patterns: List of patterns to ignore
            
        Returns:
            List of file paths relative to the repository root
        """
        if ignore_patterns is None:
            ignore_patterns = []
            
        # Add common ignore patterns if not already present
        common_ignores = [
            ".git", ".github", "__pycache__", ".pytest_cache", 
            ".vscode", ".idea", "node_modules", "venv", "env",
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dylib", "*.dll",
            "*.class", "*.jar", "*.war", "*.ear", "*.zip", "*.tar.gz",
            "*.log", ".DS_Store", "Thumbs.db"
        ]
        
        for pattern in common_ignores:
            if pattern not in ignore_patterns:
                ignore_patterns.append(pattern)
                
        all_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in ignore_patterns)]
            
            for file in files:
                # Skip ignored files
                if any(fnmatch.fnmatch(file, pattern) for pattern in ignore_patterns):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Filter out test files
                if 'test' in rel_path.lower():
                    continue
                
                # Skip binary files and files that are too large
                try:
                    if os.path.getsize(file_path) > self.max_file_size:
                        continue
                        
                    # Try to read the first few bytes to check if it's a text file
                    with open(file_path, 'rb') as f:
                        content = f.read(min(1024, os.path.getsize(file_path)))
                        if b'\0' in content:  # Binary file check
                            continue
                            
                    all_files.append(rel_path)
                except Exception as e:
                    logger.warning(f"Error processing file {rel_path}: {str(e)}")
                    
        return all_files 