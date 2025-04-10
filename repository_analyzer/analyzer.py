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
from utils import info_msg, warning_msg
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
        Parse a file for AST and extract information.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Dictionary with parsed information or None if parsing failed
        """
        try:
            # Ensure file_path is absolute and relative paths are properly converted
            if not os.path.isabs(file_path):
                full_path = os.path.join(self.repo_path, file_path)
            else:
                full_path = file_path
                
            # Check if file exists
            if not os.path.isfile(full_path):
                logger.warning(f"File does not exist: {full_path}")
                return None
                
            # Get file language
            language = self._get_language(full_path)
            if not language:
                logger.debug(f"Unsupported language for file: {full_path}")
                return None
                
            # Read file content
            try:
                with open(full_path, 'rb') as f:
                    content = f.read()
            except (IOError, PermissionError) as e:
                logger.warning(f"Cannot read file {full_path}: {str(e)}")
                return None
                
            # Parse AST
            try:
                tree = language.parse(content)
                root_node = tree.root_node
                
                # Extract AST information
                ast_info = self._extract_ast_info(root_node, content)
                
                # Add file name
                ast_info["name"] = os.path.basename(full_path)
                
                # Add entry point information
                file_ext = os.path.splitext(full_path)[1]
                content_str = self._safe_decode(content)
                ast_info["is_entry_point"] = self._detect_entry_point(full_path, content_str, file_ext)
                
                return ast_info
            except Exception as e:
                logger.warning(f"Failed to parse AST for {full_path}: {str(e)}")
                return None
                
        except Exception as e:
            self._safe_exception_handler("_parse_file", e)
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
            
            # Save the supported extensions to the results
            analysis_results["supported_extensions"] = self.languages.keys()
            
            return analysis_results
        except Exception as e:
            logger.error(f"Failed to analyze repository: {str(e)}")
            raise RepositoryAnalyzerError(f"Repository analysis failed: {str(e)}")
    
    def analyze_code(self) -> Dict[str, Any]:
        """
        Analyze code in the repository and return results.
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Verify repository is available
            if not self.verify_repository():
                raise RepositoryAnalyzerError(f"Repository path {self.repo_path} is not valid")
            
            # Setup parser with detected languages
            if self.parser is None:
                info_msg("Initializing parser...")
                languages = self._detect_languages_in_repo()
                self._setup_tree_sitter(languages)
            
            # Initialize embedding store if needed
            if self.embedding_store is None:
                from .embedding_store import EmbeddingStore
                info_msg("Creating new embedding store")
                
                # Check for environment variable to control device
                force_cpu = os.environ.get("FORCE_CPU", "").lower() in ["true", "1", "yes"]
                force_gpu = os.environ.get("FORCE_GPU", "").lower() in ["true", "1", "yes"]
                
                # Determine device based on environment and availability
                device = None
                if force_cpu:
                    device = "cpu"
                elif force_gpu:
                    device = "cuda"
                    
                # Create store with appropriate device setting
                self.embedding_store = EmbeddingStore(device=device, gpu_id=self.gpu_ids[0] if self.gpu_ids else None)
            
            # Load existing embeddings if present
            self.embedding_store.load()
            
            # Get all code files
            info_msg("Scanning repository for code files...")
            ignore_patterns = [
                '**/node_modules/**', '**/.git/**', '**/venv/**', '**/.venv/**',
                '**/__pycache__/**', '**/dist/**', '**/build/**', '**/.idea/**',
                '**/.vscode/**', '**/tests/**', '**/test/**', '**/vendor/**',
                '**/*.min.js', '**/*.min.css', '**/jquery*.js', '**/bootstrap*.js',
                '**/bootstrap*.css', '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif',
                '**/*.ico', '**/*.svg', '**/*.woff', '**/*.woff2', '**/*.ttf',
                '**/*.eot', '**/*.map', '**/*.pb', '**/*.pyc', '**/*.pyo', '**/*.pyd',
                '**/*.so', '**/*.dylib', '**/*.dll', '**/*.exe', '**/*.bin',
                '**/*.obj', '**/*.o', '**/*.a', '**/*.lib', '**/*.out', '**/*.class',
                '**/*.jar', '**/*.war', '**/*.ear', '**/*.zip', '**/*.tar',
                '**/*.tar.gz', '**/*.tar.bz2', '**/*.rar', '**/*.7z'
            ]
            
            # Get all code files using a generator to avoid loading all files in memory at once
            files_to_analyze = self._get_all_files(str(self.repo_path), ignore_patterns)
            
            # Count total files for progress tracking
            file_count = sum(1 for f in self._get_all_files(str(self.repo_path), ignore_patterns))
            info_msg(f"Found {file_count} files to analyze")
            
            # Reinitialize all_files generator
            files_to_analyze = self._get_all_files(str(self.repo_path), ignore_patterns)
            
            # Skip empty repositories
            if file_count == 0:
                warning_msg("No files found in repository, skipping analysis")
                return {
                    "components": [],
                    "data_flows": [],
                    "architecture": self._analyze_architecture({})
                }
            
            # Analyze file types
            info_msg("Analyzing code files...")
            file_types = {}
            for file_path in self._get_all_files(str(self.repo_path), ignore_patterns):
                ext = Path(file_path).suffix.lower()
                if ext:
                    if ext not in file_types:
                        file_types[ext] = 0
                    file_types[ext] += 1
                    
            # Log file type distribution
            info_msg("File type distribution:")
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                info_msg(f"  {ext}: {count} files")
            
            # Initialize results
            components = []
            data_flows = []
            file_relationships = {}  # Dictionary to track file relationships
            
            # Initialize directory structure for architecture analysis
            directory_structure = {}
            
            # Track entry points and critical files
            entry_points = []
            
            # Configure batch processing to handle GPU memory better
            batch_size = 10  # Files per batch
            batch_count = (file_count + batch_size - 1) // batch_size
            info_msg(f"Processing in {batch_count} batches of up to {batch_size} files each")
            
            # Process each batch of files
            with tqdm(total=file_count, desc="Analyzing code files") as pbar:
                for i in range(0, len(files_to_analyze), batch_size):
                    batch_files = files_to_analyze[i:i+batch_size]
                    current_batch = []
                    
                    # Process each file in the batch
                    for file_path in batch_files:
                        try:
                            # Convert to absolute path if necessary
                            if not os.path.isabs(file_path):
                                abs_file_path = os.path.join(self.repo_path, file_path)
                            else:
                                abs_file_path = file_path
                                
                            # Verify the file exists before processing
                            if not os.path.isfile(abs_file_path):
                                logger.warning(f"File does not exist: {abs_file_path}")
                                pbar.update(1)
                                continue
                                
                            # Read file content - needed for embedding and entry point detection
                            content = None
                            try:
                                with open(abs_file_path, 'rb') as f:
                                    file_bytes = f.read()
                                    content = self._safe_decode(file_bytes)
                            except Exception as file_error:
                                logger.warning(f"Error reading file {abs_file_path}: {str(file_error)}")
                                pbar.update(1)
                                continue
                                
                            # Parse file AST
                            file_info = self._parse_file(abs_file_path)
                            if file_info:
                                # Create component from file info
                                component = {
                                    "name": os.path.basename(file_path),
                                    "path": file_path,  # Keep relative path for output
                                    "type": self._get_language_name(os.path.splitext(file_path)[1]),
                                    "classes": file_info.get("classes", []),
                                    "functions": file_info.get("functions", []),
                                    "imports": file_info.get("imports", []),
                                    "dependencies": file_info.get("dependencies", []),
                                    "metadata": {
                                        "is_entry_point": file_info.get("is_entry_point", False),
                                        "is_critical": False  # Will be updated later
                                    }
                                }
                                
                                current_batch.append((component, file_info, content))
                                
                                # Store entry points
                                if file_info.get("is_entry_point", False):
                                    entry_points.append(file_path)
                                    
                                # Add to directory structure for architecture analysis
                                rel_path = os.path.relpath(file_path, str(self.repo_path))
                                dir_path = os.path.dirname(rel_path)
                                if dir_path not in directory_structure:
                                    directory_structure[dir_path] = []
                                directory_structure[dir_path].append({
                                    "file": os.path.basename(file_path),
                                    "language": component["type"],
                                    "is_entry_point": file_info.get("is_entry_point", False)
                                })
                                
                            # Update progress bar
                            pbar.update(1)
                                
                        except Exception as e:
                            self._safe_exception_handler("analyze_file", e, severity="warning")
                            pbar.update(1)
                            continue
                    
                    # Process the batch
                    self._process_file_batch(current_batch, components, data_flows, file_relationships)
                    
                    # Memory cleanup for GPU accelerated code
                    if self.embedding_store and hasattr(self.embedding_store, 'device') and self.embedding_store.device == 'cuda':
                        try:
                            import torch
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                        except:
                            pass
            
            # Add enhanced architecture information
            architecture = self._analyze_architecture({
                "file_types": file_types,
                "directory_structure": directory_structure,
                "entry_points": entry_points,
            })
            
            # Identify critical files based on architecture analysis
            critical_files = self._identify_critical_files(architecture)
            
            # Update component metadata with critical file status
            for component in components:
                if component["path"] in critical_files:
                    component["metadata"]["is_critical"] = True
                    
                    # Add to embedding store metadata if not already there
                    if self.embedding_store:
                        self._update_file_metadata(component["path"], {"is_critical": True})
                    
            # Log entry points
            info_msg(f"Detected {len(entry_points)} entry points")
            for entry_point in entry_points[:5]:  # Show only a few
                info_msg(f"  {entry_point}")
            if len(entry_points) > 5:
                info_msg(f"  ... and {len(entry_points) - 5} more")
            
            # Log critical files
            info_msg(f"Identified {len(critical_files)} critical files")
            for critical_file in critical_files[:5]:  # Show only a few
                info_msg(f"  {critical_file}")
            if len(critical_files) > 5:
                info_msg(f"  ... and {len(critical_files) - 5} more")
            
            # Update the embedding store
            if self.embedding_store:
                # Save embeddings before returning
                try:
                    info_msg("Saving embedding store...")
                    self.embedding_store.save()
                    info_msg("Embedding store saved successfully")
                except Exception as e:
                    warning_msg(f"Failed to save embedding store: {str(e)}")
                    
                # Release GPU memory after saving
                try:
                    import torch
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    info_msg("Released GPU memory after saving embeddings")
                except:
                    pass
            
            # Return results
            return {
                "components": components,
                "data_flows": data_flows,
                "architecture": architecture,
                "entry_points": entry_points,
                "critical_files": critical_files,
                "file_relationships": file_relationships
            }
            
        except Exception as e:
            self._safe_exception_handler("analyze_code", e, severity="error", reraise=True)
            return {}  # This line will only be reached if reraise is set to False

    def _process_file_batch(self, batch, components, data_flows, file_relationships):
        """Process a batch of files for analysis and embedding."""
        for component, file_info, content in batch:
            # Add component
            components.append(component)
            
            # Extract data flows
            self._extract_data_flows(component, file_info, data_flows)
            
            # Extract file relationships
            imports = file_info.get("imports", [])
            file_path = component["path"]
            
            # Convert imports to file paths if possible
            related_files = []
            for imp in imports:
                # Simple heuristic: check if the repository contains a file with this name
                if "/" in imp or "\\" in imp:
                    # This looks like a path
                    imp_parts = imp.replace("\\", "/").split("/")
                    imp_name = imp_parts[-1]
                    
                    # Check for files that might match this import
                    repo_root = str(self.repo_path)
                    for root, _, files in os.walk(repo_root):
                        for file in files:
                            if file == imp_name or file == imp_name + ".py" or file == imp_name + ".js":
                                related_path = os.path.join(root, file)
                                related_files.append(related_path)
                                break
            
            # Store file relationships
            if related_files:
                file_relationships[file_path] = related_files
            
            # Add to embedding store if available
            if self.embedding_store and content:
                try:
                    enhanced_content = content
                    
                    # Add some context from file info to improve embeddings
                    if file_info.get("classes") or file_info.get("functions"):
                        # Create a summary of the file structure
                        summary = f"File: {component['name']}\nLanguage: {component['type']}\n\n"
                        
                        # Add classes
                        if file_info.get("classes"):
                            summary += "Classes:\n"
                            for cls in file_info.get("classes")[:5]:  # Limit to first 5
                                summary += f"- {cls.get('name')}\n"
                            if len(file_info.get("classes")) > 5:
                                summary += f"... and {len(file_info.get('classes')) - 5} more\n"
                        
                        # Add functions
                        if file_info.get("functions"):
                            summary += "\nFunctions:\n"
                            for func in file_info.get("functions")[:5]:  # Limit to first 5
                                summary += f"- {func.get('name')}\n"
                            if len(file_info.get("functions")) > 5:
                                summary += f"... and {len(file_info.get('functions')) - 5} more\n"
                        
                        # Add imports
                        if file_info.get("imports"):
                            summary += "\nImports:\n"
                            for imp in file_info.get("imports")[:5]:  # Limit to first 5
                                summary += f"- {imp}\n"
                            if len(file_info.get("imports")) > 5:
                                summary += f"... and {len(file_info.get('imports')) - 5} more\n"
                        
                        # Enhance content with summary
                        enhanced_content = summary + "\n\n" + content[:50000]  # Limit content to 50K chars
                    
                    # Add file to embedding store with metadata
                    self.embedding_store.add_file(
                        file_path, 
                        enhanced_content, 
                        metadata={
                            "language": component["type"],
                            "is_entry_point": component["metadata"].get("is_entry_point", False),
                            "extension": os.path.splitext(file_path)[1].lower()[1:],  # Extension without dot
                            "classes": [cls.get("name") for cls in file_info.get("classes", [])],
                            "functions": [func.get("name") for func in file_info.get("functions", [])],
                            "imports": file_info.get("imports", [])
                        }
                    )
                except Exception as e:
                    warning_msg(f"Error adding {file_path} to embedding store: {str(e)}")
                
            # Release GPU memory after each file if using CUDA
            if self.embedding_store and hasattr(self.embedding_store, 'device') and self.embedding_store.device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

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
            "*.class", "*.jar", "*.war", "*.ear", "*.zip", ".tar.gz",
            "*.log", ".DS_Store", "Thumbs.db"
        ]
        
        for pattern in common_ignores:
            if pattern not in ignore_patterns:
                ignore_patterns.append(pattern)
                
        all_files = []
        
        # Ensure repo_path is an absolute path
        abs_repo_path = os.path.abspath(repo_path)
        if not os.path.exists(abs_repo_path):
            logger.warning(f"Repository path does not exist: {abs_repo_path}")
            return all_files
            
        for root, dirs, files in os.walk(abs_repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in ignore_patterns)]
            
            for file in files:
                # Skip ignored files
                if any(fnmatch.fnmatch(file, pattern) for pattern in ignore_patterns):
                    continue
                    
                file_path = os.path.join(root, file)
                # Convert to absolute path if not already
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
                
                # Check if file exists
                if not os.path.isfile(file_path):
                    continue
                    
                rel_path = os.path.relpath(file_path, abs_repo_path)
                
                # Filter out test files
                if 'test' in rel_path.lower():
                    continue
                
                # Skip binary files and files that are too large
                try:
                    if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10 MB
                        logger.info(f"Skipping large file: {rel_path} ({os.path.getsize(file_path) / 1024 / 1024:.2f} MB)")
                        continue
                        
                    # Check if file is binary
                    is_binary = False
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read(1024)  # Try to read as text
                    except UnicodeDecodeError:
                        is_binary = True
                    
                    if is_binary:
                        continue
                        
                    all_files.append(rel_path)
                except (IOError, OSError) as e:
                    logger.warning(f"Error checking file {rel_path}: {str(e)}")
                    continue
                    
        return all_files

    def _analyze_architecture(self, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the architecture of the codebase.
        
        Args:
            architecture_data: Dictionary containing architecture information
            
        Returns:
            Dictionary with enhanced architecture information
        """
        try:
            # Extract architecture data
            file_types = architecture_data.get("file_types", {})
            directory_structure = architecture_data.get("directory_structure", {})
            entry_points = architecture_data.get("entry_points", [])
            
            # Create base architecture object
            architecture = {
                "file_types": file_types,
                "directory_structure": directory_structure,
                "entry_points": entry_points,
            }
            
            # Determine dominant language based on file types
            dominant_language = None
            max_count = 0
            for ext, count in file_types.items():
                if count > max_count:
                    max_count = count
                    dominant_language = self._get_language_name(ext)
            
            if dominant_language:
                architecture["dominant_language"] = dominant_language
            
            # Identify key directories
            key_directories = []
            for dir_path, files in directory_structure.items():
                # Skip empty directories or those with just one file
                if len(files) <= 1:
                    continue
                    
                # Check if directory contains entry points
                contains_entry_point = any(file["is_entry_point"] for file in files)
                
                # Check for directories that might indicate architectural significance
                significant_name = any(name in dir_path.lower() for name in [
                    "src", "lib", "core", "api", "controller", "model", "view", "service",
                    "util", "common", "main", "app", "module", "component"
                ])
                
                if contains_entry_point or significant_name:
                    key_directories.append({
                        "path": dir_path,
                        "file_count": len(files),
                        "languages": list(set(file["language"] for file in files)),
                        "contains_entry_point": contains_entry_point
                    })
            
            architecture["key_directories"] = key_directories
            
            return architecture
            
        except Exception as e:
            self._safe_exception_handler("analyze_architecture", e, severity="warning")
            # Return minimal architecture data to prevent further errors
            return {
                "file_types": architecture_data.get("file_types", {}),
                "directory_structure": architecture_data.get("directory_structure", {}),
                "entry_points": architecture_data.get("entry_points", [])
            } 