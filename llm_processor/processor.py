"""
LLM Processor Module

This module provides functionality for processing code analysis results
using a local LLM to generate threat models and security insights.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from repository_analyzer.embedding_store import EmbeddingStore

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable
from utils.model_utils import download_model, validate_model_path, check_model_file, detect_gpu_capabilities
from utils.model_config import get_default_model_path, get_model_info

logger = logging.getLogger(__name__)

class LLMProcessorError(Exception):
    """Custom exception for LLM processor errors."""
    pass

class Threat(BaseModel):
    """Model representing a security threat."""
    type: str
    description: str
    severity: str
    impact: str
    mitigation: str
    code_snippet: Optional[str] = None

class CodeAnalysis(BaseModel):
    """Model representing code analysis results."""
    component: str
    purpose: str
    key_functions: List[str]
    dependencies: List[str]
    security_considerations: List[str]

class LLMProcessor:
    """
    Processes code analysis results using a local LLM.
    
    This class handles threat analysis, code understanding, and
    security recommendations using CodeLlama.
    """
    
    def __init__(self, model_path_or_embedding_store: Union[str, EmbeddingStore], 
                distributed: bool = False, gpu_id: Optional[int] = None, 
                gpu_ids: Optional[List[int]] = None, memory_limit: Optional[float] = None,
                device: str = None):
        """
        Initialize the LLM processor.
        
        Args:
            model_path_or_embedding_store: Either a path to the model file or an instance of EmbeddingStore
            distributed: Whether to use distributed processing (multi-GPU) when available
            gpu_id: Specific GPU ID to use if multiple GPUs are available
            gpu_ids: List of specific GPU IDs to use if multiple GPUs are available
            memory_limit: Memory limit per GPU in GB
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        # Initialize core attributes
        self.distributed = distributed
        self.gpu_id = gpu_id
        self.gpu_ids = gpu_ids
        self.memory_limit = memory_limit
        self.device = device
        
        # Set a default max_context_size to prevent errors
        self.max_context_size = 2048
        self.max_tokens_out = 1024
        
        # Handle model path or embedding store
        if isinstance(model_path_or_embedding_store, str):
            self.embedding_store = EmbeddingStore()
            self.model_path = model_path_or_embedding_store
        else:
            self.embedding_store = model_path_or_embedding_store
            self.model_path = get_env_variable("LLM_MODEL_PATH", get_default_model_path())
        
        self.llm = None
        self._setup_llm()
        
    def _setup_llm(self) -> None:
        """Set up the local LLM with appropriate configuration."""
        try:
            # Import the GPU detection function
            from utils.model_utils import detect_gpu_capabilities
            
            # Import model config functions to get context size
            from utils.model_config import get_model_info
            
            # Get model info to determine context size
            model_info = get_model_info()
            model_ctx_size = model_info.get("context_length", 4096)
            
            # Get environment variable for force GPU usage (override auto-detection)
            force_gpu = os.environ.get("FORCE_GPU", "").lower() in ["true", "1", "yes"]
            force_cpu = os.environ.get("FORCE_CPU", "").lower() in ["true", "1", "yes"]
            
            # Detect GPU capabilities with forced settings if specified
            gpu_config = detect_gpu_capabilities(force_gpu=force_gpu, force_cpu=force_cpu)
            
            # First try to use LlamaCpp, which is better for GGUF files
            info_msg("Loading model using LlamaCpp (better for GGUF files)")
            try:
                from langchain_community.llms import LlamaCpp
                
                # Get model path from environment
                model_path = self.model_path
                info_msg(f"Loading model from path: {model_path}")
                
                if not os.path.exists(model_path):
                    warning_msg(f"Model file {model_path} does not exist")
                    raise FileNotFoundError(f"Model file {model_path} not found")
                
                # Configure LlamaCpp options
                # Use model context size from configuration, override with environment if specified
                n_ctx = int(get_env_variable("LLM_N_CTX", str(model_ctx_size)))
                info_msg(f"Using context window size: {n_ctx}")
                
                n_batch = int(get_env_variable("LLM_N_BATCH", "512"))  # Batch size
                n_gpu_layers = gpu_config['n_gpu_layers']  # Use detected GPU layers
                temperature = float(get_env_variable("LLM_TEMPERATURE", "0.7"))
                max_tokens = int(get_env_variable("LLM_MAX_TOKENS", "1024"))
                top_p = float(get_env_variable("LLM_TOP_P", "0.95"))
                
                # Detect architecture (x86 vs ARM)
                is_arm = False
                try:
                    import platform
                    machine = platform.machine().lower()
                    is_arm = "arm" in machine or "aarch" in machine
                except:
                    pass
                
                # Set environment variables to help CUDA detection in llama.cpp
                if n_gpu_layers > 0 and gpu_config['use_gpu']:
                    # Force CUDA detection on with environment variables
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
                    
                    # Unset CPU-specific environment variables that might interfere
                    for env_var in ["LLAMA_CLBLAST", "LLAMA_METAL", "LLAMA_CPU_ONLY"]:
                        if env_var in os.environ:
                            del os.environ[env_var]
                
                # Log GPU usage
                if n_gpu_layers > 0 and gpu_config['use_gpu']:
                    if is_arm:
                        # For ARM architectures, Metal might be better than CUDA
                        success_msg(f"Using GPU acceleration: {gpu_config['gpu_info']} with {n_gpu_layers} layers")
                        # Consider adding Metal-specific options on ARM if needed
                    else:
                        # For x86 architectures with NVIDIA GPUs
                        success_msg(f"Using GPU acceleration: {gpu_config['gpu_info']} with {n_gpu_layers} layers")
                        # Make sure CUDA is enabled in llama-cpp-python
                        os.environ["LLAMA_CUBLAS"] = "1"
                else:
                    warning_msg("Running on CPU only (no GPU acceleration)")
                
                # Save context size for token counting logic
                self.max_context_size = n_ctx
                self.max_tokens_out = max_tokens
                
                # Create LlamaCpp instance with proper configuration
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_gpu_layers=n_gpu_layers,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    verbose=True,
                    f16_kv=True,  # Use half-precision for key/value cache
                    streaming=False,  # Disable streaming for now as it can cause issues
                    seed=42,  # Set a fixed seed for reproducibility
                    use_mlock=True  # Use mlock to keep the model in memory
                )
                
                success_msg("Successfully loaded model using LlamaCpp")
                return  # Exit the method since we successfully loaded the model
                
            except Exception as llama_e:
                error_msg(f"Failed to load model with LlamaCpp: {str(llama_e)}")
                warning_msg("Falling back to Hugging Face transformers approach")
            
            # If LlamaCpp fails, try the transformers approach
            info_msg("Loading model directly from Hugging Face")
            
            # Force CPU offloading to avoid memory issues and segmentation faults
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
            # Load model directly from Hugging Face using transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline
            import accelerate
            
            # Try safer initialization with lower memory footprint
            try:
                # Initialize tokenizer and model
                model_id = get_env_variable("LLM_MODEL_ID", "TheBloke/CodeLlama-7B-Instruct-GGUF")
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.tokenizer = tokenizer  # Store tokenizer for token counting later
                
                # Set max context size based on model capabilities
                self.max_context_size = int(get_env_variable("LLM_N_CTX", "2048"))  # Default to 2048 tokens
                
                # Check for environment variable to control loading parameters
                use_float16 = get_env_variable("LLM_USE_FLOAT16", "true").lower() == "true"
                low_mem = get_env_variable("LLM_LOW_MEM_USAGE", "true").lower() == "true"
                use_quantization = get_env_variable("LLM_USE_QUANTIZATION", "true").lower() == "true"
                
                # Add torch import to ensure it's available
                import torch
                
                # Use accelerate for better model loading
                from accelerate import init_empty_weights
                
                # Determine device based on GPU detection
                device = gpu_config['device'] if gpu_config['use_gpu'] else "cpu"
                
                # Skip quantization if disabled via environment variable
                if not use_quantization:
                    info_msg("Quantization disabled via LLM_USE_QUANTIZATION, using standard loading")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if use_float16 else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map=device if gpu_config['use_gpu'] else "cpu"
                    )
                else:
                    try:
                        # Install bitsandbytes if not available
                        warning_msg("bitsandbytes not found or incompatible, attempting to install latest version")
                        try:
                            import subprocess
                            import sys  # Add missing sys import
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"])
                            import bitsandbytes as bnb
                            from transformers import BitsAndBytesConfig
                            
                            # Set up quantization config with more robust defaults
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                bnb_4bit_compute_dtype=torch.float32,
                                bnb_4bit_use_double_quant=False,
                                bnb_8bit_use_double_quant=False,
                                bnb_8bit_enable_fp32_cpu_offload=True
                            )
                            
                            # Try again with updated bitsandbytes
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                quantization_config=quantization_config,
                                device_map=device if gpu_config['use_gpu'] else "cpu",
                                low_cpu_mem_usage=True
                            )
                        except Exception as install_e:
                            warning_msg(f"Failed to install or use bitsandbytes: {str(install_e)}")
                            raise  # Let the outer exception handler fall back to standard loading
                    except Exception as quant_e:
                        # If 8-bit quantization fails, try standard loading
                        warning_msg(f"8-bit quantization failed: {str(quant_e)}, trying standard loading")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if use_float16 else torch.float32,
                            low_cpu_mem_usage=True,
                            device_map=device if gpu_config['use_gpu'] else "cpu"
                        )
                
                # Create a pipeline with memory-efficient settings
                max_new_tokens = int(get_env_variable("LLM_MAX_TOKENS", "1024"))  # Reduced from default
                self.max_tokens_out = max_new_tokens
                temperature = float(get_env_variable("LLM_TEMPERATURE", "0.7"))
                
                pipe = pipeline(
                    "text-generation",
                    model=model, 
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                
                # Create the LangChain wrapper
                self.llm = HuggingFacePipeline(pipeline=pipe)
                
                # Report GPU status
                if gpu_config['use_gpu']:
                    success_msg(f"Successfully loaded model using Hugging Face with {gpu_config['device']} acceleration ({gpu_config['gpu_info']})")
                else:
                    success_msg("Successfully loaded model directly from Hugging Face (CPU only)")
            except Exception as local_e:
                error_msg(f"Failed to load model with standard approach: {str(local_e)}")
                
                # Try loading a smaller model if the main one fails
                try:
                    warning_msg("Attempting to load smaller model as fallback")
                    # Use a 1B parameter model that's much smaller
                    fallback_model_id = "gpt2"  # Very small model that should work on most systems
                    
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
                    model = AutoModelForCausalLM.from_pretrained(
                        fallback_model_id,
                        low_cpu_mem_usage=True,
                        device_map="cpu"  # Force CPU to avoid GPU memory issues
                    )
                    
                    # Create a simple pipeline with minimal settings
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=100,  # Very small output to avoid memory issues
                        temperature=0.7
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    warning_msg("Loaded smaller fallback model with limited capabilities")
                except Exception as fallback_e:
                    error_msg(f"Fallback loading also failed: {str(fallback_e)}")
                    self.llm = None
                
        except Exception as e:
            error_msg(f"Failed to set up LLM: {str(e)}")
            self.llm = None
    
    def _get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            if hasattr(self, 'tokenizer'):
                # Use the stored tokenizer if available
                return len(self.tokenizer.encode(text))
            else:
                # Estimate based on a simple heuristic if no tokenizer
                # This is a very rough estimate - about 4 chars per token for English
                return len(text) // 4
        except Exception as e:
            # Fallback to a very conservative estimate
            logger.warning(f"Failed to count tokens: {str(e)}")
            return len(text) // 3  # Even more conservative estimate
    
    def _check_tokens_and_truncate(self, prompt: str, max_output_tokens: int = None) -> str:
        """
        Check if a prompt will fit within the context window and truncate if needed.
        
        Args:
            prompt: The full prompt text
            max_output_tokens: Maximum tokens for the expected output
            
        Returns:
            Truncated prompt that will fit in the context window
        """
        if not max_output_tokens:
            max_output_tokens = self.max_tokens_out
            
        # Reserve space for output tokens
        available_tokens = self.max_context_size - max_output_tokens - 50  # 50 token safety buffer
        
        # Get token count for prompt
        token_count = self._get_token_count(prompt)
        
        if token_count <= available_tokens:
            return prompt
        
        # If prompt is too long, we need to truncate
        logger.warning(f"Prompt too large: {token_count} tokens exceeds available {available_tokens} tokens")
        
        # For now, implement a simple truncation strategy
        # This approach works best with prompts that have context at the end
        # For more sophisticated truncation, we would need to analyze the prompt structure
        
        # Simple truncation - keep instructions and final content
        if "\n\n" in prompt:
            # Split into sections
            sections = prompt.split("\n\n")
            instructions = sections[0]  # Keep the first section (usually instructions)
            
            # Figure out how many tokens we can keep
            instruction_tokens = self._get_token_count(instructions)
            if instruction_tokens <= 0:
                instruction_tokens = 1  # Avoid division by zero
                
            remaining_tokens = available_tokens - instruction_tokens - 20  # Additional buffer
            
            if remaining_tokens <= 0:
                # Instructions alone are too long
                return instructions[:len(instructions) // 2] + "..."
            
            # Join a subset of the remaining sections
            remaining_content = ""
            for section in sections[1:]:
                section_tokens = self._get_token_count(section)
                if section_tokens <= remaining_tokens:
                    remaining_content += "\n\n" + section
                    remaining_tokens -= section_tokens
                else:
                    # Last section is too big, truncate it
                    char_limit = int(remaining_tokens * 4)  # Rough character estimate
                    remaining_content += "\n\n" + section[:char_limit] + "..."
                    break
            
            return instructions + remaining_content
        else:
            # No clear sections, just truncate the middle
            chars_per_token = 4  # Rough estimate
            total_chars = len(prompt)
            keep_chars = available_tokens * chars_per_token
            
            if keep_chars >= total_chars:
                return prompt
            
            # Keep start and end, remove middle
            keep_start = keep_chars // 2
            keep_end = keep_chars // 2
            
            return prompt[:keep_start] + " [...content truncated...] " + prompt[-keep_end:]

    def _get_context_for_path(self, path: str, max_results: int = 5, is_php: bool = False, enhanced_context: str = None, filter_metadata: Dict[str, Any] = None) -> List[str]:
        """
        Helper method to retrieve context for a specific path.
        
        Args:
            path: Path to get context for
            max_results: Maximum number of context chunks to retrieve
            is_php: Whether this is a PHP file (needs special handling)
            enhanced_context: Pre-supplied context if available
            filter_metadata: Optional filter for context retrieval
            
        Returns:
            List of context chunks
        """
        context = []
        
        # Use enhanced context if available
        if enhanced_context:
            context.append(f"File Content:\n{enhanced_context}")
            return context
            
        # Otherwise get relevant context from embedding store
        if self.embedding_store:
            # For PHP files, try to get more context to compensate for potential AST parsing limitations
            actual_max = max_results * 2 if is_php else max_results
            retrieved_context = self.embedding_store.get_relevant_context(
                path,
                max_results=actual_max,
                filter_metadata=filter_metadata
            )
            context.extend(retrieved_context)
            
        return context

    def analyze_code_structure(self, component: Dict[str, Any]) -> CodeAnalysis:
        """
        Analyze the structure of a code component.
        
        Args:
            component: Dictionary containing component information
            
        Returns:
            CodeAnalysis object containing structural analysis
        """
        try:
            if not self.llm:
                raise LLMProcessorError("LLM is not available")
            
            # Extract component information
            name = component.get("name", "Unknown")
            path = component.get("path", "Unknown")
            classes = component.get("classes", [])
            functions = component.get("functions", [])
            imports = component.get("imports", [])
            dependencies = component.get("dependencies", [])
            
            # Get only first few classes and functions if there are many
            # This helps prevent exceeding token limits
            max_classes = 3  # Reduced from 5 to 3
            max_functions = 5  # Reduced from 10 to 5
            max_imports = 5   # Add limit for imports
            
            if len(classes) > max_classes:
                logger.warning(f"Component {name} has {len(classes)} classes, limiting to {max_classes}")
                classes = classes[:max_classes]
                
            if len(functions) > max_functions:
                logger.warning(f"Component {name} has {len(functions)} functions, limiting to {max_functions}")
                functions = functions[:max_functions]
                
            if len(imports) > max_imports:
                logger.warning(f"Component {name} has {len(imports)} imports, limiting to {max_imports}")
                imports = imports[:max_imports]
            
            # Retrieve file context - just 1 small chunk to save tokens
            file_contexts = self._get_context_for_path(path, max_results=1)
            
            # Limit context size to prevent token explosion
            context_text = "\n".join(file_contexts)
            if len(context_text) > 2000:  # Limit context to ~2000 chars 
                context_text = context_text[:2000] + "... [truncated]"
            
            # Construct a prompt for the LLM - simplified to save tokens
            prompt_template = """Analyze this code component:

Component: {name} ({path})

Classes: {classes}
Functions: {functions}
Imports: {imports}
Code snippet: {code_context}

For this component, analyze:
1. Purpose?
2. Key functions?
3. Dependencies?
4. Security considerations?

Return concisely as:
Purpose: <description>
Key Functions: <comma-separated list>
Dependencies: <comma-separated list>
Security Considerations: <comma-separated list>
"""
            
            # Format the classes and functions for the prompt - more concisely
            classes_text = ""
            for i, cls in enumerate(classes):
                cls_name = cls.get("name", "Unknown")
                methods = cls.get("methods", [])
                # Just list the class and method count rather than all methods
                classes_text += f"- {cls_name} ({len(methods)} methods)\n"
            
            functions_text = ""
            for i, func in enumerate(functions):
                func_name = func.get("name", "Unknown")
                params = func.get("parameters", [])
                # Just name and param count to save tokens
                functions_text += f"- {func_name} ({len(params)} params)\n"
            
            # Format imports more concisely
            imports_text = ", ".join(imports[:max_imports])
            if len(imports) > max_imports:
                imports_text += f" ... and {len(imports) - max_imports} more"
                
            # Check if the combined text would be too long for the context window
            combined_text = (
                f"Component: {name} ({path})\n\n"
                f"Classes:\n{classes_text}\n"
                f"Functions:\n{functions_text}\n"
                f"Imports: {imports_text}\n"
                f"Code snippet: {context_text}"
            )
            
            token_count = self._get_token_count(combined_text)
            max_tokens = self.max_context_size - 800  # Reserve 800 tokens for the prompt and output
            
            if token_count > max_tokens:
                # If too long, truncate the context which is usually the longest part
                logger.warning(f"Code analysis context too large: {token_count} tokens, truncating")
                reduction_factor = max_tokens / token_count
                new_context_length = int(len(context_text) * reduction_factor * 0.7)  # 30% extra margin
                if new_context_length < len(context_text):
                    context_text = context_text[:new_context_length] + "... [truncated]"
            
            # Create chain and run analysis
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({
                "name": name,
                "path": path,
                "classes": classes_text or "None",
                "functions": functions_text or "None",
                "imports": imports_text or "None",
                "code_context": context_text or "No context available"
            })
            
            # Extract results manually to save tokens in future calls
            purpose = self._extract_section(result, "Purpose")
            key_functions = self._extract_list(result, "Key Functions")
            dependencies = self._extract_list(result, "Dependencies")
            security_considerations = self._extract_list(result, "Security Considerations")
            
            # Create and return code analysis object
            return CodeAnalysis(
                component=name,
                purpose=purpose,
                key_functions=key_functions,
                dependencies=dependencies,
                security_considerations=security_considerations
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze code structure: {str(e)}")
            raise LLMProcessorError(f"Code structure analysis failed: {str(e)}")
    
    def analyze_component(self, component: Dict[str, Any]) -> List[Threat]:
        """
        Analyze a component for security threats.
        
        Args:
            component: Component information from repository analysis
            
        Returns:
            List of identified threats
        """
        try:
            # Apply aggressive token limiting to component
            limited_component = self._limit_component_context(component)
            
            # First, analyze the code structure
            code_analysis = self.analyze_code_structure(limited_component)
            
            # Get appropriate context using the helper method - REDUCING MAX_RESULTS FROM 5 TO 2
            context = self._get_context_for_path(
                limited_component["path"],
                max_results=2,  # Reduced from 5 to 2 to prevent token limit issues
                enhanced_context=limited_component.get("enhanced_context")
            )
            
            # Create prompt parameters first so we can check tokens before invocation
            prompt_params = {
                "component_name": limited_component["name"],
                "component_path": limited_component["path"],
                "purpose": code_analysis.purpose,
                "key_functions": ", ".join(code_analysis.key_functions[:5]),  # Limit to 5 key functions
                "dependencies": ", ".join(code_analysis.dependencies[:5]),   # Limit to 5 dependencies
                "security_considerations": ", ".join(code_analysis.security_considerations[:5]),  # Limit to 5
                "context": "\n".join(context),
                "file_extension": Path(limited_component["path"]).suffix.lstrip('.')
            }
            
            # Check if combined context is too large and reduce if needed
            combined_text = (
                f"Component: {prompt_params['component_name']}\n"
                f"Path: {prompt_params['component_path']}\n\n"
                f"Component Purpose: {prompt_params['purpose']}\n"
                f"Key Functions: {prompt_params['key_functions']}\n"
                f"Dependencies: {prompt_params['dependencies']}\n"
                f"Security Considerations: {prompt_params['security_considerations']}\n\n"
                f"Context:\n{prompt_params['context']}"
            )
            
            # Calculate token count for combined text
            token_count = self._get_token_count(combined_text)
            
            # If token count is too high, truncate context
            max_allowed = self.max_context_size - 1500  # Reserve 1500 tokens for prompt instructions and output
            if token_count > max_allowed:
                logger.warning(f"Component context too large: {token_count} tokens exceeds {max_allowed}. Truncating to fit in context window.")
                
                # Calculate how much we need to reduce
                reduction_factor = max_allowed / token_count
                
                # Get approximate character count for the new context
                original_chars = len(prompt_params["context"])
                new_chars = int(original_chars * reduction_factor * 0.7)  # Adding 30% extra margin
                
                # Truncate context while keeping beginning and end parts
                if new_chars < original_chars:
                    if original_chars > 0:
                        # Keep the first 40% and last 30% of the context
                        first_part = int(new_chars * 0.6)
                        last_part = int(new_chars * 0.4)
                        
                        if first_part + last_part < original_chars:
                            context_text = prompt_params["context"]
                            prompt_params["context"] = (
                                context_text[:first_part] + 
                                "\n\n[...content truncated...]\n\n" + 
                                context_text[-last_part:]
                            )
            
            # Construct prompt for threat analysis with more detailed instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a security expert analyzing code for potential threats using the STRIDE methodology. 
Be thorough but concise in your analysis."""),
                ("human", """
                Component: {component_name}
                Path: {component_path}
                
                Component Purpose: {purpose}
                Key Functions: {key_functions}
                Dependencies: {dependencies}
                Security Considerations: {security_considerations}
                
                Context:
                {context}
                
                Please analyze this component for security threats using STRIDE:
                - Spoofing
                - Tampering
                - Repudiation
                - Information disclosure
                - Denial of service
                - Elevation of privilege
                
                For each threat, provide:
                - Type
                - Description
                - Severity (Low/Medium/High)
                - Impact
                - Mitigation
                - Relevant code snippet (if any)
                
                Be specific about relevant threats based on the code structure and language ({file_extension}).
                """)
            ])
            
            # Create chain and run analysis
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke(prompt_params)
            
            # Parse and return threats
            threats = self._parse_threats(result)
            return threats
            
        except Exception as e:
            logger.error(f"Failed to analyze component: {str(e)}")
            raise LLMProcessorError(f"Component analysis failed: {str(e)}")
    
    def analyze_data_flow(self, data_flow: Dict[str, Any], additional_context: str = None) -> List[Threat]:
        """
        Analyze a data flow for security threats.
        
        Args:
            data_flow: Data flow information from repository analysis
            additional_context: Additional context about the data flow (optional)
            
        Returns:
            List of identified threats
        """
        try:
            # Get relevant context
            contexts = []
            
            # Add provided additional context if any
            if additional_context:
                contexts.append(additional_context)
                
            # Add context from embedding store using the helper method
            if self.embedding_store:
                retrieved_contexts = self._get_context_for_path(
                    data_flow["source"],
                    max_results=5
                )
                contexts.extend(retrieved_contexts)
            
            # Construct prompt for data flow analysis with enhanced instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a security expert analyzing data flows for potential threats. 
Focus on identifying how data moves through the system and potential security issues at each point.
Pay special attention to:
- Input validation and sanitization
- Authentication and authorization checks
- Data encryption and protection
- Error handling and information leakage
- Access control enforcement
- Injection vulnerabilities (SQL, command, etc.)
- Cross-site scripting opportunities
- File operation security"""),
                ("human", """
                Data Flow:
                Source: {source}
                Function: {function}
                Parameters: {parameters}
                
                Context:
                {context}
                
                Please analyze this data flow for security threats, focusing on:
                1. Data validation
                2. Authentication
                3. Authorization
                4. Encryption
                5. Error handling
                
                For each threat, provide:
                - Type
                - Description
                - Severity (Low/Medium/High)
                - Impact
                - Mitigation
                - Relevant code snippet (if any)
                
                Also identify how this data flow interacts with other components and any security boundaries it crosses.
                """)
            ])
            
            # Create chain and run analysis
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "source": data_flow["source"],
                "function": data_flow["function"],
                "parameters": ", ".join(data_flow.get("parameters", [])),
                "context": "\n".join(contexts)
            })
            
            # Parse and return threats
            return self._parse_threats(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze data flow: {str(e)}")
            raise LLMProcessorError(f"Data flow analysis failed: {str(e)}")
    
    def _generate_dummy_threat_model(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a dummy threat model when LLM is not available.
        This provides a basic structure that the rest of the program can work with.
        
        Args:
            analysis_results: Results from repository analysis
            
        Returns:
            Dictionary containing a minimal threat model
        """
        logger.info("Generating dummy threat model (no LLM available)")
        
        components = analysis_results.get("components", [])
        data_flows = analysis_results.get("data_flows", [])
        
        # Create basic threat model structure
        threat_model = {
            "components": [],
            "data_flows": [],
            "overall_risk": "Unknown (LLM unavailable)",
            "architecture": {
                "description": "Architecture analysis not available (LLM unavailable)",
                "layers": [],
                "frameworks": [],
                "tech_stack": []
            },
            "code_flow": [],
            "security_boundaries": [],
            "vulnerabilities": [],
            "cross_boundary_flows": []
        }
        
        # Add basic component info
        for component in components:
            threat_model["components"].append({
                "name": component["name"],
                "type": component["type"],
                "path": component["path"],
                "priority": False,
                "threats": [{
                    "type": "Information Disclosure",
                    "description": "This is a placeholder threat (LLM unavailable)",
                    "severity": "Medium",
                    "impact": "Unknown (LLM unavailable)",
                    "mitigation": "Please download the LLM model to get real threat analysis"
                }]
            })
        
        # Add basic data flow info
        for flow in data_flows:
            threat_model["data_flows"].append({
                "source": flow["source"],
                "destination": flow.get("destination", "Unknown"),
                "function": flow["function"],
                "description": "Data flow (details unavailable without LLM)",
                "threats": [{
                    "type": "Information Disclosure",
                    "description": "This is a placeholder threat (LLM unavailable)",
                    "severity": "Medium",
                    "impact": "Unknown (LLM unavailable)",
                    "mitigation": "Please download the LLM model to get real threat analysis"
                }]
            })
        
        # Add security boundaries
        for ext in [".php", ".py", ".js"]:
            files = [c["path"] for c in components if c["path"].endswith(ext)]
            if files:
                threat_model["security_boundaries"].append({
                    "name": f"{ext[1:].upper()} Components",
                    "description": f"Automatically detected boundary for {ext} files",
                    "files": files[:5]  # Just include a few as examples
                })
        
        return threat_model

    def generate_threat_model(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive threat model from code analysis results.
        
        Args:
            analysis_results: Code analysis results from repository analyzer
            
        Returns:
            Dictionary containing the threat model
        """
        try:
            logger.info("Generating comprehensive threat model with RAG")
            
            # Check if LLM is available
            if not self.llm:
                logger.warning("LLM not available, generating minimal threat model")
                return self._generate_minimal_threat_model(analysis_results)
                
            # Extract components, data flows and architecture info
            components = analysis_results.get("components", [])
            data_flows = analysis_results.get("data_flows", [])
            file_types = analysis_results.get("architecture", {}).get("file_types", {})
            directory_structure = analysis_results.get("architecture", {}).get("directory_structure", {})
            entry_points = analysis_results.get("architecture", {}).get("entry_points", [])
            file_relationships = analysis_results.get("file_relationships", {})
            
            logger.info("Performing overall codebase assessment with RAG")
            
            # Check if embedding store is available and reload if needed
            if self.embedding_store:
                try:
                    self.embedding_store.load()
                    logger.info("Embedding store reloaded")
                except Exception as e:
                    logger.warning(f"Failed to reload embedding store: {str(e)}")
            
            # Determine repository size and complexity
            num_components = len(components)
            num_data_flows = len(data_flows)
            large_repo = num_components > 50
            
            if self.embedding_store:
                num_files = len(self.embedding_store.file_mapping)
                logger.info(f"Found {num_files} unique files in embedding store")
                
                # With 100K context window, don't limit files - use all of them
                file_list = []
                
                # Add all entry points
                for entry_point in entry_points:
                    file_list.append(entry_point)
                
                # Add all security-related files
                security_files = self.embedding_store.search_by_file_path("*security*") + \
                               self.embedding_store.search_by_file_path("*auth*") + \
                               self.embedding_store.search_by_file_path("*login*") + \
                               self.embedding_store.search_by_file_path("*user*") + \
                               self.embedding_store.search_by_file_path("*password*") + \
                               self.embedding_store.search_by_file_path("*credential*")
                # Add unique security files
                for sec_file in security_files:
                    if sec_file not in file_list:
                        file_list.append(sec_file)
                
                # Add all core/model files
                model_files = self.embedding_store.search_by_file_path("*model*") + \
                            self.embedding_store.search_by_file_path("*database*") + \
                            self.embedding_store.search_by_file_path("*schema*") + \
                            self.embedding_store.search_by_file_path("*repository*")
                # Add unique model files
                for model_file in model_files:
                    if model_file not in file_list:
                        file_list.append(model_file)
                
                # Add all API/controller files
                api_files = self.embedding_store.search_by_file_path("*api*") + \
                          self.embedding_store.search_by_file_path("*controller*") + \
                          self.embedding_store.search_by_file_path("*service*") + \
                          self.embedding_store.search_by_file_path("*route*")
                # Add unique API files
                for api_file in api_files:
                    if api_file not in file_list:
                        file_list.append(api_file)
                
                # Add remaining files from the embedding store
                for info in self.embedding_store.file_mapping:  # Changed from dict.items() to list iteration
                    file_path = info.get("file_path", "")
                    if file_path and file_path not in file_list:
                        file_list.append(file_path)
                
                logger.info(f"Using all {len(file_list)} files for architecture analysis with 100K context window")
            else:
                # If no embedding store, use all component paths
                file_list = [c["path"] for c in components]
            
            # Determine technology stack
            tech_stack = self._determine_tech_stack(file_types)
            
            # Architecture analysis with limited file list
            try:
                architecture_analysis = self._explore_architecture(
                    tech_stack, 
                    file_list, 
                    entry_points,
                    file_relationships
                )
            except Exception as e:
                logger.warning(f"Error in final architecture analysis: {str(e)}")
                architecture_analysis = {
                    "description": "Limited analysis due to repository size",
                    "key_components": entry_points[:10],
                    "security_boundaries": []
                }
            
            # Extract security boundaries
            security_boundaries = architecture_analysis.get("security_boundaries", [])
            
            # Create initial threat model structure
            threat_model = {
                "overall_risk": "Medium",  # Default, will be calculated later
                "architecture": architecture_analysis,
                "components": [],
                "data_flows": [],
                "security_boundaries": security_boundaries,
                "code_flow": [],
                "vulnerabilities": [],
                "cross_boundary_flows": []
            }
            
            # Analyze components with RAG for dynamic context
            logger.info("Analyzing components with dynamic RAG")
                
            # With 100K context window, we can analyze more components
            max_components = 500 if large_repo else 1000  # Previously 100/200
            priority_components = []
            
            # Define priority patterns based on security relevance
            priority_patterns = [
                "auth", "login", "user", "admin", "password", "hash", "crypt", 
                "token", "jwt", "oauth", "permission", "role", "security", "key",
                "secret", "credential", "session", "cookie", "api", "controller",
                "upload", "file", "sql", "query", "database", "model", "payment",
                "validator", "request", "response", "config", "settings"
            ]
            
            # First find components matching priority patterns
            for component in components:
                name = component.get("name", "").lower()
                path = component.get("path", "").lower()
                
                is_priority = False
                # Check for priority patterns in name or path
                for pattern in priority_patterns:
                    if pattern in name or pattern in path:
                        is_priority = True
                        break
                        
                # Also consider entry points as priority components
                if path in entry_points:
                    is_priority = True
                    
                if is_priority:
                    priority_components.append(component)
                    
                # If we've reached our limit for priority components, stop
                if len(priority_components) >= max_components:
                    break
                    
            # If we still have room, add non-priority components up to the limit
            if len(priority_components) < max_components:
                remaining_slots = max_components - len(priority_components)
                non_priority = [c for c in components if c not in priority_components]
                priority_components.extend(non_priority[:remaining_slots])
                
            logger.info(f"Analyzing {len(priority_components)} priority components first")
            
            # Process components in batches to avoid memory issues
            batch_size = 20  # Process 20 components at a time
            components_analyzed = 0
            
            # Analyze components with RAG for dynamic context
            for component in priority_components:
                component_threats = self._analyze_component_with_exploration(component, analysis_results)
                threat_model["components"].append({
                    "name": component["name"],
                    "type": component["type"],
                    "path": component["path"],
                    "priority": True,
                    "threats": [threat.dict() for threat in component_threats]
                })
            
            # Analyze data flows with enhanced focus on security boundaries
            logger.info("Analyzing data flows with focus on security boundaries")
            data_flows_by_component = {}
            for data_flow in data_flows:
                source = data_flow["source"]
                if source not in data_flows_by_component:
                    data_flows_by_component[source] = []
                data_flows_by_component[source].append(data_flow)
            
            # Process data flows by component for better context
            for source, flows in data_flows_by_component.items():
                # Get combined context for all flows from this source
                contexts = []
                if self.embedding_store:
                    contexts = self._get_context_for_path(
                        source,
                        max_results=10  # Increased to get better context
                    )
                
                # Check if this flow crosses security boundaries
                crosses_boundary = False
                for flow in flows:
                    for cross_flow in threat_model["cross_boundary_flows"]:
                        if flow["source"] == cross_flow["source"] and flow["function"] == cross_flow["function"]:
                            crosses_boundary = True
                            break
                    if crosses_boundary:
                        break
                
                # Analyze each flow with the combined context
                for data_flow in flows:
                    # Add boundary crossing information to the flow
                    if crosses_boundary:
                        data_flow["crosses_boundary"] = True
                    
                    flow_threats = self._analyze_data_flow_with_exploration(
                        data_flow, 
                        "\n\n".join(contexts),
                        threat_model["security_boundaries"]
                    )
                    
                    threat_model["data_flows"].append({
                        "source": data_flow["source"],
                        "function": data_flow["function"],
                        "parameters": data_flow.get("parameters", []),
                        "crosses_boundary": crosses_boundary,
                        "threats": [threat.dict() for threat in flow_threats]
                    })
            
            # Perform cross-file vulnerability analysis
            logger.info("Performing cross-file vulnerability analysis")
            vulnerabilities = self._analyze_cross_file_vulnerabilities(
                analysis_results, 
                threat_model["components"],
                threat_model["data_flows"],
                threat_model["cross_boundary_flows"]
            )
            
            threat_model["vulnerabilities"] = vulnerabilities
            
            # Calculate overall risk based on all identified threats and vulnerabilities
            threat_model["overall_risk"] = self._calculate_enhanced_risk(threat_model)
            
            # Save threat model to file
            output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "threat_model.json", "w") as f:
                json.dump(threat_model, f, indent=2)
            
            return threat_model
            
        except Exception as e:
            logger.error(f"Failed to generate threat model: {str(e)}")
            raise LLMProcessorError(f"Threat model generation failed: {str(e)}")
            
    def _generate_minimal_threat_model(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a minimal threat model when no components are found.
        
        Args:
            analysis_results: Results from repository analysis
            
        Returns:
            Dictionary containing a minimal threat model
        """
        try:
            # Create basic threat model structure
            minimal_model = {
                "components": [],
                "data_flows": [],
                "overall_risk": "Unknown",
                "architecture": {
                    "architecture_pattern": "Unknown",
                    "components": [],
                    "entry_points": [],
                    "data_stores": []
                },
                "code_flow": [],
                "security_boundaries": [],
                "vulnerabilities": [],
                "cross_boundary_flows": []
            }
            
            # Add file types if available
            if "architecture" in analysis_results and "file_types" in analysis_results["architecture"]:
                file_types = analysis_results["architecture"]["file_types"]
                if file_types:
                    tech_stack = ", ".join([f"{ext} ({count} files)" for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)])
                    minimal_model["architecture"]["tech_stack"] = tech_stack
                    
                    # Determine likely architecture pattern based on file types
                    if ".php" in file_types:
                        minimal_model["architecture"]["architecture_pattern"] = "PHP Web Application"
                    elif ".py" in file_types:
                        minimal_model["architecture"]["architecture_pattern"] = "Python Application"
                    elif ".js" in file_types or ".jsx" in file_types or ".ts" in file_types:
                        minimal_model["architecture"]["architecture_pattern"] = "JavaScript/TypeScript Application"
            
            # Save the minimal threat model
            output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "threat_model.json", "w") as f:
                json.dump(minimal_model, f, indent=2)
                
            return minimal_model
            
        except Exception as e:
            logger.error(f"Failed to generate minimal threat model: {str(e)}")
            # Return an extremely minimal model as fallback
            return {
                "components": [],
                "data_flows": [],
                "overall_risk": "Unknown",
                "architecture": {},
                "vulnerabilities": []
            }
    
    def _explore_architecture(self, tech_stack: str, file_list: List[str], entry_points: List[str], file_relationships: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Explore architecture to identify security boundaries and zones.
        
        Args:
            tech_stack: Identified technology stack
            file_list: List of key files
            entry_points: List of entry point files
            file_relationships: Dictionary of file relationships
            
        Returns:
            Dictionary with architecture information
        """
        try:
            # Filter out test files from file list
            filtered_file_list = [file for file in file_list if 'test' not in file.lower()]
            if len(filtered_file_list) < len(file_list):
                logger.info(f"Filtered out {len(file_list) - len(filtered_file_list)} test files from architecture analysis")
                file_list = filtered_file_list
            
            # Filter out test files from entry points
            filtered_entry_points = [file for file in entry_points if 'test' not in file.lower()]
            if len(filtered_entry_points) < len(entry_points):
                logger.info(f"Filtered out {len(entry_points) - len(filtered_entry_points)} test files from entry points")
                entry_points = filtered_entry_points
            
            # Filter out test files from file relationships
            filtered_relationships = {}
            for file, related in file_relationships.items():
                if 'test' not in file.lower():
                    filtered_relationships[file] = [f for f in related if 'test' not in f.lower()]
            if len(filtered_relationships) < len(file_relationships):
                logger.info(f"Filtered out test files from file relationships")
                file_relationships = filtered_relationships
            
            # With 100K context window, we can analyze many more files
            # We'll still apply a high limit to prevent extremely rare cases of enormous repos
            max_files_to_check = 10000  # Previously 500, now much higher for 100K context
            
            if len(file_list) > max_files_to_check:
                logger.warning(f"Extremely large repository with {len(file_list)} files, limiting architecture analysis to {max_files_to_check} files")
                
                # Include a good mix of files - first prioritize entry points
                limited_files = []
                # Include all entry points 
                for entry in entry_points:
                    if entry not in limited_files:
                        limited_files.append(entry)
                        
                # Then prioritize by filename patterns
                priority_patterns = [
                    "security", "auth", "login", "user", "admin", "config", 
                    "main", "app", "index", "api", "controller", "model"
                ]
                
                # Add files matching priority patterns
                for pattern in priority_patterns:
                    matching_files = [f for f in file_list if pattern in f.lower()]
                    for file in matching_files:
                        if file not in limited_files and len(limited_files) < max_files_to_check:
                            limited_files.append(file)
                
                # Fill remaining slots with other files
                for file in file_list:
                    if file not in limited_files and len(limited_files) < max_files_to_check:
                        limited_files.append(file)
                
                file_list = limited_files
            
            # Use all entry points without limiting
            
            # Include all file relationships without limiting
            simplified_relationships = {}
            if file_relationships:
                # Only include relationships for files in our file list
                for file in file_list:
                    if file in file_relationships:
                        # With 100K context, we can include many more related files
                        related = [f for f in file_relationships[file] if f in file_list]
                        if related:
                            simplified_relationships[file] = related
            
            # Initialize results to build incrementally
            description = ""
            security_boundaries = []
            boundary_files = {}
            
            # Process files in batches, focusing on high priority files first
            for batch_idx in range(0, len(file_list), 100):
                batch_end = min(batch_idx + 100, len(file_list))
                current_batch = file_list[batch_idx:batch_end]
                
                logger.info(f"Processing architecture batch {batch_idx//100 + 1}: files {batch_idx+1}-{batch_end} of {len(file_list)}")
                
                # For all batches after the first, we have reduced analysis scope
                # to preserve important discoveries while adding new files
                is_first_batch = (batch_idx == 0)
                
                # Limit entry points for non-first batches
                batch_entry_points = entry_points if is_first_batch else entry_points[:3]
                
                # Limit file relationships for current batch
                batch_relationships = {}
                if file_relationships:
                    # Only include relationships for files in this batch
                    for file in current_batch:
                        if file in file_relationships:
                            # For first batch, include more relationships
                            rel_limit = 5 if is_first_batch else 3
                            related = [f for f in file_relationships[file] if f in current_batch][:rel_limit]
                            if related:
                                batch_relationships[file] = related
                
                # Format batch data for prompt
                file_list_formatted = "\n".join([f"- {file}" for file in current_batch])
                entry_points_formatted = "\n".join([f"- {ep}" for ep in batch_entry_points])
                
                # Format file relationships
                relationships_formatted = []
                for file, related in batch_relationships.items():
                    if related:  # Only include files with actual relationships
                        related_str = ", ".join(related)
                        relationships_formatted.append(f"- {file} -> {related_str}")
                
                # Limit relationships to prevent token explosion
                if len(relationships_formatted) > 20:
                    relationships_formatted = relationships_formatted[:20]
                    relationships_formatted.append("- ... (additional relationships omitted)")
                    
                relationships_text = "\n".join(relationships_formatted)
                
                # Create the prompt - modified for incremental analysis
                system_prompt = """You are an expert in software architecture and security analysis.
Your task is to identify the architecture layers and security boundaries in a codebase.
Be concise and focus on the most important security boundaries."""

                # Modify human prompt based on whether this is the first batch
                if is_first_batch:
                    human_prompt = """
I need you to analyze the architecture of a software project.

Technology stack: {tech_stack}

Key entry points:
{entry_points}

Here are some key files:
{file_list}

Some important file relationships:
{file_relationships}

Please identify:
1. The overall architecture description (1-2 sentences)
2. 2-4 key security boundaries/zones (e.g., frontend/backend, authenticated/unauthenticated, admin/user)
3. Which files belong in which security boundaries

Return your analysis in this format:
DESCRIPTION:
<1-2 sentence description>

SECURITY_BOUNDARIES:
- <boundary name>: <description>
- <boundary name>: <description>

FILES_BY_BOUNDARY:
- <boundary name>: <comma-separated file list>
- <boundary name>: <comma-separated file list>
"""
                else:
                    # For subsequent batches, include previous findings and focus on new files
                    human_prompt = """
I need you to continue analyzing the architecture of a software project. 
I've already analyzed some files and identified initial boundaries. Now I need you to look at additional files.

Technology stack: {tech_stack}

Previous findings:
{previous_description}

Previously identified security boundaries:
{previous_boundaries}

Additional files to analyze:
{file_list}

Some important file relationships:
{file_relationships}

Please:
1. Update or refine the architecture description if needed
2. Assign these additional files to existing boundaries or create new boundaries if necessary

Return your analysis in this format:
DESCRIPTION:
<1-2 sentence updated description>

SECURITY_BOUNDARIES:
- <boundary name>: <description>
- <boundary name>: <description>

FILES_BY_BOUNDARY:
- <boundary name>: <comma-separated file list>
- <boundary name>: <comma-separated file list>
"""

                # Prepare the prompt based on batch
                prompt_template = system_prompt if is_first_batch else system_prompt
                
                # Format the prompt parameters for this batch
                if is_first_batch:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ])
                    
                    chain = prompt | self.llm | StrOutputParser()
                    result = chain.invoke({
                        "tech_stack": tech_stack,
                        "entry_points": entry_points_formatted,
                        "file_list": file_list_formatted,
                        "file_relationships": relationships_text
                    })
                else:
                    # Format previous security boundaries for the prompt
                    previous_boundaries_text = "\n".join([f"- {b['name']}: {b['description']}" for b in security_boundaries])
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ])
                    
                    chain = prompt | self.llm | StrOutputParser()
                    result = chain.invoke({
                        "tech_stack": tech_stack,
                        "previous_description": description,
                        "previous_boundaries": previous_boundaries_text,
                        "file_list": file_list_formatted,
                        "file_relationships": relationships_text
                    })
                
                # Parse the result
                batch_description = self._extract_section(result, "DESCRIPTION")
                
                # Update overall description if meaningful
                if batch_description and (not description or len(batch_description) > 10):
                    description = batch_description
                
                # Extract security boundaries section
                boundaries_section = self._extract_section(result, "SECURITY_BOUNDARIES")
                
                # Parse security boundaries
                if boundaries_section and is_first_batch:
                    # For first batch, use the boundaries as-is
                    boundary_lines = boundaries_section.split("\n")
                    for line in boundary_lines:
                        line = line.strip()
                        if line.startswith("- "):
                            line = line[2:]  # Remove the "- " prefix
                            if ":" in line:
                                boundary_name, boundary_desc = line.split(":", 1)
                                security_boundaries.append({
                                    "name": boundary_name.strip(),
                                    "description": boundary_desc.strip(),
                                    "files": []  # Will fill this later
                                })
                elif boundaries_section and not is_first_batch:
                    # For subsequent batches, merge with existing boundaries
                    boundary_lines = boundaries_section.split("\n")
                    for line in boundary_lines:
                        line = line.strip()
                        if line.startswith("- "):
                            line = line[2:]  # Remove the "- " prefix
                            if ":" in line:
                                boundary_name, boundary_desc = line.split(":", 1)
                                boundary_name = boundary_name.strip()
                                
                                # Check if this boundary already exists
                                existing = False
                                for boundary in security_boundaries:
                                    if boundary["name"] == boundary_name:
                                        existing = True
                                        # Update description if new one is more detailed
                                        if len(boundary_desc.strip()) > len(boundary["description"]):
                                            boundary["description"] = boundary_desc.strip()
                                        break
                                
                                # If not found, add as new boundary
                                if not existing:
                                    security_boundaries.append({
                                        "name": boundary_name.strip(),
                                        "description": boundary_desc.strip(),
                                        "files": []  # Will fill this later
                                    })
                
                # Parse files by boundary
                files_by_boundary_section = self._extract_section(result, "FILES_BY_BOUNDARY")
                if files_by_boundary_section:
                    boundary_files_lines = files_by_boundary_section.split("\n")
                    for line in boundary_files_lines:
                        line = line.strip()
                        if line.startswith("- "):
                            line = line[2:]  # Remove the "- " prefix
                            if ":" in line:
                                boundary_name, file_list_str = line.split(":", 1)
                                boundary_name = boundary_name.strip()
                                
                                # Find the boundary in our list
                                for boundary in security_boundaries:
                                    if boundary["name"] == boundary_name:
                                        # Parse comma-separated file list
                                        new_files = [f.strip() for f in file_list_str.split(",")]
                                        
                                        # Add to existing files, avoiding duplicates
                                        existing_files = boundary["files"]
                                        for new_file in new_files:
                                            if new_file and new_file not in existing_files:
                                                existing_files.append(new_file)
                                        
                                        # Update the boundary
                                        boundary["files"] = existing_files
                                        break
            
            # Create architecture result
            architecture = {
                "description": description,
                "tech_stack": tech_stack.split(", "),
                "key_components": entry_points,
                "security_boundaries": security_boundaries,
            }
            
            # Add default boundary if none identified
            if not security_boundaries:
                architecture["security_boundaries"] = [{
                    "name": "Default Boundary",
                    "description": "Default security boundary for all components",
                    "files": file_list[:30]  # Limit to 30 files for default boundary
                }]
                
            return architecture
            
        except Exception as e:
            logger.error(f"Failed to explore architecture: {str(e)}")
            # Return a simplified structure
            return {
                "description": "Architecture analysis failed due to an error",
                "tech_stack": tech_stack.split(", "),
                "key_components": entry_points[:5],
                "security_boundaries": [{
                    "name": "Default Boundary",
                    "description": "Default security boundary for all components",
                    "files": file_list[:20]  # Limit to 20 files
                }]
            }
    
    def _query_rag_for_answer(self, question: str) -> str:
        """
        Query the RAG system to answer a specific question about the codebase.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer to the question
        """
        try:
            # Get relevant context from embedding store
            if not self.embedding_store:
                return "No embedding store available to retrieve context."
                
            contexts = self.embedding_store.get_relevant_context(question, max_results=5)
            if not contexts:
                return "No relevant information found."
            
            # Construct QA prompt
            qa_prompt_template = """
            You are a code analysis expert answering questions about a codebase based on the provided context.

            Question: {question}
            
            Context from codebase:
            {context}
            
            Please answer the question concisely based only on the provided context.
            """
            
            # Format the prompt
            prompt = qa_prompt_template.format(
                question=question,
                context="\n\n".join(contexts)
            )
            
            # Check token count and truncate if necessary
            prompt = self._check_tokens_and_truncate(prompt, max_output_tokens=300)
            
            # Call LLM directly
            answer = self.llm.invoke(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            return "Error retrieving information."
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        try:
            import re
            
            # Find JSON pattern in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.warning(f"Failed to extract JSON: {str(e)}")
            return {}
    
    def _identify_cross_boundary_flows(self, security_boundaries: List[Dict[str, Any]], data_flows: List[Dict[str, Any]], file_relationships: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Identify data flows that cross security boundaries.
        
        Args:
            security_boundaries: List of security boundaries
            data_flows: List of data flows
            file_relationships: Dictionary of file relationships
            
        Returns:
            List of cross-boundary flows
        """
        cross_flows = []
        
        # Build a map of components to their boundaries
        component_to_boundary = {}
        for boundary in security_boundaries:
            for component in boundary.get("components", []):
                component_to_boundary[component] = boundary["name"]
        
        # Check each data flow
        for flow in data_flows:
            source = flow.get("source", "")
            function = flow.get("function", "")
            
            # Skip if missing source or function
            if not source or not function:
                continue
            
            # Check if source and any related components are in different boundaries
            source_boundary = component_to_boundary.get(source)
            if not source_boundary:
                continue
            
            # Look for relationships that cross boundaries
            related_components = []
            if "." in source:
                # Handle class methods
                base_component = source.split(".")[0]
                if base_component in file_relationships:
                    related_components.extend(file_relationships[base_component])
            elif source in file_relationships:
                related_components.extend(file_relationships[source])
            
            for related in related_components:
                related_boundary = component_to_boundary.get(related)
                if related_boundary and related_boundary != source_boundary:
                    cross_flows.append({
                        "source": source,
                        "function": function,
                        "source_boundary": source_boundary,
                        "destination": related,
                        "destination_boundary": related_boundary,
                        "parameters": flow.get("parameters", [])
                    })
        
        return cross_flows
    
    def _analyze_component_with_exploration(self, component: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[Threat]:
        """
        Analyze a component for security threats with dynamic exploration.
        
        Args:
            component: Component information from repository analysis
            analysis_results: Overall analysis results
            
        Returns:
            List of identified threats
        """
        try:
            # First, analyze the code structure
            code_analysis = self.analyze_code_structure(component)
            
            # Get path and determine language
            path = component.get("path", "")
            ext = Path(path).suffix.lower() if path else ""
            
            # Get initial context
            contexts = []
            
            # For PHP files, check if we have embedded content
            if ext == ".php" and "enhanced_context" in component:
                contexts.append(f"File Content:\n{component['enhanced_context']}")
            # Otherwise get relevant context from embedding store
            elif self.embedding_store:
                retrieved_contexts = self.embedding_store.get_relevant_context(
                    path,
                    max_results=3,
                    filter_metadata={"extension": ext} if ext else None
                )
                contexts.extend(retrieved_contexts)
            
            # If this is an entry point or critical file, perform a deeper analysis
            metadata = component.get("metadata", {})
            is_priority = metadata.get("is_entry_point") or metadata.get("is_critical")
            
            # Generate questions for dynamic exploration
            exploration_questions = self._generate_component_questions(component, code_analysis, is_priority)
            
            # Get answers to questions
            for question in exploration_questions:
                answer = self._query_rag_for_answer(f"{path} {question}")
                if answer and answer != "No relevant information found." and answer != "Error retrieving information.":
                    # Add the answer to the context
                    contexts.append(f"Q: {question}\nA: {answer}")
            
            # Check for relationships with other files
            file_relationships = analysis_results.get("file_relationships", {})
            if path in file_relationships:
                related_files = file_relationships[path]
                contexts.append(f"Related files: {', '.join(related_files)}")
                
                # Get additional context from most important related file
                if related_files and self.embedding_store:
                    # Prioritize similar file types
                    similar_ext_files = [f for f in related_files if Path(f).suffix.lower() == ext]
                    target_file = similar_ext_files[0] if similar_ext_files else related_files[0]
                    
                    related_contexts = self.embedding_store.get_relevant_context(
                        target_file,
                        max_results=1
                    )
                    if related_contexts:
                        contexts.append(f"Related file content ({target_file}):\n{related_contexts[0]}")
            
            # Construct threat analysis prompt template
            prompt_template = f"""You are a security expert analyzing code for potential threats using the STRIDE methodology. 
Focus specifically on {ext[1:].upper() if ext else "code"} security vulnerabilities.
Be thorough and specific in your analysis, looking for:
- SQL injection
- Cross-site scripting (XSS)
- CSRF vulnerabilities
- Remote file inclusion
- Unsafe file operations
- Insufficient input validation
- Session handling vulnerabilities
- Hardcoded credentials
- Lack of encryption
- Authentication/Authorization bypass
- Sensitive data exposure

Component: {component["name"]}
Path: {component["path"]}

Component Purpose: {code_analysis.purpose}
Key Functions: {', '.join(code_analysis.key_functions)}
Dependencies: {', '.join(code_analysis.dependencies)}
Security Considerations: {', '.join(code_analysis.security_considerations)}

Is Entry Point or Critical: {str(is_priority)}

Context:
{chr(10).join(contexts)}

Please analyze this component for security threats using STRIDE:
- Spoofing
- Tampering
- Repudiation
- Information disclosure
- Denial of service
- Elevation of privilege

For each threat, provide:
- Type
- Description
- Severity (Low/Medium/High)
- Impact
- Mitigation
- Relevant code snippet (if any)

Be specific about relevant threats based on the code structure and language ({ext.lstrip('.') if ext else "unknown"}).
For PHP code, pay special attention to input validation, SQL injection, XSS, and file operations.
"""
            
            # Check token count and truncate if necessary
            prompt = self._check_tokens_and_truncate(prompt_template, max_output_tokens=600)
            
            # Call LLM directly
            result = self.llm.invoke(prompt)
            
            # Parse and return threats
            threats = self._parse_threats(result)
            
            return threats
            
        except Exception as e:
            logger.error(f"Failed to analyze component: {str(e)}")
            raise LLMProcessorError(f"Component analysis failed: {str(e)}")
    
    def _generate_component_questions(self, component: Dict[str, Any], code_analysis: CodeAnalysis, is_priority: bool) -> List[str]:
        """Generate questions for dynamically exploring a component."""
        questions = []
        
        # Basic questions for all components
        questions.append("How is user input validated?")
        questions.append("Are there any authentication checks?")
        
        # Add language-specific questions
        path = component.get("path", "")
        ext = Path(path).suffix.lower() if path else ""
        
        if ext == ".php":
            questions.append("How are database queries constructed?")
            questions.append("Is user input properly sanitized?")
            questions.append("Are there any file operations?")
            questions.append("How are sessions managed?")
        
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            questions.append("Are there any DOM manipulation functions?")
            questions.append("How is AJAX/fetch used?")
            questions.append("Is there input sanitization?")
        
        elif ext == ".py":
            questions.append("How are database queries constructed?")
            questions.append("Are there any shell commands or exec calls?")
            questions.append("How is authentication handled?")
        
        # Add questions based on code analysis
        if "database" in code_analysis.purpose.lower() or "db" in code_analysis.purpose.lower():
            questions.append("How is SQL injection prevented?")
            questions.append("Are prepared statements used?")
        
        if "auth" in code_analysis.purpose.lower() or "login" in code_analysis.purpose.lower():
            questions.append("How are passwords stored?")
            questions.append("Is MFA implemented?")
            questions.append("How are sessions managed?")
        
        if is_priority:
            questions.append("What security checks are performed?")
            questions.append("Are there any hardcoded credentials?")
            questions.append("How is sensitive data handled?")
        
        # Limit to avoid too many queries
        return questions[:5]
    
    def _analyze_data_flow_with_exploration(self, data_flow: Dict[str, Any], additional_context: str = None, security_boundaries: List[Dict[str, Any]] = None) -> List[Threat]:
        """
        Analyze a data flow for security threats with boundary awareness.
        
        Args:
            data_flow: Data flow information from repository analysis
            additional_context: Additional context about the data flow (optional)
            security_boundaries: List of security boundaries
            
        Returns:
            List of identified threats
        """
        try:
            # Get relevant context
            contexts = []
            
            # Add provided additional context if any
            if additional_context:
                contexts.append(additional_context)
                
            # Add context from embedding store
            if self.embedding_store:
                stored_contexts = self.embedding_store.get_relevant_context(
                    data_flow["source"],
                    max_results=5
                )
                contexts.extend(stored_contexts)
            
            # Check if this flow crosses security boundaries
            crosses_boundary = data_flow.get("crosses_boundary", False)
            boundary_context = ""
            
            if crosses_boundary and security_boundaries:
                # Add boundary information
                for boundary in security_boundaries:
                    components = boundary.get("components", [])
                    if data_flow["source"] in components:
                        boundary_context += f"Source is within security boundary: {boundary['name']} - {boundary.get('description', '')}\n"
                
                boundary_context += "WARNING: This data flow crosses security boundaries!"
            
            if boundary_context:
                contexts.append(boundary_context)
            
            # Add parameters to allow better analysis
            parameters = data_flow.get("parameters", [])
            if parameters:
                contexts.append(f"Flow parameters: {', '.join(parameters)}")
            
            # Generate dynamic questions about this flow
            flow_questions = [
                "How is data validated before processing?",
                "What security checks are performed on the parameters?",
                "Is the data sanitized before use?"
            ]
            
            if crosses_boundary:
                flow_questions.append("What authentication/authorization is performed?")
                flow_questions.append("How is the data protected when crossing boundaries?")
            
            # Get answers to questions
            for question in flow_questions[:2]:  # Limit to 2 questions
                source = data_flow["source"]
                function = data_flow["function"]
                answer = self._query_rag_for_answer(f"{source} {function} {question}")
                if answer and answer != "No relevant information found." and answer != "Error retrieving information.":
                    contexts.append(f"Q: {question}\nA: {answer}")
            
            # Construct prompt for data flow analysis with enhanced instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a security expert analyzing data flows for potential threats. 
Focus on identifying how data moves through the system and potential security issues at each point.
{'This data flow CROSSES SECURITY BOUNDARIES, which requires extra scrutiny!' if crosses_boundary else ''}
Pay special attention to:
- Input validation and sanitization
- Authentication and authorization checks
- Data encryption and protection
- Error handling and information leakage
- Access control enforcement
- Injection vulnerabilities (SQL, command, etc.)
- Cross-site scripting opportunities
- File operation security"""),
                ("human", """
                Data Flow:
                Source: {source}
                Function: {function}
                Parameters: {parameters}
                Crosses Security Boundary: {crosses_boundary}
                
                Context:
                {context}
                
                Please analyze this data flow for security threats, focusing on:
                1. Data validation
                2. Authentication
                3. Authorization
                4. Encryption
                5. Error handling
                
                For each threat, provide:
                - Type
                - Description
                - Severity (Low/Medium/High)
                - Impact
                - Mitigation
                - Relevant code snippet (if any)
                
                Also identify how this data flow interacts with other components and any security boundaries it crosses.
                """)
            ])
            
            # Create chain and run analysis
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "source": data_flow["source"],
                "function": data_flow["function"],
                "parameters": ", ".join(data_flow.get("parameters", [])),
                "crosses_boundary": str(crosses_boundary),
                "context": "\n".join(contexts)
            })
            
            # Parse and return threats
            return self._parse_threats(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze data flow: {str(e)}")
            raise LLMProcessorError(f"Data flow analysis failed: {str(e)}")
    
    def _analyze_cross_file_vulnerabilities(self, analysis_results: Dict[str, Any], components: List[Dict[str, Any]], data_flows: List[Dict[str, Any]], cross_boundary_flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze cross-file vulnerabilities.
        
        Args:
            analysis_results: Overall analysis results
            components: Analyzed components
            data_flows: Analyzed data flows
            cross_boundary_flows: Flows that cross security boundaries
            
        Returns:
            List of cross-file vulnerabilities
        """
        try:
            # Create context with key information
            context = []
            
            # Add high-risk components
            high_risk_components = []
            for component in components:
                for threat in component.get("threats", []):
                    if threat.get("severity", "").lower() == "high":
                        high_risk_components.append(component["name"])
                        break
            
            if high_risk_components:
                context.append(f"High risk components: {', '.join(high_risk_components)}")
            
            # Add cross-boundary flows information
            if cross_boundary_flows:
                flows_text = []
                for flow in cross_boundary_flows[:5]:  # Limit to 5 flows
                    flows_text.append(f"{flow['source']} -> {flow['destination']} (from {flow['source_boundary']} to {flow['destination_boundary']})")
                context.append("Cross-boundary flows:\n- " + "\n- ".join(flows_text))
            
            # Get file relationships
            file_relationships = analysis_results.get("file_relationships", {})
            if file_relationships:
                # Filter to relationships involving high-risk components
                risky_relationships = {}
                for comp in high_risk_components:
                    if comp in file_relationships:
                        risky_relationships[comp] = file_relationships[comp]
                
                if risky_relationships:
                    rel_text = []
                    for source, targets in list(risky_relationships.items())[:5]:
                        rel_text.append(f"{source} -> {', '.join(targets[:3])}" + ("..." if len(targets) > 3 else ""))
                    context.append("Risky file relationships:\n- " + "\n- ".join(rel_text))
            
            # Get entry points
            entry_points = analysis_results.get("architecture", {}).get("entry_points", [])
            if entry_points:
                context.append(f"Entry points: {', '.join(entry_points[:5])}")
            
            # Analyze cross-file vulnerabilities
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a security expert performing cross-file vulnerability analysis. Your task is to identify vulnerabilities that span multiple files and components."),
                ("human", """
                Based on the following information about a codebase:
                
                {context}
                
                Please identify potential cross-file vulnerabilities, focusing on:
                1. Insecure data passing between components
                2. Authentication/authorization bypasses
                3. Cross-boundary data leaks
                4. End-to-end injection vulnerabilities
                5. Race conditions
                
                For each vulnerability, provide:
                - Type
                - Description
                - Severity (Low/Medium/High)
                - Components involved
                - Attack vector
                - Mitigation
                
                Format as JSON:
                ```json
                [
                  {
                    "type": "vulnerability_type",
                    "description": "description",
                    "severity": "severity",
                    "components": ["component1", "component2"],
                    "attack_vector": "attack_vector",
                    "mitigation": "mitigation"
                  }
                ]
                ```
                
                Return ONLY the JSON without explanation.
                """)
            ])
            
            # Get vulnerabilities - Updated for LangChain 0.2.x
            chain = prompt | self.llm | StrOutputParser()
            vulnerabilities_response = chain.invoke({
                "context": "\n\n".join(context)
            })
            
            # Extract JSON response
            vulnerabilities = self._extract_json(vulnerabilities_response)
            if not isinstance(vulnerabilities, list):
                return []
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Cross-file vulnerability analysis failed: {str(e)}")
            return []
    
    def _calculate_enhanced_risk(self, threat_model: Dict[str, Any]) -> str:
        """Calculate the overall risk level based on all threats and vulnerabilities."""
        try:
            severity_scores = {
                "Low": 1,
                "Medium": 2,
                "High": 3
            }
            
            max_score = 0
            total_score = 0
            threat_count = 0
            
            # Check component threats
            for component in threat_model.get("components", []):
                for threat in component.get("threats", []):
                    score = severity_scores.get(threat.get("severity", "Low"), 1)
                    max_score = max(max_score, score)
                    total_score += score
                    threat_count += 1
            
            # Check data flow threats
            for flow in threat_model.get("data_flows", []):
                for threat in flow.get("threats", []):
                    # Higher weight for boundary-crossing flows
                    weight = 1.5 if flow.get("crosses_boundary") else 1.0
                    score = severity_scores.get(threat.get("severity", "Low"), 1) * weight
                    max_score = max(max_score, score)
                    total_score += score
                    threat_count += 1
            
            # Check cross-file vulnerabilities
            for vuln in threat_model.get("vulnerabilities", []):
                score = severity_scores.get(vuln.get("severity", "Low"), 1) * 2.0  # Higher weight
                max_score = max(max_score, score)
                total_score += score
                threat_count += 1
            
            # Calculate average severity (if any threats exist)
            avg_score = total_score / threat_count if threat_count > 0 else 0
            
            # Determine overall risk based on max and average scores
            if max_score >= 3 or avg_score >= 2.0:
                return "High"
            elif max_score >= 2 or avg_score >= 1.5:
                return "Medium"
            return "Low"
            
        except Exception as e:
            logger.warning(f"Failed to calculate enhanced risk: {str(e)}")
            return "Unknown"
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a specific section from the LLM response."""
        try:
            start = text.find(f"{section}:")
            if start == -1:
                return ""
            start += len(section) + 1
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        except Exception as e:
            logger.warning(f"Failed to extract section {section}: {str(e)}")
            return ""
    
    def _extract_list(self, text: str, section: str) -> List[str]:
        """Extract a list from the LLM response."""
        try:
            section_text = self._extract_section(text, section)
            if not section_text:
                return []
            return [item.strip("- ").strip() for item in section_text.split("\n") if item.strip()]
        except Exception as e:
            logger.warning(f"Failed to extract list from {section}: {str(e)}")
            return []
    
    def _parse_threats(self, text: str) -> List[Threat]:
        """Parse threats from the LLM response."""
        try:
            threats = []
            current_threat = {}
            
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    if current_threat and "type" in current_threat:
                        try:
                            # Ensure all required fields are present
                            required_fields = ["type", "description", "severity", "impact", "mitigation"]
                            for field in required_fields:
                                if field not in current_threat:
                                    current_threat[field] = "Not specified"
                            
                            threats.append(Threat(**current_threat))
                        except Exception as e:
                            logger.warning(f"Failed to create threat from {current_threat}: {str(e)}")
                        finally:
                            current_threat = {}
                    continue
                
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    if key in ["type", "description", "severity", "impact", "mitigation", "code_snippet"]:
                        current_threat[key] = value.strip()
            
            # Handle last threat if present
            if current_threat and "type" in current_threat:
                try:
                    # Ensure all required fields are present
                    required_fields = ["type", "description", "severity", "impact", "mitigation"]
                    for field in required_fields:
                        if field not in current_threat:
                            current_threat[field] = "Not specified"
                    
                    threats.append(Threat(**current_threat))
                except Exception as e:
                    logger.warning(f"Failed to create threat from {current_threat}: {str(e)}")
            
            return threats
            
        except Exception as e:
            logger.warning(f"Failed to parse threats: {str(e)}")
            return []
    
    def _determine_tech_stack(self, file_types: Dict[str, int]) -> str:
        """
        Determine the technology stack based on file types.
        
        Args:
            file_types: Dictionary mapping file extensions to counts
            
        Returns:
            String describing the tech stack
        """
        if not file_types:
            return "Unknown technology stack"
            
        # Create a simple description based on file types
        # Get the top 3 extensions by count
        top_extensions = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Format as a string
        tech_stack = ", ".join([f"{ext} ({count} files)" for ext, count in top_extensions])
        
        return tech_stack or "Unknown technology stack"
    
    def _limit_component_context(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply aggressive token limiting to component analysis.
        
        Args:
            component: The component to limit
            
        Returns:
            Component with limited content to prevent token overflow
        """
        try:
            # Create a copy to avoid modifying the original
            limited_component = component.copy()
            
            # Limit classes to maximum 3
            if "classes" in limited_component and len(limited_component["classes"]) > 3:
                limited_component["classes"] = limited_component["classes"][:3]
                
            # Limit functions to maximum 5
            if "functions" in limited_component and len(limited_component["functions"]) > 5:
                limited_component["functions"] = limited_component["functions"][:5]
                
            # Limit imports to maximum 5
            if "imports" in limited_component and len(limited_component["imports"]) > 5:
                limited_component["imports"] = limited_component["imports"][:5]
                
            # Limit dependencies to maximum 5
            if "dependencies" in limited_component and len(limited_component["dependencies"]) > 5:
                limited_component["dependencies"] = limited_component["dependencies"][:5]
                
            # Check if there's enhanced context and limit it
            if "enhanced_context" in limited_component:
                context = limited_component["enhanced_context"]
                if isinstance(context, str) and len(context) > 2000:
                    # Keep start and end, truncate middle
                    limited_component["enhanced_context"] = (
                        context[:1000] + 
                        "\n\n[...content truncated...]\n\n" + 
                        context[-1000:]
                    )
                    
            # For each method, limit parameters to 5
            if "classes" in limited_component:
                for cls in limited_component["classes"]:
                    if "methods" in cls:
                        # Limit number of methods
                        if len(cls["methods"]) > 5:
                            cls["methods"] = cls["methods"][:5]
                        
                        # Limit parameters for each method
                        for method in cls["methods"]:
                            if "parameters" in method and len(method["parameters"]) > 5:
                                method["parameters"] = method["parameters"][:5]
            
            # For each function, limit parameters to 5
            if "functions" in limited_component:
                for func in limited_component["functions"]:
                    if "parameters" in func and len(func["parameters"]) > 5:
                        func["parameters"] = func["parameters"][:5]
                        
            return limited_component
            
        except Exception as e:
            logger.warning(f"Error limiting component context: {str(e)}")
            return component  # Return original if anything fails