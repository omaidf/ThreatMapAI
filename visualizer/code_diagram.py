"""
Code Diagram Generator

This module provides functionality for generating code structure diagrams,
including class diagrams and data flow diagrams.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import networkx as nx
import pygraphviz as pgv
import re
import tempfile

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.file_utils import check_output_directory

logger = logging.getLogger(__name__)

class DiagramGenerationError(Exception):
    """Custom exception for diagram generation errors."""
    pass

class CodeDiagramGenerator:
    """
    Generates code structure diagrams.
    
    This class handles the generation of:
    - Class diagrams: Show relationships between classes and components
    - Flow diagrams: Show data flow between system components
    """
    
    def __init__(self):
        """Initialize the code diagram generator."""
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        check_output_directory(str(self.output_dir))
        
    def generate_class_diagram(self, components: List[Dict[str, Any]]) -> str:
        """
        Generate a class diagram in Mermaid format.
        
        Args:
            components: List of component dictionaries from analysis
            
        Returns:
            Mermaid class diagram content
        """
        try:
            # Start with diagram header
            diagram = ["classDiagram"]
            
            # Track classes to avoid duplicates
            processed_classes = set()
            
            # Process each component
            for component in components:
                if "name" not in component:
                    continue
                
                component_name = component["name"]
                
                # Handle classes
                if component.get("type") == "class":
                    if component_name in processed_classes:
                        continue
                    
                    processed_classes.add(component_name)
                    
                    # Add class definition
                    diagram.append(f"    class {component_name}")
                    
                    # Add methods if available
                    for method in component.get("methods", []):
                        method_name = method.get("name", "unknown")
                        diagram.append(f"    {component_name} : +{method_name}()")
                    
                    # Add attributes if available
                    for attr in component.get("attributes", []):
                        diagram.append(f"    {component_name} : +{attr}")
                
                # Handle dependencies
                for dependency in component.get("dependencies", []):
                    diagram.append(f"    {component_name} ..> {dependency} : uses")
                
                # Handle inheritance
                if "parent_class" in component:
                    parent = component["parent_class"]
                    diagram.append(f"    {parent} <|-- {component_name} : inherits")
            
            # Join all lines
            return "\n".join(diagram)
            
        except Exception as e:
            error_msg(f"Failed to generate class diagram: {str(e)}")
            # Return minimal diagram with error message
            return "classDiagram\n    class DiagramError\n    DiagramError : +error()\n    note for DiagramError \"Error generating diagram\""
    
    def generate_flow_diagram(self, data_flows: List[Dict[str, Any]]) -> str:
        """
        Generate a data flow diagram in Mermaid format.
        
        Args:
            data_flows: List of data flow dictionaries from analysis
            
        Returns:
            Mermaid flow diagram content
        """
        try:
            # Start with diagram header
            diagram = ["flowchart TD"]
            
            # Create a directed graph for flow analysis
            G = nx.DiGraph()
            
            # Process data flows
            edges = set()
            for flow in data_flows:
                if "source" not in flow or "function" not in flow:
                    continue
                
                source = flow["source"].replace("/", "_").replace(".", "_")
                target = flow["function"].replace("/", "_").replace(".", "_")
                
                # Add safe IDs for mermaid
                source_id = f"A{hash(source) % 10000}"
                target_id = f"B{hash(target) % 10000}"
                
                # Add nodes and edge to graph
                G.add_node(source_id, label=source)
                G.add_node(target_id, label=target)
                G.add_edge(source_id, target_id)
                
                # Add edge to output set if not already there
                edge_key = f"{source_id}-->{target_id}"
                if edge_key not in edges:
                    edges.add(edge_key)
                    
                    # Add description if available
                    description = flow.get("description", "")
                    if description:
                        diagram.append(f"    {source_id}[{source}] --> |{description}| {target_id}[{target}]")
                    else:
                        diagram.append(f"    {source_id}[{source}] --> {target_id}[{target}]")
            
            # If no edges were found, add a placeholder
            if not edges:
                diagram.append("    A[No Data Flows] --> B[Found]")
            
            # Join all lines
            return "\n".join(diagram)
            
        except Exception as e:
            error_msg(f"Failed to generate flow diagram: {str(e)}")
            # Return minimal diagram with error message
            return "flowchart TD\n    A[Error] --> B[Generating Flow Diagram]"
    
    def save_diagram(self, diagram: str, filename: str) -> str:
        """
        Save a diagram to a file.
        
        Args:
            diagram: Diagram content
            filename: Name of the file to save
            
        Returns:
            Path to the saved diagram
        """
        try:
            # Ensure output directory exists
            check_output_directory(str(self.output_dir))
            
            # Create file path
            file_path = self.output_dir / filename
            
            # Write diagram to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(diagram)
            
            success_msg(f"Diagram saved to {file_path}")
            return str(file_path)
        except Exception as e:
            error_msg(f"Failed to save diagram: {str(e)}")
            
            # Try to save to alternative location
            try:
                alt_path = Path(tempfile.gettempdir()) / filename
                with open(alt_path, 'w', encoding='utf-8') as f:
                    f.write(diagram)
                warning_msg(f"Diagram saved to alternative location: {alt_path}")
                return str(alt_path)
            except Exception as inner_e:
                error_msg(f"Failed to save diagram to alternative location: {str(inner_e)}")
                return None 