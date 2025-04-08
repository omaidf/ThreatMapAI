"""
Threat Model Visualizer Module

This module provides functionality for visualizing threat models
and generating security reports from analysis results.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import jinja2
import markdown
from datetime import datetime
import tempfile

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.env_utils import get_env_variable
from utils.file_utils import check_output_directory

logger = logging.getLogger(__name__)

class VisualizerError(Exception):
    """Custom exception for visualization errors."""
    pass

class ThreatModelVisualizer:
    """
    Visualizes threat models and generates security reports.
    
    This class handles the generation of:
    - Threat model diagrams
    - Security reports
    - Risk assessment summaries
    """
    
    def __init__(self):
        """Initialize the threat model visualizer."""
        self.output_dir = Path(get_env_variable("OUTPUT_DIR", "output"))
        check_output_directory(str(self.output_dir))
        
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the template file exists
        template_file = self.template_dir / "report_template.md"
        if not template_file.exists():
            self._create_default_template(template_file)
            
        self._setup_templates()
    
    def _create_default_template(self, template_path: Path) -> None:
        """Create a default template if it doesn't exist."""
        try:
            info_msg(f"Creating default template at {template_path}")
            default_template = """# {{ title }}

**Date:** {{ date }}

## Overview

This report presents the security analysis of the codebase, including identified threats, risk assessment, and recommendations.

## Risk Assessment

**Overall Risk Level:** {{ overall_risk }}

## Components Analysis

{% for component in components %}
### {{ component.name }} ({{ component.type }})

{% if component.threats %}
#### Identified Threats
{% for threat in component.threats %}
- **Type:** {{ threat.type }}
- **Severity:** {{ threat.severity }}
- **Description:** {{ threat.description }}
- **Impact:** {{ threat.impact }}
- **Mitigation:** {{ threat.mitigation }}
{% if threat.code_snippet %}
```python
{{ threat.code_snippet }}
```
{% endif %}
{% endfor %}
{% else %}
No threats identified.
{% endif %}
{% endfor %}

## Data Flow Analysis

{% for flow in data_flows %}
### {{ flow.source }} -> {{ flow.function }}

{% if flow.threats %}
#### Identified Threats
{% for threat in flow.threats %}
- **Type:** {{ threat.type }}
- **Severity:** {{ threat.severity }}
- **Description:** {{ threat.description }}
- **Impact:** {{ threat.impact }}
- **Mitigation:** {{ threat.mitigation }}
{% if threat.code_snippet %}
```python
{{ threat.code_snippet }}
```
{% endif %}
{% endfor %}
{% else %}
No threats identified.
{% endif %}
{% endfor %}

## Recommendations

{% for rec in recommendations %}
### {{ rec.type }}: {{ rec.target }}
- **Priority:** {{ rec.priority }}
- **Description:** {{ rec.description }}
- **Action:** {{ rec.action }}
{% endfor %}
"""
            template_path.parent.mkdir(parents=True, exist_ok=True)
            with open(template_path, 'w') as f:
                f.write(default_template)
            success_msg(f"Default template created successfully")
        except Exception as e:
            error_msg(f"Failed to create default template: {str(e)}")
            raise VisualizerError(f"Template creation failed: {str(e)}")
    
    def _setup_templates(self) -> None:
        """Set up Jinja2 templates for report generation."""
        try:
            if not self.template_dir.exists():
                raise VisualizerError(f"Template directory not found: {self.template_dir}")
                
            template_file = self.template_dir / "report_template.md"
            if not template_file.exists():
                raise VisualizerError(f"Report template not found: {template_file}")
                
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                autoescape=True
            )
            
            # Verify we can load the template
            self.env.get_template("report_template.md")
            info_msg("Templates loaded successfully")
        except jinja2.exceptions.TemplateError as e:
            error_msg(f"Template error: {str(e)}")
            raise VisualizerError(f"Template error: {str(e)}")
        except Exception as e:
            error_msg(f"Failed to setup templates: {str(e)}")
            raise VisualizerError(f"Template setup failed: {str(e)}")
    
    def generate_report(self, threat_model: Dict[str, Any]) -> str:
        """
        Generate a comprehensive security report from the threat model.
        
        Args:
            threat_model: Threat model data
            
        Returns:
            Path to the generated report
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare report data
            report_data = {
                "title": "Security Threat Analysis Report",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_risk": threat_model.get("overall_risk", "Unknown"),
                "components": [],
                "data_flows": [],
                "recommendations": []
            }
            
            # Process components
            for component in threat_model.get("components", []):
                comp_data = {
                    "name": component.get("name", "Unknown"),
                    "type": component.get("type", "Unknown"),
                    "threats": []
                }
                
                for threat in component.get("threats", []):
                    comp_data["threats"].append({
                        "type": threat.get("type", "Unknown"),
                        "description": threat.get("description", "No description"),
                        "severity": threat.get("severity", "Unknown"),
                        "impact": threat.get("impact", "Unknown"),
                        "mitigation": threat.get("mitigation", "Unknown"),
                        "code_snippet": threat.get("code_snippet")
                    })
                
                report_data["components"].append(comp_data)
            
            # Process data flows
            for flow in threat_model.get("data_flows", []):
                flow_data = {
                    "source": flow.get("source", "Unknown"),
                    "function": flow.get("function", "Unknown"),
                    "threats": []
                }
                
                for threat in flow.get("threats", []):
                    flow_data["threats"].append({
                        "type": threat.get("type", "Unknown"),
                        "description": threat.get("description", "No description"),
                        "severity": threat.get("severity", "Unknown"),
                        "impact": threat.get("impact", "Unknown"),
                        "mitigation": threat.get("mitigation", "Unknown"),
                        "code_snippet": threat.get("code_snippet")
                    })
                
                report_data["data_flows"].append(flow_data)
            
            # Generate recommendations
            report_data["recommendations"] = self._generate_recommendations(threat_model)
            
            # Render report
            try:
                template = self.env.get_template("report_template.md")
                report_content = template.render(**report_data)
            except jinja2.exceptions.TemplateError as e:
                logger.error(f"Template rendering error: {str(e)}")
                # Generate simple report if template fails
                report_content = f"# Security Threat Analysis Report\n\n* Date: {report_data['date']}\n* Overall Risk: {report_data['overall_risk']}\n\n## No detailed threat information available"
            
            # Convert to HTML
            try:
                html_content = markdown.markdown(
                    report_content,
                    extensions=['tables', 'fenced_code', 'codehilite']
                )
            except Exception as e:
                logger.error(f"Markdown conversion error: {str(e)}")
                html_content = f"<h1>Error generating report</h1><pre>{report_content}</pre>"
            
            # Save report with additional error handling
            report_path = self.output_dir / "threat_analysis_report.html"
            try:
                # Ensure parent directory exists
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file with explicit mode and encoding
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                logger.info(f"Report generated at {report_path}")
                return str(report_path)
                
            except IOError as e:
                logger.error(f"IO error while saving report: {str(e)}")
                # Try an alternative location if the primary fails
                alt_path = Path(tempfile.gettempdir()) / "threat_analysis_report.html"
                with open(alt_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.warning(f"Report saved to alternative location: {alt_path}")
                return str(alt_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            # Create a minimal report instead of failing
            try:
                minimal_html = f"""
                <html>
                <head><title>Security Threat Analysis Report</title></head>
                <body>
                <h1>Security Threat Analysis Report</h1>
                <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Overall Risk: {threat_model.get("overall_risk", "Unknown")}</p>
                <div style="padding: 10px; margin: 10px; border: 1px solid #ccc; background-color: #f8f8f8;">
                <h2>Limited Analysis Available</h2>
                <p>The analysis could not generate complete results.</p>
                <p>Error: {str(e)}</p>
                </div>
                </body>
                </html>
                """
                # Try standard location first
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    report_path = self.output_dir / "threat_analysis_report.html"
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(minimal_html)
                    logger.warning(f"Generated minimal report due to error: {str(e)}")
                    return str(report_path)
                except Exception:
                    # If that fails, try temp directory
                    alt_path = Path(tempfile.gettempdir()) / "threat_analysis_report.html"
                    with open(alt_path, "w", encoding="utf-8") as f:
                        f.write(minimal_html)
                    logger.warning(f"Generated minimal report in temp directory: {alt_path}")
                    return str(alt_path)
            except Exception as inner_e:
                logger.error(f"Failed to create minimal report: {str(inner_e)}")
                raise VisualizerError(f"Report generation failed: {str(e)}")
    
    def generate_report_from_dir(self, output_dir: str) -> str:
        """
        Generate a security report from existing analysis results.
        
        Args:
            output_dir: Directory containing analysis results
            
        Returns:
            Path to the generated report
        """
        try:
            # Set output directory
            self.output_dir = Path(output_dir)
            
            # Make sure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load threat model from file
            threat_model_path = self.output_dir / "threat_model.json"
            if not threat_model_path.exists():
                logger.warning(f"Threat model file not found: {threat_model_path}")
                # Create a default minimal threat model
                default_threat_model = {
                    "components": [],
                    "data_flows": [],
                    "overall_risk": "Unknown",
                    "architecture": {
                        "architecture_pattern": "Unknown",
                        "components": [],
                        "entry_points": []
                    }
                }
                # Load analysis results for additional context if available
                analysis_path = self.output_dir / "analysis_results.json"
                if analysis_path.exists():
                    try:
                        with open(analysis_path, "r") as f:
                            analysis_results = json.load(f)
                        # Extract file types if available
                        if "architecture" in analysis_results and "file_types" in analysis_results["architecture"]:
                            file_types = analysis_results["architecture"]["file_types"]
                            tech_stack = ", ".join([f"{ext} ({count} files)" for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)])
                            default_threat_model["tech_stack"] = tech_stack
                    except Exception as e:
                        logger.warning(f"Failed to load analysis results: {str(e)}")
                
                # Generate report with default model
                return self.generate_report(default_threat_model)
            
            with open(threat_model_path, "r") as f:
                threat_model = json.load(f)
            
            # Generate report
            return self.generate_report(threat_model)
        
        except Exception as e:
            logger.error(f"Failed to generate report from directory: {str(e)}")
            # Create a minimal report as fallback
            try:
                minimal_html = f"""
                <html>
                <head><title>Security Threat Analysis Report</title></head>
                <body>
                <h1>Security Threat Analysis Report</h1>
                <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Status: Error</p>
                <div style="padding: 10px; margin: 10px; border: 1px solid #ccc; background-color: #f8f8f8;">
                <h2>Report Generation Error</h2>
                <p>The report could not be generated from the directory: {output_dir}</p>
                <p>Error: {str(e)}</p>
                </div>
                </body>
                </html>
                """
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                report_path = self.output_dir / "threat_analysis_report.html"
                with open(report_path, "w") as f:
                    f.write(minimal_html)
                logger.warning(f"Generated minimal report due to error: {str(e)}")
                return str(report_path)
            except Exception as inner_e:
                logger.error(f"Failed to create minimal report: {str(inner_e)}")
                raise VisualizerError(f"Report generation failed: {str(e)}")
    
    def generate_class_diagram(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a class diagram from analysis results.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Mermaid class diagram string
        """
        try:
            diagram = ["classDiagram"]
            
            # Add classes
            for component in analysis_results.get("components", []):
                if component.get("type") == "class":
                    diagram.append(f"    class {component['name']} {{")
                    
                    # Add class members
                    for member in component.get("members", []):
                        if member.get("type") == "method":
                            diagram.append(f"        +{member['name']}()")
                        elif member.get("type") == "attribute":
                            diagram.append(f"        +{member['name']}")
                    
                    diagram.append("    }")
            
            # Add relationships
            for component in analysis_results.get("components", []):
                for dependency in component.get("dependencies", []):
                    diagram.append(f"    {component['name']} --> {dependency}")
            
            return "\n".join(diagram)
            
        except Exception as e:
            logger.error(f"Failed to generate class diagram: {str(e)}")
            raise VisualizerError(f"Class diagram generation failed: {str(e)}")
    
    def generate_flow_diagram(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a flow diagram from analysis results.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Mermaid flow diagram string
        """
        try:
            diagram = ["flowchart TD"]
            diagram.append("    %% Data Flow Diagram")
            
            # Define styles for different types of nodes
            diagram.append("    classDef component fill:#f9f,stroke:#333,stroke-width:2px;")
            diagram.append("    classDef dataflow fill:#bbf,stroke:#33c,stroke-width:1px;")
            diagram.append("    classDef security fill:#fbb,stroke:#c33,stroke-width:2px,stroke-dasharray: 5 5;")
            
            # Track all nodes we've added to avoid duplicates
            added_nodes = set()
            
            # Add components as boxes
            for component in analysis_results.get("components", []):
                component_id = component["name"].replace(" ", "_").replace(".", "_")
                if component_id not in added_nodes:
                    component_label = component.get("name", "Unknown")
                    diagram.append(f"    {component_id}[{component_label}]")
                    diagram.append(f"    class {component_id} component;")
                    added_nodes.add(component_id)
            
            # Add data flows as directed lines
            flow_count = 0
            for flow in analysis_results.get("data_flows", []):
                source = flow.get("source", "").replace(" ", "_").replace(".", "_")
                function = flow.get("function", "").replace(" ", "_").replace(".", "_")
                
                # Only add if both source and function exist
                if source and function and source in added_nodes:
                    # Create a flow node for the function if it doesn't exist
                    if function not in added_nodes:
                        diagram.append(f"    {function}({flow.get('function', 'Unknown')})")
                        diagram.append(f"    class {function} dataflow;")
                        added_nodes.add(function)
                    
                    # Add flow with label showing parameters if any
                    params = ", ".join(flow.get("parameters", []))
                    flow_id = f"flow_{flow_count}"
                    if params:
                        diagram.append(f"    {source} -->|{params}| {function}")
                    else:
                        diagram.append(f"    {source} --> {function}")
                    flow_count += 1
            
            # If architecture and code flow data is available, add it
            if hasattr(analysis_results, "get") and "architecture" in analysis_results.get("__all__", []):
                # Add architecture components
                arch = analysis_results.get("architecture", {})
                for comp in arch.get("components", []):
                    comp_id = comp.get("name", "").replace(" ", "_").replace(".", "_")
                    if comp_id and comp_id not in added_nodes:
                        comp_label = f"{comp.get('name', 'Unknown')}\n({comp.get('purpose', '')})"
                        diagram.append(f"    {comp_id}[{comp_label}]")
                        diagram.append(f"    class {comp_id} component;")
                        added_nodes.add(comp_id)
                
                # Add architecture data flows
                for flow in arch.get("data_flows", []):
                    source = flow.get("source", "").replace(" ", "_").replace(".", "_")
                    dest = flow.get("destination", "").replace(" ", "_").replace(".", "_")
                    
                    if source and dest and source in added_nodes:
                        if dest not in added_nodes:
                            dest_label = flow.get("destination", "Unknown")
                            diagram.append(f"    {dest}[{dest_label}]")
                            added_nodes.add(dest)
                        
                        data_type = flow.get("data_type", "")
                        if data_type:
                            diagram.append(f"    {source} -->|{data_type}| {dest}")
                        else:
                            diagram.append(f"    {source} --> {dest}")
                
                # Add security boundaries
                for boundary in arch.get("security_boundaries", []):
                    boundary_id = f"boundary_{boundary.get('name', '').replace(' ', '_')}"
                    boundary_label = boundary.get("name", "Security Boundary")
                    diagram.append(f"    {boundary_id}[/{boundary_label}/]")
                    diagram.append(f"    class {boundary_id} security;")
            
            # If diagram is too simple, add a note
            if len(diagram) < 5:
                diagram.append("    note[No detailed flow information available]")
            
            return "\n".join(diagram)
            
        except Exception as e:
            logger.error(f"Failed to generate flow diagram: {str(e)}")
            raise VisualizerError(f"Flow diagram generation failed: {str(e)}")
    
    def generate_visualizations_from_dir(self, output_dir: str) -> Dict[str, str]:
        """
        Generate visualizations from existing analysis results.
        
        Args:
            output_dir: Directory containing analysis results
            
        Returns:
            Dictionary of generated diagrams
        """
        try:
            # Set output directory
            self.output_dir = Path(output_dir)
            
            # Load analysis results from file
            analysis_path = self.output_dir / "analysis_results.json"
            if not analysis_path.exists():
                raise VisualizerError(f"Analysis results file not found: {analysis_path}")
            
            with open(analysis_path, "r") as f:
                analysis_results = json.load(f)
            
            # Load threat model from file
            threat_model_path = self.output_dir / "threat_model.json"
            if not threat_model_path.exists():
                raise VisualizerError(f"Threat model file not found: {threat_model_path}")
            
            with open(threat_model_path, "r") as f:
                threat_model = json.load(f)
            
            # Generate diagrams
            class_diagram = self.generate_class_diagram(analysis_results)
            flow_diagram = self.generate_flow_diagram(analysis_results)
            threat_diagram = self.generate_threat_diagram(threat_model)
            
            # Save diagrams
            diagrams = {}
            diagrams["Class Diagram"] = self.save_diagram(class_diagram, "class_diagram.mmd")
            diagrams["Flow Diagram"] = self.save_diagram(flow_diagram, "flow_diagram.mmd")
            diagrams["Threat Diagram"] = self.save_diagram(threat_diagram, "threat_diagram.mmd")
            
            return diagrams
        
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            raise VisualizerError(f"Visualization generation failed: {str(e)}")
    
    def _generate_recommendations(self, threat_model: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate security recommendations based on the threat model.
        
        Args:
            threat_model: Threat model data
            
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Process component threats
            for component in threat_model.get("components", []):
                for threat in component.get("threats", []):
                    if threat.get("severity") in ["High", "Medium"]:
                        recommendations.append({
                            "type": "Component",
                            "target": component.get("name", "Unknown"),
                            "description": f"Address {threat.get('type', 'Unknown')} threat: {threat.get('description', 'No description')}",
                            "priority": threat.get("severity", "Unknown"),
                            "action": threat.get("mitigation", "No mitigation provided")
                        })
            
            # Process data flow threats
            for flow in threat_model.get("data_flows", []):
                for threat in flow.get("threats", []):
                    if threat.get("severity") in ["High", "Medium"]:
                        recommendations.append({
                            "type": "Data Flow",
                            "target": f"{flow.get('source', 'Unknown')} -> {flow.get('function', 'Unknown')}",
                            "description": f"Address {threat.get('type', 'Unknown')} threat: {threat.get('description', 'No description')}",
                            "priority": threat.get("severity", "Unknown"),
                            "action": threat.get("mitigation", "No mitigation provided")
                        })
            
            # Sort by priority
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {str(e)}")
            return []
    
    def generate_threat_diagram(self, threat_model: Dict[str, Any]) -> str:
        """
        Generate a threat diagram from threat model.
        
        Args:
            threat_model: Threat model data
            
        Returns:
            Mermaid diagram string
        """
        try:
            # Use Mermaid flowchart for more flexibility
            diagram = ["flowchart TD"]
            diagram.append("    %% Threat Model Diagram")
            
            # Define styles for different node types
            diagram.append("    classDef component fill:#f9f,stroke:#333,stroke-width:2px;")
            diagram.append("    classDef dataflow fill:#bbf,stroke:#33c,stroke-width:1px;")
            diagram.append("    classDef threat fill:#fbb,stroke:#c33,stroke-width:2px;")
            diagram.append("    classDef highThreat fill:#f88,stroke:#c33,stroke-width:3px;")
            diagram.append("    classDef mediumThreat fill:#fc8,stroke:#c83,stroke-width:2px;")
            diagram.append("    classDef lowThreat fill:#ff8,stroke:#cc3,stroke-width:1px;")
            diagram.append("    classDef boundary fill:none,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;")
            diagram.append("    classDef vulnerability fill:#f66,stroke:#900,stroke-width:3px;")
            diagram.append("    classDef critical fill:#f00,stroke:#900,stroke-width:4px;")
            
            # Track all nodes to avoid duplicates
            added_nodes = set()
            
            # Add security boundaries first (as subgraphs)
            for i, boundary in enumerate(threat_model.get("security_boundaries", [])):
                boundary_id = f"security_boundary_{i}"
                boundary_name = boundary.get("name", f"Security Boundary {i+1}")
                diagram.append(f"    subgraph {boundary_id}[{boundary_name}]")
                
                # Add components within this boundary
                for component_name in boundary.get("components", []):
                    comp_id = component_name.replace(" ", "_").replace(".", "_")
                    if comp_id not in added_nodes:
                        diagram.append(f"        {comp_id}[{component_name}]")
                        diagram.append(f"        class {comp_id} component;")
                        added_nodes.add(comp_id)
                
                diagram.append("    end")
                diagram.append(f"    class {boundary_id} boundary;")
            
            # Add priority components outside of boundaries
            for component in threat_model.get("components", []):
                comp_id = component.get("name", "Unknown").replace(" ", "_").replace(".", "_")
                if comp_id not in added_nodes:
                    # Check if this is a priority component
                    is_priority = component.get("priority", False)
                    comp_label = component.get("name", "Unknown")
                    
                    # Add component
                    diagram.append(f"    {comp_id}[{comp_label}]")
                    if is_priority:
                        diagram.append(f"    class {comp_id} critical;")
                    else:
                        diagram.append(f"    class {comp_id} component;")
                    added_nodes.add(comp_id)
                
                # Add threats for this component
                for i, threat in enumerate(component.get("threats", [])):
                    threat_id = f"{comp_id}_threat_{i}"
                    threat_label = f"{threat.get('type', 'Unknown Threat')}\n({threat.get('severity', 'Unknown')})"
                    diagram.append(f"    {threat_id}{{{{ {threat_label} }}}}")
                    
                    # Style based on severity
                    severity = threat.get("severity", "").lower()
                    if severity == "high":
                        diagram.append(f"    class {threat_id} highThreat;")
                    elif severity == "medium":
                        diagram.append(f"    class {threat_id} mediumThreat;")
                    else:
                        diagram.append(f"    class {threat_id} lowThreat;")
                    
                    # Connect component to threat
                    diagram.append(f"    {comp_id} --> {threat_id}")
            
            # Add data flows with special emphasis on cross-boundary flows
            for flow in threat_model.get("data_flows", []):
                source = flow.get("source", "").replace(" ", "_").replace(".", "_")
                function = flow.get("function", "").replace(" ", "_").replace(".", "_")
                
                # Create nodes if they don't exist
                if source and source not in added_nodes:
                    diagram.append(f"    {source}[{flow.get('source', 'Unknown')}]")
                    diagram.append(f"    class {source} component;")
                    added_nodes.add(source)
                
                if function and function not in added_nodes:
                    diagram.append(f"    {function}({flow.get('function', 'Unknown')})")
                    diagram.append(f"    class {function} dataflow;")
                    added_nodes.add(function)
                
                # Connect source to function
                if source and function:
                    flow_label = ", ".join(flow.get("parameters", []))
                    
                    # Highlight cross-boundary flows
                    crosses_boundary = flow.get("crosses_boundary", False)
                    if crosses_boundary:
                        if flow_label:
                            diagram.append(f"    {source} -->|🔴 {flow_label}| {function}")
                        else:
                            diagram.append(f"    {source} -->|🔴 CROSSES BOUNDARY| {function}")
                    else:
                        if flow_label:
                            diagram.append(f"    {source} -->|{flow_label}| {function}")
                        else:
                            diagram.append(f"    {source} --> {function}")
                
                # Add threats for this flow
                for i, threat in enumerate(flow.get("threats", [])):
                    threat_id = f"flow_{source}_{function}_threat_{i}"
                    threat_label = f"{threat.get('type', 'Unknown')}\n({threat.get('severity', 'Unknown')})"
                    diagram.append(f"    {threat_id}{{{{ {threat_label} }}}}")
                    
                    # Style based on severity
                    severity = threat.get("severity", "").lower()
                    if severity == "high":
                        diagram.append(f"    class {threat_id} highThreat;")
                    elif severity == "medium":
                        diagram.append(f"    class {threat_id} mediumThreat;")
                    else:
                        diagram.append(f"    class {threat_id} lowThreat;")
                    
                    # Connect flow to threat - if function exists use that, otherwise source
                    if function:
                        diagram.append(f"    {function} --> {threat_id}")
                    elif source:
                        diagram.append(f"    {source} --> {threat_id}")
            
            # Add cross-boundary flows
            for i, flow in enumerate(threat_model.get("cross_boundary_flows", [])):
                source = flow.get("source", "").replace(" ", "_").replace(".", "_")
                destination = flow.get("destination", "").replace(" ", "_").replace(".", "_")
                
                # Create nodes if they don't exist
                if source and source not in added_nodes:
                    diagram.append(f"    {source}[{flow.get('source', 'Unknown')}]")
                    diagram.append(f"    class {source} component;")
                    added_nodes.add(source)
                
                if destination and destination not in added_nodes:
                    diagram.append(f"    {destination}[{flow.get('destination', 'Unknown')}]")
                    diagram.append(f"    class {destination} component;")
                    added_nodes.add(destination)
                
                # Connect with boundary labels
                if source and destination:
                    source_boundary = flow.get("source_boundary", "")
                    dest_boundary = flow.get("destination_boundary", "")
                    
                    flow_id = f"cross_flow_{i}"
                    label = f"Crosses: {source_boundary} → {dest_boundary}"
                    
                    diagram.append(f"    {source} -->|{label}| {destination}")
            
            # Add cross-file vulnerabilities with special styling
            if "vulnerabilities" in threat_model:
                for i, vuln in enumerate(threat_model.get("vulnerabilities", [])):
                    vuln_id = f"vulnerability_{i}"
                    vuln_type = vuln.get("type", "Unknown Vulnerability")
                    severity = vuln.get("severity", "Medium")
                    vuln_label = f"{vuln_type}\n({severity})"
                    
                    diagram.append(f"    {vuln_id}[/{vuln_label}/]")
                    diagram.append(f"    class {vuln_id} vulnerability;")
                    
                    # Connect vulnerability to components
                    for comp in vuln.get("components", []):
                        comp_id = comp.replace(" ", "_").replace(".", "_")
                        if comp_id not in added_nodes:
                            diagram.append(f"    {comp_id}[{comp}]")
                            diagram.append(f"    class {comp_id} component;")
                            added_nodes.add(comp_id)
                        
                        diagram.append(f"    {comp_id} -.-> {vuln_id}")
            
            # Add legend
            diagram.append("    subgraph Legend[Legend]")
            diagram.append("        comp[Component]:::component")
            diagram.append("        flow([Data Flow]):::dataflow")
            diagram.append("        high_threat{{High Threat}}:::highThreat")
            diagram.append("        med_threat{{Medium Threat}}:::mediumThreat")
            diagram.append("        low_threat{{Low Threat}}:::lowThreat")
            diagram.append("        prio[Priority Component]:::critical")
            diagram.append("        vuln[/Vulnerability/]:::vulnerability")
            diagram.append("        legend_boundary[Security Boundary]:::boundary")
            diagram.append("    end")
            
            # If diagram is too simple, add a note
            if len(diagram) < 8:  # Just header, styles, and possibly a note
                diagram.append("    note[No detailed threat information available]")
            
            return "\n".join(diagram)
            
        except Exception as e:
            logger.error(f"Failed to generate threat diagram: {str(e)}")
            raise VisualizerError(f"Threat diagram generation failed: {str(e)}")
    
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
        except IOError as e:
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
        except Exception as e:
            error_msg(f"Failed to save diagram: {str(e)}")
            return None 