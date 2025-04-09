"""
Utilities for handling diagrams in the AI Threat Map Generator.

This module provides functions for finding, viewing, and serving diagrams.
"""

import os
import webbrowser
import threading
import time
from pathlib import Path
from typing import List, Optional

# Import view_diagram functions when needed
def import_view_diagram_module():
    """Import view_diagram module to avoid circular imports"""
    global convert_to_html, start_server
    try:
        from view_diagram import convert_to_html, start_server
    except ImportError:
        # Provide fallback implementations if needed
        def convert_to_html(diagram_file, output_dir=None):
            """Fallback implementation"""
            raise ImportError("view_diagram module not available")
            
        def start_server(html_files, port=8000):
            """Fallback implementation"""
            raise ImportError("view_diagram module not available")

def find_diagrams(output_dir: str) -> List[Path]:
    """Find Mermaid diagram files in the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        List of Path objects for diagram files
    """
    diagram_files = []
    output_path = Path(output_dir)
    
    if output_path.exists() and output_path.is_dir():
        for file in output_path.glob("**/*.mmd"):
            diagram_files.append(file)
    
    return diagram_files

def start_server_and_open_diagrams(diagram_paths: List[str], output_dir: str) -> None:
    """Convert diagrams to HTML and start a local server to view them.
    
    Args:
        diagram_paths: List of paths to diagram files
        output_dir: Output directory for generated HTML files
    """
    import_view_diagram_module()
    
    try:
        # Convert diagrams to HTML
        html_files = []
        for path in diagram_paths:
            html_file = convert_to_html(path, output_dir)
            if html_file:
                html_files.append(html_file)
        
        if not html_files:
            print("No diagram files were converted to HTML.")
            return
            
        # Start server in a background thread
        server_thread = threading.Thread(
            target=start_server,
            args=(html_files,),
            daemon=True
        )
        server_thread.start()
        
        # Open each diagram in the browser
        for html_file in html_files:
            webbrowser.open(f"http://localhost:8000/{os.path.basename(html_file)}")
        
    except Exception as e:
        print(f"Error starting diagram viewer: {str(e)}")

def view_diagrams(output_dir: str, port: int = 8000) -> None:
    """Find diagrams in the output directory and view them in a browser.
    
    Args:
        output_dir: Directory containing diagrams
        port: Port for the local server
    """
    import_view_diagram_module()
    
    # Find diagrams
    diagram_paths = find_diagrams(output_dir)
    
    if not diagram_paths:
        print(f"No diagram files found in {output_dir}")
        return
    
    print(f"Found {len(diagram_paths)} diagram(s):")
    for path in diagram_paths:
        print(f"  - {path}")
    
    # Convert diagrams to HTML and start server
    start_server_and_open_diagrams([str(p) for p in diagram_paths], output_dir)
    
    # Keep the server running until user interrupts
    try:
        print("\nDiagram viewer server running on port 8000. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...") 