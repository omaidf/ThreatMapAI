"""
Diagram Utilities for the AI Threat Model Map Generator.

This module provides utilities for working with Mermaid diagrams,
including converting them to HTML and displaying them in a browser.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time
import tempfile
import click
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Import from utils package
from utils.common import success_msg, error_msg, warning_msg, info_msg, find_files_by_pattern

def convert_to_html(mermaid_file: str, output_dir: Optional[str] = None, open_browser: bool = True) -> Optional[str]:
    """
    Convert a Mermaid diagram file to HTML.
    
    Args:
        mermaid_file: Path to the Mermaid diagram file
        output_dir: Directory to save the HTML file (default: same as Mermaid file)
        open_browser: Whether to open the generated HTML in a browser
        
    Returns:
        Path to the generated HTML file or None if conversion failed
    """
    try:
        mermaid_path = Path(mermaid_file)
        
        if not mermaid_path.exists():
            error_msg(f"{mermaid_file} not found")
            return None
        
        with open(mermaid_path, 'r') as f:
            mermaid_content = f.read()
        
        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / f"{mermaid_path.stem}.html"
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_path = mermaid_path.with_suffix('.html')
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{mermaid_path.stem}</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .mermaid {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <h1>{mermaid_path.stem.replace('_', ' ').title()}</h1>
            <div class="mermaid">
            {mermaid_content}
            </div>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    securityLevel: 'loose',
                    flowchart: {{ useMaxWidth: false }},
                    fontSize: 16
                }});
            </script>
        </body>
        </html>
        """
        
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write HTML file with explicit encoding
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            success_msg(f"Created HTML diagram at {output_path}")
            
            # Open in browser if requested
            if open_browser:
                webbrowser.open(f"file://{output_path.resolve()}")
                
            return str(output_path)
            
        except IOError as e:
            error_msg(f"Error writing HTML file: {str(e)}")
            # Try alternate location
            alt_path = Path(tempfile.gettempdir()) / f"{mermaid_path.stem}.html"
            with open(alt_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            warning_msg(f"Created HTML diagram at alternate location: {alt_path}")
            
            if open_browser:
                webbrowser.open(f"file://{alt_path.resolve()}")
                
            return str(alt_path)
            
    except Exception as e:
        error_msg(f"Error converting diagram to HTML: {str(e)}")
        return None

def start_server(directory: str, port: int = 8000) -> None:
    """
    Start a simple HTTP server in the specified directory.
    
    Args:
        directory: Directory to serve files from
        port: Port to listen on (default: 8000)
    """
    try:
        # Change to the directory
        os.chdir(directory)
        
        # Create a request handler
        handler = http.server.SimpleHTTPRequestHandler
        
        # Try to create a server
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            success_msg(f"Server started at http://localhost:{port}")
            httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                warning_msg(f"Port {port} is already in use. Server might already be running.")
                warning_msg(f"Try accessing http://localhost:{port} in your browser.")
            else:
                error_msg(f"Failed to start server: {str(e)}")
    except Exception as e:
        error_msg(f"Server error: {str(e)}")

def find_diagrams(output_dir: str) -> List[Path]:
    """
    Find Mermaid diagram files in the output directory.
    
    Args:
        output_dir: Directory to search for diagrams
        
    Returns:
        List of paths to diagram files
    """
    return find_files_by_pattern(output_dir, "**/*.mmd")

def start_server_and_open_diagrams(diagrams: List[str], output_dir: str, port: int = 8000) -> None:
    """
    Start a server and open diagrams in the browser.
    
    Args:
        diagrams: List of diagram paths to open
        output_dir: Directory containing the diagrams
        port: Port for the server
    """
    if not diagrams:
        warning_msg(f"No diagrams to display")
        return
        
    # Start HTTP server in the output directory
    server_dir = str(Path(output_dir).resolve())
    server_thread = threading.Thread(target=start_server, args=(server_dir, port), daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Give the server a moment to start
    
    # Convert diagrams to HTML and open them
    html_files = []
    for diagram_path in diagrams:
        if diagram_path and Path(diagram_path).exists():
            html_path = convert_to_html(diagram_path, output_dir, open_browser=False)
            if html_path:
                html_files.append(html_path)
    
    # Open HTML files in browser
    if html_files:
        info_msg("Opening diagrams in browser...")
        for html_file in html_files:
            html_name = Path(html_file).name
            webbrowser.open(f"http://localhost:{port}/{html_name}")
            time.sleep(0.5)  # Small delay between opening tabs
    
    # Return without blocking so callers can decide what to do next
    return

def view_diagrams(output_dir: str, port: int = 8000) -> None:
    """
    Interactive function to view diagrams in a browser.
    
    Args:
        output_dir: Directory containing diagrams
        port: Port for local server
    """
    with tqdm(total=1, desc="Finding diagrams") as progress:
        diagrams = find_diagrams(output_dir)
        progress.update(1)
    
    if not diagrams:
        warning_msg(f"No diagrams found in {output_dir}")
        return
    
    # Show menu of diagrams
    info_msg("Available diagrams:")
    for i, diagram in enumerate(diagrams, 1):
        click.echo(f"{i}. {diagram.stem}")
    
    # Let user choose which diagram to view
    choice = click.prompt(
        "Which diagram would you like to view? (number or 'all')", 
        default="all"
    )
    
    if choice.lower() == 'all':
        # Convert and display all diagrams
        start_server_and_open_diagrams(diagrams, output_dir, port)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(diagrams):
                # Convert and display selected diagram
                start_server_and_open_diagrams([diagrams[idx]], output_dir, port)
            else:
                warning_msg(f"Invalid choice: {choice}")
        except ValueError:
            warning_msg(f"Invalid choice: {choice}")
    
    # Keep the server running until user presses Ctrl+C
    try:
        info_msg("Diagram viewer server running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        info_msg("Shutting down server...") 