#!/usr/bin/env python3
"""
Mermaid Diagram Viewer

This script converts Mermaid diagram files (.mmd) to HTML for easy viewing.
It is retained for backwards compatibility. New code should use the visualizer.diagram_utils module.
"""

import os
import sys
import argparse
from pathlib import Path

# Import from utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from visualizer.diagram_utils import convert_to_html, start_server, view_diagrams

def main():
    parser = argparse.ArgumentParser(description='Convert Mermaid diagrams to HTML for viewing')
    parser.add_argument('file', help='Path to the Mermaid diagram file', nargs='?')
    parser.add_argument('--dir', '-d', help='Directory containing Mermaid diagrams')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Port for the HTTP server')
    parser.add_argument('--serve', '-s', action='store_true', help='Start a local server for viewing diagrams')
    parser.add_argument('--open', '-o', action='store_true', help='Open the diagram in the default browser')
    args = parser.parse_args()
    
    # If a specific file is provided, convert and display it
    if args.file:
        # Check if file exists
        if not os.path.exists(args.file):
            error_msg(f"File not found: {args.file}")
            sys.exit(1)
        
        # Get output directory (same as file by default)
        output_dir = os.path.dirname(args.file) or '.'
        
        # Convert to HTML and optionally open in browser
        html_path = convert_to_html(args.file, output_dir, args.open)
        if html_path:
            success_msg(f"Converted {args.file} to HTML: {html_path}")
            
            # Start server if requested
            if args.serve:
                start_server(output_dir, args.port)
        else:
            error_msg(f"Failed to convert {args.file}")
            sys.exit(1)
    
    # If a directory is provided or no file specified, view diagrams interactively
    elif args.dir or not args.file:
        dir_path = args.dir or '.'
        
        # Check if directory exists
        if not os.path.isdir(dir_path):
            error_msg(f"Directory not found: {dir_path}")
            sys.exit(1)
        
        # Interactive viewing
        view_diagrams(dir_path, args.port)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 