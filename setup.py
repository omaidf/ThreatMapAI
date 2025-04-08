"""
AI Threat Model Map Generator Setup

This module provides the setup configuration for the
AI Threat Model Map Generator package.
"""

import os
from typing import List
from setuptools import setup, find_packages

def read_requirements() -> List[str]:
    """Read requirements from requirements.txt."""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Package metadata
NAME = "aithreatmap"
DESCRIPTION = "AI-powered threat model generator for code repositories"
AUTHOR = "AI Threat Map Generator Team"
AUTHOR_EMAIL = "info@aithreatmap.org"
URL = "https://github.com/aithreatmap/AIThreatMap"
VERSION = "0.1.0"

# List of package data files and directories
package_data = {
    'visualizer': ['templates/*'],
}

# Main setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=read_requirements(),
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data=package_data,
    entry_points={
        'console_scripts': [
            'aithreatmap=cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Security',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    
    # Additional modules to include
    py_modules=[
        'utils',
        'model_utils',
        'file_utils',
        'diagram_utils',
        'view_diagram',
        'cli'
    ],
    
    # Add long description from README
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
) 