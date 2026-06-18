"""
Setup configuration for the Fake News Intelligence System.

This module provides the package configuration for installing and distributing
the AI-powered Fake News Intelligence System. It reads dependencies from
requirements.txt and configures the package metadata for PyPI compatibility.

The system uses NLP and machine learning to classify news articles as real or fake,
providing explainable predictions, sentiment analysis, clickbait detection, and
credibility scoring through an interactive Streamlit dashboard.
"""

import os
from setuptools import setup, find_packages


def read_requirements(filename="requirements.txt"):
    """Read and parse requirements from a requirements file.

    Reads the specified requirements file line by line, stripping whitespace
    and filtering out empty lines and comment lines (starting with '#').

    Args:
        filename (str): Path to the requirements file relative to this
            setup.py. Defaults to 'requirements.txt'.

    Returns:
        list[str]: A list of requirement strings suitable for use in
            install_requires.
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    requirements = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


def read_long_description():
    """Read the long description from README.md if it exists.

    Attempts to read the README.md file in the same directory as this
    setup.py to use as the package's long description on PyPI.

    Returns:
        str: The contents of README.md, or an empty string if the file
            does not exist.
    """
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


long_description = read_long_description()

setup(
    name="fake-news-intelligence",
    version="1.0.0",
    description="AI-powered Fake News Intelligence System",
    long_description=long_description,
    long_description_content_type="text/markdown" if long_description else "text/plain",
    author="Data Science Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
    ],
)
