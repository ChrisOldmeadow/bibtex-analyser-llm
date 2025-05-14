"""
Setup script for the Bibtex Analyzer package.
"""

from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="bibtex-analyzer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing and visualizing BibTeX bibliographies using AI-powered tagging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bibtex-analyzer",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "bibtex_analyzer": ["*.json", "*.txt"],
    },
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "bibtex-analyzer=bibtex_analyzer.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
