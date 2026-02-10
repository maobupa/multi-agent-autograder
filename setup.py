"""
Setup script for Adversarial Grader package.
"""

from setuptools import setup, find_packages

setup(
    name="adversarial-grader",
    version="1.0.0",
    description="An adversarial grading system using LLM-based grader and critic",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "krippendorff>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
