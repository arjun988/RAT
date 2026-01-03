"""
Setup script for RAT package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rat-transformer",
    version="0.1.2",
    author="RAT Team",
    author_email="team@rat-transformer.ai",
    description="RAT: Reinforced Adaptive Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ReinforcedAdaptiveTransformer-RAT/RAT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="transformer attention reinforcement-learning nlp language-model rat ml ai machine-learning LLM pytorch",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.21.0",
        "datasets>=2.7.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=5.0.0",
        ],
        "training": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
            "accelerate>=0.16.0",
        ],
        "serving": [
            "fastapi>=0.88.0",
            "uvicorn>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rat-train=rat.cli:train_command",
            "rat-generate=rat.cli:generate_command",
            "rat-eval=rat.cli:eval_command",
            "rat-test=rat.cli:test_command",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
