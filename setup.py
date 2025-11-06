"""
PackTron - Efficient Sentence Packing for Large Language Model Training

A lightweight library for efficient sentence packing that eliminates padding waste
in LLM training, providing accurate token counting and better training efficiency.
"""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import subprocess
import sys
from pathlib import Path

# Ensure we use the correct Python for compilation
os.environ.setdefault("PYTHON", sys.executable)

# Read README if exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Parse requirements.txt and convert fixed versions to >=
def parse_requirements():
    """Parse requirements.txt and convert fixed versions to minimum versions"""
    requirements = []
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        return requirements
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Handle different version specifiers
            if ">=" in line:
                # Already has >=, keep as is
                requirements.append(line)
            elif "==" in line:
                # Convert == to >=
                package, version = line.split("==", 1)
                requirements.append(f"{package.strip()}>={version.strip()}")
            elif "~=" in line:
                # Compatible release, keep as is
                requirements.append(line)
            else:
                # No version specified, keep as is
                requirements.append(line)
    
    return requirements


class BuildCppExtension(build_ext):
    """Custom build_ext to compile C++ extension using Makefile"""
    
    def run(self):
        """Compile the C++ extension"""
        # First run the standard build_ext
        super().run()
        
        # Then compile our C++ extension
        utils_dir = Path(__file__).parent / "packtron" / "utils"
        makefile_path = utils_dir / "Makefile"
        helpers_cpp_path = utils_dir / "helpers.cpp"
        
        if not makefile_path.exists() or not helpers_cpp_path.exists():
            print("Warning: Makefile or helpers.cpp not found, skipping C++ extension compilation")
            print("You can manually compile later by running: make -C packtron/utils/")
            return
        
        print("\nCompiling C++ extension (helpers_cpp)...")
        try:
            # Use the same Python interpreter that's running setup.py
            import sys
            python_exe = sys.executable
            result = subprocess.run(
                ["make", "-C", str(utils_dir)],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHON": python_exe}  # Ensure Makefile uses correct Python
            )
            if result.stdout:
                print(result.stdout)
            print("✓ Successfully compiled helpers_cpp")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error compiling C++ extension:")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            print("\nNote: Installation will continue, but you may need to manually compile:")
            print("      cd packtron/utils/ && make")
            print("\nMake sure you have:")
            print("  - pybind11 installed: pip install pybind11")
            print("  - C++ compiler (g++ or clang++)")
            print("  - Python development headers")
            # Don't fail the installation, just warn
        except FileNotFoundError:
            print("✗ Warning: 'make' command not found.")
            print("  Please install build tools:")
            print("    - Linux: sudo apt-get install build-essential")
            print("    - macOS: Install Xcode Command Line Tools")
            print("  Then manually compile: cd packtron/utils/ && make")


setup(
    name="packtron",
    version="0.1.0",
    author="JimmyAwoe",
    author_email="",  # Add your email if desired
    description="Efficient sentence packing for large language model training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JimmyAwoe/PackTron",  # Update with your actual repo URL
    packages=find_packages(exclude=["test", "tests", "*.test", "*.tests", "example", "datasets"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.7.0",
        "numpy>=1.21.0",
        "transformers>=4.36.0",
        "pybind11>=2.10.0",
    ],
    entry_points={
        "console_scripts": [
            "packtron-preprocess=packtron.preprocess_data:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={
        "build_ext": BuildCppExtension,
    },
    # Note: C++ extension compilation happens during build_ext
    # Users can also manually compile by running: make -C packtron/utils/
    include_package_data=True,
    zip_safe=False,
)

