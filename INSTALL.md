# PackTron Installation Guide

## Quick Install

```bash
# Clone the repository
git clone https://github.com/JimmyAwoe/PackTron.git
cd PackTron

# Install with pip (recommended)
pip install -e .

# Or install in development mode
pip install -e ".[dev]"
```

## Prerequisites

### Required
- Python >= 3.8
- pip

### For C++ Extension Compilation
- **pybind11**: `pip install pybind11`
- **C++ Compiler**: 
  - Linux: `g++` or `clang++` (install via `sudo apt-get install build-essential`)
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio Build Tools
- **Python Development Headers**: Usually included with Python, but may need `python3-dev` on Linux

## Installation Methods

### Method 1: Using setup.py (Recommended)

```bash
pip install -e .
```

This will:
- Install all dependencies from `requirements.txt` (with minimum version requirements)
- Automatically compile the C++ extension (`helpers_cpp`)
- Install PackTron in editable mode

### Method 2: Manual Installation

If automatic compilation fails, you can manually compile:

```bash
# Install dependencies
pip install -r requirements.txt

# Manually compile C++ extension
cd utils/
make
cd ..
```

### Method 3: Development Installation

For development with additional tools:

```bash
pip install -e ".[dev]"
```

## Verify Installation

After installation, verify that the C++ extension is compiled:

```bash
python -c "from utils.helpers_cpp import build_sample_idx_int32; print('✓ C++ extension loaded successfully')"
```

If you see an import error, compile manually:
```bash
cd utils/ && make
```

## Troubleshooting

### C++ Extension Compilation Fails

1. **Check pybind11 is installed**: `pip install pybind11`
2. **Check compiler is available**: `g++ --version` or `clang++ --version`
3. **Manually compile**: `cd utils/ && make`
4. **Check error messages**: The setup.py will show detailed error messages

### Import Errors

If you see import errors for `helpers_cpp`:
- The C++ extension may not be compiled
- Run `make -C utils/` to compile manually
- Check that the `.so` file exists in `utils/` directory

### Dependency Issues

If you encounter dependency conflicts:
- The setup.py uses minimum version requirements (>=)
- You can manually adjust versions in `requirements.txt` if needed
- Consider using a virtual environment

## Building from Source

For development:

```bash
# Clone repository
git clone https://github.com/JimmyAwoe/PackTron.git
cd PackTron

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"

# Compile C++ extension (if not done automatically)
cd utils/
make
cd ..
```

## Next Steps

After installation, see the examples in `example/` directory:
- `example/llama_train.py` - Training example with LLaMA model
- `scripts/preprocess_data.sh` - Data preprocessing script

For usage examples, refer to the main README.md file.

