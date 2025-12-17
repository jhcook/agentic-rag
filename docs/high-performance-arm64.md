# High-Performance Python on macOS ARM64

This project utilizes a **hybrid configuration** to maximize performance on Apple Silicon (ARM64) while maintaining compatibility for other platforms.

## Architecture

| Platform | Python Version | NumPy Version | Features |
|----------|---------------|---------------|----------|
| **macOS ARM64** | **3.13** | **>= 2.0** | Stable, NEON optimizations |
| **Intel Mac** | 3.11 | < 2.0 | Legacy Stability |
| **Linux/Windows** | 3.11 | < 2.0 | Legacy Stability |

## Setup

The `start.py` script automatically handles version selection.

### 1. Prerequisites (macOS ARM64)
You need a Python 3.13 environment. `uv` handles this automatically.

### 2. Auto-Configuration
When running `python start.py`, the script detects:
- `sys.platform == 'darwin'`
- `platform.machine() == 'arm64'`

It sets `UV_PYTHON=3.13` and installs the `numpy>=2.0` dependency set.

## Verification

To verify you are running in the optimized environment:

1. **Check Python Version**:
   ```bash
   uv run python --version
   # Output: Python 3.13.x
   ```

2. **Check NumPy**:
   ```bash
   uv run python -c "import numpy; print(f'NumPy {numpy.__version__}')"
   # Output: NumPy 2.x.x
   ```

## Troubleshooting

- **Dependency Conflicts**: If you switch machines, you may need to delete the `.venv` folder and run `start.py` again.
