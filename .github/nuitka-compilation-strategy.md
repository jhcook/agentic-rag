# Nuitka Compilation Strategy for Agentic RAG

## Overview

This document outlines the strategy for compiling the Agentic RAG Python codebase to native executables using Nuitka, protecting intellectual property while maintaining multi-platform support (macOS, Linux, Windows).

## Why Nuitka?

- **IP Protection**: Source code compiled to C, then to native machine code (not easily reversible)
- **Performance**: 10-30% speed improvement over interpreted Python
- **Single Binary**: Can bundle entire application including dependencies
- **Multi-Platform**: Native builds for macOS, Linux, Windows
- **Ecosystem Compatibility**: Works with FAISS, LangChain, transformers, FastAPI
- **Active Development**: Well-maintained, commercial support available

## Architecture Components to Compile

### 1. Backend Core (`src/core/`)
**Priority**: HIGHEST (proprietary RAG logic)

- `rag_core.py` - Core RAG implementation
- `factory.py` - Backend orchestration
- `ollama_config.py` - Ollama integration
- `google_backend.py`, `openai_assistants_backend.py` - Cloud backends
- All other core modules

### 2. API Servers (`src/servers/`)
**Priority**: HIGH (business logic exposure)

- `rest_server.py` - FastAPI REST API
- `mcp_server.py` - Model Context Protocol server

### 3. Client Applications
**Priority**: MEDIUM

- `start.py` - Service launcher
- `src/clients/cli_agent.py` - CLI tool

### 4. UI Bundle
**Priority**: LOW (already minified/obfuscated by Vite)

- Keep React/Vite build as-is, serve from compiled backend

## Implementation Plan

### Phase 1: Development Setup (Week 1)

1. **Install Nuitka**
```bash
pip install nuitka ordered-set zstandard
```

2. **Create Build Scripts Directory**
```bash
mkdir -p build_scripts/nuitka
```

3. **Test Compilation** (Simple module first)
```bash
python -m nuitka --standalone --onefile src/core/models.py
```

### Phase 2: Core Module Compilation (Week 1-2)

**Target**: `src/core/` → single shared library

**Build Command**:
```bash
python -m nuitka \
  --module \
  --include-package=src.core \
  --include-data-dir=cache=cache \
  --include-data-dir=config=config \
  --nofollow-import-to=pytest \
  --nofollow-import-to=tests \
  --output-dir=build/core \
  src/core
```

**Configuration File**: `build_scripts/nuitka/core.nuitka-project`
```yaml
# Nuitka Project: Core RAG Module
[nuitka]
module-name: src.core
standalone: true
include-package: src.core
include-data-dir: cache=cache
include-data-dir: config=config
follow-imports: yes
python-flag: no_site
warn-implicit-exceptions: yes
warn-unusual-code: yes
```

### Phase 3: Server Compilation (Week 2-3)

**REST Server**:
```bash
python -m nuitka \
  --standalone \
  --onefile \
  --include-package=src.core \
  --include-package=src.servers \
  --include-package-data=fastapi \
  --include-package-data=pydantic \
  --enable-plugin=anti-bloat \
  --output-filename=agentic-rag-server \
  src/servers/rest_server.py
```

**MCP Server**:
```bash
python -m nuitka \
  --standalone \
  --onefile \
  --include-package=src.core \
  --include-package=src.servers \
  --output-filename=agentic-rag-mcp \
  src/servers/mcp_server.py
```

### Phase 4: Launcher Compilation (Week 3)

**start.py** → Platform-specific launcher:
```bash
python -m nuitka \
  --standalone \
  --onefile \
  --include-package=src \
  --include-data-dir=ui/dist=ui/dist \
  --include-data-dir=config=config \
  --include-data-file=requirements.txt=requirements.txt \
  --enable-plugin=multiprocessing \
  --output-filename=agentic-rag \
  start.py
```

### Phase 5: Platform-Specific Builds (Week 4)

**macOS**:
```bash
# Universal binary (Intel + Apple Silicon)
python -m nuitka \
  --standalone \
  --onefile \
  --macos-create-app-bundle \
  --macos-app-icon=assets/icon.icns \
  --macos-app-name="Agentic RAG" \
  --include-package=src \
  start.py
```

**Linux**:
```bash
# Static binary with minimal dependencies
python -m nuitka \
  --standalone \
  --onefile \
  --linux-onefile-icon=assets/icon.png \
  --include-package=src \
  start.py
```

**Windows**:
```bash
# Windows executable with icon
python -m nuitka \
  --standalone \
  --onefile \
  --windows-icon-from-ico=assets/icon.ico \
  --windows-company-name="Your Company" \
  --windows-product-name="Agentic RAG" \
  --windows-file-version=1.0.0 \
  --include-package=src \
  start.py
```

## Dependency Handling

### Machine Learning Libraries

**FAISS**:
- Include via `--include-package-data=faiss`
- Test GPU support separately on each platform

**Sentence Transformers**:
```bash
--include-package-data=sentence_transformers \
--include-data-dir=cache/sentence_transformers=cache/sentence_transformers
```

**PyTorch** (if used):
- Add `--nofollow-import-to=torch.distributed` (reduces size)
- Platform-specific wheels: compile on target platform

### Data Files

```bash
--include-data-dir=cache=cache \
--include-data-dir=config=config \
--include-data-dir=ui/dist=ui/dist \
--include-data-file=.env=.env
```

### Exclude Development Dependencies

```bash
--nofollow-import-to=pytest \
--nofollow-import-to=black \
--nofollow-import-to=mypy \
--nofollow-import-to=isort
```

## Build Automation

### Build Script: `build_scripts/compile_all.py`

```python
#!/usr/bin/env python3
"""
Automated Nuitka compilation for all Agentic RAG components.
"""

import subprocess
import sys
from pathlib import Path

BUILDS = {
    "core": {
        "module": "src/core",
        "type": "module",
        "output": "build/core.so",
    },
    "rest-server": {
        "module": "src/servers/rest_server.py",
        "type": "standalone",
        "output": "build/agentic-rag-server",
    },
    "mcp-server": {
        "module": "src/servers/mcp_server.py",
        "type": "standalone",
        "output": "build/agentic-rag-mcp",
    },
    "launcher": {
        "module": "start.py",
        "type": "standalone",
        "output": "build/agentic-rag",
    },
}

def compile_component(name: str, config: dict):
    """Compile a single component with Nuitka."""
    print(f"Compiling {name}...")
    
    cmd = [
        sys.executable,
        "-m", "nuitka",
        "--standalone" if config["type"] == "standalone" else "--module",
        "--onefile" if config["type"] == "standalone" else "",
        "--include-package=src",
        f"--output-filename={Path(config['output']).name}",
        config["module"],
    ]
    
    # Remove empty strings
    cmd = [c for c in cmd if c]
    
    subprocess.run(cmd, check=True)
    print(f"✓ {name} compiled successfully")

if __name__ == "__main__":
    for name, config in BUILDS.items():
        compile_component(name, config)
```

### CI/CD Integration: `.github/workflows/nuitka-build.yml`

```yaml
name: Nuitka Multi-Platform Build

on:
  push:
    branches: [main, release/*]
  pull_request:
    branches: [main]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11"]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install nuitka ordered-set zstandard
      
      - name: Compile with Nuitka
        run: |
          python build_scripts/compile_all.py
      
      - name: Test compiled binary
        run: |
          ./build/agentic-rag --help
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: agentic-rag-${{ matrix.os }}
          path: build/agentic-rag*
```

## Testing Strategy

### 1. Unit Tests on Compiled Code

```python
# tests/test_compiled.py
import subprocess
import sys

def test_compiled_core_import():
    """Test that compiled core module can be imported."""
    result = subprocess.run(
        [sys.executable, "-c", "import src.core; print('OK')"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "OK" in result.stdout

def test_compiled_server_health():
    """Test compiled server responds to health check."""
    # Start server, check /health endpoint
    pass
```

### 2. Integration Tests

```bash
# Run full workflow with compiled binaries
./build/agentic-rag --role server &
sleep 5
curl http://localhost:8001/health
# Expected: {"status": "healthy"}
```

### 3. Performance Benchmarks

```python
# Compare compiled vs interpreted performance
pytest tests/benchmark_monolith.py --compare-compiled
```

## Security Considerations (SOC 2 / GDPR)

### Secrets Protection

**Before Compilation**:
- Ensure no secrets in `config/settings.json`
- Verify `.env` is gitignored
- Check `secrets/` directory is excluded from builds

**Build-time Exclusions**:
```bash
--nofollow-import-to=secrets \
--exclude-data-files=secrets/* \
--exclude-data-files=.env
```

### Logging & Audit Trail

- Compiled binaries still log to `log/` directory
- Ensure `_redact_api_key()` functions remain in compiled code
- Test log redaction post-compilation:
```python
# tests/test_compiled_security.py
def test_compiled_logging_redacts_secrets():
    # Trigger API call with key, check logs
    pass
```

### License Management

**Option A**: Embed license check in `start.py` before compilation:
```python
def verify_license():
    """Check license validity before starting services."""
    # Load license file, verify signature
    pass
```

**Option B**: Hardware-based licensing:
```python
import uuid
def get_machine_id():
    return uuid.UUID(int=uuid.getnode())
```

## Build Artifacts Organization

```
build/
├── core/
│   └── src.core.so  # Compiled core module
├── linux/
│   ├── agentic-rag
│   ├── agentic-rag-server
│   └── agentic-rag-mcp
├── macos/
│   ├── agentic-rag.app/
│   ├── agentic-rag-server
│   └── agentic-rag-mcp
└── windows/
    ├── agentic-rag.exe
    ├── agentic-rag-server.exe
    └── agentic-rag-mcp.exe
```

## Distribution Strategy

### 1. Standalone Installers

**macOS**:
- Create DMG with `create-dmg`
- Code sign with Apple Developer certificate
- Notarize for Gatekeeper

**Windows**:
- Use Inno Setup or NSIS
- Code sign with Authenticode certificate

**Linux**:
- AppImage (self-contained)
- DEB/RPM packages for major distros
- Snap/Flatpak for universal distribution

### 2. Update Mechanism

- Keep UI separate for fast updates (Electron auto-updater)
- Backend updates: download new compiled binary
- Version check endpoint: `/api/version`

## Rollout Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Development setup + core compilation | Compiled `src/core/` module |
| 2 | Server compilation | Standalone REST + MCP binaries |
| 3 | Launcher compilation | All-in-one `agentic-rag` binary |
| 4 | Platform-specific builds | macOS .app, Windows .exe, Linux binary |
| 5 | CI/CD automation | GitHub Actions workflow |
| 6 | Testing + documentation | Test suite + deployment guide |

## Known Limitations

1. **Dynamic Imports**: Any `importlib.import_module()` calls need special handling
2. **Plugin Systems**: MCP tools may need explicit inclusion
3. **File Paths**: Use `sys._MEIPASS` for bundled resource paths
4. **Size**: Standalone binaries are 200-500MB (includes Python runtime)
5. **First Run**: Slower startup (10-20s) due to extraction

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PyInstaller | Easy setup | Easier to decompile | ❌ Lower security |
| PyArmor | License management | Requires obfuscation layer | ⚠️ Backup option |
| Cython | Good performance | Requires C compilation setup | ❌ More complex |
| Go Rewrite | Maximum protection | 3-6 months effort | ❌ Too costly |
| **Nuitka** | **Balance of all factors** | **Some size overhead** | ✅ **Selected** |

## Next Steps

1. **Immediate**: Install Nuitka, test-compile `src/core/models.py`
2. **Short-term**: Create `build_scripts/compile_all.py`
3. **Medium-term**: Set up CI/CD pipeline
4. **Long-term**: Implement licensing system pre-compilation

## Resources

- [Nuitka Documentation](https://nuitka.net/doc/user-manual.html)
- [Nuitka Commercial Support](https://nuitka.net/pages/commercial.html)
- [Python Packaging Guide](https://packaging.python.org/)
- [Code Signing Guide](https://developer.apple.com/support/code-signing/)

## Questions & Escalation

For issues during implementation:
1. Check [Nuitka GitHub Issues](https://github.com/Nuitka/Nuitka/issues)
2. Test on target platform early (don't assume cross-compilation)
3. Consider commercial Nuitka support for production deployment

---

**Last Updated**: 2025-12-13  
**Owner**: Platform Team  
**Status**: DRAFT - Ready for Review
