#!/usr/bin/env python3
"""
Cross-platform startup script for Lauren AI (Agentic RAG)

Works on Windows, macOS, and Linux. Handles service startup, process management,
and configuration loading with automatic platform detection.

Author: Justin Cook
"""

import sys
import os
import platform
import subprocess
import time
import json
import signal
import argparse
import socket
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Color codes for terminal output (cross-platform with colorama fallback)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RESET = Style.RESET_ALL
except ImportError:
    # Fallback to no colors if colorama not available
    RED = GREEN = YELLOW = RESET = ""

# Detect platform
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()
LOG_DIR = ROOT_DIR / "log"
CONFIG_DIR = ROOT_DIR / "config"
VENV_DIR = ROOT_DIR / ".venv"

# Started processes tracking
STARTED_PROCESSES: List[Tuple[subprocess.Popen, str]] = []


def print_error(msg: str) -> None:
    """Print error message in red."""
    print(f"{RED}{msg}{RESET}", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print success message in green."""
    print(f"{GREEN}{msg}{RESET}")


def print_warning(msg: str) -> None:
    """Print warning message in yellow."""
    print(f"{YELLOW}{msg}{RESET}")


def load_env_file(env_file: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_file.exists():
        print_success(f"Loading configuration from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    else:
        print_warning(f"Warning: {env_file} file not found, using defaults")
    return env_vars


def is_port_free(host: str, port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return True


def wait_for_port(host: str, port: int, service_name: str, max_wait: int = 180) -> bool:
    """Wait for a service to be ready on a specific port."""
    print_warning(f"Waiting for {service_name} to be ready on port {port}...")
    for _ in range(max_wait):
        if not is_port_free(host, port):
            print_success(f"{service_name} is ready")
            return True
        time.sleep(1)
    print_error(f"Error: {service_name} did not start within {max_wait} seconds")
    return False


def get_python_executable(venv_path: Path = None) -> str:
    """Get the appropriate Python executable path."""
    if venv_path is None:
        venv_path = VENV_DIR
    
    if venv_path.exists():
        if IS_WINDOWS:
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            return str(python_exe)
    
    # Fallback to system Python
    return sys.executable


def create_venv(python_cmd: str = None, venv_path: Path = None) -> bool:
    """Create virtual environment."""
    if python_cmd is None:
        python_cmd = sys.executable
    
    if venv_path is None:
        venv_path = VENV_DIR
    
    print_warning(f"Creating virtual environment at {venv_path}...")
    
    # Remove old venv if it exists
    if venv_path.exists():
        import shutil
        print_warning("Removing existing virtual environment...")
        shutil.rmtree(venv_path)
    
    try:
        subprocess.run([python_cmd, "-m", "venv", str(venv_path)], check=True)
        print_success("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def install_requirements(venv_path: Path = None) -> bool:
    """Install Python requirements."""
    python_exe = get_python_executable(venv_path)
    requirements_file = ROOT_DIR / "requirements.txt"
    
    if not requirements_file.exists():
        print_error("Error: requirements.txt not found")
        return False
    
    print_warning("Installing requirements...")
    print(f"Python: {python_exe}")
    
    try:
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([python_exe, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        print_success("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        return False


def check_ollama() -> bool:
    """Check if Ollama is available."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ollama: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    return False


def start_ollama(host: str, port: int, skip_model_pull: bool = False) -> Optional[subprocess.Popen]:
    """Start Ollama server."""
    print_warning("Starting Ollama server...")
    
    if not is_port_free(host, port):
        print_warning(f"Ollama already running on port {port}")
        return None
    
    # Set environment variables
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{host}:{port}"
    env["OLLAMA_KEEP_ALIVE"] = env.get("OLLAMA_KEEP_ALIVE", "-1")
    
    log_file = LOG_DIR / "ollama_server.log"
    
    try:
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=f,
                stderr=f,
                env=env,
                cwd=str(ROOT_DIR)
            )
        print(f"Ollama PID: {proc.pid}")
        
        if not wait_for_port(host, port, "Ollama"):
            print_error("Failed to start Ollama")
            proc.terminate()
            return None
        
        # Pull model if needed
        if not skip_model_pull:
            pull_ollama_models()
        
        return proc
    except Exception as e:
        print_error(f"Failed to start Ollama: {e}")
        return None


def pull_ollama_models() -> None:
    """Pull required Ollama models."""
    settings_file = CONFIG_DIR / "settings.json"
    model_name = None
    
    if settings_file.exists():
        try:
            with open(settings_file) as f:
                settings = json.load(f)
                model_name = settings.get("model", "").replace("ollama/", "")
        except Exception as e:
            print_warning(f"Could not read settings.json: {e}")
    
    if not model_name:
        model_name = os.getenv("LLM_MODEL_NAME", "llama3.2:1b").replace("ollama/", "")
    
    if model_name:
        print_warning(f"Pulling Ollama model '{model_name}'...")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model_name in result.stdout:
                print(f"✓ Model already available: {model_name}")
            else:
                subprocess.run(["ollama", "pull", model_name], check=True)
                print(f"✓ Model ready: {model_name}")
        except subprocess.CalledProcessError:
            print_warning(f"Warning: Failed to pull model '{model_name}'")


def start_mcp_server(host: str, port: int, venv_path: Path = None) -> Optional[subprocess.Popen]:
    """Start MCP server."""
    print_warning("Starting HTTP/MCP server...")
    
    if not is_port_free(host, port):
        print_warning(f"HTTP/MCP server already running on port {port}")
        return None
    
    python_exe = get_python_executable(venv_path)
    log_file = LOG_DIR / "mcp_server.log"
    
    env = os.environ.copy()
    env["MCP_HOST"] = host
    env["MCP_PORT"] = str(port)
    
    try:
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                [python_exe, "-m", "src.servers.mcp_server"],
                stdout=f,
                stderr=f,
                env=env,
                cwd=str(ROOT_DIR)
            )
        print(f"HTTP Server PID: {proc.pid}")
        
        if not wait_for_port(host, port, "HTTP/MCP server"):
            print_error("Failed to start HTTP/MCP server")
            proc.terminate()
            return None
        
        return proc
    except Exception as e:
        print_error(f"Failed to start MCP server: {e}")
        return None


def start_rest_server(host: str, port: int, venv_path: Path = None) -> Optional[subprocess.Popen]:
    """Start REST API server."""
    print_warning("Starting REST API server...")
    
    if not is_port_free(host, port):
        print_warning(f"REST API server already running on port {port}")
        return None
    
    python_exe = get_python_executable(venv_path)
    log_file = LOG_DIR / "rest_server.log"
    
    env = os.environ.copy()
    env["RAG_HOST"] = host
    env["RAG_PORT"] = str(port)
    
    try:
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                [python_exe, "-m", "uvicorn", "src.servers.rest_server:app",
                 "--host", host, "--port", str(port), "--no-access-log"],
                stdout=f,
                stderr=f,
                env=env,
                cwd=str(ROOT_DIR)
            )
        print(f"REST API Server PID: {proc.pid}")
        
        if not wait_for_port(host, port, "REST API server"):
            print_error("Failed to start REST API server")
            proc.terminate()
            return None
        
        return proc
    except Exception as e:
        print_error(f"Failed to start REST API server: {e}")
        return None


def start_ui_server(host: str, port: int) -> Optional[subprocess.Popen]:
    """Start UI dev server."""
    print_warning("Starting UI dev server...")
    
    if not is_port_free(host, port):
        print_warning(f"UI dev server already running on port {port}")
        return None
    
    ui_dir = ROOT_DIR / "ui"
    log_file = LOG_DIR / "ui_server.log"
    
    # Check if npm is available
    npm_cmd = "npm.cmd" if IS_WINDOWS else "npm"
    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print_error("Error: npm not found (required to start UI)")
        return None
    
    # Install dependencies if needed
    if not (ui_dir / "node_modules").exists():
        print_warning("Installing UI dependencies...")
        subprocess.run([npm_cmd, "install"], cwd=str(ui_dir), check=True)
    
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["VITE_PORT"] = str(port)
    
    try:
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                [npm_cmd, "run", "dev", "--", "--host", host, "--port", str(port), "--strictPort"],
                stdout=f,
                stderr=f,
                env=env,
                cwd=str(ui_dir)
            )
        print(f"UI Dev Server PID: {proc.pid}")
        
        if not wait_for_port(host, port, "UI dev server"):
            print_error("Failed to start UI dev server")
            proc.terminate()
            return None
        
        return proc
    except Exception as e:
        print_error(f"Failed to start UI dev server: {e}")
        return None


def cleanup_processes() -> None:
    """Terminate all started processes."""
    if STARTED_PROCESSES:
        print_warning(f"\nStopping {len(STARTED_PROCESSES)} service(s)...")
        for proc, name in STARTED_PROCESSES:
            if proc.poll() is None:  # Still running
                print(f"  Stopping {name} (PID: {proc.pid})...")
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                    print(f"  ✓ {name} stopped")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"  ✓ {name} force killed")
                except Exception as e:
                    print_warning(f"  Warning: Could not stop {name}: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n\nReceived interrupt signal, shutting down...")
    cleanup_processes()
    sys.exit(0)


def main():
    """Main startup logic."""
    parser = argparse.ArgumentParser(description="Start Lauren AI services")
    parser.add_argument("--env", default=".env", help="Environment file to load (default: .env)")
    parser.add_argument("--venv", default=".venv", help="Virtual environment path (default: .venv)")
    parser.add_argument("--python", default=sys.executable, help="Python command to use (default: current Python)")
    parser.add_argument("--role", choices=["monolith", "server", "client"], default="monolith",
                       help="Startup role (default: monolith)")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama startup")
    parser.add_argument("--skip-mcp", action="store_true", help="Skip MCP server")
    parser.add_argument("--skip-rest", action="store_true", help="Skip REST API server")
    parser.add_argument("--skip-ui", action="store_true", help="Skip UI dev server")
    parser.add_argument("--skip-model-pull", action="store_true", help="Skip Ollama model pulling")
    parser.add_argument("--recreate-venv", action="store_true", help="Recreate virtual environment")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Use custom venv path
    global VENV_DIR
    VENV_DIR = Path(args.venv)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print_success("=== Lauren AI Startup Script ===")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Role: {args.role}")
    print(f"Virtual environment: {VENV_DIR}")
    print()
    
    # Apply role-based configuration
    if args.role == "server":
        args.skip_ui = True
    elif args.role == "client":
        args.skip_ollama = True
        args.skip_mcp = True
    
    # Load environment
    env_file = ROOT_DIR / args.env
    env_vars = load_env_file(env_file)
    os.environ.update(env_vars)
    
    # Configuration
    ollama_host = env_vars.get("OLLAMA_HOST", "127.0.0.1")
    ollama_port = int(env_vars.get("OLLAMA_PORT", "11434"))
    mcp_host = env_vars.get("MCP_HOST", "127.0.0.1")
    mcp_port = int(env_vars.get("MCP_PORT", "8000"))
    rag_host = env_vars.get("RAG_HOST", "0.0.0.0")
    rag_port = int(env_vars.get("RAG_PORT", "8001"))
    ui_host = env_vars.get("UI_HOST", "0.0.0.0")
    ui_port = int(env_vars.get("UI_PORT", "5173"))
    
    # Setup virtual environment
    if args.recreate_venv or not VENV_DIR.exists():
        if not create_venv(args.python, VENV_DIR):
            return 1
        if not install_requirements(VENV_DIR):
            return 1
    elif not (VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")).exists():
        print_warning("Virtual environment appears invalid, recreating...")
        if not create_venv(args.python, VENV_DIR):
            return 1
        if not install_requirements(VENV_DIR):
            return 1
    else:
        print_success(f"Virtual environment exists: {VENV_DIR}")
    
    print()
    print_success("=== Starting Services ===")
    print()
    
    try:
        # Start Ollama
        if not args.skip_ollama:
            if check_ollama():
                proc = start_ollama(ollama_host, ollama_port, args.skip_model_pull)
                if proc:
                    STARTED_PROCESSES.append((proc, "Ollama"))
            else:
                print_warning("Ollama not found, skipping (use --skip-ollama to suppress this warning)")
                print_warning("Install: https://ollama.ai/download")
        else:
            print_warning("Ollama skipped (--skip-ollama)")
        print()
        
        # Start MCP Server
        if not args.skip_mcp:
            proc = start_mcp_server(mcp_host, mcp_port, VENV_DIR)
            if proc:
                STARTED_PROCESSES.append((proc, "HTTP/MCP Server"))
        print()
        
        # Start REST API
        if not args.skip_rest:
            proc = start_rest_server(rag_host, rag_port, VENV_DIR)
            if proc:
                STARTED_PROCESSES.append((proc, "REST API Server"))
        print()
        
        # Start REST API
        if not args.skip_rest:
            proc = start_rest_server(rag_host, rag_port)
            if proc:
                STARTED_PROCESSES.append((proc, "REST API Server"))
        print()
        
        # Start UI
        if not args.skip_ui:
            proc = start_ui_server(ui_host, ui_port)
            if proc:
                STARTED_PROCESSES.append((proc, "UI Dev Server"))
        print()
        # Summary
        print_success("=== All Services Started Successfully ===")
        print()
        print("Services running:")
        if not args.skip_ollama and check_ollama():
            print(f"  - Ollama: http://{ollama_host}:{ollama_port} (log: log/ollama_server.log)")
        if not args.skip_mcp:
            print(f"  - HTTP/MCP Server: http://{mcp_host}:{mcp_port} (log: log/mcp_server.log)")
        if not args.skip_rest:
            print(f"  - REST API: http://{rag_host}:{rag_port} (log: log/rest_server.log)")
        if not args.skip_ui:
            ui_url = f"http://{ui_host if ui_host != '0.0.0.0' else 'localhost'}:{ui_port}"
            print(f"  - UI Dev Server: {ui_url} (log: log/ui_server.log)")
            
            # Open browser automatically unless disabled
            if not args.no_browser:
                print()
                print_success(f"Opening browser to {ui_url}...")
                time.sleep(1)  # Give UI a moment to fully initialize
                webbrowser.open(ui_url)
        print()
        print("Press Ctrl+C to stop all services")
        print("Press Ctrl+C to stop all services")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Check if any process died
            for proc, name in STARTED_PROCESSES:
                if proc.poll() is not None:
                    print_error(f"\n{name} has stopped unexpectedly!")
                    cleanup_processes()
                    return 1
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
        cleanup_processes()
        return 0
    except Exception as e:
        print_error(f"\nError during startup: {e}")
        cleanup_processes()
        return 1


if __name__ == "__main__":
    sys.exit(main())
