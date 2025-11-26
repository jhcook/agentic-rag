#!/usr/bin/env python3
"""
Cross-platform stop script for Lauren AI (Agentic RAG)

Works on Windows, macOS, and Linux. Stops all running services.

Author: Justin Cook
"""

import sys
import platform
import subprocess
import time
import signal
from pathlib import Path

# Color codes for terminal output (cross-platform with colorama fallback)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RESET = Style.RESET_ALL
except ImportError:
    RED = GREEN = YELLOW = RESET = ""

IS_WINDOWS = platform.system() == "Windows"

# Service ports
DEFAULT_PORTS = {
    "Ollama": 11434,
    "HTTP/MCP Server": 8000,
    "REST API Server": 8001,
    "UI Dev Server": 5173,
}


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{RED}{msg}{RESET}", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}{msg}{RESET}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}{msg}{RESET}")


def find_process_by_port(port: int) -> list:
    """Find PIDs using a specific port."""
    pids = []
    
    if IS_WINDOWS:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids.append(int(pid))
        except Exception as e:
            print_warning(f"Could not check port {port}: {e}")
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip().isdigit():
                        pids.append(int(line.strip()))
        except Exception as e:
            print_warning(f"Could not check port {port}: {e}")
    
    return pids


def kill_process(pid: int, name: str, force: bool = False) -> bool:
    """Kill a process by PID."""
    try:
        if IS_WINDOWS:
            flag = "/F" if force else "/T"
            subprocess.run(["taskkill", flag, "/PID", str(pid)], 
                         capture_output=True, check=True)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            import os
            os.kill(pid, sig)
        return True
    except Exception as e:
        if not force:
            print_warning(f"Could not stop {name} (PID {pid}): {e}")
        return False


def stop_service(name: str, port: int) -> bool:
    """Stop a service by finding and killing processes on its port."""
    pids = find_process_by_port(port)
    
    if not pids:
        print(f"  {name}: Not running")
        return True
    
    print(f"  Stopping {name} (port {port}, PID(s): {', '.join(map(str, pids))})...")
    
    # Try graceful shutdown first
    success = True
    for pid in pids:
        if not kill_process(pid, name, force=False):
            success = False
    
    # Wait a bit
    time.sleep(2)
    
    # Check if still running and force kill if needed
    remaining_pids = find_process_by_port(port)
    if remaining_pids:
        print_warning(f"  Force killing {name}...")
        for pid in remaining_pids:
            kill_process(pid, name, force=True)
        time.sleep(1)
        
        # Final check
        if find_process_by_port(port):
            print_error(f"  ✗ Failed to stop {name}")
            return False
    
    print_success(f"  ✓ {name} stopped")
    return True


def stop_process_controller() -> None:
    """Stop process controller instances if any."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *process_controller*"],
                capture_output=True
            )
        except Exception:
            pass
    else:
        try:
            subprocess.run(
                ["pkill", "-f", "process_controller.py"],
                capture_output=True
            )
        except Exception:
            pass


def main():
    """Main stop logic."""
    print_success("=== Stopping Lauren AI Services ===")
    print()
    
    all_stopped = True
    
    # Stop each service
    for name, port in DEFAULT_PORTS.items():
        if not stop_service(name, port):
            all_stopped = False
    
    # Stop process controllers
    print("  Stopping process controllers...")
    stop_process_controller()
    
    print()
    if all_stopped:
        print_success("=== All Services Stopped Successfully ===")
        return 0
    else:
        print_warning("=== Some services could not be stopped ===")
        print_warning("You may need to manually kill remaining processes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
