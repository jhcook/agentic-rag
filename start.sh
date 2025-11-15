#!/usr/bin/env bash
#
# Start the Agentic RAG application
#
# Author: Justin Cook

set -euo pipefail

# Function to add timestamps to log output and strip color codes
log_with_timestamp() {
    while IFS= read -r line; do
        # Strip ANSI color codes for log file (sed is the right tool for regex substitution)
        clean_line=$(printf '%s\n' "$line" | sed $'s/\033\\[[0-9;]*m//g')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $clean_line"
    done
}

# Log all output to file with timestamps while still showing on screen with colors
mkdir -p log 2>/dev/null || true
exec > >(tee >(log_with_timestamp >> log/start.log)) 2>&1

# Color output (for terminal only, will be stripped in log file)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track started processes for cleanup
declare -a STARTED_PIDS=()
declare -a STARTED_SERVICES=()

# Cleanup function for rollback on failure
cleanup_on_failure() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}ERROR: Script failed with exit code $exit_code${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""

        # Log system information
        echo -e "${YELLOW}Collecting diagnostic information...${NC}"
        echo "Timestamp: $(date)"
        echo "Working directory: $(pwd)"
        echo "User: $(whoami)"
        echo "Shell: $SHELL"
        echo "PATH: $PATH"
        echo ""

        # Log environment variables
        echo -e "${YELLOW}Relevant environment variables:${NC}"
        env | grep -E "(OLLAMA|MCP|REST|PYTHON|VENV)" || true
        echo ""

        # Check if services are still running
        echo -e "${YELLOW}Checking service status:${NC}"
        for port in "${OLLAMA_PORT:-11434}" "${MCP_PORT:-8000}" "${REST_PORT:-8001}"; do
            if lsof -ti:"$port" > /dev/null 2>&1; then
                echo "  Port $port: OCCUPIED"
                lsof -ti:"$port" | xargs ps -p 2>/dev/null || true
            else
                echo "  Port $port: FREE"
            fi
        done
        echo ""

        # Show recent log entries if they exist
        echo -e "${YELLOW}Recent log entries:${NC}"
        for log_file in log/ollama_server.log log/http_server.log log/rest_server.log; do
            if [[ -f "$log_file" ]]; then
                echo "--- Last 10 lines of $log_file ---"
                tail -10 "$log_file" 2>/dev/null || true
                echo ""
            fi
        done

        # Cleanup started processes
        if [[ ${#STARTED_PIDS[@]} -gt 0 ]]; then
            echo -e "${YELLOW}Rolling back: Stopping ${#STARTED_PIDS[@]} started service(s)...${NC}"
            for i in "${!STARTED_PIDS[@]}"; do
                pid="${STARTED_PIDS[$i]}"
                service="${STARTED_SERVICES[$i]}"

                if kill -0 "$pid" 2>/dev/null; then
                    echo "  Stopping $service (PID: $pid)..."
                    kill "$pid" 2>/dev/null || true
                    sleep 1

                    # Force kill if still running
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "  Force killing $service (PID: $pid)..."
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                    echo "  ✓ $service stopped"
                else
                    echo "  $service (PID: $pid) already stopped"
                fi
            done
            echo -e "${GREEN}Rollback complete${NC}"
        fi

        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Startup failed. Check log/start.log for details${NC}"
        echo -e "${RED}========================================${NC}"
    fi
}

# Set trap for cleanup on exit, error, or interrupt
trap cleanup_on_failure EXIT INT TERM

# Help function
show_help() {
    cat << 'EOF'
Agentic RAG Startup Script
================================

DESCRIPTION:
    This script manages the complete startup sequence for the Agentic RAG application.
    It handles virtual environment setup, dependency installation, and sequential service
    startup with proper error handling and automatic rollback on failure.

USAGE:
    ./start.sh [OPTIONS]

OPTIONS:
    -h, --help
        Display this help message and exit

    --env ENV_FILE
        Specify the environment file to load configuration from
        Default: .env
        Examples: .env.production, .env.development, config/prod.env

    --venv VENV_NAME
        Specify the name/path of the virtual environment to use
        Default: .venv
        If this flag is provided, the virtual environment will be recreated

    --python PYTHON_CMD
        Specify the Python command/version to use for creating the virtual environment
        Default: python3.11
        Examples: python3.12, /usr/local/bin/python3.11, python3

    --recreate-venv
        Force recreation of the virtual environment even if it already exists
        Useful when dependencies are corrupted or Python version needs to change

ENVIRONMENT CONFIGURATION (.env):
    The script reads configuration from a .env file in the current directory.
    If .env is not found, default values are used.

    Key environment variables:
        OLLAMA_API_BASE       Ollama server URL (default: http://127.0.0.1:11434)
        MCP_HOST              MCP server host (default: 127.0.0.1)
        MCP_PORT              MCP server port (default: 8000)
        REST_HOST             REST API host (default: 127.0.0.1)
        REST_PORT             REST API port (default: 8001)

    Note: If OLLAMA_API_BASE points to a remote server (not 127.0.0.1 or localhost),
          the script will skip starting Ollama locally and use the remote instance.

SERVICES STARTED (in order):
    1. Ollama (if local) - AI model serving backend
       Port: Extracted from OLLAMA_API_BASE
       Log: log/ollama_server.log

    2. HTTP Server - MCP (Model Context Protocol) server
       Port: MCP_PORT
       Log: log/http_server.log

    3. REST API Server - REST API for document search and RAG operations
       Port: REST_PORT
       Log: log/rest_server.log

ERROR HANDLING:
    - The script uses strict error checking (set -euo pipefail)
    - On failure, all started services are automatically stopped (rollback)
    - Detailed diagnostic information is logged to log/start.log including:
        * System information and environment variables
        * Service status and port usage
        * Recent log entries from failed services
        * Process information for debugging

EXAMPLES:
    # Basic startup with defaults
    ./start.sh

    # Use a different environment file
    ./start.sh --env .env.production

    # Use a different Python version
    ./start.sh --python python3.12

    # Recreate virtual environment
    ./start.sh --recreate-venv

    # Use custom virtual environment name with specific Python
    ./start.sh --venv myenv --python python3.11

    # Force recreate with custom settings
    ./start.sh --venv .venv --python python3.12 --recreate-venv

    # Production setup with custom environment file
    ./start.sh --env .env.production --venv .venv-prod

STOPPING SERVICES:
    To stop all running services:
        pkill -f 'ollama serve|http_server.py|rest_server'

VIEWING LOGS:
    To monitor all service logs in real-time:
        tail -f log/*.log

    To view startup script log:
        less log/start.log

REQUIREMENTS:
    - Python 3.11+ (or specify version with --python)
    - uv (Python package installer)
    - Ollama (if using local Ollama instance)
    - uvicorn (installed via requirements.txt)
    - All dependencies listed in requirements.txt

FILES:
    .env                    Environment configuration (optional)
    requirements.txt        Python dependencies
    log/start.log          Startup script output
    log/ollama_server.log  Ollama service logs
    log/http_server.log    MCP server logs
    log/rest_server.log    REST API logs

EXIT CODES:
    0    Success - all services started
    1    Failure - see log/start.log for details

AUTHOR:
    Justin Cook

EOF
    exit 0
}

# Default values
ENV_FILE=".env"
VENV_NAME=".venv"
PYTHON_CMD="python3.11"
RECREATE_VENV=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --recreate-venv)
            RECREATE_VENV=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage: $0 [-h|--help] [--env ENV_FILE] [--venv VENV_NAME] [--python PYTHON_CMD] [--recreate-venv]"
            echo ""
            echo "Run './start.sh --help' for detailed information"
            exit 1
            ;;
    esac
done

# Create log directory if it doesn't exist
mkdir -p log

# Load environment variables from specified env file if it exists
if [[ -f "$ENV_FILE" ]]; then
    echo -e "${GREEN}Loading configuration from $ENV_FILE${NC}"
    set -a  # Automatically export all variables
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a  # Disable automatic export
else
    echo -e "${YELLOW}Warning: $ENV_FILE file not found, using defaults${NC}"
fi

# Set default values if not in .env
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://127.0.0.1:11434}
MCP_HOST=${MCP_HOST:-127.0.0.1}
MCP_PORT=${MCP_PORT:-8000}
REST_HOST=${REST_HOST:-127.0.0.1}
REST_PORT=${REST_PORT:-8001}

# Extract host and port from OLLAMA_API_BASE
OLLAMA_HOST=$(echo "$OLLAMA_API_BASE" | sed -E 's|https?://([^:/]+).*|\1|')
OLLAMA_PORT=$(echo "$OLLAMA_API_BASE" | sed -E 's|.*:([0-9]+).*|\1|')
if [[ "$OLLAMA_PORT" == "$OLLAMA_API_BASE" ]]; then
    # No port in URL, use default
    OLLAMA_PORT=11434
fi

# Determine if Ollama should be started locally
START_OLLAMA=false
if [[ "$OLLAMA_HOST" == "127.0.0.1" ]] || [[ "$OLLAMA_HOST" == "localhost" ]]; then
    START_OLLAMA=true
fi

echo -e "${GREEN}=== Agentic RAG Startup Script ===${NC}"
echo "Virtual environment: $VENV_NAME"
echo "Python command: $PYTHON_CMD"
echo "Ollama API: $OLLAMA_API_BASE (start locally: $START_OLLAMA)"
echo "MCP Server: http://${MCP_HOST}:${MCP_PORT}"
echo "REST API: http://${REST_HOST}:${REST_PORT}"
echo ""

# Check requirements
echo -e "${YELLOW}Checking requirements...${NC}"

# Check Python version
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_CMD not found${NC}"
    echo "Please install Python or specify a different version with --python"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "✓ Python: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    echo -e "${YELLOW}Warning: Python 3.11+ recommended, found $PYTHON_VERSION${NC}"
fi

# Check pip availability
if ! "$PYTHON_CMD" -m pip --version &> /dev/null; then
    echo -e "${RED}Error: pip not available in $PYTHON_CMD${NC}"
    echo "Please ensure pip is installed: $PYTHON_CMD -m ensurepip"
    exit 1
fi

PIP_VERSION=$("$PYTHON_CMD" -m pip --version 2>&1 | awk '{print $2}')
echo "✓ pip: $PIP_VERSION"

# Check if requirements.txt exists (uv will be installed from here)
if [[ ! -f "requirements.txt" ]]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo "✓ requirements.txt found"

# Check Ollama if starting locally
if [[ "$START_OLLAMA" == true ]]; then
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}Error: ollama not found${NC}"
        echo "Ollama is required for local startup"
        echo "Install: brew install ollama"
        echo "Or use remote Ollama by setting OLLAMA_API_BASE in $ENV_FILE to a remote URL"
        exit 1
    fi

    OLLAMA_VERSION=$(ollama --version 2>&1 | head -1)
    echo "✓ ollama: $OLLAMA_VERSION"
else
    echo "✓ Using remote Ollama (local installation not required)"
fi

echo -e "${GREEN}All requirements satisfied${NC}"
echo ""

# Function to check if virtual environment exists and is valid
check_venv() {
    if [[ -d "$VENV_NAME" ]] && [[ -f "$VENV_NAME/bin/activate" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    echo -e "${YELLOW}Creating virtual environment '$VENV_NAME' with $PYTHON_CMD...${NC}"

    # Check if python command exists
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo -e "${RED}Error: $PYTHON_CMD not found. Please install it or specify a different Python version with --python${NC}"
        exit 1
    fi

    # Remove old venv if recreating
    if [[ -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_NAME"
    fi

    # Create new venv
    echo "Executing: $PYTHON_CMD -m venv $VENV_NAME"
    if ! "$PYTHON_CMD" -m venv "$VENV_NAME" 2>&1; then
        echo -e "${RED}Error: Failed to create virtual environment${NC}"
        echo -e "${RED}Python version: $("$PYTHON_CMD" --version 2>&1)${NC}"
        exit 1
    fi

    echo -e "${GREEN}Virtual environment created successfully${NC}"
}

# Check if venv exists, create if needed or if --venv was specified
if check_venv; then
    if [[ "$RECREATE_VENV" == true ]]; then
        create_venv
    else
        echo -e "${GREEN}Virtual environment '$VENV_NAME' exists${NC}"
    fi
else
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' not found${NC}"
    create_venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
# shellcheck source=/dev/null
if ! source "$VENV_NAME/bin/activate"; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated${NC}"

# Clear Python bytecode cache to ensure fresh code execution
echo -e "${YELLOW}Clearing Python bytecode cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}Bytecode cache cleared${NC}"

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
echo "Python version: $(python --version 2>&1)"
echo "Pip version: $(pip --version 2>&1)"
if ! pip install -r requirements.txt 2>&1; then
    echo -e "${RED}Error: Failed to install requirements${NC}"
    echo -e "${YELLOW}Try running with --recreate-venv flag to recreate the virtual environment${NC}"
    echo -e "${YELLOW}Requirements file: $(cat requirements.txt 2>&1 | head -20)${NC}"
    exit 1
fi
echo -e "${GREEN}Requirements installed successfully${NC}"
echo ""

# Function to check if a process is running
check_process() {
    local port=$1
    lsof -ti:"$port" > /dev/null 2>&1
    return $?
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_wait=180
    local count=0

    echo -e "${YELLOW}Waiting for $service_name to be ready on port $port...${NC}"
    while ! check_process "$port"; do
        sleep 1
        count=$((count + 1))
        if [[ $count -ge $max_wait ]]; then
            echo -e "${RED}Error: $service_name did not start within ${max_wait} seconds${NC}"
            return 1
        fi
    done
    echo -e "${GREEN}$service_name is ready${NC}"
    return 0
}

# Start processes in order
echo -e "${GREEN}=== Starting Services ===${NC}"
echo ""

# 1. Start Ollama (only if local)
if [[ "$START_OLLAMA" == true ]]; then
    echo -e "${YELLOW}Starting Ollama server...${NC}"
    if check_process "$OLLAMA_PORT"; then
        echo -e "${YELLOW}Ollama already running on port $OLLAMA_PORT${NC}"
    else
        nohup ollama serve > log/ollama_server.log 2>&1 &
        OLLAMA_PID=$!
        STARTED_PIDS+=("$OLLAMA_PID")
        STARTED_SERVICES+=("Ollama")
        echo "Ollama PID: $OLLAMA_PID"

        if ! wait_for_service "$OLLAMA_PORT" "Ollama"; then
            echo -e "${RED}Failed to start Ollama${NC}"
            echo -e "${RED}Check log/ollama_server.log for details${NC}"
            exit 1
        fi
    fi
    echo ""
else
    echo -e "${GREEN}Using remote Ollama at $OLLAMA_API_BASE${NC}"
    echo ""
fi

# 2. Start HTTP Server (MCP Server)
echo -e "${YELLOW}Starting HTTP server...${NC}"
if check_process "$MCP_PORT"; then
    echo -e "${YELLOW}HTTP server already running on port $MCP_PORT${NC}"
else
    nohup uv run python http_server.py > log/http_server.log 2>&1 &
    HTTP_PID=$!
    STARTED_PIDS+=("$HTTP_PID")
    STARTED_SERVICES+=("HTTP Server")
    echo "HTTP Server PID: $HTTP_PID"

    if ! wait_for_service "$MCP_PORT" "HTTP server"; then
        echo -e "${RED}Failed to start HTTP server${NC}"
        echo -e "${RED}Check log/http_server.log for details${NC}"
        exit 1
    fi
fi
echo ""

# 3. Start REST API Server
echo -e "${YELLOW}Starting REST API server...${NC}"
if check_process "$REST_PORT"; then
    echo -e "${YELLOW}REST API server already running on port $REST_PORT${NC}"
else
    nohup uvicorn rest_server:app --host "$REST_HOST" --port "$REST_PORT" > log/rest_server.log 2>&1 &
    REST_PID=$!
    STARTED_PIDS+=("$REST_PID")
    STARTED_SERVICES+=("REST API Server")
    echo "REST API Server PID: $REST_PID"

    if ! wait_for_service "$REST_PORT" "REST API server"; then
        echo -e "${RED}Failed to start REST API server${NC}"
        echo -e "${RED}Check log/rest_server.log for details${NC}"
        exit 1
    fi
fi
echo ""

# All services started successfully - clear trap so cleanup doesn't run
trap - EXIT

# Summary
echo -e "${GREEN}=== All Services Started Successfully ===${NC}"
echo ""
echo "Services running:"
if [[ "$START_OLLAMA" == true ]]; then
    echo "  - Ollama: $OLLAMA_API_BASE (log: log/ollama_server.log)"
else
    echo "  - Ollama: $OLLAMA_API_BASE (remote)"
fi
echo "  - HTTP Server: http://${MCP_HOST}:${MCP_PORT} (log: log/http_server.log)"
echo "  - REST API: http://${REST_HOST}:${REST_PORT} (log: log/rest_server.log)"
echo ""
echo "To stop all services, run: pkill -f 'ollama serve|http_server.py|rest_server'"
echo "To view logs: tail -f log/*.log"
