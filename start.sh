#!/usr/bin/env bash
#
# Start the Agentic RAG application
#
# Author: Justin Cook

set -euo pipefail

# Ensure we run from repo root (so docker compose and relative paths work)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

# Deactivate inherited virtual environment to prevent conflicts during recreation
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    # Remove VIRTUAL_ENV/bin from PATH
    # We use a simple substitution; strictness isn't critical here as we just want to prefer system python
    PATH=${PATH//${VIRTUAL_ENV}\/bin:/}
    PATH=${PATH//:${VIRTUAL_ENV}\/bin/}
    unset VIRTUAL_ENV
fi

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
        for port in "${OLLAMA_PORT:-11434}" "${MCP_PORT:-8000}" "${RAG_PORT:-8001}"; do
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
        for log_file in log/ollama_server.log log/mcp_server.log log/rest_server.log log/ui_server.log; do
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

        # Exit with the original error code to ensure script terminates
        exit "$exit_code"
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

    --skip-model-pull
        Skip automatic Ollama model pulling
        Useful for offline usage or when models are already available

    --skip-ollama
        Do not start Ollama locally
        Useful when using Ollama Cloud or an external Ollama endpoint
        Note: This does NOT disable the Ollama backend (needed for Ollama Cloud)

    --skip-ui
        Do not start the UI dev server

    --skip-rest
        Do not start the REST API server

    --role ROLE
        Specify the node role: monolith, server, or client
        Default: monolith
        - monolith: Starts all services (Ollama, MCP, REST, UI)
        - server: Starts backend services (Ollama, MCP, REST). Skips UI.
        - client: Starts frontend services (UI, REST proxy). Skips Ollama, MCP.

ENVIRONMENT CONFIGURATION (.env):
    The script reads configuration from a .env file in the current directory.
    If .env is not found, default values are used.

    Key environment variables:
        OLLAMA_API_BASE       Ollama server URL (default: http://127.0.0.1:11434)
        OLLAMA_KEEP_ALIVE     Duration that models stay loaded (-1 = indefinite, 24h = 24 hours, 5m = 5 minutes)
        MCP_HOST              MCP server host (default: 127.0.0.1)
        MCP_PORT              MCP server port (default: 8000)
        RAG_HOST              REST API host (default: 127.0.0.1)
        RAG_PORT              REST API port (default: 8001)
        UI_PORT               UI dev server port (default: 5173)

    Note: If OLLAMA_API_BASE points to a remote server (not 127.0.0.1 or localhost),
          the script will skip starting Ollama locally and use the remote instance.

SERVICES STARTED (in order):
    1. Ollama (if local) - AI model serving backend
       Port: Extracted from OLLAMA_API_BASE
       Log: log/ollama_server.log

    2. HTTP Server - MCP (Model Context Protocol) server
       Port: MCP_PORT
       Log: log/mcp_server.log

    3. REST API Server - REST API for document search and RAG operations
       Port: RAG_PORT
       Log: log/rest_server.log

    4. UI Dev Server (Vite) - React UI
       Port: UI_PORT
       Log: log/ui_server.log

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

    # Skip model pulling for offline usage
    ./start.sh --skip-model-pull

    # Skip Ollama (use only OpenAI Assistants or Google Gemini)
    ./start.sh --skip-ollama

    # Force recreate with custom settings
    ./start.sh --venv .venv --python python3.12 --recreate-venv

    # Production setup with custom environment file
    ./start.sh --env .env.production --venv .venv-prod

STOPPING SERVICES:
    To stop all running services:
        pkill -f 'ollama serve|mcp_server.py|rest_server|npm run dev'

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
    - node + npm (for UI dev server, unless --skip-ui)
    - All dependencies listed in requirements.txt

FILES:
    .env                    Environment configuration (optional)
    requirements.txt        Python dependencies
    log/start.log          Startup script output
    log/ollama_server.log  Ollama service logs
    log/mcp_server.log    MCP server logs
    log/rest_server.log    REST API logs
    log/ui_server.log      UI dev server logs

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
SKIP_MODEL_PULL=false
SKIP_OLLAMA=false
ALLOW_LOCAL_BACKEND=true
START_UI=true
START_REST=true
ROLE="monolith"
START_MCP=true

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
        --role)
            ROLE="$2"
            shift 2
            ;;
        --recreate-venv)
            RECREATE_VENV=true
            shift
            ;;
        --skip-model-pull)
            SKIP_MODEL_PULL=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --skip-ui)
            START_UI=false
            shift
            ;;
        --skip-rest)
            START_REST=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage: $0 [-h|--help] [--env ENV_FILE] [--venv VENV_NAME] [--python PYTHON_CMD] [--role ROLE] [--recreate-venv] [--skip-model-pull] [--skip-ollama] [--skip-ui]"
            echo ""
            echo "Run './start.sh --help' for detailed information"
            exit 1
            ;;
    esac
done

if [[ "$SKIP_OLLAMA" == true ]]; then
    ALLOW_LOCAL_BACKEND=false
fi

# Apply role-based configuration
case $ROLE in
    monolith)
        # Default settings: everything on
        ;;
    server)
        START_UI=false
        ;;
    client)
        # Client role is UI-only (no backend services).
        SKIP_OLLAMA=true
        START_MCP=false
        START_REST=false
        ;;
    *)
        echo -e "${RED}Error: Unknown role '$ROLE'. Valid roles: monolith, server, client${NC}"
        exit 1
        ;;
esac

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

# Respect skip-Ollama toggles for downstream processes.
# Important: --skip-ollama means "don't start a local Ollama process".
# It must NOT disable the Ollama backend, otherwise Ollama Cloud cannot be used.
if [[ "$SKIP_OLLAMA" == true ]]; then
    export SKIP_OLLAMA=1
fi

ARCH_NAME=$(uname -m)

# Export local backend flag so downstream scripts know the intention
export ALLOW_LOCAL_BACKEND

# Set default values if not in .env
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://127.0.0.1:11434}
MCP_HOST=${MCP_HOST:-127.0.0.1}
MCP_PORT=${MCP_PORT:-8000}
RAG_HOST=${RAG_HOST:-${REST_HOST:-0.0.0.0}}
RAG_PORT=${RAG_PORT:-${REST_PORT:-8001}}
UI_HOST=${UI_HOST:-0.0.0.0}
UI_PORT=${UI_PORT:-5173}

# Pgvector/PostgreSQL (Docker Compose)
PGVECTOR_HOST=${PGVECTOR_HOST:-127.0.0.1}
PGVECTOR_PORT=${PGVECTOR_PORT:-5432}
PGVECTOR_DB=${PGVECTOR_DB:-agentic_rag}
PGVECTOR_USER=${PGVECTOR_USER:-agenticrag}

# Intel/MKL Optimizations
# Required for NumPy on some systems to prevent crashes and optimize performance
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Extract host and port from OLLAMA_API_BASE
OLLAMA_HOST=$(echo "$OLLAMA_API_BASE" | sed -E 's|https?://([^:/]+).*|\1|')
OLLAMA_PORT=$(echo "$OLLAMA_API_BASE" | sed -E 's|.*:([0-9]+).*|\1|')
if [[ "$OLLAMA_PORT" == "$OLLAMA_API_BASE" ]]; then
    # No port in URL, use default
    OLLAMA_PORT=11434
fi

# Determine if Ollama should be started locally
START_OLLAMA=false
if [[ "$SKIP_OLLAMA" == false ]] && [[ "$ROLE" != "client" ]]; then
    if [[ "$OLLAMA_HOST" == "127.0.0.1" ]] || [[ "$OLLAMA_HOST" == "localhost" ]]; then
        START_OLLAMA=true
    fi
fi

# Auto-tune Ollama context length when not provided explicitly
if [[ -z "${OLLAMA_NUM_CTX:-}" ]]; then
    OS_TYPE=$(uname -s)
    if [[ "$ARCH_NAME" == "arm64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
        # Check memory size on macOS
        TOTAL_MEM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        # If RAM > 20GB (approx 21474836480 bytes), assume 32GB+ machine and allow larger context
        if [[ "$TOTAL_MEM" -gt 21474836480 ]]; then
            export OLLAMA_NUM_CTX=4096
            echo "Detected Apple Silicon with high RAM (>20GB); default OLLAMA_NUM_CTX=4096"
        else
            export OLLAMA_NUM_CTX=1024
            echo "Detected Apple Silicon with standard RAM; default OLLAMA_NUM_CTX=1024 (optimized for monolith)"
        fi
    elif [[ "$ARCH_NAME" == "arm64" ]]; then
        export OLLAMA_NUM_CTX=1024
        echo "Detected ARM64 (Linux); default OLLAMA_NUM_CTX=1024"
    else
        export OLLAMA_NUM_CTX=1024
        echo "Detected $ARCH_NAME CPU; defaulting OLLAMA_NUM_CTX=1024 for lower memory/CPU usage"
    fi
else
    echo "Using configured OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX}"
fi

echo -e "${GREEN}=== Agentic RAG Startup Script ===${NC}"
echo "Role: $ROLE"
echo "Virtual environment: $VENV_NAME"
echo "Python command: $PYTHON_CMD"
echo "Ollama API: $OLLAMA_API_BASE (start locally: $START_OLLAMA)"
echo "MCP Server: http://${MCP_HOST}:${MCP_PORT}"
echo "REST API: http://${RAG_HOST}:${RAG_PORT}"
if [[ "$START_REST" == false ]]; then
    echo "REST API: skipped (--skip-rest)"
fi
if [[ "$START_UI" == true ]]; then
    echo "UI Dev Server: http://${UI_HOST}:${UI_PORT}"
else
    echo "UI Dev Server: skipped (--skip-ui)"
fi
echo ""

# Check requirements
echo -e "${YELLOW}Checking requirements...${NC}"

# Docker + pgvector are only required when starting backend services.
if [[ "$START_MCP" == true || "$START_REST" == true ]]; then
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: docker not found${NC}"
        echo "Docker is required to run the pgvector PostgreSQL container."
        echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: docker compose not available${NC}"
        echo "Docker Compose (v2) is required."
        echo "If you're using Docker Desktop, it should be included."
        exit 1
    fi

    if [[ -z "${PGVECTOR_PASSWORD:-}" ]]; then
        echo -e "${RED}Error: PGVECTOR_PASSWORD is not set${NC}"
        echo "Set PGVECTOR_PASSWORD in $ENV_FILE to start the pgvector container."
        exit 1
    fi

    mkdir -p cache/vector

    echo -e "${YELLOW}Starting pgvector (PostgreSQL) container...${NC}"
    docker compose -f "$ROOT_DIR/docker-compose.yml" up -d pgvector

    echo -e "${YELLOW}Waiting for pgvector to become healthy...${NC}"
    max_wait_pg=60
    count_pg=0
    while [[ $count_pg -lt $max_wait_pg ]]; do
        status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}nohealth{{end}}' agentic-rag-pgvector 2>/dev/null || echo "missing")
        if [[ "$status" == "healthy" ]]; then
            echo -e "${GREEN}✓ pgvector is healthy${NC}"
            break
        fi
        if [[ "$status" == "unhealthy" ]]; then
            echo -e "${RED}Error: pgvector container is unhealthy${NC}"
            docker logs --tail 50 agentic-rag-pgvector 2>/dev/null || true
            exit 1
        fi
        sleep 1
        count_pg=$((count_pg + 1))
    done

    if [[ $count_pg -ge $max_wait_pg ]]; then
        echo -e "${RED}Error: pgvector did not become healthy within ${max_wait_pg}s${NC}"
        docker logs --tail 50 agentic-rag-pgvector 2>/dev/null || true
        exit 1
    fi
else
    echo -e "${GREEN}Skipping pgvector startup (client-only / no backend services)${NC}"
fi

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
        echo "Or skip Ollama with --skip-ollama (use OpenAI Assistants or Google Gemini only)"
        exit 1
    fi

    OLLAMA_VERSION=$(ollama --version 2>&1 | head -1)
    echo "✓ ollama: $OLLAMA_VERSION"
else
    if [[ "$SKIP_OLLAMA" == true ]]; then
        echo "✓ Ollama skipped (--skip-ollama flag)"
    else
        echo "✓ Using remote Ollama (local installation not required)"
    fi
fi

# Check for libomp on macOS (often required by Torch)
if [[ "$(uname -s)" == "Darwin" ]]; then
    if brew list libomp &>/dev/null; then
        echo "✓ libomp: found (via brew)"
    else
        # Check standard locations if not in brew or brew missing
        if [[ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ]] || [[ -f "/usr/local/opt/libomp/lib/libomp.dylib" ]]; then
             echo "✓ libomp: found (manual check)"
        else
            echo -e "${YELLOW}Warning: libomp not found. This may be required for Torch on macOS.${NC}"
            echo -e "${YELLOW}Attempting to install via Homebrew...${NC}"
            if command -v brew &>/dev/null; then
                if brew install libomp; then
                    echo -e "${GREEN}✓ libomp installed successfully${NC}"
                else
                    echo -e "${RED}Error: Failed to install libomp. Please install manually: brew install libomp${NC}"
                    # We don't exit here strictly, as some setups might have it elsewhere, but it's risky
                fi
            else
                echo -e "${RED}Error: Homebrew not found. Please install libomp manually.${NC}"
            fi
        fi
    fi
fi

if [[ "$START_UI" == true ]]; then
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: node not found (required to start UI)${NC}"
        echo "Install: https://nodejs.org/ (v18+ recommended)"
        exit 1
    fi
    echo "✓ node: $(node -v)"

    if ! command -v npm &> /dev/null; then
        echo -e "${RED}Error: npm not found (required to start UI)${NC}"
        exit 1
    fi
    echo "✓ npm: $(npm -v)"
fi

extract_ollama_model() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        return
    fi
    # Strip provider prefix like "ollama/" if present
    echo "$raw" | sed -E 's|^ollama/||'
}

pull_ollama_models() {
    # Only pull when running local Ollama
    if [[ "$START_OLLAMA" != true ]]; then
        return
    fi

    # Ensure variables are bound to avoid set -u errors
    LLM_MODEL_NAME="${LLM_MODEL_NAME:-}"
    ASYNC_LLM_MODEL_NAME="${ASYNC_LLM_MODEL_NAME:-}"

    # Skip if user requested to skip model pulling
    if [[ "$SKIP_MODEL_PULL" == true ]]; then
        echo -e "${YELLOW}Skipping Ollama model pulling (--skip-model-pull)${NC}"
        return
    fi

    # Fallback to settings.json if LLM_MODEL_NAME is not set
    if [[ -z "$LLM_MODEL_NAME" ]] && [[ -f "config/settings.json" ]]; then
        echo -e "${YELLOW}LLM_MODEL_NAME not set, reading from config/settings.json...${NC}"
        LLM_MODEL_NAME=$($PYTHON_CMD -c "import json; print(json.load(open('config/settings.json')).get('model', ''))" 2>/dev/null || echo "")
    fi

    local models=()
    models+=("$(extract_ollama_model "$LLM_MODEL_NAME")")
    models+=("$(extract_ollama_model "$ASYNC_LLM_MODEL_NAME")")

    # Deduplicate and skip empty entries
    local seen=""
    for m in "${models[@]}"; do
        if [[ -z "$m" ]]; then
            continue
        fi
        if [[ "$seen" == *"|$m|"* ]]; then
            continue
        fi
        seen+="|$m|"

        # Check if model is already available locally
        if ollama list 2>/dev/null | grep -q "^${m}[[:space:]]"; then
            echo "✓ Model already available: ${m}"
            continue
        fi

        echo -e "${YELLOW}Pulling Ollama model '$m'...${NC}"
        if ! ollama pull "$m" >> log/ollama_server.log 2>&1; then
            echo -e "${RED}Warning: failed to pull model '$m'. Check log/ollama_server.log${NC}"
            echo -e "${YELLOW}Note: You can run 'ollama pull $m' manually when online, or use a remote Ollama instance${NC}"
        else
            echo "✓ Model ready: $m"
        fi
    done
}

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

# Ensure PyTorch enables the NumPy array API so the `_ARRAY_API` warning disappears when torch loads NumPy.
# export PYTORCH_ENABLE_NUMPY_ARRAY_API=1

# Optional torch pinning based on platform
TORCH_VERSION=${TORCH_VERSION:-}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
if [[ -z "$TORCH_VERSION" ]]; then
    if [[ "$ARCH_NAME" == "x86_64" ]]; then
        TORCH_VERSION="2.2.2"
        echo "Detected x86_64; preferring torch==$TORCH_VERSION (override with TORCH_VERSION env var)"
    else
        TORCH_VERSION="2.5.1"
        echo "Detected non-x86; installing torch==$TORCH_VERSION from PyTorch wheels"
    fi
fi

if [[ -n "$TORCH_VERSION" ]]; then
    TORCH_ARGS=("torch==${TORCH_VERSION}")
    # Prefer the CPU/MKL channel on Intel unless a custom index is provided
    if [[ -n "$TORCH_INDEX_URL" ]]; then
        TORCH_ARGS+=("--index-url" "$TORCH_INDEX_URL")
    elif [[ "$ARCH_NAME" == "x86_64" ]]; then
        TORCH_ARGS+=("--index-url" "https://download.pytorch.org/whl/cpu")
    fi
    echo -e "${YELLOW}Installing pinned torch: ${TORCH_ARGS[*]}${NC}"
    if ! pip install "${TORCH_ARGS[@]}" 2>&1; then
        echo -e "${RED}Warning: Failed to install torch (${TORCH_ARGS[*]}); will fall back to requirements.txt default${NC}"
        TORCH_VERSION=""
    fi
fi

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

# Verify uv CLI after install (required for process launches)
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: 'uv' CLI not available after install${NC}"
    echo "Ensure requirements.txt includes uv or install manually: pip install uv"
    exit 1
fi
echo "uv version: $(uv --version 2>/dev/null | head -1)"
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
        # Ensure OLLAMA_KEEP_ALIVE is set for model persistence
        export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}
        echo "Ollama keep-alive setting: $OLLAMA_KEEP_ALIVE"

        nohup ollama serve >> log/ollama_server.log 2>&1 &
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
    # Pull required models after server is reachable
    pull_ollama_models
    echo ""
else
    if [[ "$SKIP_OLLAMA" == true ]]; then
        echo -e "${YELLOW}Ollama skipped (--skip-ollama). Using OpenAI Assistants or Google Gemini only.${NC}"
    else
        echo -e "${GREEN}Using remote Ollama at $OLLAMA_API_BASE${NC}"
    fi
    echo ""
fi

# 2. Start HTTP Server (MCP Server)
if [[ "$START_MCP" == true ]]; then
    echo -e "${YELLOW}Starting HTTP/MCP server...${NC}"
    if check_process "$MCP_PORT"; then
        echo -e "${YELLOW}HTTP/MCP server already running on port $MCP_PORT${NC}"
    else
        nohup uv run python -m src.servers.mcp_server >> log/mcp_server.log 2>&1 &
        HTTP_PID=$!
        STARTED_PIDS+=("$HTTP_PID")
        STARTED_SERVICES+=("HTTP Server")
        echo "HTTP Server PID: $HTTP_PID"

        if ! wait_for_service "$MCP_PORT" "HTTP/MCP server"; then
            echo -e "${RED}Failed to start HTTP/MCP server${NC}"
            echo -e "${RED}Check log/mcp_server.log for details${NC}"
            exit 1
        fi
    fi
    echo ""
else
    echo -e "${GREEN}Skipping HTTP/MCP server (client role)${NC}"
    echo ""
fi

# 3. Start REST API Server
if [[ "$START_REST" == true ]]; then
    echo -e "${YELLOW}Starting REST API server...${NC}"
    if check_process "$RAG_PORT"; then
        echo -e "${YELLOW}REST API server already running on port $RAG_PORT${NC}"
    else
        # Run uvicorn directly to avoid the uv wrapper logging twice
        nohup python -m uvicorn src.servers.rest_server:app --host "$RAG_HOST" --port "$RAG_PORT" --no-access-log >> log/rest_server.log 2>&1 &
        REST_PID=$!
        STARTED_PIDS+=("$REST_PID")
        STARTED_SERVICES+=("REST API Server")
        echo "REST API Server PID: $REST_PID"

        if ! wait_for_service "$RAG_PORT" "REST API server"; then
            echo -e "${RED}Failed to start REST API server${NC}"
            echo -e "${RED}Check log/rest_server.log for details${NC}"
            exit 1
        fi
    fi
    echo ""
else
    echo -e "${GREEN}Skipping REST API server (--skip-rest)${NC}"
    echo ""
fi

# 4. Start UI dev server (Vite)
if [[ "$START_UI" == true ]]; then
    echo -e "${YELLOW}Starting UI dev server...${NC}"
    if check_process "$UI_PORT"; then
        echo -e "${YELLOW}UI dev server already running on port $UI_PORT${NC}"
    else
        # Ensure dependencies are installed; npm will no-op if already present
        (cd ui && npm install >> ../log/ui_server.log 2>&1)
        PORT="$UI_PORT" VITE_PORT="$UI_PORT" nohup npm --prefix ui run dev -- --host "$UI_HOST" --port "$UI_PORT" --strictPort >> log/ui_server.log 2>&1 &
        UI_PID=$!
        STARTED_PIDS+=("$UI_PID")
        STARTED_SERVICES+=("UI Dev Server")
        echo "UI Dev Server PID: $UI_PID"

        if ! wait_for_service "$UI_PORT" "UI dev server"; then
            echo -e "${RED}Failed to start UI dev server${NC}"
            echo -e "${RED}Check log/ui_server.log for details${NC}"
            exit 1
        fi
    fi
    echo ""
fi

# All services started successfully - clear trap so cleanup doesn't run
trap - EXIT

# Summary
echo -e "${GREEN}=== All Services Started Successfully ===${NC}"
echo ""
echo "Services running:"
if [[ "$START_OLLAMA" == true ]]; then
    echo "  - Ollama: $OLLAMA_API_BASE (log: log/ollama_server.log)"
else
    echo "  - Ollama: $OLLAMA_API_BASE (remote or skipped)"
fi

if [[ "$START_MCP" == true ]]; then
    echo "  - HTTP/MCP Server: http://${MCP_HOST}:${MCP_PORT} (log: log/mcp_server.log)"
else
    echo "  - HTTP/MCP Server: skipped"
fi

if [[ "$START_REST" == true ]]; then
    echo "  - REST API: http://${RAG_HOST}:${RAG_PORT} (log: log/rest_server.log)"
else
    echo "  - REST API: skipped"
fi

if [[ "$START_UI" == true ]]; then
    echo "  - UI Dev Server: http://${UI_HOST}:${UI_PORT} (log: log/ui_server.log)"
fi
echo ""
echo "To stop all services, run: pkill -f 'ollama serve|mcp_server.py|rest_server|npm run dev'"
echo "To view logs: tail -f log/*.log"
