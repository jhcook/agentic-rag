#!/usr/bin/env bash
#
# Stop the Agentic RAG application services
#
# Author: Justin Cook

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment file if specified or use default
ENV_FILE=".env"
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            cat << 'EOF'
Stop Agentic RAG Services

USAGE:
    ./stop.sh [--env ENV_FILE]

OPTIONS:
    --env ENV_FILE
        Specify the environment file to load port configuration
        Default: .env

DESCRIPTION:
    Stops all Agentic RAG services including:
    - Ollama server (if running locally)
    - HTTP/MCP server
    - REST API server
    - UI dev server

    The script will verify each service is fully stopped before completing.

EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--env ENV_FILE]"
            echo "Run './stop.sh --help' for more information"
            exit 1
            ;;
    esac
done

# Load environment variables if file exists
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

# Set default ports
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://127.0.0.1:11434}
MCP_PORT=${MCP_PORT:-8000}
RAG_PORT=${RAG_PORT:-${REST_PORT:-8001}}
UI_PORT=${UI_PORT:-5173}

# Extract Ollama host and port
OLLAMA_HOST=$(echo "$OLLAMA_API_BASE" | sed -E 's|https?://([^:/]+).*|\1|')
OLLAMA_PORT=$(echo "$OLLAMA_API_BASE" | sed -E 's|.*:([0-9]+).*|\1|')
if [[ "$OLLAMA_PORT" == "$OLLAMA_API_BASE" ]]; then
    OLLAMA_PORT=11434
fi

# Check if Ollama is local
STOP_OLLAMA=false
if [[ "$OLLAMA_HOST" == "127.0.0.1" ]] || [[ "$OLLAMA_HOST" == "localhost" ]]; then
    STOP_OLLAMA=true
fi

echo -e "${GREEN}=== Stopping Agentic RAG Services ===${NC}"
echo ""

# Stop pgvector container (Docker Compose)
if command -v docker &> /dev/null && docker compose version &> /dev/null && [[ -f "$ROOT_DIR/docker-compose.yml" ]]; then
    echo -e "${YELLOW}Stopping pgvector (PostgreSQL) container...${NC}"
    docker compose -f "$ROOT_DIR/docker-compose.yml" stop pgvector 2>/dev/null || true
else
    echo -e "${YELLOW}pgvector container: Not managed (docker/compose missing or compose file not found)${NC}"
fi

# Function to get PID on port
get_pid_on_port() {
    local port=$1
    lsof -ti:"$port" -s TCP:LISTEN 2>/dev/null || echo ""
}

# Function to get process name
get_process_name() {
    local pid=$1
    ps -p "$pid" -o comm= 2>/dev/null || echo "unknown"
}

# Function to stop process by port
stop_service_on_port() {
    local port=$1
    local service_name=$2
    local max_wait=10
    local pid
    local proc_name

    pid=$(get_pid_on_port "$port")

    if [[ -z "$pid" ]]; then
        echo -e "${YELLOW}$service_name (port $port): Not running${NC}"
        return 0
    fi

    proc_name=$(get_process_name "$pid")
    echo -e "${YELLOW}Stopping $service_name (port $port, PID $pid: $proc_name)...${NC}"

    # Try graceful shutdown (SIGTERM)
    kill "$pid" 2>/dev/null || true

    # Wait for process to stop
    local count=0
    while [[ $count -lt $max_wait ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name stopped${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}  Forcing stop with SIGKILL...${NC}"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1

        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name force stopped${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to stop $service_name (PID $pid)${NC}"
            return 1
        fi
    fi
}

# Stop services
failed=0

# Stop UI dev server
stop_service_on_port "$UI_PORT" "UI Dev Server" || failed=$((failed + 1))

# Stop REST API Server
stop_service_on_port "$RAG_PORT" "REST API Server" || failed=$((failed + 1))

# Stop HTTP/MCP Server
stop_service_on_port "$MCP_PORT" "HTTP/MCP Server" || failed=$((failed + 1))

# Stop Ollama if local
if [[ "$STOP_OLLAMA" == true ]]; then
    stop_service_on_port "$OLLAMA_PORT" "Ollama" || failed=$((failed + 1))
else
    echo -e "${YELLOW}Ollama (remote): Not stopping${NC}"
fi

# Stop any running process controllers (they also terminate managed children)
controller_pids=$(pgrep -f "src/utils/process_controller.py" || true)
if [[ -n "$controller_pids" ]]; then
    echo -e "${YELLOW}Stopping process controllers...${NC}"
    while read -r cpid; do
        if [[ -n "$cpid" ]]; then
            kill "$cpid" 2>/dev/null || true
        fi
    done <<< "$controller_pids"
    sleep 1
    controller_pids=$(pgrep -f "src/utils/process_controller.py" || true)
    if [[ -n "$controller_pids" ]]; then
        echo -e "${YELLOW}  Forcing controller shutdown...${NC}"
        while read -r cpid; do
            if [[ -n "$cpid" ]]; then
                kill -9 "$cpid" 2>/dev/null || true
            fi
        done <<< "$controller_pids"
    fi
else
    echo -e "${YELLOW}Process controllers: Not running${NC}"
fi

echo ""

# Summary
if [[ $failed -eq 0 ]]; then
    echo -e "${GREEN}=== All Services Stopped Successfully ===${NC}"
    exit 0
else
    echo -e "${RED}=== Failed to stop $failed service(s) ===${NC}"
    echo ""
    echo "You may need to manually kill processes:"
    echo "  ps aux | grep -E 'ollama|mcp_server|rest_server|npm run dev'"
    echo "  kill -9 <PID>"
    exit 1
fi
