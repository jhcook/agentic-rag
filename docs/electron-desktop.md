# Electron Desktop Wrapper (Windows / macOS / Linux)

This wraps the existing services (`start.py`) and Vite UI into a single desktop app with an icon and clean shutdown.

## Prerequisites
- Node.js 18+ and npm.
- Python 3.10+ on PATH, or set `PYTHON_EXE` to your interpreter/venv.
- `.env` configured as usual (ports, hosts, etc.). `start.py` still drives the backend.
- Icons placed in `electron/build/`: `icon.ico` (Windows), `icon.icns` (macOS), `icon.png` (Linux/general).

## One-time setup (from repo root)
```bash
# Build the UI (re-run this when the frontend changes)
cd ui
npm install
npm run build

# Install desktop dependencies
cd ../electron
npm install
```

## Run in development (desktop shell)
```bash
cd electron
npm run dev
```
- Launches the packaged UI (`ui/dist/index.html`).
- Spawns `python start.py --skip-ui --no-browser` in the repo root.
- Uses `RAG_HOST`/`RAG_PORT`/`RAG_PATH` from the environment for health checks (defaults: `127.0.0.1`, `8001`, `api`).

## Build installers
```bash
cd electron
npm run build:win    # Windows NSIS installer (.exe)
npm run build:mac    # macOS dmg
npm run build:linux  # Linux AppImage
# or all targets from the current OS:
npm run build
```
Outputs land in `electron/dist/`.
- Packaging now copies the Python backend (`start.py`, `src/**`, `config/**`, `scripts/**`) beside the Electron app, and `asar` is disabled so the Python files remain regular files under `Contents/Resources/app`. Bring your `.env` along if the backend needs it at runtime.

### Platform notes
- **Windows:** ensure `PYTHON_EXE` points to your venv (`C:\path\to\venv\Scripts\python.exe`) if needed. `windowsHide` prevents an extra console window.
- **macOS:** requires an `.icns` icon. Code signing/notarization is not configured; add your Apple signing settings if distributing broadly.
- **Linux:** AppImage target uses `icon.png` from `electron/build/`. You can add other targets in `electron/package.json`.

## Configuration knobs
- Backend port/host/path: `RAG_HOST`, `RAG_PORT`, `RAG_PATH` (via `.env` or environment when launching Electron).
- UI port is unused in desktop mode (`--skip-ui` is passed to `start.py`).
- Python path: set `PYTHON_EXE` before running `npm run dev` or `npm run build*`.
- Backend readiness timeout: set `BACKEND_READY_TIMEOUT_MS` (ms) if startup takes longer on your machine (default ~60s).

## Troubleshooting
- **“Backend did not become ready”**: increase the wait, e.g. `BACKEND_READY_TIMEOUT_MS=90000 npm run dev`, and verify `curl http://127.0.0.1:8001/api/health` returns 200. Make sure your `.venv` exists or set `PYTHON_EXE` to a valid Python with deps installed.
- **Icons missing**: place `icon.ico`/`icon.icns`/`icon.png` in `electron/build/` before packaging.
- **Google auth TLS errors (CERTIFICATE_VERIFY_FAILED)**: the Python backend does not read macOS Keychain by default. Export your CA bundle PEM path before launching (`export REQUESTS_CA_BUNDLE=/path/to/corp-root.pem`) so the OAuth token exchange trusts the proxy/issuer. Do this in the shell before `npm run dev` or add it to the environment Electron passes to the backend.

## What got added
- `electron/package.json`: Electron + electron-builder config and scripts.
- `electron/main.js`: creates the window, starts/stops the Python backend, waits for health.
- `electron/python-runner.js`: spawns `start.py --skip-ui --no-browser`, kills it on exit.
- `electron/preload.js`: placeholder for IPC if you need it later.
