const { spawn } = require('child_process')
const path = require('path')

let child = null

function pickPython() {
  if (process.env.PYTHON_EXE) return process.env.PYTHON_EXE
  // Prefer repo venv if present
  const repoRoot = path.join(__dirname, '..')
  const venvPy = process.platform === 'win32'
    ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, '.venv', 'bin', 'python3')
  try {
    if (require('fs').existsSync(venvPy)) return venvPy
  } catch (_) {
    // ignore
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

function startBackend() {
  if (child) return Promise.resolve(child)

  const repoRoot = path.join(__dirname, '..')
  const pythonExe = pickPython()

  return new Promise((resolve, reject) => {
    child = spawn(pythonExe, ['start.py', '--skip-ui', '--no-browser'], {
      cwd: repoRoot,
      env: { ...process.env },
      stdio: 'inherit',
      windowsHide: true
    })

    child.once('error', err => {
      child = null
      reject(err)
    })

    child.once('exit', code => {
      if (!child) return
      console.log(`Backend exited with code ${code}`)
      child = null
    })

    resolve(child)
  })
}

function stopBackend() {
  return new Promise(resolve => {
    if (!child) return resolve()

    const proc = child
    child = null

    const cleanup = () => resolve()
    proc.once('exit', cleanup)

    try {
      proc.kill('SIGTERM')
    } catch (err) {
      console.warn('Failed to send SIGTERM to backend:', err)
      return cleanup()
    }

    setTimeout(() => {
      if (proc.killed) return
      try {
        proc.kill('SIGKILL')
      } catch (err) {
        console.warn('Failed to force-kill backend:', err)
      }
    }, 4000)
  })
}

async function waitForBackend(timeoutMs = 20000) {
  const host = process.env.RAG_HOST || '127.0.0.1'
  const port = process.env.RAG_PORT || '8001'
  const basePath = (process.env.RAG_PATH || 'api').replace(/^\/+|\/+$/g, '')
  const url = `http://${host}:${port}/${basePath}/health`
  const envTimeout = Number(process.env.BACKEND_READY_TIMEOUT_MS || timeoutMs || 60000)
  const maxWait = Number.isFinite(envTimeout) && envTimeout > 0 ? envTimeout : 60000
  const start = Date.now()

  while (Date.now() - start < maxWait) {
    try {
      const res = await fetch(url)
      if (res.ok) return
    } catch {
      // ignore until timeout
    }
    await new Promise(r => setTimeout(r, 500))
  }
  throw new Error(`Backend did not become ready: ${url}`)
}

module.exports = { startBackend, stopBackend, waitForBackend }
