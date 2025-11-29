const { app, BrowserWindow, dialog } = require('electron')
const path = require('path')
const { startBackend, stopBackend, waitForBackend } = require('./python-runner')

let mainWindow = null
let shuttingDown = false
let isCreatingWindow = false

function getAppRoot() {
  // When packaged, app.getAppPath() points to the app.asar; in dev we want the repo root.
  return app.isPackaged ? app.getAppPath() : path.join(__dirname, '..')
}

function createWindow(appRoot) {
  // Prevent creating multiple windows
  if (mainWindow !== null || isCreatingWindow) {
    return
  }
  
  isCreatingWindow = true
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    icon: path.join(__dirname, 'build', 'icon.png'), // Supply .png/.icns/.ico in build/
    title: 'LaurenAI',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  const indexHtml = path.join(appRoot, 'ui', 'dist', 'index.html')
  mainWindow.loadFile(indexHtml).catch(err => {
    dialog.showErrorBox('Failed to load UI', err?.message || String(err))
    app.quit()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
    isCreatingWindow = false
  })
  
  isCreatingWindow = false
}

app.on('ready', async () => {
  const appRoot = getAppRoot()
  try {
    await startBackend({ appRoot })
    await waitForBackend()
    createWindow(appRoot)
  } catch (err) {
    const message = err?.message || String(err)
    dialog.showErrorBox('Startup failed', message)
    app.quit()
  }
})

app.on('window-all-closed', async () => {
  // On macOS, keep the app running even when all windows are closed
  if (process.platform !== 'darwin') {
    await stopBackend()
    app.quit()
  }
})

app.on('before-quit', async event => {
  if (shuttingDown) return
  shuttingDown = true
  event.preventDefault()
  await stopBackend()
  app.quit()
})

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked and no windows are open
  // Only create if we're not already creating one and mainWindow is null
  if (mainWindow === null && !isCreatingWindow) {
    const appRoot = getAppRoot()
    // If backend is already started, just create window
    // Otherwise, wait for ready event to handle startup
    if (!shuttingDown) {
      createWindow(appRoot)
    }
  }
})
