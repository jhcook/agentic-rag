const { app, BrowserWindow, dialog } = require('electron')
const path = require('path')
const { startBackend, stopBackend, waitForBackend } = require('./python-runner')

let mainWindow = null
let shuttingDown = false

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    icon: path.join(__dirname, 'build', 'icon.png'), // Supply .png/.icns/.ico in build/
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  const indexHtml = path.join(__dirname, '..', 'ui', 'dist', 'index.html')
  mainWindow.loadFile(indexHtml)

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.on('ready', async () => {
  try {
    await startBackend()
    await waitForBackend()
    createWindow()
  } catch (err) {
    const message = err?.message || String(err)
    dialog.showErrorBox('Startup failed', message)
    app.quit()
  }
})

app.on('window-all-closed', async () => {
  await stopBackend()
  app.quit()
})

app.on('before-quit', async event => {
  if (shuttingDown) return
  shuttingDown = true
  event.preventDefault()
  await stopBackend()
  app.quit()
})

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
})
