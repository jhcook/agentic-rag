import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { File as FileIcon, Folder, Upload, Cloud, HardDrive, RefreshCw, Trash, Eye, ArrowUp, ChevronRight, Calendar, ArrowDownAZ, ArrowUpAZ, Plus, Server, CheckSquare, Square } from 'lucide-react'
import { toast } from 'sonner'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog'

type DriveFile = {
  id: string
  name: string
  mimeType: string
  size?: string
  webViewLink?: string
  iconLink?: string
  createdTime?: string
  modifiedTime?: string
}

type SortOption = 'name' | 'date' | 'size'
type SortDirection = 'asc' | 'desc'

type LocalFile = {
  uri: string
  size_bytes: number
}

export function FileManager({ config, activeMode }: { config: any, activeMode?: string }) {
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [localFiles, setLocalFiles] = useState<LocalFile[]>([])
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [previewFile, setPreviewFile] = useState<DriveFile | null>(null)
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null)
  const [folderStack, setFolderStack] = useState<{id: string | null, name: string}[]>([{id: null, name: 'Root'}])
  const [sortBy, setSortBy] = useState<SortOption>('date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [deleteConfirm, setDeleteConfirm] = useState<{id: string, name: string, type: 'file' | 'folder'} | null>(null)
  const [purgeConfirm, setPurgeConfirm] = useState(false)
  const [createFolderDialog, setCreateFolderDialog] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [searchText, setSearchText] = useState('')
  const [useRegex, setUseRegex] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const directoryInputRef = useRef<HTMLInputElement>(null)

  const mode = activeMode || 'local'
  const isLocalMode = mode === 'local' || mode === 'openai_assistants'

  const fetchDriveFiles = async (folderId: string | null = null) => {
    setLoading(true)
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const url = new URL(`http://${host}:${port}/${base}/drive/files`)
      if (folderId) {
        url.searchParams.append('folder_id', folderId)
      }
      const res = await fetch(url.toString())
      if (res.ok) {
        const data = await res.json()
        setDriveFiles(data.files || [])
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (e: any) {
      console.error("Failed to fetch drive files", e)
      if (e.message.includes("401") || e.message.includes("403")) {
         toast.error("Access denied. Please re-authenticate.")
      } else {
         toast.error("Failed to load Drive files")
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchLocalFiles = async () => {
    setLoading(true)
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/documents`)
      if (res.ok) {
        const data = await res.json()
        setLocalFiles(data.documents || [])
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (e: any) {
      console.error("Failed to fetch local files", e)
      toast.error("Failed to load indexed files")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isLocalMode) {
      fetchLocalFiles()
    } else {
      fetchDriveFiles(currentFolderId)
    }
  }, [config, currentFolderId, activeMode])

  const handleNavigate = (folder: DriveFile) => {
    if (folder.mimeType.includes('folder')) {
      setFolderStack(prev => [...prev, { id: folder.id, name: folder.name }])
      setCurrentFolderId(folder.id)
    }
  }

  const handleNavigateUp = () => {
    if (folderStack.length > 1) {
      const newStack = [...folderStack]
      newStack.pop()
      setFolderStack(newStack)
      setCurrentFolderId(newStack[newStack.length - 1].id)
    }
  }

  const handleBreadcrumbClick = (index: number) => {
    const newStack = folderStack.slice(0, index + 1)
    setFolderStack(newStack)
    setCurrentFolderId(newStack[newStack.length - 1].id)
  }

  const sortedDriveFiles = [...driveFiles].sort((a, b) => {
    let comparison = 0
    if (sortBy === 'name') {
      comparison = a.name.localeCompare(b.name)
    } else if (sortBy === 'date') {
      const dateA = new Date(a.modifiedTime || 0).getTime()
      const dateB = new Date(b.modifiedTime || 0).getTime()
      comparison = dateA - dateB
    } else if (sortBy === 'size') {
      const sizeA = Number(a.size || 0)
      const sizeB = Number(b.size || 0)
      comparison = sizeA - sizeB
    }
    return sortDirection === 'asc' ? comparison : -comparison
  })

  const sortedLocalFiles = [...localFiles]
    .filter(file => {
      if (!searchText) return true
      
      if (useRegex) {
        try {
          const regex = new RegExp(searchText, 'i')
          return regex.test(file.uri)
        } catch (e) {
          // Invalid regex - fall back to plain text search
          return file.uri.toLowerCase().includes(searchText.toLowerCase())
        }
      }
      
      return file.uri.toLowerCase().includes(searchText.toLowerCase())
    })
    .sort((a, b) => {
      let comparison = 0
      if (sortBy === 'name') {
        comparison = a.uri.localeCompare(b.uri)
      } else if (sortBy === 'size') {
        comparison = a.size_bytes - b.size_bytes
      }
      return sortDirection === 'asc' ? comparison : -comparison
    })

  const handleUploadToDrive = async (file: File) => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const formData = new FormData()
    formData.append('file', file)
    if (currentFolderId) {
      formData.append('folder_id', currentFolderId)
    }
    
    const toastId = toast.loading(`Uploading ${file.name}...`)
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/drive/upload`, {
        method: 'POST',
        body: formData
      })
      
      if (!res.ok) throw new Error('Upload failed')
      
      toast.success(`Uploaded ${file.name}`, { id: toastId })
      fetchDriveFiles(currentFolderId)
    } catch (e) {
      toast.error(`Failed to upload ${file.name}`, { id: toastId })
    }
  }

  const handleDeleteDriveFile = async (fileId: string) => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const toastId = toast.loading('Deleting...')
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/drive/files/${fileId}`, {
        method: 'DELETE'
      })
      
      if (!res.ok) throw new Error('Delete failed')
      
      toast.success('Deleted successfully', { id: toastId })
      fetchDriveFiles(currentFolderId)
      setDeleteConfirm(null)
    } catch (e) {
      toast.error('Failed to delete', { id: toastId })
    }
  }

  const handleDeleteLocalFile = async (uri: string) => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const toastId = toast.loading('Queueing deletion...')
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/documents/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uris: [uri] })
      })
      
      if (!res.ok) throw new Error('Delete failed')
      
      const data = await res.json()
      if (data.status === 'queued') {
        toast.success(`Queued for deletion (${data.queue_size} in queue)`, { id: toastId })
      } else {
        toast.success('Deleted successfully', { id: toastId })
      }
      fetchLocalFiles()
      setDeleteConfirm(null)
    } catch (e) {
      toast.error('Failed to delete', { id: toastId })
    }
  }

  const handleDeleteSelectedFiles = async () => {
    if (selectedFiles.size === 0) return
    
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const uris = Array.from(selectedFiles)
    const toastId = toast.loading(`Queueing ${uris.length} file(s) for deletion...`)
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/documents/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uris })
      })
      
      if (!res.ok) throw new Error('Delete failed')
      
      const data = await res.json()
      if (data.status === 'queued') {
        toast.success(`Queued ${uris.length} file(s) for deletion (${data.queue_size} in queue)`, { id: toastId })
      } else {
        toast.success('Deleted successfully', { id: toastId })
      }
      setSelectedFiles(new Set())
      fetchLocalFiles()
    } catch (e) {
      toast.error('Failed to delete', { id: toastId })
    }
  }

  const toggleFileSelection = (uri: string) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev)
      if (newSet.has(uri)) {
        newSet.delete(uri)
      } else {
        newSet.add(uri)
      }
      return newSet
    })
  }

  const selectAllFiles = () => {
    if (selectedFiles.size === sortedLocalFiles.length) {
      setSelectedFiles(new Set())
    } else {
      setSelectedFiles(new Set(sortedLocalFiles.map(f => f.uri)))
    }
  }

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return
    
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const formData = new FormData()
    formData.append('name', newFolderName.trim())
    if (currentFolderId) {
      formData.append('parent_id', currentFolderId)
    }
    
    const toastId = toast.loading('Creating folder...')
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/drive/folders`, {
        method: 'POST',
        body: formData
      })
      
      if (!res.ok) throw new Error('Create failed')
      
      toast.success('Folder created', { id: toastId })
      fetchDriveFiles(currentFolderId)
      setCreateFolderDialog(false)
      setNewFolderName('')
    } catch (e) {
      toast.error('Failed to create folder', { id: toastId })
    }
  }

  const handlePurgeIndex = async () => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    const toastId = toast.loading('Purging index...')
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/flush_cache`, {
        method: 'POST'
      })
      
      if (!res.ok) throw new Error('Purge failed')
      
      toast.success('Index purged successfully', { id: toastId })
      fetchLocalFiles()
      setPurgeConfirm(false)
    } catch (e) {
      toast.error('Failed to purge index', { id: toastId })
    }
  }

  const handleDropToDrive = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).forEach(handleUploadToDrive)
    }
  }

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1]
        resolve(base64)
      }
      reader.onerror = reject
    })
  }

  const handleAddFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const visibleFiles = Array.from(files).filter(f => !f.name.startsWith('.'))
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/upsert_document`
    const toastId = 'upload-files'
    toast.loading('Queuing files for indexing...', { id: toastId })

    let success = 0
    let skipped = files.length - visibleFiles.length

    for (const file of visibleFiles) {
      try {
        const isBinary = /\.(pdf|docx?|pages)$/i.test(file.name)
        let payload: Record<string, unknown> = { uri: file.name }
        if (isBinary) {
          payload.binary_base64 = await fileToBase64(file)
        } else {
          payload.text = await file.text()
        }
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        const data = await res.json()
        if (!res.ok || data?.error) {
          throw new Error(data?.error || `HTTP ${res.status}`)
        }
        success += 1
      } catch (err) {
        console.error('Upload failed for', file.name, err)
        skipped += 1
      }
    }

    toast.success(`Queued ${success} file(s) for indexing${skipped ? `, skipped ${skipped}` : ''}`, { id: toastId })
    fetchLocalFiles()
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleAddDirectory = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const visibleFiles = Array.from(files).filter(f => !f.name.startsWith('.'))
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/upsert_document`
    const toastId = 'upload-dir'
    toast.loading('Queuing directory for indexing...', { id: toastId })

    let success = 0
    let skipped = files.length - visibleFiles.length

    for (const file of visibleFiles) {
      try {
        const isBinary = /\.(pdf|docx?|pages)$/i.test(file.name)
        // Always use just the filename, not the full path with directories
        let payload: Record<string, unknown> = { uri: file.name }
        if (isBinary) {
          payload.binary_base64 = await fileToBase64(file)
        } else {
          payload.text = await file.text()
        }
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        const data = await res.json()
        if (!res.ok || data?.error) {
          throw new Error(data?.error || `HTTP ${res.status}`)
        }
        success += 1
      } catch (err) {
        console.error('Upload failed for', file.name, err)
        skipped += 1
      }
    }

    toast.success(`Queued ${success} file(s) for indexing${skipped ? `, skipped ${skipped}` : ''}`, { id: toastId })
    fetchLocalFiles()
    // Reset input
    if (directoryInputRef.current) {
      directoryInputRef.current.value = ''
    }
  }

  const formatSize = (bytes?: string | number) => {
    if (!bytes) return 'N/A'
    const b = Number(bytes)
    if (b === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(b) / Math.log(k))
    return parseFloat((b / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleDateString()
  }

  const getFileName = (uri: string) => {
    return uri.split('/').pop() || uri
  }

  return (
    <div className="h-[600px]">
      <Card className="flex flex-col h-full">
        <CardHeader className="pb-3">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                {isLocalMode ? (
                  <>
                    <Server className="h-5 w-5" />
                    Local Indexed Files
                  </>
                ) : (
                  <>
                    <Cloud className="h-5 w-5" />
                    Google Drive
                  </>
                )}
              </CardTitle>
              <div className="flex items-center gap-2">
                {isLocalMode ? (
                  <>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      multiple
                      accept=".txt,.pdf,.doc,.docx,.md,.json,.csv,.xml"
                      onChange={handleAddFiles}
                    />
                    <input
                      type="file"
                      ref={directoryInputRef}
                      className="hidden"
                      {...({ webkitdirectory: '' } as any)}
                      multiple
                      onChange={handleAddDirectory}
                    />
                    {selectedFiles.size > 0 && (
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={handleDeleteSelectedFiles}
                      >
                        <Trash className="h-4 w-4 mr-2" />
                        Delete {selectedFiles.size} Selected
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <FileIcon className="h-4 w-4 mr-2" />
                      Add Files
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => directoryInputRef.current?.click()}
                    >
                      <Folder className="h-4 w-4 mr-2" />
                      Add Directory
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => setPurgeConfirm(true)}
                    >
                      <Trash className="h-4 w-4 mr-2" />
                      Purge Index
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setCreateFolderDialog(true)}
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      New Folder
                    </Button>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      multiple
                      onChange={(e) => {
                        if (e.target.files) {
                          Array.from(e.target.files).forEach(handleUploadToDrive)
                        }
                      }}
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Upload
                    </Button>
                  </>
                )}
                <Select value={sortBy} onValueChange={(v: any) => setSortBy(v)}>
                  <SelectTrigger className="w-[120px] h-8">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name</SelectItem>
                    {!isLocalMode && <SelectItem value="date">Date</SelectItem>}
                    <SelectItem value="size">Size</SelectItem>
                  </SelectContent>
                </Select>
                <Button 
                  size="sm" 
                  variant="ghost" 
                  onClick={() => setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')}
                  className="h-8 w-8 p-0"
                >
                  {sortDirection === 'asc' ? <ArrowUpAZ className="h-4 w-4" /> : <ArrowDownAZ className="h-4 w-4" />}
                </Button>
                <Button 
                  size="sm" 
                  variant="ghost" 
                  onClick={() => isLocalMode ? fetchLocalFiles() : fetchDriveFiles(currentFolderId)} 
                  disabled={loading}
                  className="h-8 w-8 p-0"
                  title="Refresh file list"
                  aria-label="Refresh file list"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </Button>
              </div>
            </div>
            
            {isLocalMode && (
              <div className="flex items-center gap-2">
                <div className="flex-1 relative">
                  <Input
                    type="text"
                    placeholder={useRegex ? "Search with regex (e.g., \\.pdf$)" : "Search files..."}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    className="h-8 pr-20"
                  />
                  {searchText && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSearchText('')}
                      className="absolute right-12 top-1/2 -translate-y-1/2 h-6 w-6 p-0"
                      title="Clear search"
                    >
                      ×
                    </Button>
                  )}
                </div>
                <Button
                  size="sm"
                  variant={useRegex ? "default" : "outline"}
                  onClick={() => setUseRegex(!useRegex)}
                  className="h-8 px-3 whitespace-nowrap"
                  title={useRegex ? "Using regex mode" : "Using plain text search"}
                >
                  .*
                </Button>
              </div>
            )}
            
            {!isLocalMode && (
              <div className="flex items-center gap-2 text-sm bg-muted/30 p-2 rounded-md">
                <Button 
                  size="sm" 
                  variant="ghost" 
                  disabled={folderStack.length <= 1}
                  onClick={handleNavigateUp}
                  className="h-6 w-6 p-0"
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
                <div className="flex items-center gap-1 overflow-hidden">
                  {folderStack.map((folder, index) => (
                    <div key={index} className="flex items-center">
                      {index > 0 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                      <button 
                        onClick={() => handleBreadcrumbClick(index)}
                        className={`hover:underline truncate max-w-[150px] ${index === folderStack.length - 1 ? 'font-semibold' : 'text-muted-foreground'}`}
                      >
                        {folder.name}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0">
          {isLocalMode ? (
            <div className="h-full">
              {sortedLocalFiles.length === 0 && !loading ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                  <HardDrive className="h-10 w-10 mb-2 opacity-50" />
                  {searchText ? (
                    <>
                      <p>No files match your search</p>
                      <p className="text-xs mt-1">Try a different search term or pattern</p>
                    </>
                  ) : (
                    <>
                      <p>No indexed files</p>
                      <p className="text-xs mt-1">Files will appear here after indexing</p>
                    </>
                  )}
                </div>
              ) : (
                <ScrollArea className="h-full">
                  <div className="space-y-2 p-1">
                    {localFiles.length > 0 && (
                      <div className="flex items-center gap-2 p-2 border-b">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={selectAllFiles}
                          className="h-8"
                        >
                          {selectedFiles.size === localFiles.length ? (
                            <CheckSquare className="h-4 w-4" />
                          ) : (
                            <Square className="h-4 w-4" />
                          )}
                        </Button>
                        <span className="text-xs text-muted-foreground">
                          {selectedFiles.size > 0 ? `${selectedFiles.size} selected` : 'Select all'}
                        </span>
                      </div>
                    )}
                    {sortedLocalFiles.map((f) => (
                      <div 
                        key={f.uri} 
                        className="flex items-center justify-between p-2 rounded border bg-card hover:bg-accent cursor-pointer group"
                      >
                        <div className="flex items-center gap-3 overflow-hidden flex-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => toggleFileSelection(f.uri)}
                            className="h-8 w-8 p-0 shrink-0"
                          >
                            {selectedFiles.has(f.uri) ? (
                              <CheckSquare className="h-4 w-4 text-primary" />
                            ) : (
                              <Square className="h-4 w-4" />
                            )}
                          </Button>
                          <FileIcon className="h-8 w-8 shrink-0 text-gray-500" />
                          <div className="truncate flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{getFileName(f.uri)}</p>
                            <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
                              <span>{formatSize(f.size_bytes)}</span>
                              <span className="truncate">{f.uri}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            onClick={() => setDeleteConfirm({ id: f.uri, name: getFileName(f.uri), type: 'file' })}
                          >
                            <Trash className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </div>
          ) : (
            <div 
              className="h-full border-2 border-transparent rounded-lg transition-colors hover:bg-muted/10"
              onDragOver={(e) => {
                e.preventDefault()
                e.currentTarget.classList.add('border-primary/50', 'bg-primary/5')
              }}
              onDragLeave={(e) => {
                e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5')
              }}
              onDrop={(e) => {
                e.preventDefault()
                e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5')
                handleDropToDrive(e)
              }}
            >
              {sortedDriveFiles.length === 0 && !loading ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                  <Cloud className="h-10 w-10 mb-2 opacity-50" />
                  <p>No files found in this folder</p>
                  <p className="text-xs mt-1">Drop files here to upload</p>
                </div>
              ) : (
                <ScrollArea className="h-full">
                  <div className="space-y-2 p-1">
                    {sortedDriveFiles.map((f) => (
                      <div 
                        key={f.id} 
                        className="flex items-center justify-between p-2 rounded border bg-card hover:bg-accent cursor-pointer group"
                        onDoubleClick={() => f.mimeType.includes('folder') ? handleNavigate(f) : setPreviewFile(f)}
                      >
                        <div className="flex items-center gap-3 overflow-hidden flex-1">
                          {f.mimeType.includes('folder') ? (
                            <Folder className="h-8 w-8 shrink-0 text-blue-500 fill-blue-500/20" />
                          ) : (
                            <FileIcon className="h-8 w-8 shrink-0 text-gray-500" />
                          )}
                          <div className="truncate flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{f.name}</p>
                            <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
                              <span>{formatSize(f.size)}</span>
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                {formatDate(f.modifiedTime)}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          {f.mimeType.includes('folder') ? (
                            <>
                              <Button size="sm" variant="ghost" onClick={() => handleNavigate(f)}>
                                Open
                              </Button>
                              <Button 
                                size="sm" 
                                variant="ghost" 
                                onClick={() => setDeleteConfirm({ id: f.id, name: f.name, type: 'folder' })}
                              >
                                <Trash className="h-4 w-4 text-destructive" />
                              </Button>
                            </>
                          ) : (
                            <>
                              <Button size="sm" variant="ghost" onClick={() => setPreviewFile(f)}>
                                <Eye className="h-4 w-4" />
                              </Button>
                              <Button 
                                size="sm" 
                                variant="ghost" 
                                onClick={() => setDeleteConfirm({ id: f.id, name: f.name, type: 'file' })}
                              >
                                <Trash className="h-4 w-4 text-destructive" />
                              </Button>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={!!previewFile} onOpenChange={(o) => !o && setPreviewFile(null)}>
        <DialogContent className="max-w-3xl h-[80vh]">
          <DialogHeader>
            <DialogTitle>{previewFile?.name}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 h-full min-h-0 bg-muted/20 rounded-md p-4 overflow-hidden flex flex-col items-center justify-center">
            {previewFile?.webViewLink ? (
              <iframe src={previewFile.webViewLink.replace('view', 'preview')} className="w-full h-full border-0" />
            ) : (
              <div className="text-center">
                <FileIcon className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                <p>Preview not available</p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={createFolderDialog} onOpenChange={setCreateFolderDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Folder</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Folder name"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleCreateFolder()
              }
            }}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => {
              setCreateFolderDialog(false)
              setNewFolderName('')
            }}>
              Cancel
            </Button>
            <Button onClick={handleCreateFolder} disabled={!newFolderName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!deleteConfirm} onOpenChange={(o) => !o && setDeleteConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {deleteConfirm?.type === 'folder' ? 'Folder' : 'File'}?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deleteConfirm?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteConfirm) {
                  if (isLocalMode) {
                    handleDeleteLocalFile(deleteConfirm.id)
                  } else {
                    handleDeleteDriveFile(deleteConfirm.id)
                  }
                }
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={purgeConfirm} onOpenChange={setPurgeConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Purge Entire Index?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete ALL indexed documents? This action cannot be undone and will remove all data from the vector store.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handlePurgeIndex}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Purge All
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
