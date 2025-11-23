import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { File as FileIcon, Folder, Upload, Cloud, HardDrive, RefreshCw, Trash, Eye, ArrowLeft, ArrowUp, ChevronRight, Calendar, ArrowDownAZ, ArrowUpAZ } from 'lucide-react'
import { toast } from 'sonner'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

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

export function FileManager({ config }: { config: any }) {
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [loading, setLoading] = useState(false)
  const [previewFile, setPreviewFile] = useState<DriveFile | null>(null)
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null)
  const [folderStack, setFolderStack] = useState<{id: string | null, name: string}[]>([{id: null, name: 'Root'}])
  const [sortBy, setSortBy] = useState<SortOption>('date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

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

  useEffect(() => {
    fetchDriveFiles(currentFolderId)
  }, [config, currentFolderId])

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

  const sortedFiles = [...driveFiles].sort((a, b) => {
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

  const handleDropToDrive = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).forEach(handleUploadToDrive)
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

  return (
    <div className="h-[600px]">
      {/* Google Drive Pane */}
      <Card className="flex flex-col h-full">
        <CardHeader className="pb-3">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                Google Drive
              </CardTitle>
              <div className="flex items-center gap-2">
                <Select value={sortBy} onValueChange={(v: any) => setSortBy(v)}>
                  <SelectTrigger className="w-[120px] h-8">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="date">Date</SelectItem>
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
                <Button size="sm" variant="ghost" onClick={() => fetchDriveFiles(currentFolderId)} disabled={loading}>
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </Button>
              </div>
            </div>
            
            {/* Navigation Bar */}
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
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0">
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
            {sortedFiles.length === 0 && !loading ? (
              <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                <Cloud className="h-10 w-10 mb-2 opacity-50" />
                <p>No files found in this folder</p>
                <p className="text-xs mt-1">Drop files here to upload</p>
              </div>
            ) : (
              <ScrollArea className="h-full">
                <div className="space-y-2 p-1">
                  {sortedFiles.map((f) => (
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
                          <Button size="sm" variant="ghost" onClick={() => handleNavigate(f)}>
                            Open
                          </Button>
                        ) : (
                          <Button size="sm" variant="ghost" onClick={() => setPreviewFile(f)}>
                            <Eye className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>
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
    </div>
  )
}
