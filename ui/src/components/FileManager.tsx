import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { File as FileIcon, Folder, Upload, Cloud, HardDrive, RefreshCw, Trash, Eye, ArrowUp, ChevronRight, Calendar, ArrowDownAZ, ArrowUpAZ, Plus, Server, CheckSquare, Square, Link2, GripVertical } from 'lucide-react'
import { toast } from 'sonner'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog'

const SUPPORTED_EXTENSIONS = new Set([
  '.txt', '.pdf', '.doc', '.docx', '.md', '.markdown', '.json', '.csv', '.xml',
  '.html', '.htm', '.ppt', '.pptx', '.rtf', '.epub', '.xlsx', '.xls',
  '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.svg'
])

const isSupportedFile = (name: string) => {
  const lower = name.toLowerCase()
  const idx = lower.lastIndexOf('.')
  if (idx === -1) return false
  return SUPPORTED_EXTENSIONS.has(lower.slice(idx))
}

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

export function FileManager({ config, activeMode }: { config: any, activeMode?: string | null }) {
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [localFiles, setLocalFiles] = useState<LocalFile[]>([])
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [urlToIndex, setUrlToIndex] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [jobs, setJobs] = useState<any[]>([])
  const [showJobsOverlay, setShowJobsOverlay] = useState<boolean>(false)
  const [previewFile, setPreviewFile] = useState<DriveFile | null>(null)
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null)
  const [folderStack, setFolderStack] = useState<{ id: string | null, name: string }[]>([{ id: null, name: 'Root' }])
  const [sortBy, setSortBy] = useState<SortOption>('date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [deleteConfirm, setDeleteConfirm] = useState<{ id: string, name: string, type: 'file' | 'folder' } | null>(null)
  const [purgeConfirm, setPurgeConfirm] = useState(false)
  const [createFolderDialog, setCreateFolderDialog] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [searchText, setSearchText] = useState('')
  const [useRegex, setUseRegex] = useState(false)
  const defaultJobsPos = { x: 20, y: 250 }
  const [jobsPosition, setJobsPosition] = useState<{ x: number; y: number }>(defaultJobsPos)
  const jobsDragRef = useRef<{ dragging: boolean; startX: number; startY: number; startRight: number; startTop: number }>({ dragging: false, startX: 0, startY: 0, startRight: 0, startTop: 0 })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const directoryInputRef = useRef<HTMLInputElement>(null)

  const mode = activeMode || 'none'
  // Show Google Drive ONLY for google modes or vertex_ai_search
  // Show local file manager for everything else (local, openai_assistants, none, or when mode hasn't loaded)
  const isLocalMode = !mode || (!mode.startsWith('google') && mode !== 'vertex_ai_search')

  const fetchDriveFiles = async (folderId: string | null = null) => {
    setLoading(true)
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 15000) // 15s timeout

    try {
      const url = new URL(`http://${host}:${port}/${base}/drive/files`)
      if (folderId) {
        url.searchParams.append('folder_id', folderId)
      }
      const res = await fetch(url.toString(), { signal: controller.signal })
      if (res.ok) {
        const data = await res.json()
        setDriveFiles(data.files || [])
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error("Failed to fetch drive files", e)
        if (e.message.includes("401") || e.message.includes("403")) {
          toast.error("Access denied. Please re-authenticate.")
        } else {
          toast.error("Failed to load Drive files")
        }
      }
    } finally {
      clearTimeout(timeoutId)
      setLoading(false)
    }
  }

  const fetchLocalFiles = async () => {
    setLoading(true)
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 15000) // 15s timeout

    try {
      const res = await fetch(`http://${host}:${port}/${base}/documents`, { signal: controller.signal })
      if (res.ok) {
        const data = await res.json()
        setLocalFiles(data.documents || [])
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error("Failed to fetch local files", e)
        toast.error("Failed to load indexed files")
      }
    } finally {
      clearTimeout(timeoutId)
      setLoading(false)
    }
  }

  const fetchJobs = async () => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/jobs`)
      if (res.ok) {
        const data = await res.json()
        const allJobs = data?.jobs || []
        const activeStatuses = new Set(['queued', 'running', 'in_progress', 'pending'])
        const activeJobs = allJobs.filter((j: any) => activeStatuses.has(String(j.status).toLowerCase()))
        setJobs(activeJobs)
        if (activeJobs.length === 0) {
          setJobsPosition(defaultJobsPos)
          setShowJobsOverlay(false)
        } else {
          setShowJobsOverlay(true)
        }
      }
    } catch (e) {
      // best-effort
      setJobs([])
      setJobsPosition(defaultJobsPos)
      setShowJobsOverlay(false)
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
    let rejected = 0

    for (const file of visibleFiles) {
      if (!isSupportedFile(file.name)) {
        rejected += 1
        continue
      }
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

    if (rejected > 0) {
      toast.warning(`Rejected ${rejected} unsupported file(s)`, { id: toastId })
    } else {
      toast.success(`Queued ${success} file(s) for indexing${skipped ? `, skipped ${skipped}` : ''}`, { id: toastId })
    }
    fetchLocalFiles()
    fetchJobs()
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
    let rejected = 0

    for (const file of visibleFiles) {
      if (!isSupportedFile(file.name)) {
        rejected += 1
        continue
      }
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
        if (Array.isArray(data?.rejected) && data.rejected.length > 0) {
          toast.warning(`Rejected ${data.rejected.length} unsupported file(s)`, { id: toastId })
        }
        success += 1
      } catch (err) {
        console.error('Upload failed for', file.name, err)
        skipped += 1
      }
    }

    if (rejected > 0) {
      toast.warning(`Rejected ${rejected} unsupported file(s)`, { id: toastId })
    } else {
      toast.success(`Queued ${success} file(s) for indexing${skipped ? `, skipped ${skipped}` : ''}`, { id: toastId })
    }
    fetchLocalFiles()
    fetchJobs()
    // Reset input
    if (directoryInputRef.current) {
      directoryInputRef.current.value = ''
    }
  }

  const handleIndexUrl = async () => {
    const trimmed = urlToIndex.trim()
    if (!trimmed) {
      return
    }

    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const toastId = toast.loading('Indexing URL...')
    setUrlLoading(true)

    try {
      const res = await fetch(`http://${host}:${port}/${base}/index_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: trimmed })
      })
      const data = await res.json()
      if (!res.ok || data?.error) {
        throw new Error(data?.error || `HTTP ${res.status}`)
      }
      if (Array.isArray(data?.rejected) && data.rejected.length > 0) {
        toast.warning(`Rejected ${data.rejected.length} unsupported file(s)`, { id: toastId })
      } else {
        toast.success(`Indexed ${trimmed}`, { id: toastId })
      }
      setUrlToIndex('')
      fetchLocalFiles()
      fetchJobs()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to index URL'
      toast.error(message, { id: toastId })
    } finally {
      setUrlLoading(false)
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

  const cancelJob = async (jobId: string) => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/jobs/${jobId}/cancel`, {
        method: 'POST'
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      toast.success(`Canceled job ${jobId}`)
    } catch (e: any) {
      toast.error(`Failed to cancel job ${jobId}: ${e?.message || e}`)
    } finally {
      fetchJobs()
    }
  }

  const cancelAllJobs = async () => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/jobs/cancel_all`, { method: 'POST' })
      const data = await res.json().catch(() => ({}))
      if (res.ok) {
        const count = data?.count ?? 0
        toast.success(`Canceled ${count} job${count === 1 ? '' : 's'}`)
      } else {
        throw new Error(data?.error || `HTTP ${res.status}`)
      }
    } catch (e: any) {
      toast.error(`Failed to cancel all jobs: ${e?.message || e}`)
    } finally {
      fetchJobs()
    }
  }

  useEffect(() => {
    const handleMouseUp = () => {
      jobsDragRef.current.dragging = false
      document.body.style.userSelect = ''
    }
    const handleMouseMove = (e: MouseEvent) => {
      if (!jobsDragRef.current.dragging) return
      const deltaX = e.clientX - jobsDragRef.current.startX
      const deltaY = e.clientY - jobsDragRef.current.startY

      const nextRight = Math.max(8, jobsDragRef.current.startRight - deltaX)
      const nextTop = Math.max(8, jobsDragRef.current.startTop + deltaY)

      setJobsPosition({ x: nextRight, y: nextTop })
    }
    window.addEventListener('mouseup', handleMouseUp)
    window.addEventListener('mousemove', handleMouseMove)
    if (isLocalMode) {
      fetchLocalFiles()
      fetchJobs()
    }

    const interval = setInterval(() => {
      if (isLocalMode) {
        const hadJobs = jobs.length > 0
        fetchJobs().then(() => {
          // If we have active jobs, or if we had active jobs (meaning one might have just finished),
          // refresh the file list to show progress/completion.
          // We use the state 'jobs' from the render cycle, but inside the interval we get the fresh
          // data from fetchJobs. However, 'jobs' state updates are async.
          // A simpler heuristic: if the updated job queue is not empty, OR if we previously had jobs, refresh files.
          // Since we can't easily see the *result* of fetchJobs immediately without refactoring,
          // we can piggyback on the fact that we are polling.
          // Let's just refresh files every time we refresh jobs if there's *any* activity.
          // For simplicity and robustness, if there are jobs, we refresh files.
          // To catch the "just finished" case, we can rely on the fact that the user
          // will likely see the file appear on the next poll 8s later, or we can just always poll files
          // in local mode, but that might be heavy.
          // Better approach: If we detect active jobs, we refresh files.

          // Actually, let's just assume if we are polling jobs, we might as well poll files
          // but maybe less frequently? 
          // For now, matching the user request: "Make it auto refresh like the queued files list does."
          // The queued files list refreshes every 8s via this interval.
          // So we should just add fetchLocalFiles() here.
          fetchLocalFiles()
        })
      }
    }, 8000)
    return () => {
      window.removeEventListener('mouseup', handleMouseUp)
      window.removeEventListener('mousemove', handleMouseMove)
      clearInterval(interval)
    }
  }, [isLocalMode]) // eslint-disable-line react-hooks/exhaustive-deps

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
                {isLocalMode && showJobsOverlay && jobs.length > 0 && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={fetchJobs}
                    title="Refresh indexing jobs"
                  >
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Jobs
                  </Button>
                )}
                {isLocalMode ? (
                  <>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      multiple
                      accept=".txt,.pdf,.doc,.docx,.md,.markdown,.json,.csv,.xml,.html,.htm,.ppt,.pptx,.rtf,.epub,.xlsx,.xls,.png,.jpg,.jpeg,.tiff,.bmp"
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
                  className="h-8 w-8 p-0 relative z-10 cursor-pointer"
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

            {isLocalMode && (
              <div className="flex items-center gap-2">
                <Input
                  type="url"
                  placeholder="https://example.com/resume.pdf"
                  value={urlToIndex}
                  onChange={(e) => setUrlToIndex(e.target.value)}
                  className="h-8"
                />
                <Button
                  size="sm"
                  onClick={handleIndexUrl}
                  disabled={!urlToIndex.trim() || urlLoading}
                >
                  <Link2 className="h-4 w-4 mr-2" />
                  {urlLoading ? 'Indexing…' : 'Index URL'}
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
        <CardContent className="flex-1 min-h-0 relative">
          {isLocalMode && showJobsOverlay && jobs.length > 0 && (
            <div
              className="fixed z-50 w-full md:w-80 drop-shadow-lg pointer-events-auto max-h-[60vh] flex flex-col"
              style={{ top: jobsPosition.y, right: jobsPosition.x }}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="bg-card border rounded-lg p-2 space-y-2 flex flex-col flex-1 overflow-hidden">
                <div className="flex items-center justify-between sticky top-0 bg-card z-10 pb-1">
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 cursor-move"
                      title="Drag"
                      onMouseDown={(e) => {
                        e.stopPropagation()
                        jobsDragRef.current = {
                          dragging: true,
                          startX: e.clientX,
                          startY: e.clientY,
                          startRight: jobsPosition.x,
                          startTop: jobsPosition.y
                        }
                        document.body.style.userSelect = 'none'
                      }}
                    >
                      <GripVertical className="h-4 w-4" />
                    </Button>
                    <div className="text-sm font-semibold select-none">Indexing Jobs</div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => { e.stopPropagation(); setShowJobsOverlay(false); setJobs([]); }}
                      title="Close"
                    >
                      ×
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => { e.stopPropagation(); fetchJobs() }}
                      title="Refresh jobs"
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => { e.stopPropagation(); cancelAllJobs() }}
                      title="Cancel all jobs"
                    >
                      <Trash className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
                <ScrollArea className="flex-1 overflow-y-auto" scrollHideDelay={0}>
                  <div className="space-y-1 pr-1 pb-1">
                    {jobs.map((job) => (
                      <div key={job.id} className="flex items-center justify-between rounded-md border px-2 py-1 text-sm bg-background">
                        <div className="flex-1 min-w-0">
                          <div className="truncate font-medium max-w-[200px]">{job.uri || job.path || job.id}</div>
                          <div className="text-xs text-muted-foreground">Status: {job.status}</div>
                          {Array.isArray(job.rejected) && job.rejected.length > 0 && (
                            <div className="text-xs text-destructive">Rejected: {job.rejected.length} file(s)</div>
                          )}
                        </div>
                        {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'canceled' && (
                          <Button variant="ghost" size="icon" onClick={() => cancelJob(job.id)} title="Cancel job">
                            <Trash className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>
          )}
          <div className="flex flex-col h-full gap-3">
            {isLocalMode ? (
              <div className="flex-1 min-h-0">
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
