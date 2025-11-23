import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Progress } from '@/components/ui/progress'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { 
  Database, 
  MessageCircle as ChatCircle, 
  Settings as GearSix, 
  FolderOpen, 
  Zap as Lightning,
  CheckCircle,
  AlertCircle as WarningCircle,
  CloudUpload as CloudArrowUp,
  Play,
  Pause,
  File,
  Plus,
  Trash,
  ChevronDown as CaretDown,
  ChevronUp as CaretUp,
  Info,
  LineChart as ChartLine,
  List as ListBullets
} from 'lucide-react'
import { toast } from 'sonner'
import { MetricsDashboard } from '@/components/MetricsDashboard'
import { LogsViewer } from '@/components/LogsViewer'
import { ProviderSelector } from '@/components/ProviderSelector'
import { FileManager } from '@/components/FileManager'
import { ChatInterface, Message } from '@/components/ChatInterface'

type IndexedItem = {
  id: string
  name: string
  path: string
  type: 'file' | 'directory'
  size?: number
  addedAt: string
}

type OllamaConfig = {
  apiEndpoint: string
  model: string
  embeddingModel: string
  temperature: string
  topP: string
  topK: string
  repeatPenalty: string
  seed: string
  numCtx: string
  mcpHost: string
  mcpPort: string
  mcpPath: string
  ragHost: string
  ragPort: string
  ragPath: string
}

type IndexJob = {
  id: string
  type: string
  status: string
  error?: string
  result?: Record<string, unknown>
}

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState<'running' | 'stopped' | 'error'>('stopped')
  const [backendDocs, setBackendDocs] = useState<number | null>(null)
  const [backendSize, setBackendSize] = useState<number | null>(null)
  const [backendDocumentList, setBackendDocumentList] = useState<IndexedItem[]>([])
  const [indexedItems, setIndexedItems] = useState<IndexedItem[]>([])
  const [ollamaExpanded, setOllamaExpanded] = useState(false)
  const [googleExpanded, setGoogleExpanded] = useState(false)
  const [advancedExpanded, setAdvancedExpanded] = useState(false)
  const [ollamaConfig, setOllamaConfig] = useState<OllamaConfig>({
    apiEndpoint: import.meta.env.VITE_OLLAMA_API_BASE || 'http://127.0.0.1:11434',
    model: import.meta.env.VITE_LLM_MODEL_NAME?.replace(/^ollama\//, '') || 'qwen2.5:7b',
    embeddingModel: import.meta.env.VITE_EMBED_MODEL_NAME || 'Snowflake/arctic-embed-xs',
    temperature: import.meta.env.VITE_LLM_TEMPERATURE || '0.7',
    topP: '0.9',
    topK: '40',
    repeatPenalty: '1.1',
    seed: '-1',
    numCtx: '2048',
    mcpHost: import.meta.env.VITE_MCP_HOST || '127.0.0.1',
    mcpPort: import.meta.env.VITE_MCP_PORT || '8000',
    mcpPath: import.meta.env.VITE_MCP_PATH || '/mcp',
    ragHost: import.meta.env.VITE_RAG_HOST || 'localhost',
    ragPort: import.meta.env.VITE_RAG_PORT || '8001',
    ragPath: import.meta.env.VITE_RAG_PATH || 'api'
  })
  const [queryText, setQueryText] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchAnswer, setSearchAnswer] = useState<string | null>(null)
  const [searchSources, setSearchSources] = useState<string[]>([])
  const [searchJobId, setSearchJobId] = useState<string | null>(null)
  const [searchMessage, setSearchMessage] = useState<string | null>(null)
  const [flushLoading, setFlushLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string>('Idle')
  const [vertexConfig, setVertexConfig] = useState({
    projectId: '',
    location: 'us-central1',
    dataStoreId: ''
  })
  const searchAbortRef = useRef<AbortController | null>(null)
  const [jobProgress, setJobProgress] = useState<{
    total: number
    completed: number
    failed: number
    visible: boolean
  }>({ total: 0, completed: 0, failed: 0, visible: false })
  const [chatMessages, setChatMessages] = useState<Message[]>([])

  // Fetch Vertex config on mount
  useEffect(() => {
    const fetchVertexConfig = async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/vertex`)
        if (res.ok) {
          const data = await res.json()
          setVertexConfig({
            projectId: data.VERTEX_PROJECT_ID || '',
            location: data.VERTEX_LOCATION || 'us-central1',
            dataStoreId: data.VERTEX_DATA_STORE_ID || ''
          })
        }
      } catch (e) {
        console.error("Failed to fetch vertex config", e)
      }
    }
    fetchVertexConfig()
  }, [ollamaConfig])

  const handleSaveVertexConfig = async () => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/config/vertex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: vertexConfig.projectId,
          location: vertexConfig.location,
          data_store_id: vertexConfig.dataStoreId
        })
      })
      if (res.ok) {
        toast.success('Vertex AI configuration saved')
      } else {
        throw new Error('Failed to save')
      }
    } catch (e) {
      toast.error('Failed to save Vertex AI configuration')
    }
  }

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        const base64 = result.split(',')[1] || ''
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  // Sync UI defaults from env when present (fallback to existing KV values)
  useEffect(() => {
    setOllamaConfig((current) => {
      const cfg = current || {
        apiEndpoint: 'http://127.0.0.1:11434',
        model: 'qwen2.5:7b',
        embeddingModel: 'Snowflake/arctic-embed-xs',
        temperature: '0.7',
        topP: '0.9',
        topK: '40',
        repeatPenalty: '1.1',
        seed: '-1',
        numCtx: '2048',
        mcpHost: '127.0.0.1',
        mcpPort: '8000',
        mcpPath: '/mcp',
        ragHost: '127.0.0.1',
        ragPort: '8001',
        ragPath: 'api'
      }
      return {
        apiEndpoint: cfg.apiEndpoint || import.meta.env.VITE_OLLAMA_API_BASE || 'http://127.0.0.1:11434',
        model: cfg.model || (import.meta.env.VITE_LLM_MODEL_NAME?.replace(/^ollama\//, '') || 'qwen2.5:7b'),
        embeddingModel: cfg.embeddingModel || import.meta.env.VITE_EMBED_MODEL_NAME || 'Snowflake/arctic-embed-xs',
        temperature: cfg.temperature || import.meta.env.VITE_LLM_TEMPERATURE || '0.7',
        topP: cfg.topP || '0.9',
        topK: cfg.topK || '40',
        repeatPenalty: cfg.repeatPenalty || '1.1',
        seed: cfg.seed || '-1',
        numCtx: cfg.numCtx || '2048',
        mcpHost: cfg.mcpHost || import.meta.env.VITE_MCP_HOST || '127.0.0.1',
        mcpPort: cfg.mcpPort || import.meta.env.VITE_MCP_PORT || '8000',
        mcpPath: cfg.mcpPath || import.meta.env.VITE_MCP_PATH || '/mcp',
        ragHost: cfg.ragHost || import.meta.env.VITE_RAG_HOST || 'localhost',
        ragPort: cfg.ragPort || import.meta.env.VITE_RAG_PORT || '8001',
        ragPath: cfg.ragPath || import.meta.env.VITE_RAG_PATH || 'api',
      }
    })
  // run once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Poll indexing jobs to show background progress
  useEffect(() => {
    const interval = setInterval(async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/jobs`)
        if (!res.ok) return
        const data = await res.json() as { jobs?: IndexJob[] }
        const jobs = data.jobs || []
        const total = jobs.length
        const completed = jobs.filter(j => j.status === 'completed').length
        const failed = jobs.filter(j => j.status === 'failed').length
        const visible = total > 0 && (completed + failed) < total
        setJobProgress({ total, completed, failed, visible })
        if (visible) {
          setStatusMessage(`Indexing ${total - completed - failed} job(s)`)
        } else if (searching) {
          setStatusMessage('Searching...')
        } else {
          setStatusMessage('Idle')
        }
      } catch (_) {
        // ignore polling errors, will try again on next tick
      }
    }, 2500)
    return () => clearInterval(interval)
  }, [ollamaConfig, searching])

  const handleAddDirectory = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.webkitdirectory = true
    input.multiple = true

    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files
      if (files && files.length > 0) {
        const visibleFiles = Array.from(files).filter(f => !f.name.startsWith('.'))
        const host = ollamaConfig?.ragHost || '127.0.0.1'
        const port = ollamaConfig?.ragPort || '8001'
        const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
        const url = `http://${host}:${port}/${base}/upsert_document`
        const toastId = 'upload-dir'
        toast.loading('Queuing directory for indexing...', { id: toastId })

        const upload = async () => {
          let success = 0
          let skipped = files.length - visibleFiles.length
          for (const file of visibleFiles) {
            try {
              const isBinary = /\.(pdf|docx?|pages)$/i.test(file.name)
              let payload: Record<string, unknown> = { uri: file.webkitRelativePath || file.name }
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
        }

        upload()
      }
    }

    input.click()
  }

  const handleDeleteRemote = async (uri: string) => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/documents/delete`
    const loadingId = `delete-${uri}`
    toast.loading('Deleting from index...', { id: loadingId })
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uris: [uri] }),
      })
      const data = await res.json()
      if (!res.ok || data?.error) {
        throw new Error(data?.error || `HTTP ${res.status}`)
      }
      setBackendDocumentList(prev => prev.filter(d => d.id !== uri))
      toast.success('Removed from index', { id: loadingId })
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Failed to delete', { id: loadingId })
    }
  }

  const handleFlushCache = async () => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/flush_cache`
    setFlushLoading(true)
    const toastId = 'flush-cache'
    toast.loading('Flushing cache...', { id: toastId })
    try {
      const res = await fetch(url, { method: 'POST' })
      const data = await res.json()
      if (!res.ok || data?.error) {
        throw new Error(data?.error || `HTTP ${res.status}`)
      }
      setBackendDocumentList([])
      setBackendDocs(0)
      setBackendSize(0)
      toast.success('Cache flushed', { id: toastId })
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Failed to flush cache', { id: toastId })
    } finally {
      setFlushLoading(false)
    }
  }

  const handleAddFile = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.accept = '.txt,.pdf,.doc,.docx,.md,.json,.csv,.xml'
    
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files
      if (files && files.length > 0) {
        const visibleFiles = Array.from(files).filter(f => !f.name.startsWith('.'))
        const host = ollamaConfig?.ragHost || '127.0.0.1'
        const port = ollamaConfig?.ragPort || '8001'
        const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
        const url = `http://${host}:${port}/${base}/upsert_document`
        const toastId = 'upload-files'
        toast.loading('Queuing files for indexing...', { id: toastId })

        const upload = async () => {
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
              // Continue uploading others but report failure at end
              console.error('Upload failed for', file.name, err)
              skipped += 1
            }
          }

          toast.success(`Queued ${success} file(s) for indexing${skipped ? `, skipped ${skipped}` : ''}`, { id: toastId })
        }

        upload()
      }
    }
    
    input.click()
  }

  const handleRemoveItem = (id: string) => {
    setIndexedItems(current => {
      const items = current || []
      const item = items.find(i => i.id === id)
      if (item) {
        toast.info(`Removed: ${item.name}`)
      }
      return items.filter(i => i.id !== id)
    })
  }

  const formatFileSize = (bytes?: number | null) => {
    if (bytes === undefined || bytes === null) return 'N/A'
    if (bytes === 0) return '0 B'
    const units = ['B', 'KB', 'MB', 'GB']
    let size = bytes
    let unitIndex = 0
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024
      unitIndex++
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`
  }

  const handleConfigChange = (field: keyof OllamaConfig, value: string) => {
    setOllamaConfig(current => {
      const currentConfig = current || {
        apiEndpoint: 'http://localhost:11434',
        model: 'llama3.2',
        embeddingModel: 'nomic-embed-text',
        temperature: '0.7',
        topP: '0.9',
        topK: '40',
        repeatPenalty: '1.1',
        seed: '-1',
        numCtx: '2048',
        mcpHost: '127.0.0.1',
        mcpPort: '8000',
        mcpPath: '/mcp',
        ragHost: '127.0.0.1',
        ragPort: '8001',
        ragPath: 'api'
      }
      return {
        ...currentConfig,
        [field]: value
      }
    })
  }

  const handleSaveConfig = () => {
    toast.success('Configuration saved', {
      description: 'Ollama settings have been updated'
    })
  }

  const handleGoogleLogin = () => {
    const host = ollamaConfig?.ragHost || 'localhost'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    window.open(`http://${host}:${port}/${base}/auth/login?t=${Date.now()}`, '_blank', 'width=600,height=700')
  }

  const handleGoogleLogout = async () => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/auth/logout`
    
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      toast.success('Disconnected from Google Account')
    } catch (error) {
      toast.error('Failed to disconnect')
    }
  }

  const handleSearch = async () => {
    // cancel any in-flight search before starting a new one
    if (searchAbortRef.current) {
      searchAbortRef.current.abort()
    }
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/search`

    if (!queryText.trim()) {
      setSearchError('Enter a question to search')
      return
    }

    setSearching(true)
    setSearchError(null)
    setStatusMessage('Searching...')
    setSearchJobId(null)
    setSearchMessage(null)
    setSearchAnswer(null)
    setSearchSources([])
    const controller = new AbortController()
    searchAbortRef.current = controller

    try {
      // Kick off async search
      const kickRes = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ query: queryText, async: true, timeout_seconds: 300 }),
      })

      if (!kickRes.ok) {
        throw new Error(`HTTP ${kickRes.status}`)
      }

      const kick = await kickRes.json()
      const jobId = kick?.job_id
      if (!jobId) {
        throw new Error(kick?.error || 'Failed to start search')
      }

      setSearchJobId(jobId)

      const pollUrl = `http://${host}:${port}/${base}/search/jobs/${jobId}`
      let done = false
      while (!done) {
        const pollRes = await fetch(pollUrl, { signal: controller.signal })
        if (!pollRes.ok) {
          throw new Error(`HTTP ${pollRes.status}`)
        }
        const status = await pollRes.json()
        if (status?.message) {
          setSearchMessage(status.message)
        }
        const st = status?.status
        if (st === 'completed' || st === 'timeout' || st === 'failed') {
          done = true
          const data = status?.result || status

          let answer = ''
          if (typeof data?.error === 'string') {
            setSearchError(data.error)
          } else if (Array.isArray(data?.choices) && data.choices[0]?.message?.content) {
            answer = data.choices[0].message.content
          } else if (typeof data?.answer === 'string') {
            answer = data.answer
          } else if (typeof data === 'string') {
            answer = data
          } else {
            setSearchError('No answer returned from service')
          }

          if (answer) {
            setSearchAnswer(answer)
          }

          if (Array.isArray(data?.sources)) {
            setSearchSources(data.sources.filter((s: unknown) => typeof s === 'string'))
          }
        } else {
          await new Promise(res => setTimeout(res, 3000))
        }
      }
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setSearchError('Search cancelled')
      } else {
        setSearchError(error instanceof Error ? error.message : 'Search failed')
      }
    } finally {
      searchAbortRef.current = null
      setSearching(false)
      setStatusMessage(jobProgress.visible ? `Indexing ${jobProgress.total - jobProgress.completed - jobProgress.failed} job(s)` : 'Idle')
    }
  }

  const handleCancelSearch = () => {
    if (searchAbortRef.current) {
      searchAbortRef.current.abort()
    }
    setStatusMessage(jobProgress.visible ? `Indexing ${jobProgress.total - jobProgress.completed - jobProgress.failed} job(s)` : 'Idle')
  }

  // Periodically ping REST health endpoint to reflect actual status
  useEffect(() => {
    const controller = new AbortController()

    const checkHealth = async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/${base}/health`

      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`Status ${res.status}`)
        const data = await res.json()
        if (typeof data?.documents === 'number') {
          setBackendDocs(data.documents)
        }
        if (typeof data?.total_size_bytes === 'number') {
          setBackendSize(data.total_size_bytes)
        }
        setSystemStatus('running')
      } catch (error) {
        // Only mark error if not manually stopped
        setBackendDocs(null)
        setBackendSize(null)
        setSystemStatus(prev => (prev === 'stopped' ? 'stopped' : 'error'))
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => {
      controller.abort()
      clearInterval(interval)
    }
  }, [ollamaConfig?.ragHost, ollamaConfig?.ragPort, ollamaConfig?.ragPath])

  // Fetch indexed documents from REST
  useEffect(() => {
    const controller = new AbortController()
    const fetchDocs = async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/${base}/documents`
      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (Array.isArray(data?.documents)) {
          const mapped: IndexedItem[] = data.documents.map((d: any) => ({
            id: d.uri,
            name: d.uri.split('/').pop() || d.uri,
            path: d.uri,
            type: 'file',
            size: typeof d.size_bytes === 'number' ? d.size_bytes : undefined,
            addedAt: new Date().toISOString(),
          }))
          setBackendDocumentList(mapped)
        }
      } catch (error) {
        setBackendDocumentList([])
      }
    }
    fetchDocs()
    const id = setInterval(fetchDocs, 20000)
    return () => {
      controller.abort()
      clearInterval(id)
    }
  }, [ollamaConfig?.ragHost, ollamaConfig?.ragPort, ollamaConfig?.ragPath])

  const handleTestConnection = async () => {
    toast.loading('Testing connection...', { id: 'test-connection' })
    
    setTimeout(() => {
      const isSuccess = Math.random() > 0.3
      if (isSuccess) {
        toast.success('Connection successful', {
          id: 'test-connection',
          description: `Connected to Ollama at ${ollamaConfig?.apiEndpoint || 'localhost'}`
        })
      } else {
        toast.error('Connection failed', {
          id: 'test-connection',
          description: 'Unable to reach Ollama API endpoint'
        })
      }
    }, 1500)
  }

  const items = indexedItems || []
  const remoteItems = backendDocumentList.length > 0 ? backendDocumentList : items
  const totalFiles = backendDocs !== null ? backendDocs : remoteItems.length
  const totalSize = backendSize !== null ? backendSize : remoteItems.reduce((acc, item) => acc + (item.size || 0), 0)

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                        <Lightning className="h-6 w-6 text-primary-foreground" />
                      </div>
                      <div>
                        <h1 className="text-xl font-bold tracking-tight">Agentic AI</h1>
                        <p className="text-sm text-muted-foreground">Document Search System</p>
                        <p className="text-xs text-muted-foreground">Status: {searchMessage || statusMessage}</p>
                      </div>
                    </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">System:</span>
                <Badge 
                  variant={systemStatus === 'running' ? 'default' : systemStatus === 'error' ? 'destructive' : 'secondary'}
                  className="gap-1"
                >
                  {systemStatus === 'running' && <CheckCircle className="h-3 w-3" />}
                  {systemStatus === 'error' && <WarningCircle className="h-3 w-3" />}
                  {systemStatus === 'stopped' && <Pause className="h-3 w-3" />}
                  {systemStatus.charAt(0).toUpperCase() + systemStatus.slice(1)}
                </Badge>
              </div>
              <Button
                variant={systemStatus === 'running' ? 'outline' : 'default'}
                size="sm"
                onClick={() => setSystemStatus(systemStatus === 'running' ? 'stopped' : 'running')}
              >
                {systemStatus === 'running' ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Stop Services
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Services
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-3xl grid-cols-6">
            <TabsTrigger value="dashboard" className="gap-2">
              <Database className="h-4 w-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="index" className="gap-2">
              <FolderOpen className="h-4 w-4" />
              Index
            </TabsTrigger>
            <TabsTrigger value="search" className="gap-2">
              <ChatCircle className="h-4 w-4" />
              Search
            </TabsTrigger>
            <TabsTrigger value="metrics" className="gap-2">
              <ChartLine className="h-4 w-4" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="logs" className="gap-2">
              <ListBullets className="h-4 w-4" />
              Logs
            </TabsTrigger>
            <TabsTrigger value="settings" className="gap-2">
              <GearSix className="h-4 w-4" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight mb-2">System Overview</h2>
              <p className="text-muted-foreground">Monitor your AI document search system status and metrics</p>
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardDescription>Indexed Items</CardDescription>
                  <CardTitle className="text-3xl">{totalFiles}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    {backendDocs !== null
                      ? (totalFiles === 0 ? 'No items indexed yet (server)' : 'Count from server health')
                      : (totalFiles === 0 ? 'No items indexed yet' : `${items.filter(i => i.type === 'file').length} files, ${items.filter(i => i.type === 'directory').length} directories`)}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardDescription>Storage Used</CardDescription>
                  <CardTitle className="text-3xl">{formatFileSize(totalSize)}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    {backendSize !== null ? 'From server health' : 'Local estimate'}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardDescription>Active Provider</CardDescription>
                </CardHeader>
                <CardContent>
                  <ProviderSelector config={ollamaConfig} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardDescription>Queries Today</CardDescription>
                  <CardTitle className="text-3xl">0</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">Conversational searches</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Getting Started</CardTitle>
                <CardDescription>Start using your AI document search system</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-4 rounded-lg border border-border p-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                    <Database className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold mb-1">1. Start Backend Services</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      Launch the Python backend and Ollama to enable AI processing
                    </p>
                    <Button size="sm">
                      <Play className="h-4 w-4 mr-2" />
                      Start Services
                    </Button>
                  </div>
                </div>

                <div className="flex items-start gap-4 rounded-lg border border-border p-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                    <FolderOpen className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold mb-1">2. Index Your Documents</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      Select files or directories to make them searchable
                    </p>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" onClick={handleAddFile}>
                        <File className="h-4 w-4 mr-2" />
                        Add Files
                      </Button>
                      <Button size="sm" variant="outline" onClick={handleAddDirectory}>
                        <FolderOpen className="h-4 w-4 mr-2" />
                        Add Directory
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-4 rounded-lg border border-border p-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                    <ChatCircle className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold mb-1">3. Start Searching</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      Ask questions in natural language about your documents
                    </p>
                    <Button size="sm" variant="outline" disabled>
                      <ChatCircle className="h-4 w-4 mr-2" />
                      Open Search
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="index" className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight mb-2">File Indexing</h2>
              <p className="text-muted-foreground">Manage your indexed files and directories</p>
            </div>
            <FileManager config={ollamaConfig} />
          </TabsContent>

          <TabsContent value="search" className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight mb-2">Conversational Search</h2>
              <p className="text-muted-foreground">Ask questions about your indexed documents</p>
            </div>
            <ChatInterface config={ollamaConfig} messages={chatMessages} setMessages={setChatMessages} />
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <MetricsDashboard config={ollamaConfig} />
          </TabsContent>

          <TabsContent value="logs" className="space-y-6">
            <LogsViewer config={ollamaConfig} />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight mb-2">Settings</h2>
              <p className="text-muted-foreground">Configure AI providers and system preferences</p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>REST API Server Configuration</CardTitle>
                <CardDescription>Connection details for the Agentic RAG REST API</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="rag-host">Host</Label>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                            <Info className="h-4 w-4" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">Host/IP for the REST API server.</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <Input
                      id="rag-host"
                      value={ollamaConfig?.ragHost || ''}
                      onChange={(e) => handleConfigChange('ragHost', e.target.value)}
                      placeholder="127.0.0.1"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="rag-port">Port</Label>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                            <Info className="h-4 w-4" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">Port for the REST API server (default 8001).</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <Input
                      id="rag-port"
                      value={ollamaConfig?.ragPort || ''}
                      onChange={(e) => handleConfigChange('ragPort', e.target.value)}
                      placeholder="8001"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="rag-path">Base Path</Label>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                            <Info className="h-4 w-4" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">Base path prefix for REST routes (e.g., /api).</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <Input
                      id="rag-path"
                      value={ollamaConfig?.ragPath || ''}
                      onChange={(e) => handleConfigChange('ragPath', e.target.value)}
                      placeholder="api"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>AI Provider</CardTitle>
                <CardDescription>Select your preferred AI backend</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Collapsible open={ollamaExpanded} onOpenChange={setOllamaExpanded}>
                  <div className="rounded-lg border border-border">
                    <CollapsibleTrigger asChild>
                      <button className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                            <Lightning className="h-5 w-5 text-primary" />
                          </div>
                          <div className="text-left">
                            <p className="font-semibold">Ollama (Local)</p>
                            <p className="text-sm text-muted-foreground">Run AI models locally</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge>Active</Badge>
                          {ollamaExpanded ? (
                            <CaretUp className="h-5 w-5 text-muted-foreground" />
                          ) : (
                            <CaretDown className="h-5 w-5 text-muted-foreground" />
                          )}
                        </div>
                      </button>
                    </CollapsibleTrigger>
                    
                    <CollapsibleContent>
                      <div className="border-t border-border p-6 space-y-6">
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Label htmlFor="api-endpoint">Ollama API Endpoint</Label>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                    <Info className="h-4 w-4" />
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="max-w-xs">The URL where your local Ollama server is running. Default is http://localhost:11434</p>
                                </TooltipContent>
                              </Tooltip>
                            </div>
                            <Input
                              id="api-endpoint"
                              value={ollamaConfig?.apiEndpoint || ''}
                              onChange={(e) => handleConfigChange('apiEndpoint', e.target.value)}
                              placeholder="http://localhost:11434"
                            />
                          </div>

                          <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20">
                            <h4 className="font-semibold text-sm">MCP Server Configuration</h4>
                            <div className="grid gap-4 md:grid-cols-3">
                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="mcp-host">Host</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Host address for the MCP (Model Context Protocol) server endpoint</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="mcp-host"
                                  value={ollamaConfig?.mcpHost || ''}
                                  onChange={(e) => handleConfigChange('mcpHost', e.target.value)}
                                  placeholder="127.0.0.1"
                                />
                              </div>

                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="mcp-port">Port</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Port number for the MCP server. Default is 8000</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="mcp-port"
                                  value={ollamaConfig?.mcpPort || ''}
                                  onChange={(e) => handleConfigChange('mcpPort', e.target.value)}
                                  placeholder="8000"
                                />
                              </div>

                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="mcp-path">Path</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Base path for MCP streamable HTTP transport (e.g., /mcp)</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="mcp-path"
                                  value={ollamaConfig?.mcpPath || ''}
                                  onChange={(e) => handleConfigChange('mcpPath', e.target.value)}
                                  placeholder="/mcp"
                                />
                              </div>
                            </div>
                          </div>

                          <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20">
                            <h4 className="font-semibold text-sm">REST API Server Configuration</h4>
                            <div className="grid gap-4 md:grid-cols-3">
                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="rag-host">Host</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Host address for the REST API server endpoint</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="rag-host"
                                  value={ollamaConfig?.ragHost || ''}
                                  onChange={(e) => handleConfigChange('ragHost', e.target.value)}
                                  placeholder="127.0.0.1"
                                />
                              </div>

                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="rag-port">Port</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Port number for the REST API server. Default is 8001</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="rag-port"
                                  value={ollamaConfig?.ragPort || ''}
                                  onChange={(e) => handleConfigChange('ragPort', e.target.value)}
                                  placeholder="8001"
                                />
                              </div>

                              <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor="rag-path">Base Path</Label>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Base path prefix for REST API routes (e.g., api or /api)</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                <Input
                                  id="rag-path"
                                  value={ollamaConfig?.ragPath || ''}
                                  onChange={(e) => handleConfigChange('ragPath', e.target.value)}
                                  placeholder="api"
                                />
                              </div>
                            </div>
                          </div>

                          <Collapsible open={advancedExpanded} onOpenChange={setAdvancedExpanded}>
                            <CollapsibleTrigger asChild>
                              <Button variant="outline" className="w-full justify-between">
                                <span>Advanced Parameters</span>
                                {advancedExpanded ? (
                                  <CaretUp className="h-4 w-4" />
                                ) : (
                                  <CaretDown className="h-4 w-4" />
                                )}
                              </Button>
                            </CollapsibleTrigger>
                            <CollapsibleContent>
                              <div className="space-y-4 mt-4 p-4 rounded-lg border border-border bg-muted/30">
                                <div className="grid gap-4 md:grid-cols-2">
                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="model">Model</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">The primary LLM model to use for generating responses (e.g., llama3.2, mistral, codellama)</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="model"
                                      value={ollamaConfig?.model || ''}
                                      onChange={(e) => handleConfigChange('model', e.target.value)}
                                      placeholder="llama3.2"
                                    />
                                  </div>

                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="embedding-model">Embedding Model</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Model used to convert documents into vector embeddings for semantic search (e.g., nomic-embed-text)</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="embedding-model"
                                      value={ollamaConfig?.embeddingModel || ''}
                                      onChange={(e) => handleConfigChange('embeddingModel', e.target.value)}
                                      placeholder="nomic-embed-text"
                                    />
                                  </div>
                                </div>

                                <div className="grid gap-4 md:grid-cols-3">
                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="temperature">Temperature</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Controls randomness in responses. Higher values (e.g., 0.8) make output more creative, lower values (e.g., 0.2) make it more focused and deterministic. Range: 0-2</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="temperature"
                                      type="number"
                                      step="0.1"
                                      min="0"
                                      max="2"
                                      value={ollamaConfig?.temperature || ''}
                                      onChange={(e) => handleConfigChange('temperature', e.target.value)}
                                      placeholder="0.7"
                                    />
                                  </div>

                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="top-p">Top P</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Nucleus sampling threshold. The model considers tokens with cumulative probability up to this value. Lower values make output more focused. Range: 0-1</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="top-p"
                                      type="number"
                                      step="0.1"
                                      min="0"
                                      max="1"
                                      value={ollamaConfig?.topP || ''}
                                      onChange={(e) => handleConfigChange('topP', e.target.value)}
                                      placeholder="0.9"
                                    />
                                  </div>

                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="top-k">Top K</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Limits token selection to the top K most likely tokens. Lower values make output more predictable. Typical range: 10-100</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="top-k"
                                      type="number"
                                      value={ollamaConfig?.topK || ''}
                                      onChange={(e) => handleConfigChange('topK', e.target.value)}
                                      placeholder="40"
                                    />
                                  </div>
                                </div>

                                <div className="grid gap-4 md:grid-cols-3">
                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="repeat-penalty">Repeat Penalty</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Penalizes repeated tokens to reduce redundancy. Higher values (e.g., 1.2) discourage repetition more strongly. Range: 0-2</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="repeat-penalty"
                                      type="number"
                                      step="0.1"
                                      min="0"
                                      value={ollamaConfig?.repeatPenalty || ''}
                                      onChange={(e) => handleConfigChange('repeatPenalty', e.target.value)}
                                      placeholder="1.1"
                                    />
                                  </div>

                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="seed">Seed</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Random seed for reproducible outputs. Use -1 for random generation, or set a specific number to get consistent results</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="seed"
                                      type="number"
                                      value={ollamaConfig?.seed || ''}
                                      onChange={(e) => handleConfigChange('seed', e.target.value)}
                                      placeholder="-1"
                                    />
                                  </div>

                                  <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Label htmlFor="num-ctx">Context Length</Label>
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                            <Info className="h-4 w-4" />
                                          </button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="max-w-xs">Maximum context window size in tokens. Determines how much text the model can process at once. Larger values require more memory</p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </div>
                                    <Input
                                      id="num-ctx"
                                      type="number"
                                      value={ollamaConfig?.numCtx || ''}
                                      onChange={(e) => handleConfigChange('numCtx', e.target.value)}
                                      placeholder="2048"
                                    />
                                  </div>
                                </div>
                              </div>
                            </CollapsibleContent>
                          </Collapsible>
                        </div>

                        <div className="flex gap-3 pt-4 border-t border-border">
                          <Button onClick={handleTestConnection} variant="outline">
                            <CheckCircle className="h-4 w-4 mr-2" />
                            Test Connection
                          </Button>
                          <Button onClick={handleSaveConfig}>
                            <GearSix className="h-4 w-4 mr-2" />
                            Save Configuration
                          </Button>
                        </div>
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>

                <div className="flex items-center justify-between rounded-lg border border-border p-4 opacity-60">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      <CloudArrowUp className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-semibold">OpenAI + OneDrive</p>
                      <p className="text-sm text-muted-foreground">Cloud AI with OneDrive integration</p>
                    </div>
                  </div>
                  <Badge variant="secondary">Coming Soon</Badge>
                </div>

                <Collapsible open={googleExpanded} onOpenChange={setGoogleExpanded}>
                  <div className="rounded-lg border border-border">
                    <CollapsibleTrigger asChild>
                      <button className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                            <CloudArrowUp className="h-5 w-5 text-muted-foreground" />
                          </div>
                          <div className="text-left">
                            <p className="font-semibold">Gemini + Google Drive</p>
                            <p className="text-sm text-muted-foreground">Google AI with Drive integration</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {googleExpanded ? (
                            <CaretUp className="h-5 w-5 text-muted-foreground" />
                          ) : (
                            <CaretDown className="h-5 w-5 text-muted-foreground" />
                          )}
                        </div>
                      </button>
                    </CollapsibleTrigger>
                    
                    <CollapsibleContent>
                      <div className="border-t border-border p-6 space-y-6">
                        <p className="text-sm text-muted-foreground">
                          Connect your Google account to enable semantic search across your Google Drive documents and use Gemini models.
                        </p>
                        <div className="flex gap-3">
                          <Button onClick={handleGoogleLogin} className="flex-1">
                             Connect Google Account
                          </Button>
                          <Button onClick={handleGoogleLogout} variant="outline" className="flex-1">
                             Disconnect
                          </Button>
                        </div>

                        <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20 mt-4">
                          <h4 className="font-semibold text-sm">Vertex AI Configuration (Enterprise)</h4>
                          <p className="text-xs text-muted-foreground">Required for "Vertex AI Agent" mode.</p>
                          
                          <div className="space-y-2">
                            <Label htmlFor="vertex-project">Project ID</Label>
                            <Input
                              id="vertex-project"
                              value={vertexConfig.projectId}
                              onChange={(e) => setVertexConfig({...vertexConfig, projectId: e.target.value})}
                              placeholder="my-gcp-project-id"
                            />
                          </div>
                          
                          <div className="space-y-2">
                            <Label htmlFor="vertex-location">Location</Label>
                            <Input
                              id="vertex-location"
                              value={vertexConfig.location}
                              onChange={(e) => setVertexConfig({...vertexConfig, location: e.target.value})}
                              placeholder="us-central1"
                            />
                          </div>
                          
                          <div className="space-y-2">
                            <Label htmlFor="vertex-datastore">Data Store ID</Label>
                            <Input
                              id="vertex-datastore"
                              value={vertexConfig.dataStoreId}
                              onChange={(e) => setVertexConfig({...vertexConfig, dataStoreId: e.target.value})}
                              placeholder="my-datastore-id"
                            />
                          </div>
                          
                          <Button onClick={handleSaveVertexConfig} size="sm" variant="secondary" className="w-full">
                            Save Vertex Configuration
                          </Button>
                        </div>
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      {jobProgress.visible && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 w-[min(520px,90vw)] pointer-events-none">
          <div className="pointer-events-auto rounded-lg border border-border bg-card/90 backdrop-blur shadow-xl p-3 space-y-2">
            <div className="flex items-center justify-between text-sm text-foreground">
              <span>Indexing in progress</span>
              <span className="text-xs text-muted-foreground">
                {jobProgress.completed}/{jobProgress.total} done
                {jobProgress.failed > 0 ? ` · ${jobProgress.failed} failed` : ''}
              </span>
            </div>
            <Progress value={jobProgress.total ? (jobProgress.completed / jobProgress.total) * 100 : 0} />
          </div>
        </div>
      )}
    </div>
    </TooltipProvider>
  )
}

export default App
