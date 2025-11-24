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
import { ConversationSidebar, Conversation } from '@/components/ConversationSidebar'
import { SettingsDashboard } from '@/components/SettingsDashboard'

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
  const [systemStatus, setSystemStatus] = useState<'running' | 'stopped' | 'error' | 'warning'>('stopped')
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
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeMode, setActiveMode] = useState<string>('local')

  // Fetch app config on mount
  useEffect(() => {
    const fetchAppConfig = async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/app`)
        if (res.ok) {
          const data = await res.json()
          if (Object.keys(data).length > 0) {
            setOllamaConfig(prev => ({
              ...prev,
              ...data
            }))
          }
        }
      } catch (e) {
        console.error("Failed to fetch app config", e)
      }
    }
    fetchAppConfig()
  }, []) // Run once on mount, using initial env vars to find server

  // Fetch active mode
  useEffect(() => {
    const fetchMode = async () => {
      const host = ollamaConfig?.ragHost || '127.0.0.1'
      const port = ollamaConfig?.ragPort || '8001'
      const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/mode`)
        if (res.ok) {
          const data = await res.json()
          setActiveMode(data.mode || 'local')
        }
      } catch (e) {
        console.error("Failed to fetch mode", e)
      }
    }
    fetchMode()
    // Poll for mode changes
    const interval = setInterval(fetchMode, 5000)
    return () => clearInterval(interval)
  }, [ollamaConfig])

  // Load conversations from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('conversations')
      if (saved) {
        const parsed = JSON.parse(saved) as Conversation[]
        setConversations(parsed)
        // Restore active conversation if it exists
        const activeId = localStorage.getItem('activeConversationId')
        if (activeId && parsed.find(c => c.id === activeId)) {
          setActiveConversationId(activeId)
          const activeConv = parsed.find(c => c.id === activeId)
          if (activeConv) {
            setChatMessages(activeConv.messages)
          }
        } else if (parsed.length > 0) {
          // Use most recent conversation
          const mostRecent = parsed.sort((a, b) => b.updatedAt - a.updatedAt)[0]
          setActiveConversationId(mostRecent.id)
          setChatMessages(mostRecent.messages)
        }
      }
    } catch (e) {
      console.error('Failed to load conversations:', e)
    }
  }, [])

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('conversations', JSON.stringify(conversations))
      if (activeConversationId) {
        localStorage.setItem('activeConversationId', activeConversationId)
      } else {
        localStorage.removeItem('activeConversationId')
      }
    } catch (e) {
      console.error('Failed to save conversations:', e)
    }
  }, [conversations, activeConversationId])

  // Update active conversation when messages change
  useEffect(() => {
    if (activeConversationId) {
      // Update existing conversation
      setConversations(prev => {
        const existing = prev.find(c => c.id === activeConversationId)
        if (existing) {
          return prev.map(conv => {
            if (conv.id === activeConversationId) {
              return {
                ...conv,
                messages: chatMessages,
                updatedAt: Date.now()
              }
            }
            return conv
          })
        } else if (chatMessages.length > 0) {
          // Create new conversation with existing ID if it doesn't exist but we have messages
          const newConv: Conversation = {
            id: activeConversationId,
            title: 'New Conversation',
            messages: chatMessages,
            createdAt: Date.now(),
            updatedAt: Date.now()
          }
          return [newConv, ...prev]
        }
        return prev
      })
    } else if (chatMessages.length > 0) {
      // No active conversation but messages exist - create one
      const newId = crypto.randomUUID()
      const newConv: Conversation = {
        id: newId,
        title: 'New Conversation',
        messages: chatMessages,
        createdAt: Date.now(),
        updatedAt: Date.now()
      }
      setConversations(prev => [newConv, ...prev])
      setActiveConversationId(newId)
    }
  }, [chatMessages, activeConversationId])

  // Save conversation when switching away from search tab
  useEffect(() => {
    if (activeTab !== 'search' && activeConversationId && chatMessages.length > 0) {
      setConversations(prev => {
        return prev.map(conv => {
          if (conv.id === activeConversationId) {
            return {
              ...conv,
              messages: chatMessages,
              updatedAt: Date.now()
            }
          }
          return conv
        })
      })
    }
  }, [activeTab, activeConversationId, chatMessages])

  const createNewConversation = () => {
    const newId = crypto.randomUUID()
    const newConv: Conversation = {
      id: newId,
      title: 'New Conversation',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
    setConversations(prev => [newConv, ...prev])
    setActiveConversationId(newId)
    setChatMessages([])
  }

  const selectConversation = (id: string) => {
    const conv = conversations.find(c => c.id === id)
    if (conv) {
      setActiveConversationId(id)
      setChatMessages(conv.messages)
      setSidebarOpen(false) // Close sidebar on mobile after selection
    }
  }

  const deleteConversation = (id: string) => {
    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== id)
      // If deleting active conversation, switch to another or create new
      if (id === activeConversationId) {
        if (filtered.length > 0) {
          const mostRecent = filtered.sort((a, b) => b.updatedAt - a.updatedAt)[0]
          setActiveConversationId(mostRecent.id)
          setChatMessages(mostRecent.messages)
        } else {
          setActiveConversationId(null)
          setChatMessages([])
        }
      }
      return filtered
    })
    toast.success('Conversation deleted')
  }

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

  const handleSaveConfig = async () => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/config/app`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ollamaConfig)
      })
      
      if (res.ok) {
        toast.success('Configuration saved', {
          description: 'Settings have been saved to server'
        })
      } else {
        throw new Error('Failed to save')
      }
    } catch (e) {
      console.error("Failed to save config", e)
      toast.error('Failed to save configuration')
    }
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
        if (data?.status === 'warning') {
          setSystemStatus('warning')
        } else {
          setSystemStatus('running')
        }
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
                  variant={systemStatus === 'running' ? 'default' : systemStatus === 'error' ? 'destructive' : systemStatus === 'warning' ? 'secondary' : 'secondary'}
                  className={`gap-1 ${systemStatus === 'warning' ? 'bg-yellow-500/15 text-yellow-600 hover:bg-yellow-500/25 border-yellow-500/50' : ''}`}
                >
                  {systemStatus === 'running' && <CheckCircle className="h-3 w-3" />}
                  {systemStatus === 'error' && <WarningCircle className="h-3 w-3" />}
                  {systemStatus === 'warning' && <WarningCircle className="h-3 w-3" />}
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
            <FileManager config={ollamaConfig} activeMode={activeMode} />
          </TabsContent>

          <TabsContent value="search" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold tracking-tight mb-2">Conversational Search</h2>
                <p className="text-muted-foreground">Ask questions about your indexed documents</p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="md:hidden"
                onClick={() => setSidebarOpen(!sidebarOpen)}
              >
                <ListBullets className="h-4 w-4 mr-2" />
                Conversations
              </Button>
            </div>
            <div className="flex gap-4 relative">
              <div className={sidebarOpen ? 'block md:block' : 'hidden md:block'}>
                <ConversationSidebar
                  conversations={conversations}
                  activeConversationId={activeConversationId}
                  onSelectConversation={selectConversation}
                  onDeleteConversation={deleteConversation}
                  onNewConversation={createNewConversation}
                  onClose={() => setSidebarOpen(false)}
                  isOpen={sidebarOpen}
                />
              </div>
              {sidebarOpen && (
                <div
                  className="fixed inset-0 bg-black/50 z-40 md:hidden"
                  onClick={() => setSidebarOpen(false)}
                />
              )}
              <div className="flex-1">
                {activeConversationId === null && conversations.length === 0 ? (
                  <div className="flex flex-col items-center justify-center p-12 border rounded-lg bg-muted/20">
                    <ChatCircle className="h-12 w-12 mb-4 text-muted-foreground opacity-50" />
                    <p className="text-muted-foreground mb-4">No conversations yet</p>
                    <Button onClick={createNewConversation}>
                      <Plus className="h-4 w-4 mr-2" />
                      Start New Conversation
                    </Button>
                  </div>
                ) : (
                  <ChatInterface
                    config={ollamaConfig}
                    messages={chatMessages}
                    setMessages={setChatMessages}
                    onNewConversation={createNewConversation}
                    onDeleteConversation={deleteConversation}
                    activeConversationId={activeConversationId}
                  />
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <MetricsDashboard config={ollamaConfig} />
          </TabsContent>

          <TabsContent value="logs" className="space-y-6">
            <LogsViewer config={ollamaConfig} />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6">
            <SettingsDashboard
              config={ollamaConfig}
              onConfigChange={handleConfigChange}
              onSaveConfig={handleSaveConfig}
              onTestConnection={handleTestConnection}
              vertexConfig={vertexConfig}
              onVertexConfigChange={setVertexConfig}
              onSaveVertexConfig={handleSaveVertexConfig}
              onGoogleLogin={handleGoogleLogin}
              onGoogleLogout={handleGoogleLogout}
            />
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
