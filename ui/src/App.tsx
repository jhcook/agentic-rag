import { useCallback, useEffect, useRef, useState } from 'react'
import { Toaster } from '@/components/ui/sonner'
import { MainLayout } from '@/components/layout/MainLayout'
import { DashboardView } from '@/features/dashboard/DashboardView'
import { SettingsView, OllamaConfig, PgvectorConfig } from '@/features/settings/SettingsView'
import { SearchView } from '@/features/search/SearchView'
import { FileManager } from '@/components/FileManager'
import { LogsViewer } from '@/components/LogsViewer'
import { MetricsView } from '@/features/dashboard/MetricsView'
import { HelpView } from '@/features/help/HelpView'
import { toast } from 'sonner'
import { Message } from '@/components/ChatInterface'
import { Conversation } from '@/components/ConversationSidebar'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

const MASKED_SECRET = '***MASKED***'

const createDefaultOllamaConfig = (): OllamaConfig => ({
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
  ragPath: import.meta.env.VITE_RAG_PATH || 'api',
  debugMode: false,
  ollamaCloudApiKey: '',
  ollamaCloudEndpoint: '',
  ollamaCloudProxy: '',
  ollamaCloudCABundle: '',
  ollamaMode: 'local',
  availableModels: [],
})

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState<'running' | 'stopped' | 'error' | 'warning'>('stopped')
  const [serviceStatuses, setServiceStatuses] = useState<Record<string, 'running' | 'stopped' | 'error' | 'warning'>>({})
  const [queriesToday, setQueriesToday] = useState(0)

  // Data State
  const [ollamaConfig, setOllamaConfig] = useState<OllamaConfig>(createDefaultOllamaConfig())
  const [backendDocs, setBackendDocs] = useState<number | null>(null)
  const [backendSize, setBackendSize] = useState<number | null>(null)
  const [activeMode, setActiveMode] = useState<string>('none')

  // Search State
  const [queryText, setQueryText] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchAnswer, setSearchAnswer] = useState<string | null>(null)
  const [searchSources, setSearchSources] = useState<string[]>([])
  const [searchMessage, setSearchMessage] = useState<string | null>(null)
  const searchAbortRef = useRef<AbortController | null>(null)

  // Settings State
  const [vertexConfig, setVertexConfig] = useState({ projectId: '', location: '', dataStoreId: '' })
  const [openaiConfig, setOpenaiConfig] = useState({ apiKey: '', model: 'gpt-4-turbo-preview', assistantId: '' })
  const [hasOpenaiApiKey, setHasOpenaiApiKey] = useState(false)
  const [openaiModels, setOpenaiModels] = useState<string[]>([])
  const [hasOllamaCloudApiKey, setHasOllamaCloudApiKey] = useState(false)
  const [ollamaStatus, setOllamaStatus] = useState<any>(null)

  const [pgvectorConfig, setPgvectorConfig] = useState<PgvectorConfig>({
    host: '127.0.0.1',
    port: 5432,
    dbname: 'agentic_rag',
    user: 'agenticrag',
    password: '',
    hasPassword: false,
  })
  const [pgvectorStats, setPgvectorStats] = useState<{ status: string; documents?: number; chunks?: number; embedding_dim?: number; error?: string } | null>(null)

  // Chat State
  const [chatMessages, setChatMessages] = useState<Message[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Job Progress
  const [jobProgress, setJobProgress] = useState({ total: 0, completed: 0, failed: 0, visible: false })
  const [backendDocumentList, setBackendDocumentList] = useState<any[]>([])


  // --- Helper: Get API Base ---
  const getApiBase = useCallback(() => {
    const host = ollamaConfig?.ragHost || '127.0.0.1'
    const port = ollamaConfig?.ragPort || '8001'
    const base = (ollamaConfig?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    return { host, port, base }
  }, [ollamaConfig])

  // --- Effect: Load App Config ---
  useEffect(() => {
    const loadConfig = async () => {
      const { host, port, base } = getApiBase()
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/app`)
        if (res.ok) {
          const data = await res.json()
          setOllamaConfig(prev => ({ ...prev, ...data }))
        }

        const modeRes = await fetch(`http://${host}:${port}/${base}/config/mode`)
        if (modeRes.ok) setActiveMode((await modeRes.json()).mode)

        // Load cloud config
        const cloudRes = await fetch(`http://${host}:${port}/${base}/ollama/cloud-config`)
        if (cloudRes.ok) {
          const d = await cloudRes.json()
          const hasKey = Boolean(d.has_api_key ?? d.api_key)
          setHasOllamaCloudApiKey(hasKey)
          setOllamaConfig(prev => ({
            ...prev,
            ollamaCloudApiKey: hasKey ? MASKED_SECRET : (d.api_key ?? ''),
            ollamaCloudEndpoint: d.endpoint ?? '',
            ollamaCloudCABundle: d.ca_bundle ?? '',
            ollamaCloudProxy: d.proxy ?? '',
          }))
        }

        // Load OpenAI config
        const openaiRes = await fetch(`http://${host}:${port}/${base}/config/openai`)
        if (openaiRes.ok) {
          const d = await openaiRes.json()
          setOpenaiConfig(prev => ({
            ...prev,
            apiKey: d.api_key || '',
            model: d.model || 'gpt-4-turbo-preview',
            assistantId: d.assistant_id || ''
          }))
          setHasOpenaiApiKey(d.has_api_key)

          // If we have a key, try to load models silently
          if (d.has_api_key) {
            fetch(`http://${host}:${port}/${base}/config/openai/models`).then(async r => {
              if (r.ok) {
                const data = await r.json()
                if (data.models) setOpenaiModels(data.models.map((m: any) => m.id))
              }
            }).catch(() => { })
          }
        }

        // Load pgvector config
        const pgRes = await fetch(`http://${host}:${port}/${base}/config/pgvector`)
        if (pgRes.ok) {
          const d = await pgRes.json()
          setPgvectorConfig(prev => ({
            ...prev,
            host: d.host ?? prev.host,
            port: Number(d.port ?? prev.port),
            dbname: d.dbname ?? prev.dbname,
            user: d.user ?? prev.user,
            password: d.password ?? '',
            hasPassword: Boolean(d.has_password),
          }))
        }

        // Load pgvector stats (best-effort)
        fetch(`http://${host}:${port}/${base}/pgvector/stats`).then(async r => {
          if (r.ok) {
            const s = await r.json()
            setPgvectorStats(s)
          }
        }).catch(() => { })

        // Load chat history
        const historyRes = await fetch(`http://${host}:${port}/${base}/chat/history?limit=50`)
        if (historyRes.ok) {
          const sessions = await historyRes.json()
          const conversations: Conversation[] = sessions.map((s: any) => ({
            id: s.id,
            title: s.title || 'New Conversation',
            messages: [],
            createdAt: s.created_at * 1000,
            updatedAt: s.updated_at * 1000
          }))
          setConversations(conversations)
        }
      } catch (e) { console.error('Config load failed', e) }
    }
    loadConfig()
  }, []) // Init

  // --- Effect: Service Status Poll ---
  useEffect(() => {
    const checkServices = async () => {
      const { host, port, base } = getApiBase()
      try {
        const res = await fetch(`http://${host}:${port}/${base}/services`)
        if (res.ok) {
          const data = await res.json()
          const svcs: any[] = data.services || []
          const map: any = {}
          svcs.forEach(s => map[s.service] = s.status || 'stopped')
          setServiceStatuses(map)

          if (svcs.every(s => s.status === 'running')) setSystemStatus('running')
          else if (svcs.every(s => s.status === 'stopped')) setSystemStatus('stopped')
          else if (svcs.some(s => s.status === 'error')) setSystemStatus('error')
          else setSystemStatus('warning')
        }
      } catch { setSystemStatus('error') }
    }
    const id = setInterval(checkServices, 5000)
    checkServices()
    return () => clearInterval(id)
  }, [getApiBase])

  // --- Effect: Jobs Poll ---
  useEffect(() => {
    const { host, port, base } = getApiBase()
    const pollJobs = async () => {
      try {
        const res = await fetch(`http://${host}:${port}/${base}/jobs`)
        if (!res.ok) return
        const data = await res.json()
        const jobs = data.jobs || []
        const total = jobs.length
        const completed = jobs.filter((j: any) => j.status === 'completed').length
        const failed = jobs.filter((j: any) => j.status === 'failed').length
        const visible = total > 0 && (completed + failed) < total
        setJobProgress({ total, completed, failed, visible })
      } catch { }
    }
    const id = setInterval(pollJobs, 2500)
    return () => clearInterval(id)
  }, [getApiBase])

  // --- Effect: Stats Poll ---
  useEffect(() => {
    const { host, port, base } = getApiBase()
    const checkStats = async () => {
      try {
        const res = await fetch(`http://${host}:${port}/${base}/health`)
        if (res.ok) {
          const data = await res.json()
          setBackendDocs(data.documents)
          setBackendSize(data.total_size_bytes)
        }

        // Also fetch today's metrics
        const todayRes = await fetch(`http://${host}:${port}/${base}/metrics/today`)
        if (todayRes.ok) {
          const tData = await todayRes.json()
          setQueriesToday(tData.queries_today || 0)
        }
      } catch { }
    }
    const id = setInterval(checkStats, 10000)
    checkStats()
    return () => clearInterval(id)
  }, [getApiBase])

  // --- Handlers: Chat ---
  const handleSelectConversation = async (conversationId: string) => {
    setActiveConversationId(conversationId)
    const { host, port, base } = getApiBase()
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/chat/history/${conversationId}`)
      if (res.ok) {
        const messages = await res.json()
        setChatMessages(messages.map((m: any) => ({
          id: m.id || undefined,
          role: m.role,
          content: m.content,
          displayContent: m.display_content || undefined,
          sources: Array.isArray(m.sources) ? m.sources : undefined,
          timestamp: typeof m.created_at === 'number' ? Math.floor(m.created_at * 1000) : Date.now()
        })))
      }
    } catch (e) {
      console.error('Failed to load conversation messages', e)
      toast.error('Failed to load conversation')
    }
  }

  const refreshConversationList = async () => {
    const { host, port, base } = getApiBase()
    try {
      const historyRes = await fetch(`http://${host}:${port}/${base}/chat/history?limit=50`)
      if (!historyRes.ok) return
      const sessions = await historyRes.json()
      const conversations: Conversation[] = sessions.map((s: any) => ({
        id: s.id,
        title: s.title || 'New Conversation',
        messages: [],
        createdAt: s.created_at * 1000,
        updatedAt: s.updated_at * 1000
      }))
      setConversations(conversations)
    } catch {
      // best-effort
    }
  }

  const handleChatSessionId = (sessionId: string) => {
    setActiveConversationId(prev => prev ?? sessionId)
    // Best-effort refresh so the sidebar shows the session immediately
    refreshConversationList()
  }


  const handleSavePgvectorConfig = async () => {
    const { host, port, base } = getApiBase()

    const passwordPayload: string | null = (() => {
      if (pgvectorConfig.password && pgvectorConfig.password !== MASKED_SECRET) return pgvectorConfig.password
      if (pgvectorConfig.password === MASKED_SECRET) return MASKED_SECRET
      if (pgvectorConfig.hasPassword) return MASKED_SECRET
      return null
    })()

    const res = await fetch(`http://${host}:${port}/${base}/config/pgvector`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        host: pgvectorConfig.host,
        port: pgvectorConfig.port,
        dbname: pgvectorConfig.dbname,
        user: pgvectorConfig.user,
        password: passwordPayload,
      }),
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || 'Failed to save pgvector config')
    }

    // Reload config so password returns masked
    const cfgRes = await fetch(`http://${host}:${port}/${base}/config/pgvector`)
    if (cfgRes.ok) {
      const d = await cfgRes.json()
      setPgvectorConfig(prev => ({
        ...prev,
        host: d.host ?? prev.host,
        port: Number(d.port ?? prev.port),
        dbname: d.dbname ?? prev.dbname,
        user: d.user ?? prev.user,
        password: d.password ?? '',
        hasPassword: Boolean(d.has_password),
      }))
    }
  }

  const testPgvector = async () => {
    const { host, port, base } = getApiBase()
    const res = await fetch(`http://${host}:${port}/${base}/pgvector/test-connection`, { method: 'POST' })
    const data = await res.json()
    return { success: Boolean(data.success), message: String(data.message ?? '') }
  }

  const migratePgvector = async () => {
    const { host, port, base } = getApiBase()
    const res = await fetch(`http://${host}:${port}/${base}/pgvector/migrate`, { method: 'POST' })
    return await res.json()
  }

  const backfillPgvector = async () => {
    const { host, port, base } = getApiBase()
    const res = await fetch(`http://${host}:${port}/${base}/pgvector/backfill`, { method: 'POST' })
    return await res.json()
  }

  const refreshPgvectorStats = async () => {
    const { host, port, base } = getApiBase()
    const res = await fetch(`http://${host}:${port}/${base}/pgvector/stats`)
    const data = await res.json()
    setPgvectorStats(data)
    if (!res.ok || data.status === 'error') {
      throw new Error(data.error || 'Failed to fetch stats')
    }
  }
  const handleDeleteConversation = async (conversationId: string) => {
    const { host, port, base } = getApiBase()
    
    try {
      const res = await fetch(`http://${host}:${port}/${base}/chat/history/${conversationId}`, {
        method: 'DELETE'
      })
      
      if (res.ok) {
        // Find the next conversation to select
        const sortedConvos = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)
        const currentIndex = sortedConvos.findIndex(c => c.id === conversationId)
        const nextConversation = sortedConvos[currentIndex + 1] || sortedConvos[currentIndex - 1]
        
        // Remove from local state
        setConversations(prev => prev.filter(c => c.id !== conversationId))
        
        // If this was the active conversation, select the next one
        if (activeConversationId === conversationId) {
          if (nextConversation) {
            handleSelectConversation(nextConversation.id)
          } else {
            setActiveConversationId(null)
            setChatMessages([])
          }
        }
        
        toast.success('Conversation deleted')
      } else {
        throw new Error('Failed to delete conversation')
      }
    } catch (e) {
      console.error('Failed to delete conversation', e)
      toast.error('Failed to delete conversation')
    }
  }

  // --- Handlers: Search ---
  const handleSearch = async () => {
    if (searchAbortRef.current) searchAbortRef.current.abort()
    const { host, port, base } = getApiBase()
    if (!queryText.trim()) return

    setSearching(true); setSearchError(null); setSearchAnswer(null); setSearchSources([]); setSearchMessage('')
    const controller = new AbortController()
    searchAbortRef.current = controller

    try {
      setSearchMessage('Generating grounded answer...')

      const res = await fetch(`http://${host}:${port}/${base}/grounded_answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: queryText,
          session_id: activeConversationId || undefined,
        }),
        signal: controller.signal
      })

      if (!res.ok) {
        let errorMsg = 'Failed to generate grounded answer'
        try {
          const err = await res.json()
          errorMsg = err.detail || err.error || errorMsg
        } catch { }
        throw new Error(errorMsg)
      }

      const data = await res.json()

      if (data.error) {
        setSearchError(String(data.error))
        return
      }

      // Accept multiple payload shapes.
      const answerText: string = String(data.grounded_answer || data.answer || data.content || '')
      const sources: string[] = Array.isArray(data.sources)
        ? data.sources
        : (Array.isArray(data.citations) ? data.citations : [])

      setSearchAnswer(answerText)
      setSearchSources(sources)

      // If grounded_answer created a new persisted session (no active chat), refresh sidebar.
      if (data.session_id && !activeConversationId) {
        refreshConversationList()
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setSearchError(e.message)
    } finally {
      setSearching(false)
      searchAbortRef.current = null
      setSearchMessage('')
    }
  }



  // --- Handlers: Configuration ---
  const handleSaveConfig = async () => {
    const { host, port, base } = getApiBase()
    try {
      const { ollamaCloudApiKey, ...safe } = ollamaConfig
      await fetch(`http://${host}:${port}/${base}/config/app`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(safe)
      })

      // If cloud key provided, save it separately
      if (ollamaCloudApiKey && ollamaCloudApiKey !== MASKED_SECRET) {
        await fetch(`http://${host}:${port}/${base}/ollama/cloud-config`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            apiKey: ollamaCloudApiKey,
            endpoint: ollamaConfig.ollamaCloudEndpoint
          })
        })
      }

      // Switch mode to ollama implicitly
      await fetch(`http://${host}:${port}/${base}/config/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'ollama' })
      })
      setActiveMode('ollama')

      toast.success('Configuration saved & activated')
    } catch { toast.error('Failed to save') }
  }

  const handleSaveOpenaiConfig = async () => {
    const { host, port, base } = getApiBase()
    try {
      await fetch(`http://${host}:${port}/${base}/config/openai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(openaiConfig)
      })
      toast.success('OpenAI settings saved')
      setHasOpenaiApiKey(true)
      // Switch mode to openai automatically
      await fetch(`http://${host}:${port}/${base}/config/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'openai_assistants' })
      })
      setActiveMode('openai_assistants')
    } catch { toast.error('Failed to save OpenAI settings') }
  }

  const handleSaveVertexConfig = async () => {
    const { host, port, base } = getApiBase()
    try {
      await fetch(`http://${host}:${port}/${base}/config/vertex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vertexConfig)
      })
      toast.success('Vertex AI settings saved')
      // Switch mode
      await fetch(`http://${host}:${port}/${base}/config/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'vertex_ai_search' })
      })
      setActiveMode('vertex_ai_search')
    } catch { toast.error('Failed to save Vertex settings') }
  }

  const handleDisconnect = async (provider?: string) => {
    const { host, port, base } = getApiBase()
    if (!provider) return

    // Default to local mode
    try {
      if (provider === 'openai_assistants') {
        setOpenaiConfig(p => ({ ...p, apiKey: '' }))
        setHasOpenaiApiKey(false)
      }

      await fetch(`http://${host}:${port}/${base}/config/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'ollama' })
      })
      setActiveMode('ollama')
      toast.success(`Disconnected ${provider}. Switched to Ollama mode.`)
    } catch { toast.error('Failed to disconnect') }
  }

  const testOpenAI = async () => {
    const { host, port, base } = getApiBase()
    try {
      const res = await fetch(`http://${host}:${port}/${base}/config/openai/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey: openaiConfig.apiKey })
      })

      if (res.ok) {
        const data = await res.json()
        if (data.models && data.models.length > 0) {
          setOpenaiModels(data.models.map((m: any) => m.id))
          toast.success(`Success! Found ${data.models.length} models.`)
        } else if (data.warning) {
          toast.warning(data.warning, { description: data.message })
        } else {
          toast.error('No models found')
        }
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Connection failed')
      }
    } catch (e) {
      toast.error('Connection test failed')
    }
  }

  const testOllamaCloud = async (key: string, endpoint?: string) => {
    const { host, port, base } = getApiBase()
    try {
      const res = await fetch(`http://${host}:${port}/${base}/ollama/test-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key, endpoint })
      })
      const data = await res.json()
      return { success: data.success, message: data.message || (data.success ? 'Connection successful' : 'Failed') }
    } catch (e: any) {
      return { success: false, message: e.message }
    }
  }

  const handleFetchOllamaModels = async (key: string, endpoint?: string) => {
    const { host, port, base } = getApiBase()
    try {
      const res = await fetch(`http://${host}:${port}/${base}/ollama/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key, endpoint })
      })
      if (res.ok) {
        const data = await res.json()
        const models = data.models || []
        setOllamaConfig(prev => ({ ...prev, availableModels: models }))
        return models
      }
      return []
    } catch { return [] }
  }

  const handleGoogleLogin = () => {
    // Redirect to login endpoint
    const { host, port, base } = getApiBase()
    const url = `http://${host}:${port}/${base}/auth/login`
    
    // Open in a popup window
    const width = 600
    const height = 700
    const left = window.screen.width / 2 - width / 2
    const top = window.screen.height / 2 - height / 2
    
    window.open(
      url,
      'GoogleLogin',
      `width=${width},height=${height},top=${top},left=${left},resizable,scrollbars,status`
    )
  }

  // --- Effect: Listen for Google Auth Success ---
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data && event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        toast.success('Google Account Connected Successfully')
        // Refresh config/status if needed
        // For now, just showing the toast is good feedback
      }
    }
    
    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [])

  return (
    <>
      <Toaster />
      <MainLayout
        activeTab={activeTab}
        onTabChange={setActiveTab}
        systemStatus={systemStatus}
        config={ollamaConfig}
        activeMode={activeMode}
        onModeChange={setActiveMode}
        jobProgress={jobProgress}
      >
        {activeTab === 'dashboard' && (
          <DashboardView
            systemStatus={systemStatus}
            serviceStatuses={serviceStatuses}
            stats={{
              documents: backendDocs || 0,
              queriesToday: queriesToday,
              avgLatency: 350 // We could fetch this too if available in /metrics/today
            }}
            jobProgress={jobProgress}
            onNavigate={setActiveTab}
            config={ollamaConfig}
          />
        )}

        {activeTab === 'search' && (
          <SearchView
            queryText={queryText}
            onQueryTextChange={setQueryText}
            onSearch={handleSearch}
            onCancelSearch={() => searchAbortRef.current?.abort()}
            searching={searching}
            searchError={searchError}
            searchAnswer={searchAnswer}
            searchSources={searchSources}
            searchMessage={searchMessage}

            config={ollamaConfig}
            chatMessages={chatMessages}
            setMessages={setChatMessages}
            activeConversationId={activeConversationId}
            onSessionId={handleChatSessionId}
            conversations={conversations}
            onSelectConversation={handleSelectConversation}
            onNewConversation={() => {
              const newId = (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
                ? (crypto as any).randomUUID()
                : `${Date.now()}-${Math.random().toString(16).slice(2)}`
              setActiveConversationId(newId)
              setChatMessages([])
            }}
            onDeleteConversation={handleDeleteConversation}
            isSidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          />
        )}

        {activeTab === 'files' && (
          <FileManager
            config={ollamaConfig}
            activeMode={activeMode}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsView
            config={ollamaConfig}
            onConfigChange={(k, v) => setOllamaConfig(p => ({ ...p, [k]: v }))}
            onSaveConfig={handleSaveConfig}
            onGoogleLogin={handleGoogleLogin}
            onDisconnect={handleDisconnect}
            vertexConfig={vertexConfig}
            onVertexConfigChange={setVertexConfig}
            onSaveVertexConfig={handleSaveVertexConfig}
            openaiConfig={openaiConfig}
            onOpenaiConfigChange={setOpenaiConfig}
            onSaveOpenaiConfig={handleSaveOpenaiConfig}
            hasOpenaiApiKey={hasOpenaiApiKey}
            openaiModels={openaiModels}
            onTestOpenAI={testOpenAI}
            onTestOllamaCloud={testOllamaCloud}
            onFetchOllamaModels={handleFetchOllamaModels}
            ollamaStatus={ollamaStatus}

            pgvectorConfig={pgvectorConfig}
            onPgvectorConfigChange={setPgvectorConfig}
            onSavePgvectorConfig={handleSavePgvectorConfig}
            onTestPgvector={testPgvector}
            onMigratePgvector={migratePgvector}
            onBackfillPgvector={backfillPgvector}
            pgvectorStats={pgvectorStats}
            onRefreshPgvectorStats={refreshPgvectorStats}
          />
        )}

        {activeTab === 'logs' && (
          <Tabs defaultValue="logs" className="h-full space-y-6">
            <div className="space-between flex items-center">
              <TabsList>
                <TabsTrigger value="logs">Logs</TabsTrigger>
                <TabsTrigger value="metrics">Metrics</TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="logs">
              <LogsViewer config={ollamaConfig} />
            </TabsContent>
            <TabsContent value="metrics">
              <MetricsView config={ollamaConfig} />
            </TabsContent>
          </Tabs>
        )}
        {activeTab === 'help' && <HelpView />}
      </MainLayout>
    </>
  )
}

export default App
