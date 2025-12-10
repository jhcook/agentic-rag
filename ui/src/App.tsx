import { useCallback, useEffect, useRef, useState } from 'react'
import { Toaster } from 'sonner'
import { MainLayout } from '@/components/layout/MainLayout'
import { DashboardView } from '@/features/dashboard/DashboardView'
import { SettingsView, OllamaConfig } from '@/features/settings/SettingsView'
import { SearchView } from '@/features/search/SearchView'
import { FileManager } from '@/components/FileManager'
import { LogsViewer } from '@/components/LogsViewer'
import { HelpView } from '@/features/help/HelpView'
import { toast } from 'sonner'
import { Message } from '@/components/ChatInterface'
import { Conversation } from '@/components/ConversationSidebar'

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

  // --- Handlers: Search ---
  const handleSearch = async () => {
    if (searchAbortRef.current) searchAbortRef.current.abort()
    const { host, port, base } = getApiBase()
    if (!queryText.trim()) return

    setSearching(true); setSearchError(null); setSearchAnswer(null); setSearchSources([])
    const controller = new AbortController()
    searchAbortRef.current = controller

    try {
      const kickRes = await fetch(`http://${host}:${port}/${base}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, async: true }),
        signal: controller.signal
      })
      if (!kickRes.ok) throw new Error('Failed to start')
      const { job_id } = await kickRes.json()

      let done = false
      while (!done) {
        const pollRes = await fetch(`http://${host}:${port}/${base}/search/jobs/${job_id}`, { signal: controller.signal })
        const status = await pollRes.json()
        setSearchMessage(status.message)

        if (['completed', 'failed', 'timeout'].includes(status.status)) {
          done = true
          const res = status.result
          if (!res) {
            setSearchError(status.error || 'Unknown error')
          } else if (res.error) {
            setSearchError(res.error)
          } else {
            setSearchAnswer(res.answer || res)
            setSearchSources(res.sources || [])
          }
        } else {
          await new Promise(r => setTimeout(r, 1000))
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setSearchError(e.message)
    } finally {
      setSearching(false)
      searchAbortRef.current = null
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

      toast.success('Configuration saved & activated')
      setTimeout(() => window.location.reload(), 1000)
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
      setTimeout(() => window.location.reload(), 1000)
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
      setTimeout(() => window.location.reload(), 1000)
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
      toast.success(`Disconnected ${provider}. Switched to Ollama mode.`)
      setTimeout(() => window.location.reload(), 1000)
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
    window.location.href = `http://${host}:${port}/${base}/auth/login`
  }

  return (
    <>
      <Toaster theme="dark" position="top-right" />
      <MainLayout
        activeTab={activeTab}
        onTabChange={setActiveTab}
        systemStatus={systemStatus}
        config={ollamaConfig}
        activeMode={activeMode}
        onModeChange={setActiveMode}
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
            conversations={conversations}
            onSelectConversation={setActiveConversationId}
            onNewConversation={() => {
              setActiveConversationId(crypto.randomUUID())
              setChatMessages([])
            }}
            onDeleteConversation={() => { }}
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
          />
        )}

        {activeTab === 'logs' && <LogsViewer config={ollamaConfig} />}
        {activeTab === 'help' && <HelpView />}
      </MainLayout>
    </>
  )
}

export default App
