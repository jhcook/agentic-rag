import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { 
  Zap as Lightning, 
  Database, 
  CloudUpload as CloudArrowUp,
  Play,
  Pause,
  Download as DownloadSimple,
  Trash,
  Filter as FunnelSimple,
  Terminal,
  CheckCircle,
  XCircle,
  TriangleAlert as Warning,
  Info as InfoIcon,
  Search as MagnifyingGlass
} from 'lucide-react'
import { toast } from 'sonner'

type LogLevel = 'info' | 'warn' | 'error' | 'debug'

type LogEntry = {
  id: string
  timestamp: Date
  level: LogLevel
  source: string
  message: string
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

type LogStreamState = {
  isStreaming: boolean
  isPaused: boolean
}

const LOG_LEVEL_STYLES = {
  info: 'text-foreground',
  warn: 'text-accent',
  error: 'text-destructive',
  debug: 'text-muted-foreground'
}

const LOG_LEVEL_ICONS = {
  info: InfoIcon,
  warn: Warning,
  error: XCircle,
  debug: CheckCircle
}

const generateMockLog = (source: string): LogEntry => {
  const levels: LogLevel[] = ['info', 'warn', 'error', 'debug']
  const messages: Record<string, string[]> = {
    ollama: [
      'Model loaded successfully: llama3.2',
      'Processing inference request',
      'GPU memory allocated: 2.4 GB',
      'Generating embeddings for document chunk',
      'Request completed in 142ms',
      'Temperature set to 0.7',
      'Context window: 2048 tokens',
      'Model warming up...',
      'Cache hit: serving from memory',
      'API endpoint health check passed'
    ],
    mcp: [
      'MCP server started on port 8000',
      'Received context protocol request',
      'Processing streaming transport',
      'Context window updated',
      'Connected to Ollama backend',
      'Request validation completed',
      'Session established with client',
      'Buffering context data',
      'Health check: OK',
      'Memory usage: 256 MB / 1 GB'
    ],
    rest: [
      'REST API server listening on port 8001',
      'Query endpoint received request',
      'Vector search completed: 50ms',
      'Document chunking in progress',
      'Index updated with 24 new documents',
      'FAISS index loaded successfully',
      'Search returned 5 relevant results',
      'Embedding generation: 89ms',
      'Cache miss: generating new embedding',
      'Memory usage: 512 MB / 2 GB'
    ],
    openai: [
      'OpenAI API client initialized',
      'GPT-4 model selected',
      'Processing chat completion request',
      'Token count: 450 input, 280 output',
      'OneDrive integration active',
      'File sync completed: 12 documents',
      'API rate limit: 90% remaining',
      'Response streaming initiated',
      'Connection established',
      'Billing check passed'
    ],
    gemini: [
      'Gemini API client initialized',
      'Using gemini-pro model',
      'Processing multi-turn conversation',
      'Google Drive sync in progress',
      'Document retrieved from Drive',
      'Safety settings applied',
      'Token usage: 380 tokens',
      'Response generation complete',
      'Cache updated with new data',
      'API quota: 75% remaining'
    ]
  }

  const level = levels[Math.floor(Math.random() * levels.length)]
  const messageList = messages[source] || messages.ollama
  const message = messageList[Math.floor(Math.random() * messageList.length)]

  return {
    id: crypto.randomUUID(),
    timestamp: new Date(),
    level,
    source,
    message
  }
}

export function LogsViewer({ config }: { config: OllamaConfig }) {
  const ollamaConfig = config

  const [ollamaLogs, setOllamaLogs] = useState<LogEntry[]>([])
  const [mcpLogs, setMcpLogs] = useState<LogEntry[]>([])
  const [restLogs, setRestLogs] = useState<LogEntry[]>([])
  
  const [ollamaState, setOllamaState] = useState<LogStreamState>({ isStreaming: false, isPaused: false })
  const [mcpState, setMcpState] = useState<LogStreamState>({ isStreaming: false, isPaused: false })
  const [restState, setRestState] = useState<LogStreamState>({ isStreaming: false, isPaused: false })
  
  const [filterText, setFilterText] = useState('')
  const [selectedLevel, setSelectedLevel] = useState<LogLevel | 'all'>('all')
  
  const ollamaScrollRef = useRef<HTMLDivElement>(null)
  const mcpScrollRef = useRef<HTMLDivElement>(null)
  const restScrollRef = useRef<HTMLDivElement>(null)
  
  const intervalRefs = useRef<{
    ollama?: ReturnType<typeof setInterval>
    mcp?: ReturnType<typeof setInterval>
    rest?: ReturnType<typeof setInterval>
  }>({})

  useEffect(() => {
    return () => {
      Object.values(intervalRefs.current).forEach(interval => {
        if (interval) clearInterval(interval)
      })
    }
  }, [])

  const startStreaming = (source: 'ollama' | 'mcp' | 'rest') => {
    if (intervalRefs.current[source]) return

    const setLogs = source === 'ollama' ? setOllamaLogs : source === 'mcp' ? setMcpLogs : setRestLogs
    const setState = source === 'ollama' ? setOllamaState : source === 'mcp' ? setMcpState : setRestState

    setState(s => ({ ...s, isStreaming: true }))

    intervalRefs.current[source] = setInterval(() => {
      const newLog = generateMockLog(source)
      setLogs(prev => [...prev, newLog].slice(-200))
    }, Math.random() * 2000 + 1000)

    toast.success(`${source.toUpperCase()} log streaming started`)
  }

  const stopStreaming = (source: 'ollama' | 'mcp' | 'rest') => {
    if (intervalRefs.current[source]) {
      clearInterval(intervalRefs.current[source])
      intervalRefs.current[source] = undefined
    }

    const setState = source === 'ollama' ? setOllamaState : source === 'mcp' ? setMcpState : setRestState
    setState(s => ({ ...s, isStreaming: false, isPaused: false }))

    toast.info(`${source.toUpperCase()} log streaming stopped`)
  }

  const pauseStreaming = (source: 'ollama' | 'mcp' | 'rest') => {
    const state = source === 'ollama' ? ollamaState : source === 'mcp' ? mcpState : restState
    const setState = source === 'ollama' ? setOllamaState : source === 'mcp' ? setMcpState : setRestState

    if (state.isPaused) {
      setState(s => ({ ...s, isPaused: false }))
      startStreaming(source)
    } else {
      setState(s => ({ ...s, isPaused: true }))
      if (intervalRefs.current[source]) {
        clearInterval(intervalRefs.current[source])
        intervalRefs.current[source] = undefined
      }
    }
  }

  const clearLogs = (source: 'ollama' | 'mcp' | 'rest') => {
    const setLogs = source === 'ollama' ? setOllamaLogs : source === 'mcp' ? setMcpLogs : setRestLogs
    setLogs([])
    toast.info(`${source.toUpperCase()} logs cleared`)
  }

  const downloadLogs = (logs: LogEntry[], source: string) => {
    const logText = logs.map(log => 
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] ${log.message}`
    ).join('\n')
    
    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${source}-logs-${Date.now()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success(`${source.toUpperCase()} logs downloaded`)
  }

  const filterLogs = (logs: LogEntry[]) => {
    return logs.filter(log => {
      const matchesLevel = selectedLevel === 'all' || log.level === selectedLevel
      const matchesFilter = !filterText || 
        log.message.toLowerCase().includes(filterText.toLowerCase())
      return matchesLevel && matchesFilter
    })
  }

  const renderLogViewer = (
    logs: LogEntry[],
    source: 'ollama' | 'mcp' | 'rest',
    state: LogStreamState,
    scrollRef: React.RefObject<HTMLDivElement | null>
  ) => {
    const filteredLogs = filterLogs(logs)

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <CardTitle className="text-lg">
                {source === 'ollama' ? 'Ollama Server' : source === 'mcp' ? 'MCP Server' : 'REST API Server'}
              </CardTitle>
              <CardDescription>
                {source === 'ollama' && `${ollamaConfig?.apiEndpoint || 'localhost:11434'}`}
                {source === 'mcp' && `${ollamaConfig?.mcpHost || '127.0.0.1'}:${ollamaConfig?.mcpPort || '8000'}`}
                {source === 'rest' && `${ollamaConfig?.ragHost || '127.0.0.1'}:${ollamaConfig?.ragPort || '8001'}`}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={state.isStreaming ? 'default' : 'secondary'} className="gap-1">
                {state.isStreaming ? (
                  <>
                    <CheckCircle className="h-3 w-3" />
                    Streaming
                  </>
                ) : (
                  <>
                    <Pause className="h-3 w-3" />
                    Stopped
                  </>
                )}
              </Badge>
              <div className="flex gap-1">
                {!state.isStreaming ? (
                  <Button size="sm" onClick={() => startStreaming(source)}>
                    <Play className="h-4 w-4" />
                  </Button>
                ) : (
                  <Button size="sm" variant="outline" onClick={() => pauseStreaming(source)}>
                    <Pause className="h-4 w-4" />
                  </Button>
                )}
                <Button size="sm" variant="outline" onClick={() => stopStreaming(source)}>
                  <XCircle className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" onClick={() => downloadLogs(logs, source)}>
                  <DownloadSimple className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" onClick={() => clearLogs(source)}>
                  <Trash className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px] w-full rounded-lg border border-border bg-muted/20 p-4">
            <div ref={scrollRef} className="space-y-1 font-mono text-xs">
              {filteredLogs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-[380px] text-muted-foreground">
                  <Terminal className="h-12 w-12 mb-4 opacity-50" />
                  <p>No logs available</p>
                  <p className="text-[10px] mt-1">Start streaming to view logs</p>
                </div>
              ) : (
                filteredLogs.map(log => {
                  const Icon = LOG_LEVEL_ICONS[log.level]
                  return (
                    <div key={log.id} className="flex items-start gap-2 py-1 hover:bg-muted/50 px-2 rounded">
                      <span className="text-muted-foreground shrink-0">
                        {log.timestamp.toLocaleTimeString()}
                      </span>
                      <Icon className={`h-3.5 w-3.5 mt-0.5 shrink-0 ${LOG_LEVEL_STYLES[log.level]}`} />
                      <span className={`${LOG_LEVEL_STYLES[log.level]} break-all`}>
                        {log.message}
                      </span>
                    </div>
                  )
                })
              )}
            </div>
          </ScrollArea>
          <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
            <span>{filteredLogs.length} entries {filterText || selectedLevel !== 'all' ? `(filtered from ${logs.length})` : ''}</span>
            <span>Auto-scroll enabled</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  useEffect(() => {
    if (ollamaScrollRef.current && ollamaState.isStreaming) {
      ollamaScrollRef.current.scrollTop = ollamaScrollRef.current.scrollHeight
    }
  }, [ollamaLogs, ollamaState.isStreaming])

  useEffect(() => {
    if (mcpScrollRef.current && mcpState.isStreaming) {
      mcpScrollRef.current.scrollTop = mcpScrollRef.current.scrollHeight
    }
  }, [mcpLogs, mcpState.isStreaming])

  useEffect(() => {
    if (restScrollRef.current && restState.isStreaming) {
      restScrollRef.current.scrollTop = restScrollRef.current.scrollHeight
    }
  }, [restLogs, restState.isStreaming])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight mb-2">System Logs</h2>
        <p className="text-muted-foreground">Monitor real-time logs from your AI infrastructure</p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <MagnifyingGlass className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Filter logs..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant={selectedLevel === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedLevel('all')}
          >
            All
          </Button>
          <Button
            variant={selectedLevel === 'info' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedLevel('info')}
          >
            <InfoIcon className="h-4 w-4 mr-2" />
            Info
          </Button>
          <Button
            variant={selectedLevel === 'warn' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedLevel('warn')}
          >
            <Warning className="h-4 w-4 mr-2" />
            Warn
          </Button>
          <Button
            variant={selectedLevel === 'error' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedLevel('error')}
          >
            <XCircle className="h-4 w-4 mr-2" />
            Error
          </Button>
        </div>
      </div>

      <Tabs defaultValue="ollama" className="space-y-6">
        <TabsList className="grid w-full max-w-lg grid-cols-3">
          <TabsTrigger value="ollama" className="gap-2">
            <Lightning className="h-4 w-4" />
            Ollama
          </TabsTrigger>
          <TabsTrigger value="mcp" className="gap-2">
            <Database className="h-4 w-4" />
            MCP
          </TabsTrigger>
          <TabsTrigger value="rest" className="gap-2">
            <CloudArrowUp className="h-4 w-4" />
            REST API
          </TabsTrigger>
        </TabsList>

        <TabsContent value="ollama" className="space-y-6">
          {renderLogViewer(ollamaLogs, 'ollama', ollamaState, ollamaScrollRef)}
        </TabsContent>

        <TabsContent value="mcp" className="space-y-6">
          {renderLogViewer(mcpLogs, 'mcp', mcpState, mcpScrollRef)}
        </TabsContent>

        <TabsContent value="rest" className="space-y-6">
          {renderLogViewer(restLogs, 'rest', restState, restScrollRef)}
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Log Management Tips</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <div className="flex items-start gap-3">
            <InfoIcon className="h-5 w-5 mt-0.5 shrink-0 text-primary" />
            <p>Logs are automatically limited to the most recent 200 entries per server to maintain performance</p>
          </div>
          <div className="flex items-start gap-3">
            <DownloadSimple className="h-5 w-5 mt-0.5 shrink-0 text-primary" />
            <p>Download logs before clearing them if you need to keep a permanent record</p>
          </div>
          <div className="flex items-start gap-3">
            <FunnelSimple className="h-5 w-5 mt-0.5 shrink-0 text-primary" />
            <p>Use filters to focus on specific log levels or search for particular events</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
