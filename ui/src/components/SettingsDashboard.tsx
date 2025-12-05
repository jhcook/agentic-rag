import { useState } from 'react'
import { 
  Settings as GearSix, 
  Info, 
  CheckCircle, 
  CloudUpload as CloudArrowUp, 
  ChevronDown as CaretDown, 
  ChevronUp as CaretUp, 
  Zap as Lightning 
} from 'lucide-react'
import { toast } from 'sonner'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

export type OllamaConfig = {
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
  debugMode?: boolean
  ollamaMode?: 'local' | 'cloud' | 'auto'
  ollamaCloudApiKey?: string
  ollamaCloudEndpoint?: string
   ollamaCloudProxy?: string
   ollamaCloudCABundle?: string
}

export type VertexConfig = {
  projectId: string
  location: string
  dataStoreId: string
}

export type OpenAIConfig = {
  apiKey: string
  model: string
  assistantId: string
}

interface SettingsDashboardProps {
  config: OllamaConfig
  onConfigChange: (field: keyof OllamaConfig, value: string | boolean) => void
  onSaveConfig: () => void
  onTestConnection: () => void
  vertexConfig: VertexConfig
  onVertexConfigChange: (config: VertexConfig) => void
  onSaveVertexConfig: () => void
  onGoogleLogin: () => void
  onDisconnect: (provider?: 'google' | 'openai_assistants' | 'ollama') => void
  openaiConfig: OpenAIConfig
  openaiModels: string[]
  onOpenaiConfigChange: (config: OpenAIConfig) => void
  onSaveOpenAIConfig: () => void
  onTestOpenAIConnection: () => void
  onSwitchBackend: (mode: string) => void
  activeMode?: string | null
  availableModes: string[]
  onSetOllamaMode?: (mode: 'local' | 'cloud' | 'auto') => Promise<void>
  onTestOllamaCloudConnection?: (apiKey: string, endpoint?: string) => Promise<{ success: boolean; message: string }>
  ollamaStatus?: {
    mode: string
    endpoint: string
    cloud_available: boolean
    local_available: boolean
    cloud_status: string | null
    local_status: string | null
  } | null
}

export function SettingsDashboard({
  config,
  onConfigChange,
  onSaveConfig,
  onTestConnection,
  vertexConfig,
  onVertexConfigChange,
  onSaveVertexConfig,
  onGoogleLogin,
  onDisconnect,
  openaiConfig,
  openaiModels,
  onOpenaiConfigChange,
  onSaveOpenAIConfig,
  onTestOpenAIConnection,
  onSwitchBackend,
  activeMode = 'ollama',
  availableModes = [],
  onSetOllamaMode,
  onTestOllamaCloudConnection,
  ollamaStatus
}: SettingsDashboardProps) {
  const [ollamaExpanded, setOllamaExpanded] = useState(false)
  const [openaiExpanded, setOpenaiExpanded] = useState(false)
  const [googleExpanded, setGoogleExpanded] = useState(false)
  const [advancedExpanded, setAdvancedExpanded] = useState(false)
  const ollamaAvailable = availableModes.includes('ollama')
  const openaiAvailable = availableModes.includes('openai_assistants')
  const googleAvailable = availableModes.some((mode) => mode.startsWith('google') || mode === 'vertex_ai_search')

  const handleSaveAll = () => {
    onSaveConfig()
    onSaveOpenAIConfig()
    onSaveVertexConfig()
  }

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight mb-2">Settings</h2>
        <p className="text-muted-foreground">Configure AI providers and system preferences</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>REST API Server Configuration</CardTitle>
          <CardDescription>Connection details for the Lauren AI REST API</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-lg border border-border bg-muted/20">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="debug-mode">Debug Mode</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                        <Info className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Enable debug logging for REST and MCP servers. Logs will show DEBUG level messages for troubleshooting.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <p className="text-sm text-muted-foreground">Enable verbose debug logging</p>
            </div>
          <Switch
            id="debug-mode"
            checked={config?.debugMode || false}
            onCheckedChange={(checked) => onConfigChange('debugMode', checked)}
          />
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="rag-host">Host</Label>
                <TooltipProvider>
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
                </TooltipProvider>
              </div>
              <Input
                id="rag-host"
                value={config?.ragHost || ''}
                onChange={(e) => onConfigChange('ragHost', e.target.value)}
                placeholder="127.0.0.1"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="rag-port">Port</Label>
                <TooltipProvider>
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
                </TooltipProvider>
              </div>
              <Input
                id="rag-port"
                value={config?.ragPort || ''}
                onChange={(e) => onConfigChange('ragPort', e.target.value)}
                placeholder="8001"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="rag-path">Base Path</Label>
                <TooltipProvider>
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
                </TooltipProvider>
              </div>
              <Input
                id="rag-path"
                value={config?.ragPath || ''}
                onChange={(e) => onConfigChange('ragPath', e.target.value)}
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
                      <p className="font-semibold">Ollama</p>
                      <p className="text-sm text-muted-foreground">Run AI models locally</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {activeMode === 'ollama' && <Badge>Active</Badge>}
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
                        <TooltipProvider>
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
                        </TooltipProvider>
                      </div>
                      <Input
                        id="api-endpoint"
                        value={config?.apiEndpoint || ''}
                        onChange={(e) => onConfigChange('apiEndpoint', e.target.value)}
                        placeholder="http://localhost:11434"
                      />
                    </div>

                    <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20">
                      <h4 className="font-semibold text-sm">Ollama Mode Configuration</h4>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="ollama-mode">Mode</Label>
                          <Select
                            value={config?.ollamaMode || 'local'}
                            onValueChange={(value) => {
                              onConfigChange('ollamaMode', value)
                              if (onSetOllamaMode) {
                                onSetOllamaMode(value as 'local' | 'cloud' | 'auto').catch(console.error)
                              }
                            }}
                          >
                            <SelectTrigger id="ollama-mode">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="local">Local</SelectItem>
                              <SelectItem value="cloud">Cloud</SelectItem>
                              <SelectItem value="auto">Auto (Cloud with Local Fallback)</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-muted-foreground">
                            {config?.ollamaMode === 'local' && 'Use local Ollama instance only'}
                            {config?.ollamaMode === 'cloud' && 'Use Ollama Cloud service only'}
                            {config?.ollamaMode === 'auto' && 'Try Ollama Cloud first, fallback to local on failure'}
                          </p>
                        </div>

                        {(config?.ollamaMode === 'cloud' || config?.ollamaMode === 'auto') && (
                          <div className="space-y-4 pt-2 border-t border-border">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="ollama-cloud-api-key">Ollama Cloud API Key</Label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Your Ollama Cloud API key. Get it from your Ollama Cloud account settings.</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                            <Input
                              id="ollama-cloud-api-key"
                              type="password"
                              value={config?.ollamaCloudApiKey || ''}
                              onChange={(e) => onConfigChange('ollamaCloudApiKey', e.target.value)}
                                placeholder="Enter your API key"
                                className="font-mono text-sm"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="ollama-cloud-endpoint">Ollama Cloud Endpoint</Label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Ollama Cloud endpoint URL. Default is https://ollama.com</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                            <Input
                              id="ollama-cloud-endpoint"
                              value={config?.ollamaCloudEndpoint || ''}
                              onChange={(e) => onConfigChange('ollamaCloudEndpoint', e.target.value)}
                              placeholder="https://ollama.com"
                            />
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="ollama-cloud-proxy">HTTPS Proxy</Label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Optional HTTPS proxy URL used for Ollama Cloud requests. Stored in config/settings.json.</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                              <Input
                                id="ollama-cloud-proxy"
                                value={config?.ollamaCloudProxy || ''}
                                onChange={(e) => onConfigChange('ollamaCloudProxy', e.target.value)}
                                placeholder="https://proxy.example.com:3128"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="ollama-cloud-ca-bundle">CA Bundle (PEM)</Label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
                                        <Info className="h-4 w-4" />
                                      </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      <p className="max-w-xs">Path to a PEM CA bundle for TLS verification. Stored securely in secrets/ollama_cloud_config.json.</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                              <Input
                                id="ollama-cloud-ca-bundle"
                                value={config?.ollamaCloudCABundle || ''}
                                onChange={(e) => onConfigChange('ollamaCloudCABundle', e.target.value)}
                                placeholder="/path/to/corp-root.pem"
                              />
                            </div>
                            </div>

                            {onTestOllamaCloudConnection && (
                              <Button
                                type="button"
                                variant="outline"
                                onClick={async () => {
                                  const apiKey = config?.ollamaCloudApiKey || ''
                                  const endpoint = config?.ollamaCloudEndpoint
                                  if (!apiKey) {
                                    toast.error('Please enter an API key first')
                                    return
                                  }
                                  try {
                                    const result = await onTestOllamaCloudConnection(apiKey, endpoint)
                                    if (result.success) {
                                      toast.success('Connection successful', {
                                        description: result.message
                                      })
                                    } else {
                                      toast.error('Connection failed', {
                                        description: result.message
                                      })
                                    }
                                  } catch (error) {
                                    toast.error('Connection test failed', {
                                      description: error instanceof Error ? error.message : 'Unknown error'
                                    })
                                  }
                                }}
                                className="w-full"
                              >
                                <CheckCircle className="h-4 w-4 mr-2" />
                                Test Cloud Connection
                              </Button>
                            )}
                          </div>
                        )}

                        {ollamaStatus && (
                          <div className="space-y-2 pt-2 border-t border-border">
                            <Label className="text-sm font-semibold">Connection Status</Label>
                            <div className="grid gap-2 md:grid-cols-2">
                              <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                                <span className="text-sm">Local</span>
                                <Badge variant={ollamaStatus.local_status === 'connected' ? 'default' : 'secondary'}>
                                  {ollamaStatus.local_status || 'unknown'}
                                </Badge>
                              </div>
                              {(config?.ollamaMode === 'cloud' || config?.ollamaMode === 'auto') && (
                                <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                                  <span className="text-sm">Cloud</span>
                                  <Badge variant={ollamaStatus.cloud_status === 'connected' ? 'default' : 'secondary'}>
                                    {ollamaStatus.cloud_status || 'unknown'}
                                  </Badge>
                                </div>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground">
                              Current endpoint: {ollamaStatus.endpoint}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="space-y-4 rounded-lg border border-border p-4 bg-muted/20">
                      <h4 className="font-semibold text-sm">MCP Server Configuration</h4>
                      <div className="grid gap-4 md:grid-cols-3">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor="mcp-host">Host</Label>
                            <TooltipProvider>
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
                            </TooltipProvider>
                          </div>
                          <Input
                            id="mcp-host"
                            value={config?.mcpHost || ''}
                            onChange={(e) => onConfigChange('mcpHost', e.target.value)}
                            placeholder="127.0.0.1"
                          />
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor="mcp-port">Port</Label>
                            <TooltipProvider>
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
                            </TooltipProvider>
                          </div>
                          <Input
                            id="mcp-port"
                            value={config?.mcpPort || ''}
                            onChange={(e) => onConfigChange('mcpPort', e.target.value)}
                            placeholder="8000"
                          />
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor="mcp-path">Path</Label>
                            <TooltipProvider>
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
                            </TooltipProvider>
                          </div>
                          <Input
                            id="mcp-path"
                            value={config?.mcpPath || ''}
                            onChange={(e) => onConfigChange('mcpPath', e.target.value)}
                            placeholder="/mcp"
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
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="model"
                                value={config?.model || ''}
                                onChange={(e) => onConfigChange('model', e.target.value)}
                                placeholder="llama3.2"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="embedding-model">Embedding Model</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="embedding-model"
                                value={config?.embeddingModel || ''}
                                onChange={(e) => onConfigChange('embeddingModel', e.target.value)}
                                placeholder="nomic-embed-text"
                              />
                            </div>
                          </div>

                          <div className="grid gap-4 md:grid-cols-3">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="temperature">Temperature</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="temperature"
                                type="number"
                                step="0.1"
                                min="0"
                                max="2"
                                value={config?.temperature || ''}
                                onChange={(e) => onConfigChange('temperature', e.target.value)}
                                placeholder="0.7"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="top-p">Top P</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="top-p"
                                type="number"
                                step="0.1"
                                min="0"
                                max="1"
                                value={config?.topP || ''}
                                onChange={(e) => onConfigChange('topP', e.target.value)}
                                placeholder="0.9"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="top-k">Top K</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="top-k"
                                type="number"
                                value={config?.topK || ''}
                                onChange={(e) => onConfigChange('topK', e.target.value)}
                                placeholder="40"
                              />
                            </div>
                          </div>

                          <div className="grid gap-4 md:grid-cols-3">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="repeat-penalty">Repeat Penalty</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="repeat-penalty"
                                type="number"
                                step="0.1"
                                min="0"
                                value={config?.repeatPenalty || ''}
                                onChange={(e) => onConfigChange('repeatPenalty', e.target.value)}
                                placeholder="1.1"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="seed">Seed</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="seed"
                                type="number"
                                value={config?.seed || ''}
                                onChange={(e) => onConfigChange('seed', e.target.value)}
                                placeholder="-1"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Label htmlFor="num-ctx">Context Length</Label>
                                <TooltipProvider>
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
                                </TooltipProvider>
                              </div>
                              <Input
                                id="num-ctx"
                                type="number"
                                value={config?.numCtx || ''}
                                onChange={(e) => onConfigChange('numCtx', e.target.value)}
                                placeholder="2048"
                              />
                            </div>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  </div>

                  <div className="flex gap-3 pt-4 border-t border-border">
                    <Button onClick={onTestConnection} variant="outline" className="flex-1">
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Test Connection
                    </Button>
                    <Button
                      onClick={() => onSwitchBackend('ollama')}
                      className="flex-1"
                      disabled={!ollamaAvailable || activeMode === 'ollama'}
                    >
                      Use this Backend
                    </Button>
                    {activeMode === 'ollama' && (
                      <Button
                        onClick={() => onDisconnect('ollama')}
                        variant="outline"
                        className="flex-1 bg-red-500/10 hover:bg-red-500/20 hover:text-red-500 border-red-500/20 text-red-500"
                        disabled={!ollamaAvailable}
                      >
                        Disconnect
                      </Button>
                    )}
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>

          <Collapsible open={openaiExpanded} onOpenChange={setOpenaiExpanded}>
            <div className="rounded-lg border border-border">
              <CollapsibleTrigger asChild>
                <button className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      <Lightning className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div className="text-left">
                      <p className="font-semibold">OpenAI Assistants</p>
                      <p className="text-sm text-muted-foreground">GPT-4 orchestration with local search</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {activeMode === 'openai_assistants' && <Badge>Active</Badge>}
                    {openaiExpanded ? (
                      <CaretUp className="h-5 w-5 text-muted-foreground" />
                    ) : (
                      <CaretDown className="h-5 w-5 text-muted-foreground" />
                    )}
                  </div>
                </button>
              </CollapsibleTrigger>
              
              <CollapsibleContent>
                <div className="border-t border-border p-6 space-y-6">
                  <div className="rounded-lg bg-muted/50 p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-muted-foreground" />
                      <p className="text-sm font-medium">How it works</p>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      OpenAI Assistants orchestrates conversations using GPT-4 while keeping your documents local. 
                      When it needs information, it calls our search function to query your FAISS index. 
                      You get GPT-4 quality with local privacy.
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="openai-api-key">OpenAI API Key</Label>
                      <Input
                        id="openai-api-key"
                        type="password"
                        placeholder="sk-..."
                        value={openaiConfig.apiKey}
                        onChange={(e) => onOpenaiConfigChange({ ...openaiConfig, apiKey: e.target.value })}
                        className="font-mono text-sm"
                      />
                      <p className="text-xs text-muted-foreground">
                        Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="underline">platform.openai.com</a>
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="openai-model">Model</Label>
                      {openaiModels.length > 0 ? (
                        <Select
                          value={openaiConfig.model}
                          onValueChange={(value) => onOpenaiConfigChange({ ...openaiConfig, model: value })}
                        >
                          <SelectTrigger id="openai-model">
                            <SelectValue placeholder="Select a model" />
                          </SelectTrigger>
                          <SelectContent>
                            {openaiModels.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id="openai-model"
                          value={openaiConfig.model}
                          onChange={(e) => onOpenaiConfigChange({ ...openaiConfig, model: e.target.value })}
                          placeholder="gpt-4-turbo-preview"
                        />
                      )}
                      <p className="text-xs text-muted-foreground">
                        {openaiModels.length > 0 ? 'Select from your available models' : 'Test connection to load available models'}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Button onClick={onTestOpenAIConnection} variant="outline" className="flex-1">
                      Test Connection
                    </Button>
                    {activeMode !== 'openai_assistants' && (
                      <Button
                        onClick={() => onSwitchBackend('openai_assistants')}
                        className="flex-1"
                        disabled={!openaiAvailable}
                      >
                        Use this Backend
                      </Button>
                    )}
                    <Button
                      onClick={() => onDisconnect('openai_assistants')}
                      variant="outline"
                      className="flex-1 bg-red-500/10 hover:bg-red-500/20 hover:text-red-500 border-red-500/20 text-red-500"
                      disabled={!openaiAvailable}
                    >
                      Disconnect
                    </Button>
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>

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
                    {(activeMode === 'google_gemini' || activeMode === 'vertex_ai_search') && <Badge>Active</Badge>}
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
                    <Button onClick={onGoogleLogin} className="flex-1">
                       Connect Google Account
                    </Button>
                    {(activeMode !== 'google_gemini' && activeMode !== 'vertex_ai_search') && (
                      <Button
                        onClick={() => onSwitchBackend('google_gemini')}
                        className="flex-1"
                        disabled={!googleAvailable}
                      >
                        Use this Backend
                      </Button>
                    )}
                    <Button
                      onClick={() => onDisconnect('google')}
                      variant="outline"
                      className="flex-1 bg-red-500/10 hover:bg-red-500/20 hover:text-red-500 border-red-500/20 text-red-500"
                      disabled={!googleAvailable}
                    >
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
                        onChange={(e) => onVertexConfigChange({...vertexConfig, projectId: e.target.value})}
                        placeholder="my-gcp-project-id"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="vertex-location">Location</Label>
                      <Input
                        id="vertex-location"
                        value={vertexConfig.location}
                        onChange={(e) => onVertexConfigChange({...vertexConfig, location: e.target.value})}
                        placeholder="us-central1"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="vertex-datastore">Data Store ID</Label>
                      <Input
                        id="vertex-datastore"
                        value={vertexConfig.dataStoreId}
                        onChange={(e) => onVertexConfigChange({...vertexConfig, dataStoreId: e.target.value})}
                        placeholder="my-datastore-id"
                      />
                    </div>
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>
        </CardContent>
      </Card>
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-background/80 backdrop-blur-sm border-t border-border flex justify-center z-50">
        <Button onClick={handleSaveAll} size="lg" className="shadow-lg">
          <GearSix className="mr-2 h-5 w-5" />
          Save All Configurations
        </Button>
      </div>
    </div>
  )
}
