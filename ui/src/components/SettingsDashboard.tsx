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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

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
}

export type VertexConfig = {
  projectId: string
  location: string
  dataStoreId: string
}

interface SettingsDashboardProps {
  config: OllamaConfig
  onConfigChange: (field: keyof OllamaConfig, value: string) => void
  onSaveConfig: () => void
  onTestConnection: () => void
  vertexConfig: VertexConfig
  onVertexConfigChange: (config: VertexConfig) => void
  onSaveVertexConfig: () => void
  onGoogleLogin: () => void
  onGoogleLogout: () => void
  activeMode?: string
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
  onGoogleLogout,
  activeMode = 'local'
}: SettingsDashboardProps) {
  const [ollamaExpanded, setOllamaExpanded] = useState(false)
  const [openaiExpanded, setOpenaiExpanded] = useState(false)
  const [googleExpanded, setGoogleExpanded] = useState(false)
  const [advancedExpanded, setAdvancedExpanded] = useState(false)

  return (
    <div className="space-y-6">
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
                      <p className="font-semibold">Ollama (Local)</p>
                      <p className="text-sm text-muted-foreground">Run AI models locally</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {activeMode === 'local' && <Badge>Active</Badge>}
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
                    <Button onClick={onTestConnection} variant="outline">
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Test Connection
                    </Button>
                    <Button onClick={onSaveConfig}>
                      <GearSix className="h-4 w-4 mr-2" />
                      Save Configuration
                    </Button>
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
                        className="font-mono text-sm"
                      />
                      <p className="text-xs text-muted-foreground">
                        Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="underline">platform.openai.com</a>
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="openai-model">Model</Label>
                      <Input
                        id="openai-model"
                        defaultValue="gpt-4-turbo-preview"
                        placeholder="gpt-4-turbo-preview"
                      />
                      <p className="text-xs text-muted-foreground">
                        Recommended: gpt-4-turbo-preview or gpt-4o
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Button className="flex-1">
                      <CheckCircle className="mr-2 h-4 w-4" />
                      Save Configuration
                    </Button>
                    <Button variant="outline" className="flex-1">
                      Test Connection
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
                    {(activeMode === 'manual' || activeMode === 'vertex_ai_search') && <Badge>Active</Badge>}
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
                    <Button onClick={onGoogleLogout} variant="outline" className="flex-1">
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
                    
                    <Button onClick={onSaveVertexConfig} size="sm" variant="secondary" className="w-full">
                      Save Vertex Configuration
                    </Button>
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>
        </CardContent>
      </Card>
    </div>
  )
}
