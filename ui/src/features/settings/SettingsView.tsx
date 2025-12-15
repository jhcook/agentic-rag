import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Trash, ChevronUp, ChevronDown, CheckCircle, AlertCircle } from 'lucide-react'
import { toast } from 'sonner'

export type OllamaConfig = {
    apiEndpoint: string
    ollamaLocalModel?: string
    ollamaCloudModel?: string
    embeddingModel: string
    temperature: string
    topP: string //[0,1]
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
    activeModel?: string
    ollamaCloudApiKey?: string
    ollamaCloudEndpoint?: string
    proxy?: string
    ollamaCloudCABundle?: string
    ollamaMode?: 'local' | 'cloud' | 'auto'
    availableModels?: string[]
    availableCloudModels?: string[]
    caBundlePath?: string
}

export type PgvectorConfig = {
    host: string
    port: number
    dbname: string
    user: string
    password: string
    hasPassword: boolean
}

interface SettingsViewProps {
    config: OllamaConfig
    onConfigChange: (key: keyof OllamaConfig, value: string | boolean) => void
    onSaveConfig: () => void
    onGoogleLogin: () => void
    onDisconnect: (provider?: 'google' | 'openai_assistants' | 'ollama') => void
    vertexConfig: { projectId: string; location: string; dataStoreId: string }
    onVertexConfigChange: (cfg: { projectId: string; location: string; dataStoreId: string }) => void
    onSaveVertexConfig: () => void
    openaiConfig: { apiKey: string; model: string; assistantId: string }
    hasOpenaiApiKey: boolean
    openaiModels: string[]
    onOpenaiConfigChange: (cfg: { apiKey: string; model: string; assistantId: string }) => void
    onSaveOpenaiConfig: () => void
    onTestOpenAI: () => void
    onTestOllamaCloud: (key: string, endpoint?: string) => Promise<{ success: boolean; message: string }>
    onFetchOllamaModels?: (key: string, endpoint?: string) => Promise<string[]>
    onTestOllamaLocal?: (endpoint?: string) => Promise<{ success: boolean; message: string }>
    onFetchOllamaLocalModels?: (endpoint?: string) => Promise<string[]>
    ollamaStatus: any

    pgvectorConfig: PgvectorConfig
    onPgvectorConfigChange: (cfg: PgvectorConfig) => void
    onSavePgvectorConfig: () => Promise<void>
    onTestPgvector: () => Promise<{ success: boolean; message: string }>
    onMigratePgvector: () => Promise<{ status: string; [k: string]: any }>
    onBackfillPgvector: () => Promise<{ status: string; [k: string]: any }>
    pgvectorStats: { status: string; documents?: number; chunks?: number; embedding_dim?: number; error?: string } | null
    onRefreshPgvectorStats: () => Promise<void>
}

export function SettingsView(props: SettingsViewProps) {
    const {
        config,
        onConfigChange,
        onSaveConfig,
        onGoogleLogin,
        onDisconnect,
        vertexConfig,
        onVertexConfigChange,
        onSaveVertexConfig,
        openaiConfig,
        hasOpenaiApiKey,
        openaiModels,
        onOpenaiConfigChange,
        onSaveOpenaiConfig,
        onTestOpenAI,
        onTestOllamaCloud,
        onFetchOllamaModels,
        onTestOllamaLocal,
        onFetchOllamaLocalModels,
        ollamaStatus,
        pgvectorConfig,
        onPgvectorConfigChange,
        onSavePgvectorConfig,
        onTestPgvector,
        onMigratePgvector,
        onBackfillPgvector,
        pgvectorStats,
        onRefreshPgvectorStats
    } = props

    const [ollamaExpanded, setOllamaExpanded] = useState(false)
    const [googleExpanded, setGoogleExpanded] = useState(false)
    const [advancedExpanded, setAdvancedExpanded] = useState(false)

    // Local state for editing masked keys
    const [editingOllamaKey, setEditingOllamaKey] = useState(false)
    const [editingOpenaiKey, setEditingOpenaiKey] = useState(false)
    const [editingPgvectorPassword, setEditingPgvectorPassword] = useState(false)

    const localModel = config.ollamaLocalModel || ''
    const cloudModel = config.ollamaCloudModel || ''
    const isHybrid = (config.ollamaMode || 'local') === 'auto'
    const showCloudModelField = ['cloud', 'auto'].includes(config.ollamaMode || 'local')
    const isCloudOnly = (config.ollamaMode || 'local') === 'cloud'

    // Handlers
    const handleTestOllama = async () => {
        if (!config.ollamaCloudApiKey) return
        const res = await onTestOllamaCloud(config.ollamaCloudApiKey, config.ollamaCloudEndpoint)
        if (res.success) {
            toast.success(res.message)
            // Auto-fetch models on success
            if (onFetchOllamaModels) {
                const models = await onFetchOllamaModels(config.ollamaCloudApiKey, config.ollamaCloudEndpoint)
                if (models.length > 0) toast.success(`Found ${models.length} available models`)
            }
        }
        else toast.error(res.message)
    }

    const isOllamaKeyMasked = config.ollamaCloudApiKey === '***MASKED***'
    const isOpenaiKeyMasked = hasOpenaiApiKey && !openaiConfig.apiKey
    const isPgvectorPasswordMasked = pgvectorConfig.password === '***MASKED***'

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div>
                <h2 className="text-3xl font-bold tracking-tight">System Configuration</h2>
                <p className="text-muted-foreground mt-1">Manage AI providers and model parameters.</p>
            </div>

            <Tabs defaultValue="ollama" className="w-full">
                <TabsList className="mb-4 bg-secondary/50 p-1">
                    <TabsTrigger value="ollama" className="data-[state=active]:bg-background/80">Ollama</TabsTrigger>
                    <TabsTrigger value="google" className="data-[state=active]:bg-background/80">Google Cloud</TabsTrigger>
                    <TabsTrigger value="openai" className="data-[state=active]:bg-background/80">OpenAI</TabsTrigger>
                    <TabsTrigger value="advanced" className="data-[state=active]:bg-background/80">Advanced</TabsTrigger>
                </TabsList>

                {/* --- OLLAMA TAB --- */}
                <TabsContent value="ollama" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Ollama Configuration</CardTitle>
                            <CardDescription>Connect to Ollama instances (Local or Cloud).</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            {/* Mode Selection */}
                            <div className="grid gap-2">
                                <Label>Ollama Mode</Label>
                                <div className="flex gap-2">
                                    {['local', 'cloud', 'auto'].map((mode) => (
                                        <Button
                                            key={mode}
                                            variant={config.ollamaMode === mode ? 'default' : 'outline'}
                                            onClick={() => onConfigChange('ollamaMode', mode)}
                                            className="flex-1 capitalize"
                                        >
                                            {mode === 'auto' ? 'Hybrid (Cloud + Local)' : mode}
                                        </Button>
                                    ))}
                                </div>
                            </div>

                            {/* Hybrid summary: no inputs, just status */}
                            {isHybrid && (
                                <div className="p-4 rounded-lg border border-white/5 bg-secondary/30 flex items-center justify-between">
                                    <div>
                                        <h4 className="font-semibold text-sm">Hybrid (Cloud + Local)</h4>
                                        <p className="text-xs text-muted-foreground">Uses cloud first with automatic local fallback. No extra settings needed here.</p>
                                    </div>
                                    {ollamaStatus?.cloud_available && ollamaStatus?.local_available ? (
                                        <CheckCircle className="w-5 h-5 text-green-500" />
                                    ) : (
                                        <AlertCircle className="w-5 h-5 text-amber-500" />
                                    )}
                                </div>
                            )}

                            {/* Cloud Config Section - Show if cloud or hybrid */}
                            {!isHybrid && ['cloud', 'auto'].includes(config.ollamaMode || 'local') && (
                                <div className="bg-secondary/20 p-4 rounded-lg border border-white/5 space-y-3 animate-in fade-in slide-in-from-top-1">
                                    <div className="flex items-center gap-2">
                                        <h3 className="font-semibold text-sm">Ollama Cloud Configuration</h3>
                                        {ollamaStatus?.cloud_available && <CheckCircle className="w-4 h-4 text-green-500" />}
                                    </div>
                                    <div className="grid gap-2">
                                        <Label>Public Key (API Key)</Label>
                                        <div className="flex gap-2">
                                            {!editingOllamaKey && isOllamaKeyMasked ? (
                                                <div className="flex-1 flex gap-2">
                                                    <Input disabled value="********************************" type="password" />
                                                    <Button variant="outline" onClick={() => {
                                                        setEditingOllamaKey(true)
                                                        onConfigChange('ollamaCloudApiKey', '')
                                                    }}>Change</Button>
                                                </div>
                                            ) : (
                                                <div className="flex-1 flex gap-2">
                                                    <Input
                                                        type="password"
                                                        placeholder="ssh-ed25519 AAAAC3NzaC..."
                                                        value={config.ollamaCloudApiKey || ''}
                                                        onChange={(e) => onConfigChange('ollamaCloudApiKey', e.target.value)}
                                                    />
                                                    {isOllamaKeyMasked && <Button variant="ghost" onClick={() => setEditingOllamaKey(false)}>Cancel</Button>}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                    <div className="grid gap-2">
                                        <Label>Cloud Endpoint (Optional)</Label>
                                        <Input
                                            placeholder="https://api.ollama.com"
                                            value={config.ollamaCloudEndpoint || ''}
                                            onChange={(e) => onConfigChange('ollamaCloudEndpoint', e.target.value)}
                                        />
                                    </div>
                                    <div className="flex gap-2">
                                        <Button variant="outline" size="sm" onClick={handleTestOllama}>Test Connection</Button>
                                        {onFetchOllamaModels && (
                                            <Button variant="secondary" size="sm" onClick={async () => {
                                                if (config.ollamaCloudApiKey) {
                                                    const models = await onFetchOllamaModels(config.ollamaCloudApiKey, config.ollamaCloudEndpoint)
                                                    if (models.length > 0) toast.success(`Found ${models.length} models`)
                                                    else toast.warning('No models found or access denied')
                                                }
                                            }}>Fetch Models</Button>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Local Params - Always visible but maybe disabled if cloud only? 
                                User Requirement: "Ollama endpoint (local), cloud only, or cloud with local fallback"
                                So if Cloud Only, maybe hide local endpoint?
                            */}
                            {!isHybrid && config.ollamaMode !== 'cloud' && (
                                <div className="space-y-2">
                                    <Label>Local API Endpoint</Label>
                                    <Input
                                        value={config.apiEndpoint}
                                        onChange={(e) => onConfigChange('apiEndpoint', e.target.value)}
                                    />
                                    <div className="flex gap-2 pt-2">
                                        {onTestOllamaLocal && (
                                            <Button variant="outline" size="sm" onClick={async () => {
                                                const res = await onTestOllamaLocal(config.apiEndpoint)
                                                res.success ? toast.success(res.message) : toast.error(res.message)
                                            }}>Test Local</Button>
                                        )}
                                        {onFetchOllamaLocalModels && (
                                            <Button variant="secondary" size="sm" onClick={async () => {
                                                const models = await onFetchOllamaLocalModels(config.apiEndpoint)
                                                if (models.length > 0) toast.success(`Found ${models.length} local models`)
                                                else toast.warning('No local models found')
                                            }}>Fetch Local Models</Button>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Model Selection */}
                            {!isHybrid && (
                              <div className="space-y-4">
                                {!isCloudOnly && (
                                    <div className="space-y-2">
                                        <Label>Local Model</Label>
                                        {config.availableModels && config.availableModels.length > 0 ? (
                                            <select
                                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                                value={localModel}
                                                onChange={(e) => onConfigChange('ollamaLocalModel', e.target.value)}
                                            >
                                                {config.availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                                                {!config.availableModels.includes(localModel) && <option value={localModel}>{localModel}</option>}
                                            </select>
                                        ) : (
                                            <Input
                                                value={localModel}
                                                onChange={(e) => onConfigChange('ollamaLocalModel', e.target.value)}
                                                placeholder="e.g. ollama/qwen2.5:0.5b"
                                            />
                                        )}
                                        <p className="text-[10px] text-muted-foreground">
                                            Used in local mode and as fallback when auto mode is enabled.
                                        </p>
                                    </div>
                                )}

                                {showCloudModelField && (
                                    <div className="space-y-2">
                                        <Label>Cloud Model</Label>
                                        {config.availableCloudModels && config.availableCloudModels.length > 0 ? (
                                            <select
                                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                                value={cloudModel}
                                                onChange={(e) => onConfigChange('ollamaCloudModel', e.target.value)}
                                            >
                                                {config.availableCloudModels.map(m => <option key={m} value={m}>{m}</option>)}
                                                {!config.availableCloudModels.includes(cloudModel) && <option value={cloudModel}>{cloudModel}</option>}
                                            </select>
                                        ) : (
                                            <Input
                                                value={cloudModel}
                                                onChange={(e) => onConfigChange('ollamaCloudModel', e.target.value)}
                                                placeholder="e.g. gemini-3-pro-preview"
                                            />
                                        )}
                                        <p className="text-[10px] text-muted-foreground">Used when Ollama Cloud mode is active.</p>
                                    </div>
                                )}
                              </div>
                            )}


                            {!isHybrid && (
                                <Collapsible open={ollamaExpanded} onOpenChange={setOllamaExpanded}>
                                    <CollapsibleTrigger asChild>
                                        <Button variant="ghost" className="w-full flex justify-between">
                                            Generation Parameters
                                            {ollamaExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                        </Button>
                                    </CollapsibleTrigger>
                                    <CollapsibleContent className="space-y-4 pt-4">
                                        <div className="grid grid-cols-3 gap-4">
                                            <div className="space-y-2">
                                                <Label>Temperature</Label>
                                                <Input value={config.temperature} onChange={(e) => onConfigChange('temperature', e.target.value)} />
                                            </div>
                                            <div className="space-y-2">
                                                <Label>Context Window</Label>
                                                <Input value={config.numCtx} onChange={(e) => onConfigChange('numCtx', e.target.value)} />
                                            </div>
                                            <div className="space-y-2">
                                                <Label>Top K</Label>
                                                <Input value={config.topK} onChange={(e) => onConfigChange('topK', e.target.value)} />
                                            </div>
                                        </div>
                                    </CollapsibleContent>
                                </Collapsible>
                            )}

                            <div className="flex gap-2 pt-4">
                                <Button onClick={onSaveConfig}>Save & Activate</Button>
                                <Button variant="destructive" onClick={() => onDisconnect('ollama')}>Disconnect</Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* --- GOOGLE TAB --- */}
                <TabsContent value="google" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Google Cloud & Vertex AI</CardTitle>
                            <CardDescription>Configure Google Drive integration and Vertex AI Search.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="p-4 bg-secondary/20 rounded-lg border border-white/5 flex items-center justify-between">
                                <div>
                                    <h3 className="font-semibold">Google Account</h3>
                                    <p className="text-xs text-muted-foreground">Required for Drive access</p>
                                </div>
                                <div className="flex gap-2">
                                    <Button variant="outline" onClick={onGoogleLogin}>Login / Refresh</Button>
                                    <Button variant="ghost" size="icon" onClick={() => onDisconnect('google')}>
                                        <Trash className="w-4 h-4 text-destructive" />
                                    </Button>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <h3 className="font-semibold text-sm border-b pb-2">Vertex AI Search</h3>
                                <div className="grid gap-2">
                                    <Label>Project ID</Label>
                                    <Input
                                        value={vertexConfig.projectId}
                                        onChange={(e) => onVertexConfigChange({ ...vertexConfig, projectId: e.target.value })}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label>Location</Label>
                                        <Input
                                            value={vertexConfig.location}
                                            onChange={(e) => onVertexConfigChange({ ...vertexConfig, location: e.target.value })}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Data Store ID</Label>
                                        <Input
                                            value={vertexConfig.dataStoreId}
                                            onChange={(e) => onVertexConfigChange({ ...vertexConfig, dataStoreId: e.target.value })}
                                        />
                                    </div>
                                </div>
                                <Button className="w-full" onClick={onSaveVertexConfig}>Save & Activate Vertex</Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* --- OPENAI TAB --- */}
                <TabsContent value="openai" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>OpenAI Assistants</CardTitle>
                            <CardDescription>Use GPT-4 models via the Assistants API.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid gap-2">
                                <Label>API Key</Label>
                                <div className="flex gap-2">
                                    {!editingOpenaiKey && hasOpenaiApiKey ? (
                                        <div className="flex-1 flex gap-2">
                                            <Input disabled value="sk-********************************" type="password" />
                                            <Button variant="outline" onClick={() => {
                                                setEditingOpenaiKey(true)
                                                onOpenaiConfigChange({ ...openaiConfig, apiKey: '' })
                                            }}>Change</Button>
                                        </div>
                                    ) : (
                                        <div className="flex-1 flex gap-2">
                                            <Input
                                                type="password"
                                                placeholder="sk-..."
                                                value={openaiConfig.apiKey}
                                                onChange={(e) => onOpenaiConfigChange({ ...openaiConfig, apiKey: e.target.value })}
                                            />
                                            {hasOpenaiApiKey && <Button variant="ghost" onClick={() => setEditingOpenaiKey(false)}>Cancel</Button>}
                                            <Button variant="outline" onClick={onTestOpenAI}>Test</Button>
                                        </div>
                                    )}
                                </div>
                                {hasOpenaiApiKey && !editingOpenaiKey && <p className="text-xs text-green-500 flex items-center gap-1"><CheckCircle className="w-3 h-3" /> Key stored securely</p>}
                            </div>

                            <div className="grid gap-2">
                                <Label>Model</Label>
                                <select
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                    value={openaiConfig.model}
                                    onChange={(e) => onOpenaiConfigChange({ ...openaiConfig, model: e.target.value })}
                                >
                                    <option value="gpt-4-turbo-preview">gpt-4-turbo-preview</option>
                                    <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                                    {openaiModels.map(m => (
                                        <option key={m} value={m}>{m}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="grid gap-2">
                                <Label>Assistant ID (Optional)</Label>
                                <Input
                                    placeholder="asst_..."
                                    value={openaiConfig.assistantId}
                                    onChange={(e) => onOpenaiConfigChange({ ...openaiConfig, assistantId: e.target.value })}
                                />
                            </div>

                            <div className="flex justify-between pt-4">
                                <Button variant="ghost" className="text-destructive" onClick={() => onDisconnect('openai_assistants')}>Clear Credentials</Button>
                                <Button onClick={onSaveOpenaiConfig}>Save & Activate</Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* --- ADVANCED TAB --- */}
                <TabsContent value="advanced" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Network & Connectivity</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label>RAG Host</Label>
                                    <Input value={config.ragHost} onChange={(e) => onConfigChange('ragHost', e.target.value)} />
                                </div>
                                <div className="space-y-2">
                                    <Label>RAG Port</Label>
                                    <Input value={config.ragPort} onChange={(e) => onConfigChange('ragPort', e.target.value)} />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label>HTTPS Proxy (Optional)</Label>
                                    <Input 
                                        value={config.proxy || ''} 
                                        onChange={(e) => onConfigChange('proxy', e.target.value)} 
                                        placeholder="http://proxy.example.com:8080"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>CA Bundle Path (Optional)</Label>
                                    <Input 
                                        value={config.caBundlePath || ''} 
                                        onChange={(e) => onConfigChange('caBundlePath', e.target.value)} 
                                        placeholder="/path/to/ca-bundle.pem"
                                    />
                                </div>
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Configure proxy and custom CA bundle for outgoing connections (OpenAI, Google, etc.).
                            </p>

                            <div className="flex items-center space-x-2 pt-2">
                                <input
                                    type="checkbox"
                                    id="debugMode"
                                    className="rounded border-gray-300 text-primary focus:ring-primary"
                                    checked={config.debugMode || false}
                                    onChange={(e) => onConfigChange('debugMode', e.target.checked)}
                                />
                                <Label htmlFor="debugMode">Enable Debug Mode (Verbose Logging)</Label>
                            </div>
                            <Button className="w-full mt-4" onClick={onSaveConfig}>Update Network Settings</Button>
                        </CardContent>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Vector Store (pgvector)</CardTitle>
                            <CardDescription>Configure and manage PostgreSQL + pgvector.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label>Host</Label>
                                    <Input
                                        value={pgvectorConfig.host}
                                        onChange={(e) => onPgvectorConfigChange({ ...pgvectorConfig, host: e.target.value })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Port</Label>
                                    <Input
                                        value={String(pgvectorConfig.port)}
                                        onChange={(e) => onPgvectorConfigChange({ ...pgvectorConfig, port: Number(e.target.value || 0) })}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label>Database</Label>
                                    <Input
                                        value={pgvectorConfig.dbname}
                                        onChange={(e) => onPgvectorConfigChange({ ...pgvectorConfig, dbname: e.target.value })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>User</Label>
                                    <Input
                                        value={pgvectorConfig.user}
                                        onChange={(e) => onPgvectorConfigChange({ ...pgvectorConfig, user: e.target.value })}
                                    />
                                </div>
                            </div>

                            <div className="grid gap-2">
                                <Label>Password</Label>
                                <div className="flex gap-2">
                                    {!editingPgvectorPassword && isPgvectorPasswordMasked ? (
                                        <div className="flex-1 flex gap-2">
                                            <Input disabled value="********************************" type="password" />
                                            <Button
                                                variant="outline"
                                                onClick={() => {
                                                    setEditingPgvectorPassword(true)
                                                    onPgvectorConfigChange({ ...pgvectorConfig, password: '' })
                                                }}
                                            >
                                                Change
                                            </Button>
                                        </div>
                                    ) : (
                                        <div className="flex-1 flex gap-2">
                                            <Input
                                                type="password"
                                                placeholder="PGVECTOR_PASSWORD"
                                                value={pgvectorConfig.password}
                                                onChange={(e) => onPgvectorConfigChange({ ...pgvectorConfig, password: e.target.value })}
                                            />
                                            {isPgvectorPasswordMasked && (
                                                <Button variant="ghost" onClick={() => setEditingPgvectorPassword(false)}>
                                                    Cancel
                                                </Button>
                                            )}
                                        </div>
                                    )}
                                </div>
                                {!pgvectorConfig.hasPassword && (
                                    <p className="text-xs text-muted-foreground">
                                        No password detected. Start scripts require `PGVECTOR_PASSWORD` in .env.
                                    </p>
                                )}
                            </div>

                            <div className="flex flex-wrap gap-2 justify-end">
                                <Button
                                    variant="outline"
                                    onClick={async () => {
                                        const res = await onTestPgvector()
                                        if (res.success) toast.success(res.message)
                                        else toast.error(res.message)
                                    }}
                                >
                                    Test Connection
                                </Button>
                                <Button
                                    variant="outline"
                                    onClick={async () => {
                                        const res = await onMigratePgvector()
                                        if (res.status === 'ok') toast.success('Schema initialized')
                                        else toast.error(res.error || 'Schema init failed')
                                    }}
                                >
                                    Initialize (Migrate)
                                </Button>
                                <Button
                                    variant="outline"
                                    onClick={async () => {
                                        const res = await onBackfillPgvector()
                                        if (res.status === 'ok') toast.success('Backfill complete')
                                        else toast.error(res.error || 'Backfill failed')
                                    }}
                                >
                                    Backfill
                                </Button>
                                <Button
                                    variant="secondary"
                                    onClick={async () => {
                                        try {
                                            await onRefreshPgvectorStats()
                                            toast.success('Stats refreshed')
                                        } catch (e: any) {
                                            toast.error(e?.message || 'Stats failed')
                                        }
                                    }}
                                >
                                    Refresh Stats
                                </Button>
                                <Button onClick={async () => {
                                    try {
                                        await onSavePgvectorConfig()
                                        toast.success('pgvector settings saved')
                                    } catch (e: any) {
                                        toast.error(e?.message || 'Failed to save pgvector settings')
                                    }
                                }}>
                                    Save
                                </Button>
                            </div>

                            <div className="text-xs text-muted-foreground">
                                {pgvectorStats?.status === 'ok' ? (
                                    <div className="grid grid-cols-3 gap-2">
                                        <div>Documents: {pgvectorStats.documents ?? 0}</div>
                                        <div>Chunks: {pgvectorStats.chunks ?? 0}</div>
                                        <div>Embedding dim: {pgvectorStats.embedding_dim ?? 0}</div>
                                    </div>
                                ) : pgvectorStats?.status === 'error' ? (
                                    <div className="text-destructive">{pgvectorStats.error || 'Stats unavailable'}</div>
                                ) : (
                                    <div>Stats not loaded.</div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

            </Tabs>
        </div>
    )
}
