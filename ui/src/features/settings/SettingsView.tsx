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
    model: string
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
    ollamaCloudApiKey?: string
    ollamaCloudEndpoint?: string
    ollamaCloudProxy?: string
    ollamaCloudCABundle?: string
    ollamaMode?: 'local' | 'cloud' | 'sub-cloud'
    availableModels?: string[]
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
    ollamaStatus: any
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
        ollamaStatus
    } = props

    const [ollamaExpanded, setOllamaExpanded] = useState(false)
    const [googleExpanded, setGoogleExpanded] = useState(false)
    const [advancedExpanded, setAdvancedExpanded] = useState(false)

    // Local state for editing masked keys
    const [editingOllamaKey, setEditingOllamaKey] = useState(false)
    const [editingOpenaiKey, setEditingOpenaiKey] = useState(false)

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
                                    {['local', 'cloud', 'sub-cloud'].map((mode) => (
                                        <Button
                                            key={mode}
                                            variant={config.ollamaMode === mode ? 'default' : 'outline'}
                                            onClick={() => onConfigChange('ollamaMode', mode)}
                                            className="flex-1 capitalize"
                                        >
                                            {mode === 'sub-cloud' ? 'Hybrid (Cloud + Local)' : mode}
                                        </Button>
                                    ))}
                                </div>
                            </div>

                            {/* Cloud Config Section - Show if cloud or hybrid */}
                            {['cloud', 'sub-cloud'].includes(config.ollamaMode || 'local') && (
                                <div className="bg-secondary/20 p-4 rounded-lg border border-white/5 space-y-3 animate-in fade-in slide-in-from-top-1">
                                    <div className="flex items-center justify-between">
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
                                    <div className="flex justify-end gap-2">
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
                            {config.ollamaMode !== 'cloud' && (
                                <div className="space-y-2">
                                    <Label>Local API Endpoint</Label>
                                    <Input
                                        value={config.apiEndpoint}
                                        onChange={(e) => onConfigChange('apiEndpoint', e.target.value)}
                                    />
                                </div>
                            )}

                            {/* Model Selection */}
                            <div className="space-y-2">
                                <Label>Model Name</Label>
                                {config.availableModels && config.availableModels.length > 0 ? (
                                    <select
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                        value={config.model}
                                        onChange={(e) => onConfigChange('model', e.target.value)}
                                    >
                                        {config.availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                                        {/* Allow custom if not in list */}
                                        {!config.availableModels.includes(config.model) && <option value={config.model}>{config.model}</option>}
                                    </select>
                                ) : (
                                    <Input
                                        value={config.model}
                                        onChange={(e) => onConfigChange('model', e.target.value)}
                                        placeholder="e.g. mistral, llama3"
                                    />
                                )}
                                <p className="text-[10px] text-muted-foreground">
                                    {config.ollamaMode === 'cloud' ? 'Select a model available in your cloud account.' : 'Name of the model to run.'}
                                </p>
                            </div>


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

                            <div className="flex justify-between pt-4">
                                <Button variant="destructive" onClick={() => onDisconnect('ollama')}>Disconnect</Button>
                                <Button onClick={onSaveConfig}>Save & Activate</Button>
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
                </TabsContent>

            </Tabs>
        </div>
    )
}
