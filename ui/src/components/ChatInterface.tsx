import { useState, useRef, useEffect } from 'react'
import { Send, User, Bot, Download, Paperclip, X, Trash, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { toast } from 'sonner'
import { marked } from 'marked'
import { ModelSelector } from '@/components/ModelSelector'

export type Message = {
  role: 'user' | 'assistant'
  content: string
  displayContent?: string
  sources?: string[]
  timestamp: number
}

function MarkdownRenderer({ content, sources }: { content: string, sources?: string[] }) {
  const [html, setHtml] = useState('')
  
  useEffect(() => {
    const parse = async () => {
       try {
         let processedContent = content
         
         // If sources exist, annotate inline citations in the text
         // Look for patterns like [1], [2], etc. and make them superscript links
         if (sources && sources.length > 0) {
           // Replace [N] patterns with superscript citations
           processedContent = processedContent.replace(/\[(\d+)\]/g, (match, num) => {
             const idx = parseInt(num) - 1
             if (idx >= 0 && idx < sources.length) {
               return `<sup class="citation" data-source-idx="${idx}">[${num}]</sup>`
             }
             return match
           })
         }
         
         const result = await marked.parse(processedContent)
         setHtml(result)
       } catch (e) {
         setHtml(content)
       }
    }
    parse()
  }, [content, sources])
  
  return <div className="prose prose-sm dark:prose-invert max-w-none [&>p]:mb-2 [&>ul]:list-disc [&>ul]:pl-4 [&>ol]:list-decimal [&>ol]:pl-4 [&>a]:text-blue-500 [&>a]:underline [&_.citation]:text-blue-600 [&_.citation]:cursor-pointer [&_.citation]:font-semibold [&_.citation]:hover:text-blue-800" dangerouslySetInnerHTML={{ __html: html }} />
}

export function ChatInterface({ 
  config, 
  messages, 
  setMessages,
  onNewConversation,
  onDeleteConversation,
  activeConversationId
}: { 
  config: any
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  onNewConversation?: () => void
  onDeleteConversation?: (id: string) => void
  activeConversationId?: string | null
}) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [attachments, setAttachments] = useState<{name: string, content: string}[]>([])
  const [sendHistory, setSendHistory] = useState(true)
  const [selectedModel, setSelectedModel] = useState<string>("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (viewportRef.current) {
      viewportRef.current.scrollTop = viewportRef.current.scrollHeight
    }
  }, [messages, loading])

  // Set default for history based on model
  useEffect(() => {
    if (selectedModel && selectedModel.toLowerCase().includes('ollama')) {
      setSendHistory(false); // Default to no history for Ollama models
    } else {
      setSendHistory(true); // Default to with history for others
    }
  }, [selectedModel]);

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return
    
    let content = input
    let displayContent = input
    
    if (attachments.length > 0) {
      const context = attachments.map(a => `--- File: ${a.name} ---\n${a.content}\n--- End File ---`).join('\n\n')
      content = `Context:\n${context}\n\nQuestion: ${input}`
      displayContent = `${input}\n\n[Attached: ${attachments.map(a => a.name).join(', ')}]`
    }

    const userMsg: Message = { role: 'user', content, displayContent, timestamp: Date.now() }
    setMessages(prev => {
      const updated = [...prev, userMsg]
      // If this is the first message and onNewConversation exists, it means we need to create a conversation
      // This will be handled by the parent component's useEffect that watches messages
      return updated
    })
    setInput('')
    setAttachments([])
    setLoading(true)

    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

    const messagesToSend = sendHistory ? [...messages, userMsg] : [userMsg];

    try {
      const res = await fetch(`http://${host}:${port}/${base}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messagesToSend.map(m => ({ role: m.role, content: m.content })),
          model: selectedModel || config?.model || undefined,
          temperature: config?.temperature ? parseFloat(config.temperature) : undefined,
          config: {
            top_p: config?.topP ? parseFloat(config.topP) : undefined,
            top_k: config?.topK ? parseInt(config.topK) : undefined,
            repeat_penalty: config?.repeatPenalty ? parseFloat(config.repeatPenalty) : undefined,
            seed: config?.seed ? parseInt(config.seed) : undefined,
            num_ctx: config?.numCtx ? parseInt(config.numCtx) : undefined,
          }
        })
      })

      if (!res.ok) {
        let errorMsg = 'Chat failed'
        try {
          const errData = await res.json()
          errorMsg = errData.detail || errData.error || errorMsg
        } catch (e) {}
        throw new Error(errorMsg)
      }
      
      const data = await res.json()
      
      if (data.error) {
        let content = `Error: ${data.error}`
        if (data.error.includes("Not authenticated") || data.error.includes("credentials")) {
           content += `\n\n⚠️ **Authentication Required**\n\nPlease [click here to re-authenticate](http://${host}:${port}/${base}/auth/login) with Google.`
        }
        const botMsg: Message = { role: 'assistant', content, timestamp: Date.now() }
        setMessages(prev => [...prev, botMsg])
        return
      }

      let responseContent = data.content || data.answer || "No response";
      const sources = data.sources || [];

      // Defensive parsing to handle raw ModelResponse object string
      if (typeof responseContent === 'string' && responseContent.startsWith('ModelResponse(')) {
        const match = responseContent.match(/message=Message\(content='([^']*)'/);
        if (match && match[1]) {
          // It's a bit of a hack, but we can un-escape the string
          responseContent = match[1].replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\"/g, '"');
        }
      }

      const botMsg: Message = { 
        role: 'assistant', 
        content: responseContent,
        sources: sources.length > 0 ? sources : undefined,
        timestamp: Date.now() 
      }
      setMessages(prev => [...prev, botMsg])
    } catch (e: any) {
      console.error(e)
      toast.error("Failed to send message")
      
      let errorText = "Error: Could not connect to backend."
      let isAuthIssue = false
      
      if (e.message && e.message !== "Failed to fetch") {
        errorText = `Error: ${e.message}`
        if (e.message.includes("401") || e.message.includes("403") || e.message.toLowerCase().includes("auth")) {
            isAuthIssue = true
        }
      } else {
        // Network error usually
        errorText = "Error: Could not connect to the backend server. Please ensure the server is running."
      }

      const authUrl = `http://${host}:${port}/${base}/auth/login`
      const helpText = isAuthIssue 
        ? `\n\nIt looks like you need to sign in. [Click here to Authenticate](${authUrl})`
        : `\n\nIf this persists, you may need to [Re-authenticate](${authUrl}).`

      setMessages(prev => [...prev, { role: 'assistant', content: errorText + helpText, timestamp: Date.now() }])
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

      const formData = new FormData()
      formData.append('file', file)

      const toastId = toast.loading(`Extracting text from ${file.name}...`)

      try {
        const res = await fetch(`http://${host}:${port}/${base}/extract`, {
          method: 'POST',
          body: formData
        })

        if (!res.ok) throw new Error('Extraction failed')

        const data = await res.json()
        setAttachments(prev => [...prev, { name: file.name, content: data.text }])
        toast.success(`Attached ${file.name}`, { id: toastId })
      } catch (e) {
        toast.error(`Failed to attach ${file.name}`, { id: toastId })
      }
      
      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleDownload = (content: string) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chat-response-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const deleteMessage = (index: number) => {
    setMessages(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="flex flex-col h-[600px] border rounded-lg bg-background">
      <div className="flex items-center justify-between p-3 border-b bg-muted/30">
        <div className="text-sm font-medium text-muted-foreground">Conversation</div>
        <div className="flex items-center gap-2">
          {onNewConversation && (
            <Button variant="ghost" size="icon" onClick={onNewConversation} title="New conversation">
              <Plus className="h-4 w-4" />
            </Button>
          )}
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => {
              if (onDeleteConversation && activeConversationId) {
                onDeleteConversation(activeConversationId)
              } else {
                setMessages([])
              }
            }} 
            title={onDeleteConversation && activeConversationId ? "Delete conversation" : "Clear conversation"}
          >
            <Trash className="h-4 w-4" />
          </Button>
          <ModelSelector config={config} onModelSelect={setSelectedModel} />
        </div>
      </div>
      <ScrollArea className="flex-1 p-4" ref={scrollRef} viewportRef={viewportRef}>
        <div className="space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground opacity-50">
              <Bot className="h-12 w-12 mb-2" />
              <p>Start a conversation...</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'} group`}>
              {m.role === 'assistant' && (
                <Avatar className="h-8 w-8">
                  <AvatarFallback><Bot className="h-4 w-4" /></AvatarFallback>
                </Avatar>
              )}
              
              {m.role === 'user' && (
                 <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-destructive" onClick={() => deleteMessage(i)} title="Delete message">
                        <Trash className="h-3 w-3" />
                    </Button>
                 </div>
              )}

              <div className={`max-w-[80%] rounded-lg p-3 ${
                m.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'
              }`}>
                <MarkdownRenderer content={m.displayContent || m.content} sources={m.sources} />
                {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-border">
                    <div className="text-xs font-semibold text-muted-foreground mb-1">Sources:</div>
                    <div className="space-y-1">
                      {m.sources.map((source, idx) => (
                        <div key={idx} className="text-xs text-muted-foreground">
                          <span className="font-mono">[{idx + 1}]</span>{' '}
                          <span className="hover:text-foreground transition-colors">{source}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {m.role === 'assistant' && (
                  <div className="mt-2 flex justify-end gap-1">
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => handleDownload(m.content)} title="Download response">
                      <Download className="h-3 w-3" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6 hover:text-destructive" onClick={() => deleteMessage(i)} title="Delete message">
                      <Trash className="h-3 w-3" />
                    </Button>
                  </div>
                )}
              </div>
              {m.role === 'user' && (
                <Avatar className="h-8 w-8">
                  <AvatarFallback><User className="h-4 w-4" /></AvatarFallback>
                </Avatar>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <Avatar className="h-8 w-8">
                <AvatarFallback><Bot className="h-4 w-4" /></AvatarFallback>
              </Avatar>
              <div className="bg-muted rounded-lg p-3">
                <div className="flex gap-1">
                  <span className="animate-bounce">.</span>
                  <span className="animate-bounce delay-100">.</span>
                  <span className="animate-bounce delay-200">.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      <div className="p-4 border-t bg-card">
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {attachments.map((a, i) => (
              <div key={i} className="flex items-center gap-1 bg-muted px-2 py-1 rounded text-xs">
                <span>{a.name}</span>
                <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}>
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="flex gap-2">
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            onChange={handleUpload} 
          />
          <Button variant="outline" size="icon" onClick={() => fileInputRef.current?.click()} disabled={loading}>
            <Paperclip className="h-4 w-4" />
          </Button>
          <textarea
            className="flex-1 min-h-[40px] max-h-[120px] rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={loading}
          />
          <Button onClick={handleSend} disabled={loading || (!input.trim() && attachments.length === 0)} size="icon">
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center space-x-2 mt-3">
          <Checkbox id="send-history" checked={sendHistory} onCheckedChange={(checked) => setSendHistory(Boolean(checked))} />
          <label
            htmlFor="send-history"
            className="text-xs font-medium leading-none text-muted-foreground peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            Send conversation history
          </label>
        </div>
      </div>
    </div>
  )
}
