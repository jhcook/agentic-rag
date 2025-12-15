import { useState, useRef, useEffect } from 'react'
import { Send, User, Bot, Download, Paperclip, X, Trash2, Trash, Plus, Pencil } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { toast } from 'sonner'
import { marked } from 'marked'
import { ModelSelector } from '@/components/ModelSelector'

export type Message = {
  id?: string
  role: 'user' | 'assistant'
  content: string
  displayContent?: string
  sources?: string[]
  timestamp: number
}

function formatDateDivider(timestamp: number) {
  const date = new Date(timestamp)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  
  if (date.toDateString() === today.toDateString()) {
    return 'Today'
  }
  if (date.toDateString() === yesterday.toDateString()) {
    return 'Yesterday'
  }
  return new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'short', day: 'numeric' }).format(date)
}

function MarkdownRenderer({ content, sources }: { content: string, sources?: string[] }) {
  const [html, setHtml] = useState('')
  
  useEffect(() => {
    const parse = async () => {
       try {
         let processedContent = content
         
         // Remove "Sources:" section if it exists (it's displayed separately)
         const sourcesIndex = processedContent.indexOf('\n\nSources:')
         if (sourcesIndex !== -1) {
           processedContent = processedContent.substring(0, sourcesIndex).trim()
         }
         
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
  onDeleteConversations,
  onRenameConversation,
  activeConversationId,
  onSessionId
}: { 
  config: any
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  onNewConversation?: () => void
  onDeleteConversation?: (id: string) => void
  onDeleteConversations?: (ids: string[]) => void
  onRenameConversation?: (id: string, title: string) => void
  activeConversationId?: string | null
  onSessionId?: (sessionId: string) => void
}) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [attachments, setAttachments] = useState<{name: string, content: string}[]>([])
  const [sendHistory, setSendHistory] = useState(true)
  const [selectedModel, setSelectedModel] = useState<string>("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Keep a live reference to the currently active conversation.
  // This prevents in-flight responses from being appended to the wrong chat
  // if the user switches conversations while waiting.
  const activeConversationIdRef = useRef<string | null>(activeConversationId ?? null)
  useEffect(() => {
    activeConversationIdRef.current = activeConversationId ?? null
  }, [activeConversationId])

  const bottomRef = useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = useRef(true)

  const scrollToBottom = (behavior: ScrollBehavior = 'auto') => {
    bottomRef.current?.scrollIntoView({ behavior, block: 'end' })
  }

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return

    const onScroll = () => {
      const distanceFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
      shouldAutoScrollRef.current = distanceFromBottom < 48
    }

    viewport.addEventListener('scroll', onScroll, { passive: true })
    onScroll()
    return () => viewport.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return
    // Wait for DOM to paint new messages before scrolling
    requestAnimationFrame(() => scrollToBottom('auto'))
  }, [messages, loading])

  // Set default for history based on model
  useEffect(() => {
    if (selectedModel && selectedModel.toLowerCase().includes('ollama')) {
      setSendHistory(false); // Default to no history for Ollama models
    } else {
      setSendHistory(true); // Default to with history for others
    }
  }, [selectedModel]);

  // If the user switches conversations mid-request, don't show a global "loading"
  // state in the newly selected conversation.
  useEffect(() => {
    setLoading(false)
  }, [activeConversationId])

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return

    const conversationIdAtSend = activeConversationIdRef.current
    
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
          messages: messagesToSend.map(m => ({
            role: m.role,
            content: m.content,
            display_content: m.displayContent
          })),
          session_id: activeConversationId || undefined,
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

      // Track fallback for dual notifications (pre-response and reminder after render)
      let fallbackUsed = false
      let fallbackReason = ''
      if (data.fallback_used) {
        fallbackUsed = true
        fallbackReason = data.fallback_reason ? ` (${data.fallback_reason})` : ''
        toast.warning(`Cloud model unavailable; falling back to local model${fallbackReason}`)
        // Do not persist fallback metadata in chat state/history; popup is enough.
        delete data.fallback_used
        delete data.fallback_reason
      }

      if (data.session_id) {
        // Best-effort refresh of sidebar; does not force navigation.
        onSessionId?.(String(data.session_id))
      }

      const shouldAppendToCurrentConversation = (() => {
        const currentActive = activeConversationIdRef.current
        // Existing session: only append if we're still viewing it.
        if (conversationIdAtSend) return currentActive === conversationIdAtSend
        // New session (was null when sent): only append if user hasn't switched away.
        if (!currentActive) return true
        // Allow append if the app already switched to the newly created session.
        return Boolean(data.session_id) && String(data.session_id) === currentActive
      })()

      if (shouldAppendToCurrentConversation && data.user_message_id) {
        const persistedId = String(data.user_message_id)
        setMessages(prev => prev.map(m => {
          if (m.role !== 'user') return m
          if (m.timestamp !== userMsg.timestamp) return m
          return { ...m, id: persistedId }
        }))
      }
      
      if (data.error) {
        let content = `Error: ${data.error}`
        if (data.error.includes("Not authenticated") || data.error.includes("credentials")) {
           content += `\n\n⚠️ **Authentication Required**\n\nPlease [click here to re-authenticate](http://${host}:${port}/${base}/auth/login) with Google.`
        }
        const botMsg: Message = { role: 'assistant', content, timestamp: Date.now() }
        if (shouldAppendToCurrentConversation) {
          setMessages(prev => [...prev, botMsg])
        }
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
        id: data.assistant_message_id ? String(data.assistant_message_id) : undefined,
        sources: sources.length > 0 ? sources : undefined,
        timestamp: Date.now() 
      }
      if (shouldAppendToCurrentConversation) {
        setMessages(prev => [...prev, botMsg])
        if (fallbackUsed) {
          toast.info(`Response generated via local fallback${fallbackReason}`)
        }
      }
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

      // Only append local error text if the user is still viewing the conversation
      // where the request was initiated.
      if (activeConversationIdRef.current === conversationIdAtSend) {
        setMessages(prev => [...prev, { role: 'assistant', content: errorText + helpText, timestamp: Date.now() }])
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files)
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

      // Process all selected files
      const filePromises = files.map(async (file) => {
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
          toast.success(`Attached ${file.name}`, { id: toastId })
          return { name: file.name, content: data.text }
        } catch (e) {
          toast.error(`Failed to attach ${file.name}`, { id: toastId })
          return null
        }
      })

      // Wait for all files to be processed
      const results = await Promise.all(filePromises)
      const successfulAttachments = results.filter((r): r is {name: string, content: string} => r !== null)
      
      if (successfulAttachments.length > 0) {
        setAttachments(prev => [...prev, ...successfulAttachments])
        if (successfulAttachments.length > 1) {
          toast.success(`Attached ${successfulAttachments.length} files`)
        }
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

  const deleteMessage = async (index: number, messageId?: string) => {
    // If we don't have persistence identifiers, this is a local-only delete.
    if (!messageId || !activeConversationId) {
      setMessages(prev => prev.filter((_, i) => i !== index))
      return
    }

    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

    try {
      const res = await fetch(
        `http://${host}:${port}/${base}/chat/history/${activeConversationId}/messages/${messageId}`,
        { method: 'DELETE' }
      )

      if (!res.ok) {
        if (res.status === 503) {
          throw new Error('Deletion is unavailable (chat storage not enabled on the server).')
        }
        let errorMsg = 'Failed to delete message'
        try {
          const errData = await res.json()
          errorMsg = errData.detail || errData.error || errorMsg
        } catch (e) {}
        throw new Error(errorMsg)
      }

      setMessages(prev => prev.filter((_, i) => i !== index))
    } catch (e: any) {
      console.error(e)
      toast.error(e?.message || 'Failed to delete message')
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0 border rounded-lg bg-background">
      <div className="flex items-center justify-between p-3 border-b bg-muted/30">
        <div className="text-sm font-medium text-muted-foreground">Conversation</div>
        <div className="flex items-center gap-2">
          {onNewConversation && (
            <Button variant="ghost" size="icon" onClick={onNewConversation} title="New conversation">
              <Plus className="h-4 w-4" />
            </Button>
          )}
          {onRenameConversation && activeConversationId && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                const currentTitle = messages.find(m => m.role === 'user')?.content?.slice(0, 50) || ''
                const newTitle = window.prompt('Rename conversation', currentTitle)
                if (newTitle && newTitle.trim()) {
                  onRenameConversation(activeConversationId, newTitle.trim())
                }
              }}
              title="Rename conversation"
            >
              <Pencil className="h-4 w-4" />
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
            <Trash2 className="h-4 w-4" />
          </Button>
          <ModelSelector config={config} onModelSelect={setSelectedModel} />
        </div>
      </div>
      <ScrollArea className="flex-1 min-h-0" ref={scrollRef} viewportRef={viewportRef}>
        <div className="p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground opacity-50">
              <Bot className="h-12 w-12 mb-2" />
              <p>Start a conversation...</p>
            </div>
          )}

          {messages.map((message, index) => {
            const isUser = message.role === 'user'
            const date = new Date(message.timestamp)
            const showDateDivider = index === 0 || 
              new Date(messages[index - 1].timestamp).toDateString() !== date.toDateString()

            return (
              <div key={index} className="group">
                {showDateDivider && (
                  <div className="flex items-center justify-center my-6 sticky top-0 z-10 pointer-events-none">
                     <span className="bg-muted/80 backdrop-blur-sm border border-border px-3 py-1 rounded-full text-xs font-medium text-muted-foreground shadow-sm">
                      {formatDateDivider(message.timestamp)}
                    </span>
                  </div>
                )}
                
                <div className={`flex gap-3 mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
                  {!isUser && (
                    <Avatar className="h-8 w-8 mt-1 shrink-0">
                      <AvatarFallback className="bg-primary/10 text-primary">
                        <Bot className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                  
                  <div className={`flex flex-col max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}>
                    <div className={`rounded-2xl px-4 py-3 shadow-sm ${
                      isUser 
                        ? 'bg-primary text-primary-foreground rounded-tr-sm' 
                        : 'bg-muted/50 border border-border rounded-tl-sm'
                    }`}>
                      <MarkdownRenderer content={message.displayContent || message.content} sources={message.sources} />
                      {!isUser && message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-border/50">
                          <div className="text-xs text-muted-foreground/70 space-y-1">
                            {message.sources.map((source, i) => (
                              <div key={i} className="truncate">
                                [{i + 1}] {source.split('/').pop()}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className={`flex items-center gap-1 mt-1 opacity-0 group-hover:opacity-100 transition-opacity ${isUser ? 'flex-row-reverse' : ''}`}>
                       <span className="text-[10px] text-muted-foreground px-1">
                        {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                       {!isUser && (
                          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => handleDownload(message.content)} title="Download response">
                            <Download className="h-3 w-3" />
                          </Button>
                       )}
                       <Button variant="ghost" size="icon" className="h-6 w-6 text-destructive/50 hover:text-destructive" onClick={() => deleteMessage(index, message.id)} title="Delete message">
                         <Trash className="h-3 w-3" />
                       </Button>
                    </div>
                  </div>

                  {isUser && (
                    <Avatar className="h-8 w-8 mt-1 shrink-0">
                      <AvatarFallback className="bg-secondary text-secondary-foreground">
                        <User className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              </div>
            )
          })}
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
          <div ref={bottomRef} />
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
            multiple
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
