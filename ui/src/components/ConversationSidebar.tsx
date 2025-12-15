import { useState, useEffect } from 'react'
import { MessageSquare, Plus, Trash2, X, Pencil } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { Message } from './ChatInterface'
import { Checkbox } from '@/components/ui/checkbox'

export type Conversation = {
  id: string
  title: string
  messages: Message[]
  createdAt: number
  updatedAt: number
}

interface ConversationSidebarProps {
  conversations: Conversation[]
  activeConversationId: string | null
  onSelectConversation: (id: string) => void
  onDeleteConversation: (id: string) => void
  onDeleteConversations?: (ids: string[]) => void
  onRenameConversation?: (id: string, title: string) => void
  onNewConversation: () => void
  onClose?: () => void
  isOpen: boolean
}

export function ConversationSidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onDeleteConversations,
  onRenameConversation,
  onNewConversation,
  onClose,
  isOpen
}: ConversationSidebarProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState<string>('')

  useEffect(() => {
    // Drop selections for conversations that no longer exist
    const existing = new Set(conversations.map(c => c.id))
    setSelectedIds(prev => {
      const next = new Set<string>()
      prev.forEach(id => { if (existing.has(id)) next.add(id) })
      return next
    })
  }, [conversations])

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const getConversationTitle = (conv: Conversation) => {
    if (conv.title && conv.title !== 'New Conversation') {
      return conv.title
    }
    // Generate title from first user message
    const firstUserMessage = conv.messages.find(m => m.role === 'user')
    if (firstUserMessage) {
      const content = firstUserMessage.content || firstUserMessage.displayContent || ''
      const preview = content.substring(0, 50).trim()
      return preview || 'New Conversation'
    }
    return 'New Conversation'
  }

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    
    // Reset time to start of day for accurate day comparison
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    const messageDate = new Date(date.getFullYear(), date.getMonth(), date.getDate())
    const diffTime = today.getTime() - messageDate.getTime()
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24))
    
    const time = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    
    if (diffDays === 0) {
      return `Today ${time}`
    } else if (diffDays === 1) {
      return `Yesterday ${time}`
    } else if (diffDays < 7) {
      const dayName = date.toLocaleDateString(undefined, { weekday: 'long' })
      return `${dayName} ${time}`
    } else {
      const dateStr = date.toLocaleDateString(undefined, { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      })
      return `${dateStr} ${time}`
    }
  }

  return (
    <div
      className={cn(
        'fixed left-0 top-0 h-full w-64 bg-background border-r border-border z-50 transition-transform duration-300 ease-in-out',
        isOpen ? 'translate-x-0' : '-translate-x-full',
        'md:relative md:translate-x-0 md:z-auto'
      )}
    >
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Conversations</h2>
          <div className="flex items-center gap-1">
            {onDeleteConversations && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                title="Delete selected"
                disabled={selectedIds.size === 0}
                onClick={() => {
                  if (selectedIds.size === 0) return
                  onDeleteConversations(Array.from(selectedIds))
                  setSelectedIds(new Set())
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onNewConversation}
              title="New conversation"
            >
              <Plus className="h-4 w-4" />
            </Button>
            {onClose && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 md:hidden"
                onClick={onClose}
                title="Close sidebar"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground">
                <MessageSquare className="h-12 w-12 mb-4 opacity-50" />
                <p className="text-sm">No conversations yet</p>
                <p className="text-xs mt-1">Start a new conversation to begin</p>
              </div>
            ) : (
              conversations
                .sort((a, b) => b.updatedAt - a.updatedAt)
                .map((conv) => (
                  <div
                    key={conv.id}
                    className={cn(
                      'group relative flex items-center gap-2 p-2 pr-12 rounded-lg cursor-pointer transition-colors',
                      activeConversationId === conv.id
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-muted'
                    )}
                    onClick={() => onSelectConversation(conv.id)}
                  >
                    <Checkbox
                      checked={selectedIds.has(conv.id)}
                      onCheckedChange={() => toggleSelect(conv.id)}
                      className={cn(
                        'h-4 w-4',
                        activeConversationId === conv.id
                          ? 'border-primary-foreground data-[state=checked]:bg-primary-foreground'
                          : ''
                      )}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    <div className="flex-1 min-w-0">
                      {editingId === conv.id ? (
                        <input
                          autoFocus
                          className="w-full text-sm font-medium bg-transparent border-b border-border focus:outline-none focus:border-primary text-foreground"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onBlur={() => {
                            const newTitle = editingTitle.trim()
                            if (newTitle && onRenameConversation) {
                              onRenameConversation(conv.id, newTitle)
                            }
                            setEditingId(null)
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              const newTitle = editingTitle.trim()
                              if (newTitle && onRenameConversation) {
                                onRenameConversation(conv.id, newTitle)
                              }
                              setEditingId(null)
                            }
                            if (e.key === 'Escape') {
                              setEditingId(null)
                            }
                          }}
                        />
                      ) : (
                        <>
                          <p className="text-sm font-medium truncate">
                            {getConversationTitle(conv)}
                          </p>
                          <p className={cn(
                            'text-xs truncate',
                            activeConversationId === conv.id
                              ? 'text-primary-foreground/70'
                              : 'text-muted-foreground'
                          )}>
                            {formatDate(conv.updatedAt)}
                          </p>
                        </>
                      )}
                    </div>
                    <div className="flex items-center gap-1 opacity-70 group-hover:opacity-100 transition-opacity absolute right-2 top-1/2 -translate-y-1/2">
                      {onRenameConversation && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className={cn(
                            'h-6 w-6 shrink-0',
                            activeConversationId === conv.id
                              ? 'text-primary-foreground hover:bg-primary-foreground/20'
                              : ''
                          )}
                          onClick={(e) => {
                            e.stopPropagation()
                            setEditingId(conv.id)
                            setEditingTitle(getConversationTitle(conv))
                          }}
                          title="Rename conversation"
                        >
                          <Pencil className="h-3 w-3" />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className={cn(
                          'h-6 w-6 shrink-0',
                          activeConversationId === conv.id
                            ? 'text-primary-foreground hover:bg-primary-foreground/20'
                            : ''
                        )}
                        onClick={(e) => {
                          e.stopPropagation()
                          onDeleteConversation(conv.id)
                        }}
                        title="Delete conversation"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
