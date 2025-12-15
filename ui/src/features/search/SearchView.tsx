import { useState, useRef, useEffect, Dispatch, SetStateAction } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Search, Loader2, XCircle, Send, MessageSquare } from 'lucide-react'
import { ChatInterface, Message } from '@/components/ChatInterface'
import { ConversationSidebar, Conversation } from '@/components/ConversationSidebar'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'

interface SearchViewProps {
    // Search Props
    queryText: string
    onQueryTextChange: (text: string) => void
    onSearch: () => void
    onCancelSearch: () => void
    searching: boolean
    searchError: string | null
    searchAnswer: string | null
    searchSources: string[]
    searchMessage: string | null

    // Chat Props
    config: any
    chatMessages: Message[]
    setMessages: Dispatch<SetStateAction<Message[]>>
    activeConversationId: string | null
    onSessionId?: (id: string) => void
    conversations: Conversation[]
    onSelectConversation: (id: string) => void
    onNewConversation: () => void
    onDeleteConversation: (id: string) => void
    onRenameConversation: (id: string, title: string) => void

    isSidebarOpen: boolean
    onToggleSidebar: () => void
}

export function SearchView(props: SearchViewProps) {
    const {
        queryText,
        onQueryTextChange,
        onSearch,
        onCancelSearch,
        searching,
        searchError,
        searchAnswer,
        searchSources,
        searchMessage,
        config,
        chatMessages,
        setMessages,
        activeConversationId,
        onSessionId,
        conversations,
        onSelectConversation,
        onNewConversation,
        onDeleteConversation,
        onRenameConversation,
        isSidebarOpen,
        onToggleSidebar
    } = props

    const [mode, setMode] = useState<'search' | 'chat'>('chat')

    return (
        <div className="flex h-full gap-6">
            {/* Chat Sidebar (Conditional) */}
            {mode === 'chat' && (
                <div className={`hidden md:block w-72 shrink-0 glass-card rounded-xl overflow-hidden h-[calc(100vh-6rem)] sticky top-6 transition-all duration-300`}>
                    <ConversationSidebar
                        conversations={conversations}
                        activeConversationId={activeConversationId}
                        onSelectConversation={onSelectConversation}
                        onNewConversation={onNewConversation}
                        onDeleteConversation={onDeleteConversation}
                        onRenameConversation={onRenameConversation}
                        isOpen={true} // Always open on desktop in this layout
                        onClose={() => { }}
                    />
                </div>
            )}

            <div className="flex-1 min-w-0 space-y-6">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-3xl font-bold tracking-tight">
                        {mode === 'search' ? 'Knowledge Search' : 'AI Chat'}
                    </h2>
                    <div className="bg-secondary/50 p-1 rounded-lg border border-white/5 flex gap-1">
                        <Button
                            variant={mode === 'search' ? 'secondary' : 'ghost'}
                            size="sm"
                            onClick={() => setMode('search')}
                            className="gap-2"
                        >
                            <Search className="w-4 h-4" /> Search
                        </Button>
                        <Button
                            variant={mode === 'chat' ? 'secondary' : 'ghost'}
                            size="sm"
                            onClick={() => setMode('chat')}
                            className="gap-2"
                        >
                            <MessageSquare className="w-4 h-4" /> Chat
                        </Button>
                    </div>
                </div>

                {/* SEARCH INTERFACE */}
                {mode === 'search' && (
                    <div className="space-y-6 animate-in fade-in duration-300">
                        <div className="relative">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground w-5 h-5" />
                            <Input
                                className="pl-12 h-14 text-lg bg-card/50 backdrop-blur border-white/10 shadow-lg rounded-xl focus-visible:ring-primary/50"
                                placeholder="Ask a question about your documents..."
                                value={queryText}
                                onChange={(e) => onQueryTextChange(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && onSearch()}
                            />
                            <div className="absolute right-2 top-2 bottom-2">
                                {searching ? (
                                    <Button variant="ghost" size="icon" onClick={onCancelSearch}>
                                        <Loader2 className="w-5 h-5 animate-spin text-primary" />
                                    </Button>
                                ) : (
                                    <Button size="icon" onClick={onSearch} className="h-10 w-10 rounded-lg shadow-lg shadow-primary/20">
                                        <Send className="w-4 h-4" />
                                    </Button>
                                )}
                            </div>
                        </div>

                        {searchError && (
                            <div className="p-4 rounded-lg bg-destructive/10 text-destructive border border-destructive/20 flex items-center gap-2">
                                <XCircle className="w-5 h-5" />
                                {searchError}
                            </div>
                        )}

                        {searchMessage && !searchAnswer && !searchError && (
                            <div className="text-center text-muted-foreground py-8">
                                <Loader2 className="w-8 h-8 animate-spin mx-auto mb-3 opacity-50" />
                                <p>{searchMessage}</p>
                            </div>
                        )}

                        {searchAnswer && (
                            <Card className="glass-card overflow-hidden animate-in slide-in-from-bottom-4 duration-500">
                                <CardContent className="p-6 md:p-8 space-y-6">
                                    <div className="prose prose-invert max-w-none">
                                        <div className="whitespace-pre-wrap leading-relaxed text-lg/relaxed text-foreground/90">
                                            {searchAnswer}
                                        </div>
                                    </div>

                                    {searchSources.length > 0 && (
                                        <div className="pt-6 border-t border-white/5">
                                            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Sources</h4>
                                            <div className="flex flex-wrap gap-2">
                                                {searchSources.map((source, i) => (
                                                    <Badge variant="secondary" key={i} className="bg-secondary/50 hover:bg-secondary transition-colors cursor-pointer px-3 py-1.5">
                                                        {source.split('/').pop()}
                                                    </Badge>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        )}
                    </div>
                )}

                {/* CHAT INTERFACE */}
                {mode === 'chat' && (
                    <div className="h-[calc(100vh-12rem)] glass-card rounded-xl overflow-hidden animate-in fade-in duration-300">
                        <ChatInterface
                            config={config}
                            messages={chatMessages}
                        setMessages={setMessages}
                        onNewConversation={onNewConversation}
                        onDeleteConversation={onDeleteConversation}
                        onRenameConversation={onRenameConversation}
                        activeConversationId={activeConversationId}
                        onSessionId={onSessionId}
                    />
                </div>
            )}
            </div>
        </div>
    )
}
