import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { BookOpen, Search, Loader2, FileText, AlertCircle, Lightbulb, Settings } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { marked } from 'marked'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'

interface DocMetadata {
    filename: string
    title: string
}

export function HelpView() {
    const [docs, setDocs] = useState<DocMetadata[]>([])
    const [loading, setLoading] = useState(true)
    const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
    const [renderedContent, setRenderedContent] = useState<string>('')
    const [contentLoading, setContentLoading] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetch('/api/docs/list')
            .then(res => res.json())
            .then(data => {
                setDocs(data.documents || [])
                setLoading(false)
            })
            .catch(err => {
                console.error("Failed to load docs", err)
                setError("Failed to load documentation index.")
                setLoading(false)
            })
    }, [])

    useEffect(() => {
        if (!selectedDoc) {
            setRenderedContent('')
            return
        }

        setContentLoading(true)
        fetch(`/api/docs/content/${selectedDoc}`)
            .then(res => {
                if (!res.ok) throw new Error("Failed to load content")
                return res.text()
            })
            .then(async text => {
                try {
                    // marked.parse can be async in newer versions
                    const html = await marked.parse(text)
                    setRenderedContent(html)
                } catch (e) {
                    console.error("Markdown parse error", e)
                    setRenderedContent("<p>Error parsing document.</p>")
                }
                setContentLoading(false)
            })
            .catch(err => {
                setRenderedContent("<p>Error loading document content.</p>")
                setContentLoading(false)
            })
    }, [selectedDoc])

    const filteredDocs = docs.filter(d =>
        d.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        d.filename.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return (
        <div className="h-[calc(100vh-8rem)] min-h-[500px] flex flex-col gap-6 animate-in fade-in duration-500">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-bold tracking-tight">Documentation & Help</h2>
                <p className="text-muted-foreground">Browse system documentation and guides.</p>
            </div>

            <Tabs defaultValue="articles" className="h-full flex flex-col">
                <TabsList>
                    <TabsTrigger value="articles" className="flex items-center gap-2">
                        <BookOpen className="w-4 h-4" />
                        Articles
                    </TabsTrigger>
                    <TabsTrigger value="tips" className="flex items-center gap-2">
                        <Lightbulb className="w-4 h-4" />
                        Quick Tips
                    </TabsTrigger>
                </TabsList>

                <TabsContent value="articles" className="flex-1 mt-4 min-h-0">
                    <div className="grid grid-cols-12 gap-6 h-full">
                        {/* Sidebar List */}
                        <Card className="col-span-12 md:col-span-4 h-full flex flex-col glass-card">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <BookOpen className="w-4 h-4" />
                                    Articles
                                </CardTitle>
                                <div className="relative">
                                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                                    <Input
                                        placeholder="Search docs..."
                                        className="pl-9"
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                    />
                                </div>
                            </CardHeader>
                            <CardContent className="flex-1 p-0 overflow-hidden">
                                <ScrollArea className="h-full px-4 pb-4">
                                    {loading ? (
                                        <div className="flex justify-center p-4">
                                            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                                        </div>
                                    ) : error ? (
                                        <div className="p-4 text-sm text-destructive flex items-center gap-2">
                                            <AlertCircle className="w-4 h-4" />
                                            {error}
                                        </div>
                                    ) : filteredDocs.length === 0 ? (
                                        <p className="p-4 text-sm text-muted-foreground text-center">No documents found.</p>
                                    ) : (
                                        <div className="space-y-1">
                                            {filteredDocs.map(doc => (
                                                <button
                                                    key={doc.filename}
                                                    onClick={() => setSelectedDoc(doc.filename)}
                                                    className={`w-full text-left px-3 py-2.5 rounded-md text-sm transition-colors flex items-center justify-between group ${selectedDoc === doc.filename
                                                        ? 'bg-secondary text-secondary-foreground font-medium'
                                                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                                                        }`}
                                                >
                                                    <span className="truncate pr-2">{doc.title}</span>
                                                    {selectedDoc === doc.filename && <Badge variant="outline" className="text-[10px] h-5 px-1.5 shadow-none bg-background/50">Active</Badge>}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </ScrollArea>
                            </CardContent>
                        </Card>

                        {/* Content Viewer */}
                        <Card className="col-span-12 md:col-span-8 h-full flex flex-col glass-card overflow-hidden">
                            <CardHeader className="pb-4 border-b">
                                <div className="flex items-center justify-between">
                                    <CardTitle className="text-xl flex items-center gap-2">
                                        {selectedDoc ? (
                                            <>
                                                <FileText className="w-5 h-5 text-primary" />
                                                {docs.find(d => d.filename === selectedDoc)?.title || selectedDoc}
                                            </>
                                        ) : (
                                            <span className="text-muted-foreground">Select a document</span>
                                        )}
                                    </CardTitle>
                                    {selectedDoc && (
                                        <Badge variant="secondary" className="font-mono text-xs">
                                            {selectedDoc}
                                        </Badge>
                                    )}
                                </div>
                            </CardHeader>
                            <div className="flex-1 overflow-y-auto p-6">
                                {contentLoading ? (
                                    <div className="flex items-center justify-center h-40">
                                        <Loader2 className="w-8 h-8 animate-spin text-primary/50" />
                                    </div>
                                ) : selectedDoc ? (
                                    <div
                                        className="text-foreground max-w-none w-full break-words
                                        [&>*:first-child]:mt-0
                                        [&_h1]:hidden
                                        [&_h2]:text-2xl [&_h2]:font-semibold [&_h2]:mt-8 [&_h2]:mb-4 [&_h2]:pb-1 [&_h2]:border-b
                                        [&_h3]:text-xl [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-3
                                        [&_p]:leading-7 [&_p]:mb-4
                                        [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:mb-4
                                        [&_ol]:list-decimal [&_ol]:pl-6 [&_ol]:mb-4
                                        [&_li]:mb-1
                                        [&_code]:bg-muted [&_code]:px-[0.3rem] [&_code]:py-[0.2rem] [&_code]:rounded [&_code]:font-mono [&_code]:text-sm
                                        [&_pre]:bg-muted [&_pre]:p-4 [&_pre]:rounded-lg [&_pre]:mb-4 [&_pre]:overflow-x-auto
                                        [&_pre_code]:bg-transparent [&_pre_code]:p-0
                                        [&_a]:text-primary [&_a]:underline-offset-4 hover:[&_a]:underline
                                        [&_blockquote]:border-l-4 [&_blockquote]:border-primary [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:my-4"
                                        dangerouslySetInnerHTML={{ __html: renderedContent }}
                                        onClick={(e) => {
                                            const link = (e.target as HTMLElement).closest('a');
                                            if (!link) return;

                                            const href = link.getAttribute('href');
                                            if (!href) return;

                                            // Handle external links
                                            if (href.startsWith('http') || href.startsWith('https')) {
                                                link.target = "_blank";
                                                link.rel = "noopener noreferrer";
                                                return;
                                            }

                                            // Handle internal markdown links
                                            if (href.endsWith('.md') || !href.includes('.')) {
                                                e.preventDefault();
                                                // clean filename (simple check)
                                                const filename = href.split('/').pop() || href;
                                                setSelectedDoc(filename);
                                            }
                                        }}
                                    />
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-64 text-muted-foreground gap-4">
                                        <BookOpen className="w-12 h-12 opacity-20" />
                                        <p>Select an article from the sidebar to start reading.</p>
                                    </div>
                                )}
                            </div>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="tips" className="mt-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="grid gap-6 md:grid-cols-2">
                        <Card className="glass-card md:col-span-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Lightbulb className="w-5 h-5 text-yellow-500" />
                                    Getting Started
                                </CardTitle>
                                <CardDescription>Essential concepts to understand.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Accordion type="single" collapsible className="w-full">
                                    <AccordionItem value="item-1">
                                        <AccordionTrigger>How RAG Works</AccordionTrigger>
                                        <AccordionContent className="text-muted-foreground leading-relaxed">
                                            Retrieval-Augmented Generation (RAG) combines your documents with AI.
                                            When you ask a question, the system first <strong>searches</strong> your indexed files for relevant snippets,
                                            then sends those snippets to the AI to generate a <strong>grounded answer</strong>.
                                            The AI effectively "reads" your documents before answering.
                                        </AccordionContent>
                                    </AccordionItem>
                                    <AccordionItem value="item-2">
                                        <AccordionTrigger>Supported File Types</AccordionTrigger>
                                        <AccordionContent className="text-muted-foreground leading-relaxed">
                                            <p className="mb-2">We support a wide range of document and data formats:</p>
                                            <ul className="list-disc list-inside space-y-1 mb-2 ml-1">
                                                <li><strong>Documents:</strong> PDF, DOCX, DOC, RTF, EPUB</li>
                                                <li><strong>Data:</strong> CSV, Excel (XLSX, XLS), PowerPoint (PPTX, PPT)</li>
                                                <li><strong>Text:</strong> TXT, Markdown, HTML</li>
                                                <li><strong>Images:</strong> PNG, JPG, TIFF, BMP, SVG (via OCR)</li>
                                            </ul>
                                            <p>For best results with PDFs, ensure they are text-selectable.</p>
                                        </AccordionContent>
                                    </AccordionItem>

                                    <AccordionItem value="item-3">
                                        <AccordionTrigger>Power User Features</AccordionTrigger>
                                        <AccordionContent className="text-muted-foreground leading-relaxed">
                                            <ul className="list-disc list-inside space-y-1">
                                                <li><strong>Shortcuts:</strong> Press <kbd className="px-1 py-0.5 bg-muted border rounded text-xs">Enter</kbd> to send, <kbd className="px-1 py-0.5 bg-muted border rounded text-xs">Shift+Enter</kbd> for new lines.</li>
                                                <li><strong>Context Control:</strong> Toggle "Send conversation history" off to save tokens or start fresh without clearing the chat.</li>
                                                <li><strong>Bulk Actions:</strong> In the sidebar, select multiple conversations to delete them at once.</li>
                                                <li><strong>Renaming:</strong> Click any conversation title to rename it for better organization.</li>
                                            </ul>
                                        </AccordionContent>
                                    </AccordionItem>
                                </Accordion>
                            </CardContent>
                        </Card>

                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Search className="w-5 h-5 text-blue-500" />
                                    Search Optimization
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <h4 className="font-medium text-sm text-foreground">Query Length Matters</h4>
                                    <p className="text-sm text-muted-foreground">
                                        The system <span className="text-primary font-semibold">autotunes</span> based on your query length:
                                    </p>
                                    <ul className="list-disc list-inside text-sm text-muted-foreground ml-2 space-y-1">
                                        <li><strong>Short (&lt;10 words):</strong> Modes to "Fact Lookup". Retrieves fewer, highly specific chunks.</li>
                                        <li><strong>Long (&gt;15 words):</strong> Modes to "Research". Retrieves more context for synthesis.</li>
                                    </ul>
                                </div>
                                <div className="space-y-2">
                                    <h4 className="font-medium text-sm text-foreground">Keywords vs. Natural Language</h4>
                                    <p className="text-sm text-muted-foreground">
                                        Use natural language questions (e.g., "How do I reset my password?") rather than just keywords.
                                        The underlying semantic search understands intent better than exact matches.
                                    </p>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Settings className="w-5 h-5 text-purple-500" />
                                    Advanced Configuration
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <h4 className="font-medium text-sm text-foreground">Model Selection</h4>
                                    <p className="text-sm text-muted-foreground">
                                        <strong>Llama 3.2</strong> is optimized for speed and works great on local MacBooks.
                                        For complex reasoning, switch to <strong>GPT-4o</strong> or <strong>Claude 3.5 Sonnet</strong> in settings if you have API keys.
                                    </p>
                                </div>
                                <div className="space-y-2">
                                    <h4 className="font-medium text-sm text-foreground">Re-indexing</h4>
                                    <p className="text-sm text-muted-foreground">
                                        If you add many files at once, the index updates automatically.
                                        However, if search results feel stale, use the "Flush Cache" button in the Files tab to force a full rebuild.
                                    </p>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>
        </div >
    )
}
