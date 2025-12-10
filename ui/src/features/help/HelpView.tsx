import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Lightbulb, Keyboard, FileText, Settings, Search } from 'lucide-react'

export function HelpView() {
    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-bold tracking-tight">Help & Tips</h2>
                <p className="text-muted-foreground">Master the system with these pro tips and tricks.</p>
            </div>

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
                                    We support <strong>PDF, DOCX, TXT, and Markdown</strong> files.
                                    For best results, ensure your PDFs are text-selectable (not scanned images).
                                    Images and scanned documents require OCR which is not currently enabled for speed.
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
        </div>
    )
}
