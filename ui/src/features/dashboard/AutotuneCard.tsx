import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Gauge, Sparkles, Activity } from 'lucide-react'
import { getApiBase } from '@/lib/api'

export function AutotuneCard() {
    const [testQuery, setTestQuery] = useState('')
    const [prediction, setPrediction] = useState<{
        predicted_k: number
        predicted_context: number
        mode: string
    } | null>(null)

    // Debounce the API call
    useEffect(() => {
        const timer = setTimeout(async () => {
            if (!testQuery.trim()) {
                setPrediction(null)
                return
            }

            try {
                const { host, port, base } = getApiBase()
                const res = await fetch(`http://${host}:${port}/${base}/config/autotune?query=${encodeURIComponent(testQuery)}`)
                if (res.ok) {
                    const data = await res.json()
                    setPrediction(data)
                }
            } catch (e) {
                console.error("Autotune fetch failed", e)
            }
        }, 500)

        return () => clearTimeout(timer)
    }, [testQuery])

    return (
        <Card className="glass-card overflow-hidden">
            <CardHeader className="pb-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-b border-white/5">
                <CardTitle className="flex items-center gap-2 text-lg">
                    <Sparkles className="w-5 h-5 text-purple-400 animate-pulse" />
                    Query Autotune Diagnostics
                </CardTitle>
                <CardDescription>Visualize how the system optimizes your search parameters in real-time.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
                <div className="space-y-2">
                    <label className="text-sm font-medium text-muted-foreground">Test Query</label>
                    <Input
                        placeholder="Type a query to see optimization logic..."
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        className="bg-background/50 backdrop-blur"
                    />
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 rounded-xl bg-card/50 border border-white/5 flex flex-col items-center justify-center text-center gap-2 transition-all duration-300">
                        <Activity className="w-5 h-5 text-blue-400" />
                        <div className="text-2xl font-bold font-mono">
                            {prediction ? prediction.mode : '-'}
                        </div>
                        <div className="text-xs text-muted-foreground uppercase tracking-wider">Mode</div>
                    </div>

                    <div className="p-4 rounded-xl bg-card/50 border border-white/5 flex flex-col items-center justify-center text-center gap-2 transition-all duration-300">
                        <div className="text-3xl font-bold text-foreground">
                            {prediction ? prediction.predicted_k : '-'}
                        </div>
                        <div className="text-xs text-muted-foreground uppercase tracking-wider">Top K Chunks</div>
                    </div>

                    <div className="p-4 rounded-xl bg-card/50 border border-white/5 flex flex-col items-center justify-center text-center gap-2 transition-all duration-300">
                        <div className="text-3xl font-bold text-foreground">
                            {prediction ? (prediction.predicted_context / 1000).toFixed(1) + 'k' : '-'}
                        </div>
                        <div className="text-xs text-muted-foreground uppercase tracking-wider">Context Window</div>
                    </div>
                </div>

                {prediction && (
                    <div className="text-xs text-center text-muted-foreground animate-in fade-in slide-in-from-top-2">
                        {prediction.mode === 'Technical/Fact' && "System optimized for precision. Expecting short, specific answers."}
                        {prediction.mode === 'Complex/Research' && "System optimized for depth. Expecting to synthesize multiple sources."}
                        {prediction.mode === 'Balanced' && "System using standard balanced parameters."}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
