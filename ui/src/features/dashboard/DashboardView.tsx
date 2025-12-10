
import {
    Activity,
    Database,
    MessageCircle,
    Cpu,
    ArrowRight,
    TrendingUp,
    AlertTriangle
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface DashboardViewProps {
    systemStatus: 'running' | 'stopped' | 'error' | 'warning'
    serviceStatuses: Record<string, 'running' | 'stopped' | 'error' | 'warning'>
    stats: {
        documents: number
        queriesToday: number
        avgLatency: number
    }
    jobProgress?: {
        visible: boolean
        total: number
        completed: number
        failed: number
    }
    onNavigate: (tab: string) => void
    config: any // Kept for potential future use or broader compatibility, though not strictly used in loop anymore
}

export function DashboardView({
    systemStatus,
    serviceStatuses,
    stats,
    jobProgress,
    onNavigate,
    config
}: DashboardViewProps) {

    return (
        <div className="space-y-8 max-w-7xl mx-auto">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">System Overview</h2>
                    <p className="text-muted-foreground mt-1">Real-time metrics and service status.</p>
                </div>
                <div className="flex items-center gap-4">
                    {/* ProviderSelector removed from here, moved to Sidebar */}
                    {systemStatus === 'error' && (
                        <Badge variant="destructive" className="px-3 py-1">
                            <AlertTriangle className="w-3 h-3 mr-1" /> System Errors Detected
                        </Badge>
                    )}
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="glass-card transition-all hover:scale-[1.01] hover:shadow-2xl">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
                        <Database className="h-4 w-4 text-primary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats.documents.toLocaleString()}</div>
                        <p className="text-xs text-muted-foreground mt-1">Indexed in vector store</p>
                    </CardContent>
                </Card>

                <Card className="glass-card transition-all hover:scale-[1.01] hover:shadow-2xl">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Queries Today</CardTitle>
                        <MessageCircle className="h-4 w-4 text-blue-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats.queriesToday.toLocaleString()}</div>
                        <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                            {/* Removed hardcoded trend */}
                            Since midnight
                        </p>
                    </CardContent>
                </Card>

                <Card className="glass-card transition-all hover:scale-[1.01] hover:shadow-2xl">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
                        <Activity className="h-4 w-4 text-orange-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats.avgLatency.toFixed(0)}ms</div>
                        <p className="text-xs text-muted-foreground mt-1">Per inference request</p>
                    </CardContent>
                </Card>

                <Card className="glass-card transition-all hover:scale-[1.01] hover:shadow-2xl">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">System Services</CardTitle>
                        <Cpu className="h-4 w-4 text-purple-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold flex items-center gap-2">
                            {Object.values(serviceStatuses).filter(s => s === 'running').length}
                            <span className="text-muted-foreground text-sm font-normal">/ {Object.keys(serviceStatuses).length}</span>
                        </div>
                        <div className="flex gap-1 mt-2">
                            {Object.entries(serviceStatuses).map(([name, status]) => (
                                <div
                                    key={name}
                                    className={`h-1.5 w-full rounded-full ${status === 'running' ? 'bg-green-500/50' :
                                        status === 'warning' ? 'bg-yellow-500/50' : 'bg-red-500/50'
                                        }`}
                                    title={`${name}: ${status}`}
                                />
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Indexing Progress */}
            {jobProgress && jobProgress.visible && (
                <Card className="glass-card border-l-4 border-l-blue-500 animate-in slide-in-from-left-2 fade-in">
                    <CardContent className="pt-6">
                        <h3 className="font-semibold text-sm mb-3 flex justify-between">
                            <span className="flex items-center gap-2"><Database className="w-4 h-4 text-blue-500 animate-pulse" /> Indexing in progress...</span>
                            <span className="text-muted-foreground">{jobProgress.completed} / {jobProgress.total}</span>
                        </h3>
                        <Progress
                            value={(jobProgress.completed / jobProgress.total) * 100}
                            className="h-2"
                        />
                    </CardContent>
                </Card>
            )}

            {/* Quick Actions */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div className="col-span-2 group relative overflow-hidden rounded-xl border bg-gradient-to-br from-primary/5 via-primary/0 to-transparent p-6 hover:border-primary/50 transition-colors">
                    <div className="relative z-10">
                        <h3 className="text-xl font-bold mb-2">Start Searching</h3>
                        <p className="text-muted-foreground mb-6 max-w-lg">
                            Query your indexed documents using the latest RAG models. Switch between chat and search modes seamlessly.
                        </p>
                        <Button onClick={() => onNavigate('search')}>
                            Go to Search <ArrowRight className="ml-2 w-4 h-4" />
                        </Button>
                    </div>
                    <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                        <SearchIcon className="w-32 h-32" />
                    </div>
                </div>

                <Card className="glass-card flex flex-col justify-center gap-4 p-6">
                    <h3 className="font-semibold">System Controls</h3>
                    <div className="space-y-2">
                        <Button variant="outline" className="w-full justify-start" onClick={() => onNavigate('files')}>
                            <Database className="mr-2 w-4 h-4" /> Manage Data
                        </Button>
                        <Button variant="outline" className="w-full justify-start" onClick={() => onNavigate('settings')}>
                            <Cpu className="mr-2 w-4 h-4" /> Configure Models
                        </Button>
                    </div>
                </Card>
            </div>
        </div>
    )
}

function SearchIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
        </svg>
    )
}
