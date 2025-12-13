
import { ReactNode, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
    LayoutDashboard,
    MessageSquare,
    Settings,
    Database,
    Search,
    Zap,
    Activity,
    LogOut,
    TerminalSquare,
    HelpCircle,
    ChevronLeft,
    ChevronRight,
    PanelLeftClose,
    PanelLeftOpen
} from 'lucide-react'
import { cn } from '@/lib/utils'
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"

import { ProviderSelector } from '@/components/ProviderSelector'

interface SidebarItemProps {
    icon: ReactNode
    label: string
    isActive?: boolean
    onClick: () => void
    notifications?: number
    collapsed?: boolean
}

function SidebarItem({ icon, label, isActive, onClick, notifications, collapsed }: SidebarItemProps) {
    const content = (
        <Button
            variant="ghost"
            className={cn(
                "w-full gap-3 px-3 py-6 transition-all duration-200",
                collapsed ? "justify-center px-2" : "justify-start",
                isActive
                    ? "bg-primary/10 text-primary hover:bg-primary/15 border-r-2 border-primary rounded-r-none"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
            )}
            onClick={onClick}
        >
            {icon}
            {!collapsed && <span className="font-medium tracking-wide truncate">{label}</span>}
            {!collapsed && notifications ? (
                <Badge variant="secondary" className="ml-auto text-xs h-5 px-1.5 min-w-5 justify-center">
                    {notifications}
                </Badge>
            ) : null}
            {collapsed && notifications ? (
                <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-primary" />
            ) : null}
        </Button>
    )

    if (collapsed) {
        return (
            <Tooltip>
                <TooltipTrigger asChild>{content}</TooltipTrigger>
                <TooltipContent side="right" className="flex items-center gap-2">
                    {label}
                    {notifications ? <Badge variant="secondary" className="text-[10px] h-4 min-w-4 px-1">{notifications}</Badge> : null}
                </TooltipContent>
            </Tooltip>
        )
    }

    return content
}

interface MainLayoutProps {
    children: ReactNode
    activeTab: string
    onTabChange: (tab: string) => void
    systemStatus?: 'running' | 'stopped' | 'error' | 'warning'
    config: any
    activeMode: string
    onModeChange: (mode: string) => void
    jobProgress?: { total: number, completed: number, failed: number, visible: boolean }
}

export function MainLayout({ children, activeTab, onTabChange, systemStatus = 'stopped', config, activeMode, onModeChange, jobProgress }: MainLayoutProps) {
    const [collapsed, setCollapsed] = useState(false)

    return (
        <TooltipProvider delayDuration={0}>
            <div className="flex h-screen w-full bg-background overflow-hidden relative">
                {/* Ambient background effects */}
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-500/5 blur-[120px] rounded-full pointer-events-none" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-indigo-500/5 blur-[120px] rounded-full pointer-events-none" />

                {/* Sidebar */}
                <aside className={cn(
                    "glass border-r flex flex-col z-20 transition-all duration-300 ease-in-out",
                    collapsed ? "w-20" : "w-64"
                )}>
                    <div className={cn("p-6 pb-4 flex items-center", collapsed ? "justify-center px-2" : "justify-between")}>
                        {!collapsed && (
                            <div className="flex items-center gap-3 overflow-hidden">
                                <div className="h-8 w-8 min-w-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary">
                                    <Zap className="h-5 w-5 fill-current" />
                                </div>
                                <div className="flex flex-col truncate">
                                    <h1 className="font-bold text-lg leading-tight truncate">Agentic RAG</h1>
                                    <div className="flex items-center gap-1.5 mt-0.5">
                                        <span className={cn(
                                            "block h-2 w-2 min-w-2 rounded-full shadow-[0_0_8px]",
                                            {
                                                'bg-green-500 shadow-green-500/50': systemStatus === 'running',
                                                'bg-red-500 shadow-red-500/50': systemStatus === 'stopped' || systemStatus === 'error',
                                                'bg-yellow-500 shadow-yellow-500/50': systemStatus === 'warning',
                                            }
                                        )} />
                                        <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider truncate">
                                            {systemStatus}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}
                        {collapsed && (
                            <div className="h-8 w-8 min-w-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary mb-2">
                                <Zap className="h-5 w-5 fill-current" />
                            </div>
                        )}
                    </div>

                    <div className="flex justify-end px-2 mb-2">
                        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setCollapsed(!collapsed)}>
                            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                        </Button>
                    </div>

                    {/* Active Mode Indicator - Hide on collapse or show small icon */}
                    {!collapsed && (
                        <div className="px-6 pb-4">
                            <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-semibold mb-1">Active Mode</div>
                            <ProviderSelector config={config} activeMode={activeMode} onModeChange={onModeChange} />
                        </div>
                    )}

                    <nav className="flex-1 px-0 space-y-1 overflow-y-auto scrollbar-hide">
                        <SidebarItem
                            icon={<LayoutDashboard className="w-5 h-5" />}
                            label="Dashboard"
                            isActive={activeTab === 'dashboard'}
                            onClick={() => onTabChange('dashboard')}
                            collapsed={collapsed}
                        />
                        <SidebarItem
                            icon={<Search className="w-5 h-5" />}
                            label="Search & Chat"
                            isActive={activeTab === 'search'}
                            onClick={() => onTabChange('search')}
                            collapsed={collapsed}
                        />
                        <SidebarItem
                            icon={<Database className="w-5 h-5" />}
                            label="Knowledge Base"
                            isActive={activeTab === 'files'}
                            onClick={() => onTabChange('files')}
                            collapsed={collapsed}
                        />
                        <div className="my-4 border-t border-white/5 mx-4" />
                        <SidebarItem
                            icon={<Activity className="w-5 h-5" />}
                            label="Logs & Metrics"
                            isActive={activeTab === 'logs'}
                            onClick={() => onTabChange('logs')}
                            collapsed={collapsed}
                        />
                        <SidebarItem
                            icon={<Settings className="w-5 h-5" />}
                            label="Settings"
                            isActive={activeTab === 'settings'}
                            onClick={() => onTabChange('settings')}
                            collapsed={collapsed}
                        />
                        <div className="my-2" />
                        <SidebarItem
                            icon={<HelpCircle className="w-5 h-5" />}
                            label="Tips & Help"
                            isActive={activeTab === 'help'}
                            onClick={() => onTabChange('help')}
                            collapsed={collapsed}
                        />
                    </nav>

                    <div className="p-4 border-t border-white/5">
                        <div className={cn("glass-panel rounded-lg p-3 text-xs text-muted-foreground text-center truncate", collapsed && "p-1 text-[10px]")}>
                            {collapsed ? "v0.2" : "v0.2.0-beta"}
                        </div>
                    </div>
                </aside>

                {/* Main Content */}
                <main className="flex-1 overflow-auto relative z-10 scrollbar-hide flex flex-col">
                    <div className="flex-1 p-8 animate-in fade-in duration-500">
                        {children}
                    </div>
                    
                    {/* Global Progress Bar */}
                    {jobProgress && jobProgress.visible && (
                        <div className="sticky bottom-0 left-0 right-0 bg-background/80 backdrop-blur-md border-t border-white/10 p-3 animate-in slide-in-from-bottom-2">
                            <div className="max-w-3xl mx-auto flex items-center gap-4">
                                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground whitespace-nowrap">
                                    <Database className="w-4 h-4 text-blue-500 animate-pulse" />
                                    <span>Indexing Documents...</span>
                                    <span className="text-xs bg-secondary px-2 py-0.5 rounded-full">
                                        {jobProgress.completed} / {jobProgress.total}
                                    </span>
                                </div>
                                <Progress 
                                    value={(jobProgress.completed / jobProgress.total) * 100} 
                                    className="h-2 flex-1"
                                />
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </TooltipProvider>
    )
}
