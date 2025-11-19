import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

type TimeRange = '1h' | '6h' | '24h' | '7d' | '30d'

type MemoryData = {
  timestamp: string
  mcpUsed: number
  mcpAllocated: number
  restUsed: number
  restAllocated: number
}

export function MemoryUsageChart() {
  const [timeRange, setTimeRange] = useState<TimeRange>('1h')
  const [data, setData] = useState<MemoryData[]>([])
  const [isLoading, setIsLoading] = useState(true)

  const MCP_ALLOCATED = 512
  const REST_ALLOCATED = 512

  useEffect(() => {
    generateMockData()
  }, [timeRange])

  const generateMockData = () => {
    setIsLoading(true)
    
    const points = timeRange === '1h' ? 60 : 
                   timeRange === '6h' ? 72 : 
                   timeRange === '24h' ? 96 :
                   timeRange === '7d' ? 168 :
                   720

    const now = Date.now()
    const interval = timeRange === '1h' ? 60000 : 
                     timeRange === '6h' ? 300000 :
                     timeRange === '24h' ? 900000 :
                     timeRange === '7d' ? 3600000 :
                     3600000

    const mockData: MemoryData[] = []
    
    for (let i = points; i >= 0; i--) {
      const timestamp = new Date(now - (i * interval))
      
      const mcpBase = 200 + Math.sin(i / 10) * 50
      const mcpVariance = Math.random() * 80
      const mcpUsed = Math.min(MCP_ALLOCATED * 0.9, Math.max(150, mcpBase + mcpVariance))

      const restBase = 180 + Math.sin(i / 8) * 40
      const restVariance = Math.random() * 70
      const restUsed = Math.min(REST_ALLOCATED * 0.9, Math.max(120, restBase + restVariance))

      mockData.push({
        timestamp: timestamp.toISOString(),
        mcpUsed: Number(mcpUsed.toFixed(1)),
        mcpAllocated: MCP_ALLOCATED,
        restUsed: Number(restUsed.toFixed(1)),
        restAllocated: REST_ALLOCATED
      })
    }

    setData(mockData)
    setIsLoading(false)
  }

  const formatXAxis = (timestamp: string) => {
    const date = new Date(timestamp)
    if (timeRange === '1h' || timeRange === '6h') {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    } else if (timeRange === '24h') {
      return date.toLocaleTimeString('en-US', { hour: '2-digit' })
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    }
  }

  const currentData = data.length > 0 ? data[data.length - 1] : null
  const mcpAvg = data.reduce((acc, d) => acc + d.mcpUsed, 0) / data.length
  const restAvg = data.reduce((acc, d) => acc + d.restUsed, 0) / data.length
  const mcpPeak = Math.max(...data.map(d => d.mcpUsed))
  const restPeak = Math.max(...data.map(d => d.restUsed))

  const currentMcpPercent = currentData ? (currentData.mcpUsed / MCP_ALLOCATED) * 100 : 0
  const currentRestPercent = currentData ? (currentData.restUsed / REST_ALLOCATED) * 100 : 0

  const getStatusBadge = (percent: number) => {
    if (percent < 60) return { variant: 'secondary' as const, text: 'Normal', color: 'text-green-600' }
    if (percent < 80) return { variant: 'secondary' as const, text: 'Moderate', color: 'text-yellow-600' }
    return { variant: 'destructive' as const, text: 'High', color: 'text-red-600' }
  }

  const mcpStatus = getStatusBadge(currentMcpPercent)
  const restStatus = getStatusBadge(currentRestPercent)

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base font-semibold">Server Memory Usage</CardTitle>
            <CardDescription className="text-sm mt-1">
              Memory consumption for MCP and REST API servers
            </CardDescription>
          </div>
          <Select value={timeRange} onValueChange={(value) => setTimeRange(value as TimeRange)}>
            <SelectTrigger className="w-24 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">1 Hour</SelectItem>
              <SelectItem value="6h">6 Hours</SelectItem>
              <SelectItem value="24h">24 Hours</SelectItem>
              <SelectItem value="7d">7 Days</SelectItem>
              <SelectItem value="30d">30 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="grid gap-4 mt-4 md:grid-cols-2">
          <div className="space-y-2 rounded-lg border border-border p-3 bg-muted/30">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">MCP Server</span>
              <Badge variant={mcpStatus.variant} className="text-xs">
                {mcpStatus.text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Current</span>
                <span className="font-semibold">
                  {currentData?.mcpUsed.toFixed(1)} MB / {MCP_ALLOCATED} MB
                </span>
              </div>
              <Progress value={currentMcpPercent} className="h-2" />
              <div className="flex items-center justify-between text-xs pt-1">
                <span className="text-muted-foreground">Avg: {mcpAvg.toFixed(1)} MB</span>
                <span className="text-muted-foreground">Peak: {mcpPeak.toFixed(1)} MB</span>
              </div>
            </div>
          </div>

          <div className="space-y-2 rounded-lg border border-border p-3 bg-muted/30">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">REST API Server</span>
              <Badge variant={restStatus.variant} className="text-xs">
                {restStatus.text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Current</span>
                <span className="font-semibold">
                  {currentData?.restUsed.toFixed(1)} MB / {REST_ALLOCATED} MB
                </span>
              </div>
              <Progress value={currentRestPercent} className="h-2" />
              <div className="flex items-center justify-between text-xs pt-1">
                <span className="text-muted-foreground">Avg: {restAvg.toFixed(1)} MB</span>
                <span className="text-muted-foreground">Peak: {restPeak.toFixed(1)} MB</span>
              </div>
            </div>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {isLoading ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            Loading...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart
              data={data}
              margin={{ top: 5, right: 5, left: 0, bottom: 5 }}
            >
              <defs>
                <linearGradient id="mcpGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="restGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.89 0.01 250)" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={formatXAxis}
                stroke="oklch(0.5 0.01 250)"
                fontSize={12}
              />
              <YAxis 
                stroke="oklch(0.5 0.01 250)"
                fontSize={12}
                domain={[0, Math.max(MCP_ALLOCATED, REST_ALLOCATED)]}
                label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: 'oklch(0.5 0.01 250)' } }}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  const label = name === 'mcpUsed' ? 'MCP Server' :
                               name === 'restUsed' ? 'REST Server' :
                               name === 'mcpAllocated' ? 'MCP Allocated' :
                               'REST Allocated'
                  return [`${value.toFixed(1)} MB`, label]
                }}
                labelFormatter={(label) => new Date(label).toLocaleString()}
                contentStyle={{
                  backgroundColor: 'oklch(1 0 0)',
                  border: '1px solid oklch(0.89 0.01 250)',
                  borderRadius: '8px'
                }}
              />
              <Legend 
                formatter={(value) => {
                  if (value === 'mcpUsed') return 'MCP Server Used'
                  if (value === 'restUsed') return 'REST Server Used'
                  if (value === 'mcpAllocated') return 'MCP Allocated'
                  return 'REST Allocated'
                }}
              />
              
              <Area 
                type="monotone" 
                dataKey="mcpUsed" 
                stroke="#3b82f6" 
                fill="url(#mcpGradient)"
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="restUsed" 
                stroke="#8b5cf6" 
                fill="url(#restGradient)"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="mcpAllocated" 
                stroke="#3b82f6" 
                strokeWidth={1.5}
                strokeDasharray="5 5"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="restAllocated" 
                stroke="#8b5cf6" 
                strokeWidth={1.5}
                strokeDasharray="5 5"
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}
