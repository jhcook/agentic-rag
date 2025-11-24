import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { MetricsChart } from '@/components/MetricsChart'
import { MemoryUsageChart } from '@/components/MemoryUsageChart'
import { 
  LineChart as ChartLine, 
  Database, 
  Zap as Lightning, 
  Timer,
  Cpu,
  HardDrive as HardDrives,
  CheckCircle,
  AlertCircle as WarningCircle
} from 'lucide-react'

type OllamaConfig = {
  ragHost: string
  ragPort: string
  ragPath: string
  mcpHost: string
  mcpPort: string
  mcpPath: string
}

type HealthData = {
  documents: number
  vectors: number
  memory_mb: number
  memory_limit_mb?: number
  total_size_bytes: number
  store_file_bytes: number
}

type MCPMetrics = {
  memory_mb: number
}

type QualityMetrics = {
  total_searches: number
  failed_searches: number
  responses_with_sources: number
  total_sources: number
  fallback_responses: number
  success_rate: number
  avg_sources: number
}

export function MetricsDashboard({ config }: { config: OllamaConfig }) {
  const [healthData, setHealth] = useState<HealthData | null>(null)
  const [mcpData, setMcpMetrics] = useState<MCPMetrics | null>(null)
  const [qualityData, setQualityMetrics] = useState<QualityMetrics | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [mcpError, setMcpError] = useState<string | null>(null)
  const [qualityError, setQualityError] = useState<string | null>(null)
  const [restData, setRestMetrics] = useState<Record<string, number>>({})
  const [restMetricsError, setRestMetricsError] = useState<string | null>(null)
  const [metricHistory, setMetricHistory] = useState<Record<string, Array<{ timestamp: string; value: number }>>>({})

  useEffect(() => {
    const controller = new AbortController()
    const fetchRestMetrics = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/metrics`

      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const body = await res.text()
        const metrics: Record<string, number> = {}
        const now = new Date().toISOString()

        // Parse average latency: sum / count (convert seconds to ms)
        const sumMatch = body.match(/rest_http_request_duration_seconds_sum\{[^}]*\}\s+(\d+(?:\.\d+)?)/)
        const countMatch = body.match(/rest_http_request_duration_seconds_count\{[^}]*\}\s+(\d+(?:\.\d+)?)/)
        const sum = sumMatch ? Number(sumMatch[1]) : 0
        const count = countMatch ? Number(countMatch[1]) : 0
        if (count > 0) {
          const avgSeconds = sum / count
          metrics['rest_http_request_duration_avg'] = avgSeconds * 1000 // Convert to ms
          // Update history for latency
          setMetricHistory(prev => {
            const history = prev['query_latency'] || []
            const newHistory = [...history, { timestamp: now, value: avgSeconds * 1000 }].slice(-60) // Keep last 60 points
            return { ...prev, 'query_latency': newHistory }
          })
        }

        // Total requests (all statuses)
        const reqRegex = /rest_http_requests_total\{[^}]*\}\s+(\d+(?:\.\d+)?)/g
        let reqSum = 0
        let m: RegExpExecArray | null
        while ((m = reqRegex.exec(body)) !== null) {
          reqSum += Number(m[1])
        }
        metrics['rest_http_requests_total'] = reqSum
        // Update history for query count
        setMetricHistory(prev => {
          const history = prev['query_count'] || []
          const lastValue = history.length > 0 ? history[history.length - 1].value : 0
          const delta = reqSum - lastValue
          if (delta > 0 || history.length === 0) {
            const newHistory = [...history, { timestamp: now, value: reqSum }].slice(-60)
            return { ...prev, 'query_count': newHistory }
          }
          return prev
        })

        // Inflight gauge
        const inflight = body.match(/rest_inflight_requests\s+(\d+(?:\.\d+)?)/)
        if (inflight) {
          metrics['rest_inflight_requests'] = Number(inflight[1])
          // Update history for active connections
          setMetricHistory(prev => {
            const history = prev['active_connections'] || []
            const newHistory = [...history, { timestamp: now, value: Number(inflight[1]) }].slice(-60)
            return { ...prev, 'active_connections': newHistory }
          })
        }

        // Parse memory usage
        const memMatch = body.match(/rest_memory_usage_megabytes\s+(\d+(?:\.\d+)?)/)
        if (memMatch) {
          metrics['rest_memory_usage_mb'] = Number(memMatch[1])
        }

        // Parse documents indexed from REST metrics
        const docsMatch = body.match(/rest_documents_indexed_total\s+(\d+(?:\.\d+)?)/)
        if (docsMatch) {
          metrics['rest_documents_indexed'] = Number(docsMatch[1])
          // Update history for documents indexed
          setMetricHistory(prev => {
            const history = prev['documents_indexed'] || []
            const newHistory = [...history, { timestamp: now, value: Number(docsMatch[1]) }].slice(-60)
            return { ...prev, 'documents_indexed': newHistory }
          })
        }

        setRestMetrics(metrics)
        setRestMetricsError(null)
      } catch (err: unknown) {
        setRestMetrics({})
        setRestMetricsError(err instanceof Error ? err.message : 'Failed to load REST metrics')
      }
    }

    const fetchHealth = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/${base}/health`

      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setHealth({
          documents: Number(data?.documents || 0),
          vectors: Number(data?.vectors || 0),
          memory_mb: Number(data?.memory_mb || 0),
          memory_limit_mb: data?.memory_limit_mb ? Number(data.memory_limit_mb) : undefined,
          total_size_bytes: Number(data?.total_size_bytes || 0),
          store_file_bytes: Number(data?.store_file_bytes || 0)
        })
        setHealthError(null)
      } catch (err: unknown) {
        setHealth(null)
        setHealthError(err instanceof Error ? err.message : 'Failed to load metrics')
      }
    }

    fetchHealth()
    fetchRestMetrics()
    const id = setInterval(fetchHealth, 15000)
    const mid = setInterval(fetchRestMetrics, 15000)
    return () => {
      controller.abort()
      clearInterval(id)
      clearInterval(mid)
    }
  }, [config?.ragHost, config?.ragPort, config?.ragPath])

  useEffect(() => {
    const controller = new AbortController()
    const fetchMcpMetrics = async () => {
      const host = config?.mcpHost || '127.0.0.1'
      const port = config?.mcpPort || '8000'
      // MCP /metrics is exposed at the root (not under mcpPath)
      const url = `http://${host}:${port}/metrics`
      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const body = await res.text()
        // Parse Prometheus text for mcp_memory_usage_megabytes
        const match = body.match(/mcp_memory_usage_megabytes\s+(\d+(?:\.\d+)?)/)
        const mem = match ? Number(match[1]) : 0
        setMcpMetrics({ memory_mb: mem })
        setMcpError(null)
      } catch (err: unknown) {
        setMcpMetrics(null)
        setMcpError(err instanceof Error ? err.message : 'Failed to load MCP metrics')
      }
    }
    fetchMcpMetrics()
    const id = setInterval(fetchMcpMetrics, 20000)
    return () => {
      controller.abort()
      clearInterval(id)
    }
  }, [config?.mcpHost, config?.mcpPort, config?.mcpPath])

  useEffect(() => {
    const controller = new AbortController()
    const fetchQuality = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/${base}/metrics/quality`
      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setQualityMetrics({
          total_searches: Number(data?.total_searches || 0),
          failed_searches: Number(data?.failed_searches || 0),
          responses_with_sources: Number(data?.responses_with_sources || 0),
          total_sources: Number(data?.total_sources || 0),
          fallback_responses: Number(data?.fallback_responses || 0),
          success_rate: Number(data?.success_rate || 0),
          avg_sources: Number(data?.avg_sources || 0),
        })
        setQualityError(null)
      } catch (err: unknown) {
        setQualityMetrics(null)
        setQualityError(err instanceof Error ? err.message : 'Failed to load quality metrics')
      }
    }
    fetchQuality()
    const id = setInterval(fetchQuality, 20000)
    return () => {
      controller.abort()
      clearInterval(id)
    }
  }, [config?.ragHost, config?.ragPort, config?.ragPath])

  const formatSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let size = bytes
    let idx = 0
    while (size >= 1024 && idx < units.length - 1) {
      size /= 1024
      idx++
    }
    return `${size.toFixed(1)} ${units[idx]}`
  }

  const docCount = healthData?.documents ?? 0
  const vectorCount = healthData?.vectors ?? 0
  const memoryMb = healthData?.memory_mb ?? 0
  const sizeBytes = (healthData?.total_size_bytes ?? 0) || (healthData?.store_file_bytes ?? 0)
  const totalSize = formatSize(sizeBytes)
  const mcpMemoryMb = mcpData?.memory_mb ?? null
  const quality = qualityData
  const restAvgLatency = restData['rest_http_request_duration_avg'] || null
  const restReqCount = restData['rest_http_requests_total'] || null
  const restInflight = restData['rest_inflight_requests'] || null

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight mb-2">System Metrics</h2>
        <p className="text-muted-foreground">
          Real-time performance monitoring and analytics {healthError && `(degraded: ${healthError})`}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">Documents Indexed</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{docCount}</div>
            <p className="text-xs text-muted-foreground mt-1">From REST /health</p>
            <div className="mt-2">
              <Badge variant="secondary" className="text-xs">
                <CheckCircle className="h-3 w-3 mr-1" />
                {healthData ? 'Healthy' : 'Unknown'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">Vectors</CardTitle>
              <ChartLine className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{vectorCount}</div>
            <p className="text-xs text-muted-foreground mt-1">FAISS index total vectors</p>
            <div className="mt-2">
              <Badge variant="secondary" className="text-xs">
                {healthData ? 'Healthy' : 'Unknown'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">Store Size</CardTitle>
              <HardDrives className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalSize}</div>
            <p className="text-xs text-muted-foreground mt-1">UTF-8 text volume</p>
            <div className="mt-2">
              <Badge variant="secondary" className="text-xs">
                <CheckCircle className="h-3 w-3 mr-1" />
                {healthData ? 'Healthy' : 'Unknown'}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
      <Tabs defaultValue="performance" className="space-y-6">
        <TabsList className="grid w-full max-w-lg grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="usage">Usage</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="quality">Quality</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <MetricsChart
              title="Query Latency"
              description="Average response time for search queries"
              metricKey="query_latency"
              unit="ms"
              chartType="area"
              color="#3b82f6"
              dataOverride={metricHistory['query_latency'] || null}
              summaryOverride={restAvgLatency !== null ? {
                current: restAvgLatency,
                avg: restAvgLatency,
                max: restAvgLatency,
                min: restAvgLatency
              } : undefined}
            />
            
            <MetricsChart
              title="Embedding Generation Time"
              description="Time to generate vector embeddings"
              metricKey="embedding_time"
              unit="ms"
              chartType="area"
              color="#8b5cf6"
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <MetricsChart
              title="Vector Search Time"
              description="FAISS vector similarity search latency"
              metricKey="vector_search_time"
              unit="ms"
              chartType="line"
              color="#06b6d4"
            />
            
            <MetricsChart
              title="Cache Hit Rate"
              description="Percentage of requests served from cache"
              metricKey="cache_hit_rate"
              unit="%"
              chartType="area"
              color="#10b981"
            />
          </div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <MetricsChart
              title="Query Count"
              description="Number of search queries over time"
              metricKey="query_count"
              chartType="bar"
              color="#3b82f6"
              dataOverride={metricHistory['query_count'] || null}
              summaryOverride={restReqCount !== null ? {
                current: restReqCount,
                avg: restReqCount,
                max: restReqCount,
                min: 0
              } : undefined}
            />
            
            <MetricsChart
              title="Active Connections"
              description="Concurrent active connections to the system"
              metricKey="active_connections"
              chartType="area"
              color="#f59e0b"
              dataOverride={metricHistory['active_connections'] || null}
              summaryOverride={restInflight !== null ? {
                current: restInflight,
                avg: restInflight,
                max: restInflight,
                min: 0
              } : undefined}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <MetricsChart
              title="Documents Indexed"
              description="Total documents in the search index"
              metricKey="documents_indexed"
              chartType="line"
              color="#8b5cf6"
              dataOverride={metricHistory['documents_indexed'] || null}
              summaryOverride={docCount > 0 ? {
                current: docCount,
                avg: docCount,
                max: docCount,
                min: 0
              } : undefined}
            />
            
            <MetricsChart
              title="Token Usage"
              description="Average tokens per LLM request"
              metricKey="token_count"
              unit="tokens"
              chartType="area"
              color="#06b6d4"
            />
          </div>
        </TabsContent>
        <TabsContent value="system" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Memory (REST)</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{memoryMb.toFixed(0)} MB</div>
                <p className="text-xs text-muted-foreground mt-1">
                  Process RSS{healthData?.memory_limit_mb ? ` / Limit ${healthData.memory_limit_mb} MB` : ''}
                </p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {healthData ? 'Healthy' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Memory (MCP)</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {mcpMemoryMb !== null ? `${mcpMemoryMb.toFixed(0)} MB` : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">From MCP /metrics</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {mcpMemoryMb !== null ? 'Healthy' : 'Unknown'}
                  </Badge>
                </div>
                {mcpError && (
                  <p className="text-xs text-destructive mt-2 flex items-center gap-1">
                    <WarningCircle className="h-3 w-3" /> {mcpError}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Avg Latency</CardTitle>
                  <Timer className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {restAvgLatency !== null ? `${restAvgLatency.toFixed(0)} ms` : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">From REST /metrics</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {restAvgLatency !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
                {restMetricsError && (
                  <p className="text-xs text-destructive mt-2 flex items-center gap-1">
                    <WarningCircle className="h-3 w-3" /> {restMetricsError}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Total Requests</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {restReqCount !== null ? restReqCount.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">From REST /metrics</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {restReqCount !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Inflight Requests</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {restInflight !== null ? restInflight.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">From REST /metrics</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {restInflight !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="quality" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Quality Metrics Summary</CardTitle>
              {qualityError && (
                <p className="text-xs text-destructive mt-1">{qualityError}</p>
              )}
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Searches</p>
                <p className="text-xl font-semibold">{quality?.total_searches ?? 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Failed Searches</p>
                <p className="text-xl font-semibold text-destructive">{quality?.failed_searches ?? 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Success Rate</p>
                <p className="text-xl font-semibold">
                  {quality ? `${(quality.success_rate * 100).toFixed(1)}%` : '--'}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Responses with Sources</p>
                <p className="text-xl font-semibold">{quality?.responses_with_sources ?? 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Avg Sources / Search</p>
                <p className="text-xl font-semibold">
                  {quality ? quality.avg_sources.toFixed(2) : '--'}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Fallback Responses</p>
                <p className="text-xl font-semibold">{quality?.fallback_responses ?? 0}</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
