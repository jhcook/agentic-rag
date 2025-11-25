import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
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

type TodayMetrics = {
  queries_today: number
  documents_added_today: number
  avg_latency_today: number
}

type EmbeddingStageMetrics = {
  requests: number
  errors: number
  avgMs: number
}

type RestMetrics = {
  latencyMs: number | null
  requestsTotal: number | null
  inflight: number | null
  memoryMb: number | null
  documentsIndexed: number | null
  mcpMemoryMb: number | null
}

export function MetricsDashboard({ config }: { config: OllamaConfig }) {
  const [healthData, setHealth] = useState<HealthData | null>(null)
  const [mcpData, setMcpMetrics] = useState<MCPMetrics | null>(null)
  const [qualityData, setQualityMetrics] = useState<QualityMetrics | null>(null)
  const [todayData, setTodayMetrics] = useState<TodayMetrics | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [mcpError, setMcpError] = useState<string | null>(null)
  const [qualityError, setQualityError] = useState<string | null>(null)
  const [todayError, setTodayError] = useState<string | null>(null)
  const [restData, setRestMetrics] = useState<RestMetrics>({
    latencyMs: null,
    requestsTotal: null,
    inflight: null,
    memoryMb: null,
    documentsIndexed: null,
    mcpMemoryMb: null,
  })
  const [embeddingMetrics, setEmbeddingMetrics] = useState<Record<string, EmbeddingStageMetrics>>({})
  const [restMetricsError, setRestMetricsError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const fetchRestMetrics = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const url = `http://${host}:${port}/metrics`

      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const body = await res.text()
        const metrics: RestMetrics = {
          latencyMs: null,
          requestsTotal: null,
          inflight: null,
          memoryMb: null,
          documentsIndexed: null,
          mcpMemoryMb: null,
        }

        // Parse average latency: sum / count (convert seconds to ms)
        const sumMatch = body.match(/rest_http_request_duration_seconds_sum\{[^}]*\}\s+(\d+(?:\.\d+)?)/)
        const countMatch = body.match(/rest_http_request_duration_seconds_count\{[^}]*\}\s+(\d+(?:\.\d+)?)/)
        const sum = sumMatch ? Number(sumMatch[1]) : 0
        const count = countMatch ? Number(countMatch[1]) : 0
        if (count > 0) {
          const avgSeconds = sum / count
          metrics.latencyMs = avgSeconds * 1000
        }

        // Total requests (all statuses)
        const reqRegex = /rest_http_requests_total\{[^}]*\}\s+(\d+(?:\.\d+)?)/g
        let reqSum = 0
        let m: RegExpExecArray | null
        while ((m = reqRegex.exec(body)) !== null) {
          reqSum += Number(m[1])
        }
        metrics.requestsTotal = reqSum

        // Inflight gauge
        const inflight = body.match(/rest_inflight_requests\s+(\d+(?:\.\d+)?)/)
        if (inflight) {
          metrics.inflight = Number(inflight[1])
        }

        // Parse memory usage
        const memMatch = body.match(/rest_memory_usage_megabytes\s+(\d+(?:\.\d+)?)/)
        if (memMatch) {
          metrics.memoryMb = Number(memMatch[1])
        }

        // Parse MCP memory (proxied via REST)
        const mcpMemMatch = body.match(/mcp_memory_usage_megabytes\s+(\d+(?:\.\d+)?)/)
        if (mcpMemMatch) {
          metrics.mcpMemoryMb = Number(mcpMemMatch[1])
        }

        // Parse documents indexed from REST metrics
        const docsMatch = body.match(/rest_documents_indexed_total\s+(\d+(?:\.\d+)?)/)
        if (docsMatch) {
          metrics.documentsIndexed = Number(docsMatch[1])
        }

        // Embedding metrics by stage
        const embedRequests: Record<string, number> = {}
        const embedErrors: Record<string, number> = {}
        const embedCounts: Record<string, number> = {}
        const embedSums: Record<string, number> = {}

        const embedReqRegex = /embedding_requests_total\{[^}]*stage=\"([^\"]+)\"[^}]*\}\s+(\d+(?:\.\d+)?)/g
        const embedErrRegex = /embedding_errors_total\{[^}]*stage=\"([^\"]+)\"[^}]*\}\s+(\d+(?:\.\d+)?)/g
        const embedCountRegex = /embedding_duration_seconds_count\{[^}]*stage=\"([^\"]+)\"[^}]*\}\s+(\d+(?:\.\d+)?)/g
        const embedSumRegex = /embedding_duration_seconds_sum\{[^}]*stage=\"([^\"]+)\"[^}]*\}\s+(\d+(?:\.\d+)?)/g

        let match: RegExpExecArray | null
        while ((match = embedReqRegex.exec(body)) !== null) {
          embedRequests[match[1]] = Number(match[2])
        }
        while ((match = embedErrRegex.exec(body)) !== null) {
          embedErrors[match[1]] = Number(match[2])
        }
        while ((match = embedCountRegex.exec(body)) !== null) {
          embedCounts[match[1]] = Number(match[2])
        }
        while ((match = embedSumRegex.exec(body)) !== null) {
          embedSums[match[1]] = Number(match[2])
        }

        const stageMetrics: Record<string, EmbeddingStageMetrics> = {}
        const stages = new Set([
          ...Object.keys(embedRequests),
          ...Object.keys(embedErrors),
          ...Object.keys(embedCounts),
          ...Object.keys(embedSums),
        ])
        stages.forEach(stage => {
          const requests = embedRequests[stage] ?? 0
          const errors = embedErrors[stage] ?? 0
          const countVal = embedCounts[stage] ?? 0
          const sumVal = embedSums[stage] ?? 0
          const avgMs = countVal > 0 ? (sumVal / countVal) * 1000 : 0
          stageMetrics[stage] = { requests, errors, avgMs }
        })

        setRestMetrics(metrics)
        setEmbeddingMetrics(stageMetrics)
        setRestMetricsError(null)
        setMcpMetrics(metrics.mcpMemoryMb !== null ? { memory_mb: metrics.mcpMemoryMb } : null)
        setMcpError(null)
      } catch (err: unknown) {
        setRestMetrics({
          latencyMs: null,
          requestsTotal: null,
          inflight: null,
          memoryMb: null,
          documentsIndexed: null,
          mcpMemoryMb: null,
        })
        setEmbeddingMetrics({})
        setRestMetricsError(err instanceof Error ? err.message : 'Failed to load REST metrics')
        setMcpMetrics(null)
        setMcpError(err instanceof Error ? err.message : 'Failed to load MCP metrics')
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
    // MCP metrics now proxied via REST metrics (see fetchRestMetrics)
    const id = setInterval(() => {}, 20000)
    return () => clearInterval(id)
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

  useEffect(() => {
    const controller = new AbortController()
    const fetchToday = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      const url = `http://${host}:${port}/${base}/metrics/today`
      try {
        const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setTodayMetrics({
          queries_today: Number(data?.queries_today || 0),
          documents_added_today: Number(data?.documents_added_today || 0),
          avg_latency_today: Number(data?.avg_latency_today || 0),
        })
        setTodayError(null)
      } catch (err: unknown) {
        setTodayMetrics(null)
        setTodayError(err instanceof Error ? err.message : 'Failed to load today metrics')
      }
    }
    fetchToday()
    const id = setInterval(fetchToday, 20000)
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
  const restAvgLatency = restData.latencyMs ?? null
  const restReqCount = restData.requestsTotal ?? null
  const restInflight = restData.inflight ?? null
  const restDocsIndexed = restData.documentsIndexed ?? null
  const searchEmbedMetrics = embeddingMetrics['search_query']
  const totalEmbedRequests = Object.keys(embeddingMetrics).length
    ? Object.values(embeddingMetrics).reduce(
        (acc, stage) => acc + (stage.requests || 0),
        0
      )
    : null
  const indexedDocuments = restDocsIndexed ?? (healthData ? docCount : null)
  const docsSourceLabel = restDocsIndexed !== null
    ? 'From REST /metrics'
    : healthData
      ? 'From REST /health'
      : 'No metrics available'
  const effectiveLatencyMs = todayData?.avg_latency_today ?? restAvgLatency
  const queriesToday = todayData?.queries_today ?? null
  const documentsAddedToday = todayData?.documents_added_today ?? null

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
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Query Latency</CardTitle>
                  <Timer className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {effectiveLatencyMs !== null ? `${effectiveLatencyMs.toFixed(0)} ms` : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {todayData?.avg_latency_today ? 'Today\u2019s average latency' : 'From REST /metrics'}
                </p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {effectiveLatencyMs !== null ? 'Active' : 'Unknown'}
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
                  <CardTitle className="text-sm font-medium text-muted-foreground">Embedding Latency</CardTitle>
                  <ChartLine className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {searchEmbedMetrics ? `${searchEmbedMetrics.avgMs.toFixed(1)} ms` : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Avg encode time for search embeddings
                </p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {searchEmbedMetrics ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Success Rate</CardTitle>
                  <CheckCircle className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {quality ? `${(quality.success_rate * 100).toFixed(1)}%` : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Query success rate</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {quality ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Embedding Requests</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {totalEmbedRequests !== null ? totalEmbedRequests.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Total embedding calls (all stages)</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {totalEmbedRequests !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Total Queries</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {restReqCount !== null ? restReqCount.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">All-time requests</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {restReqCount !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Queries Today</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {queriesToday !== null ? queriesToday.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Since midnight</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {queriesToday !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
                {todayError && (
                  <p className="text-xs text-destructive mt-2">{todayError}</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Active Connections</CardTitle>
                  <Lightning className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {restInflight !== null ? restInflight.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Current inflight REST requests</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {restInflight !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Documents Indexed</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {indexedDocuments !== null ? indexedDocuments.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">{docsSourceLabel}</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {indexedDocuments !== null ? 'Healthy' : 'Unknown'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Docs Added Today</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {documentsAddedToday !== null ? documentsAddedToday.toLocaleString() : '--'}
                </div>
                <p className="text-xs text-muted-foreground mt-1">New documents</p>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    {documentsAddedToday !== null ? 'Active' : 'Unknown'}
                  </Badge>
                </div>
                {todayError && (
                  <p className="text-xs text-destructive mt-2">{todayError}</p>
                )}
              </CardContent>
            </Card>
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
