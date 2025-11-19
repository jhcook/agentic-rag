import React, { useEffect, useState } from 'react'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'

type TimeRange = '1h' | '6h' | '24h' | '7d' | '30d'

type MetricData = {
  timestamp: string
  value: number
}

type ChartType = 'line' | 'area' | 'bar'

interface MetricsChartProps {
  title: string
  description?: string
  metricKey: string
  unit?: string
  chartType?: ChartType
  color?: string
  showLegend?: boolean
  dataOverride?: MetricData[] | null
  summaryOverride?: {
    current?: number
    avg?: number
    max?: number
    min?: number
  }
}

export function MetricsChart({ 
  title, 
  description, 
  metricKey, 
  unit = '', 
  chartType = 'line',
  color = '#3b82f6',
  showLegend = false,
  dataOverride = null,
  summaryOverride,
}: MetricsChartProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('1h')
  const [data, setData] = useState<MetricData[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (dataOverride) {
      setData(dataOverride)
      setIsLoading(false)
    } else {
      // No real metrics available yet; avoid showing fake data
      setData([])
      setIsLoading(false)
    }
  }, [timeRange, metricKey, dataOverride])

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

  const formatTooltip = (value: number) => {
    if (unit === '%') {
      return `${value.toFixed(1)}%`
    } else if (unit === 'ms') {
      return `${value.toFixed(0)}ms`
    } else if (unit === 'tokens') {
      return `${value.toFixed(0)} tokens`
    }
    return value.toFixed(2)
  }

  const currentValue = summaryOverride?.current ?? (data.length > 0 ? data[data.length - 1].value : 0)
  const avgValue = summaryOverride?.avg ?? (data.length ? data.reduce((acc, d) => acc + d.value, 0) / data.length : 0)
  const maxValue = summaryOverride?.max ?? (data.length ? Math.max(...data.map(d => d.value)) : 0)
  const minValue = summaryOverride?.min ?? (data.length ? Math.min(...data.map(d => d.value)) : 0)

  const renderChart = () => {
    const commonProps = {
      data,
      margin: { top: 5, right: 5, left: 0, bottom: 5 }
    }

    if (chartType === 'area') {
      return (
        <AreaChart {...commonProps}>
          <defs>
            <linearGradient id={`gradient-${metricKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
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
          />
          <Tooltip 
            formatter={(value: number) => [formatTooltip(value), title]}
            labelFormatter={(label) => new Date(label).toLocaleString()}
            contentStyle={{
              backgroundColor: 'oklch(1 0 0)',
              border: '1px solid oklch(0.89 0.01 250)',
              borderRadius: '8px'
            }}
          />
          {showLegend && <Legend />}
          <Area 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            fill={`url(#gradient-${metricKey})`}
            strokeWidth={2}
          />
        </AreaChart>
      )
    } else if (chartType === 'bar') {
      return (
        <BarChart {...commonProps}>
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
          />
          <Tooltip 
            formatter={(value: number) => [formatTooltip(value), title]}
            labelFormatter={(label) => new Date(label).toLocaleString()}
            contentStyle={{
              backgroundColor: 'oklch(1 0 0)',
              border: '1px solid oklch(0.89 0.01 250)',
              borderRadius: '8px'
            }}
          />
          {showLegend && <Legend />}
          <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      )
    } else {
      return (
        <LineChart {...commonProps}>
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
          />
          <Tooltip 
            formatter={(value: number) => [formatTooltip(value), title]}
            labelFormatter={(label) => new Date(label).toLocaleString()}
            contentStyle={{
              backgroundColor: 'oklch(1 0 0)',
              border: '1px solid oklch(0.89 0.01 250)',
              borderRadius: '8px'
            }}
          />
          {showLegend && <Legend />}
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      )
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base font-semibold">{title}</CardTitle>
            {description && (
              <CardDescription className="text-sm mt-1">{description}</CardDescription>
            )}
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
        <div className="flex items-center gap-4 mt-3 text-xs">
          <div>
            <span className="text-muted-foreground">Current:</span>
            <span className="ml-1 font-semibold">--</span>
          </div>
          <div>
            <span className="text-muted-foreground">Avg:</span>
            <span className="ml-1 font-semibold">--</span>
          </div>
          <div>
            <span className="text-muted-foreground">Max:</span>
            <span className="ml-1 font-semibold">--</span>
          </div>
          <div>
            <span className="text-muted-foreground">Min:</span>
            <span className="ml-1 font-semibold">--</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">
            No metrics available yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            {renderChart()}
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}
