import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { toast } from 'sonner';

type PerformanceMetric = {
  timestamp: string;
  operation: string;
  duration_ms: number;
  tokens: number | null;
  model: string | null;
  error: string | null;
};

type OllamaConfig = {
  ragHost?: string;
  ragPort?: string;
  ragPath?: string;
};

export function MetricsView({ config }: { config: OllamaConfig }) {
  const [data, setData] = useState<PerformanceMetric[]>([]);
  const [loading, setLoading] = useState(true);

  const getApiBase = useCallback(() => {
    const host = config?.ragHost || '127.0.0.1';
    const port = config?.ragPort || '8001';
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '');
    return `http://${host}:${port}/${base}`;
  }, [config]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiBase = getApiBase();
        const res = await fetch(`${apiBase}/metrics/performance?hours=24`);
        if (res.ok) {
          const metrics = await res.json();
          setData(metrics);
        } else {
          toast.error('Failed to fetch performance metrics');
        }
      } catch (error) {
        toast.error('Failed to connect to the server');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, [getApiBase]);

  if (loading) {
    return <div>Loading...</div>;
  }

  const errorRateData = data.reduce((acc, metric) => {
    const op = metric.operation;
    if (!acc[op]) {
      acc[op] = { operation: op, success: 0, error: 0 };
    }
    if (metric.error) {
      acc[op].error++;
    } else {
      acc[op].success++;
    }
    return acc;
  }, {} as Record<string, { operation: string, success: number, error: number }>);

  const durationData = data.reduce((acc, metric) => {
    const op = metric.operation;
    if (!acc[op]) {
      acc[op] = { operation: op, totalDuration: 0, count: 0 };
    }
    acc[op].totalDuration += metric.duration_ms;
    acc[op].count++;
    return acc;
  }, {} as Record<string, { operation: string, totalDuration: number, count: number }>);

  const avgDurationByOp = Object.values(durationData).map(d => ({
    operation: d.operation,
    avgDuration: Math.round(d.totalDuration / d.count)
  }));

  const tokenData = data.reduce((acc, metric) => {
    // Handle both 'tokens' (new) and 'token_count' (legacy/UI-compat) keys
    const val = (metric as any).tokens ?? (metric as any).token_count;

    if (val !== null && val !== undefined) {
      const op = metric.operation;
      if (!acc[op]) {
        acc[op] = { operation: op, totalTokens: 0 };
      }
      acc[op].totalTokens += val;
    }
    return acc;
  }, {} as Record<string, { operation: string, totalTokens: number }>);

  const tokensByOp = Object.values(tokenData);

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold tracking-tight">Performance Metrics</h2>

      <Card>
        <CardHeader>
          <CardTitle>Avg Duration by Operation (Last 24 hours)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={avgDurationByOp}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="operation" />
              <YAxis label={{ value: 'Avg Duration (ms)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="avgDuration" fill="#8884d8" name="Avg Duration (ms)" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Total Tokens by Operation (Last 24 hours)</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Tracks prompt and completion tokens from LLM API calls
          </p>
        </CardHeader>
        <CardContent>
          {tokensByOp.length === 0 ? (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              <div className="text-center space-y-2">
                <p className="font-medium">No token count data yet</p>
                <p className="text-sm">Token usage will appear here after chat or search operations</p>
                <p className="text-xs">Supported: Ollama, Google Gemini, OpenAI, and compatible APIs</p>
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={tokensByOp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="operation" />
                <YAxis label={{ value: 'Tokens', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="totalTokens" fill="#82ca9d" name="Total Tokens" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Error Rate by Operation (Last 24 hours)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={Object.values(errorRateData)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="operation" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="success" stackId="a" fill="#82ca9d" name="Success" />
              <Bar dataKey="error" stackId="a" fill="#ffc658" name="Error" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
