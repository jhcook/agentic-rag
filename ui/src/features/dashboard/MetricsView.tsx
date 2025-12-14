import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { toast } from 'sonner';

type PerformanceMetric = {
  timestamp: string;
  operation: string;
  duration_ms: number;
  token_count: number | null;
  model: string | null;
  error: string | null;
};

export function MetricsView() {
  const [data, setData] = useState<PerformanceMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/metrics/performance?hours=24');
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
  }, []);

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

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold tracking-tight">Performance Metrics</h2>

      <Card>
        <CardHeader>
          <CardTitle>Operation Duration (Last 24 hours)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} />
              <YAxis yAxisId="left" label={{ value: 'Duration (ms)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="duration_ms" name="Duration (ms)" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Token Count (Last 24 hours)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.filter(d => d.token_count !== null)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} />
              <YAxis label={{ value: 'Tokens', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="token_count" name="Token Count" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
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
