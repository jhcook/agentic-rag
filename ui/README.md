# Agentic RAG Control Panel
Single-page console for managing an agentic document retrieval-and-chat system. The UI mocks end-to-end workflows—indexing local files, configuring an Ollama backend, monitoring metrics, and watching logs—so you can plug in real services when ready.

## What’s here
- Dashboards: system status cards and onboarding steps to start services, add content, and begin querying.
- Indexing: pick files or directories from disk, track size/date metadata, and remove items; state is persisted via `useKV` storage.
- Search surface: conversational search layout with empty states and placeholders for query/reply flows.
- Metrics & health: Grafana-style charts (mock data) for latency, usage, quality, CPU/memory, and cache rates.
- Logs: tabbed log viewers for Ollama, MCP, and REST API streams with start/pause/stop, filtering, download, and clear.
- Settings: Ollama connection + model parameters (temperature, top-p, top-k, context, seed) with save/test actions; future OpenAI/OneDrive and Gemini/Drive entries marked as “coming soon.”

## Tech stack
- React 19 + TypeScript + Vite
- Tailwind CSS 4 + GitHub Spark UI primitives (shadcn-style components)
- Recharts for charts, Sonner for toasts, Phosphor icons for system glyphs

## Run it locally
1. Install deps: `npm install`
2. Start dev server: `npm run dev` (Vite, typically on http://localhost:5173)
3. Lint: `npm run lint`
4. Build: `npm run build`, preview: `npm run preview`

Requires Node 18+.

## Project map
- `src/App.tsx`: top-level tabs (Dashboard, Index, Search, Metrics, Logs, Settings) and feature wiring.
- `src/components/MetricsDashboard.tsx`: metric cards and chart groupings.
- `src/components/LogsViewer.tsx`: mock streaming log panels with controls and filters.
- `src/components/MetricsChart.tsx`: reusable recharts wrapper with mock data generation.
- `src/styles`, `src/index.css`, `src/main.css`: theme and layout styles; `main.tsx` mounts the app with Spark providers.

## Extending to a real stack
- Replace mock data generators (`MetricsChart`, `LogsViewer`) with API calls to your observability and log backends.
- Swap `useKV` storage (Spark-provided KV persistence) for real index/query endpoints; hook `handleAddFile/Directory` and search submission to your ingestion + RAG services.
- Wire `ollamaConfig` values into real connectivity checks and start/stop controls backed by your orchestrator.

## License
MIT (see `LICENSE`).
