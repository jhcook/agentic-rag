# Agentic RAG Control Panel
Single-page console for managing the agentic document retrieval-and-chat system. The UI connects to the backend services to manage indexing, configuration, and search operations.

## What’s here
- **Dashboards**: System status cards and onboarding steps.
- **Indexing**: File manager to index local files and directories. Connects to the backend to process documents.
- **Search surface**: Conversational interface for RAG-based search and chat.
- **Metrics & health**: Real-time visualization of system metrics.
- **Logs**: Log viewers for system components.
- **Settings**: Dynamic configuration management. Update models, temperature, and system prompts in real-time without restarting services.

## Tech stack
- React 19 + TypeScript + Vite
- Tailwind CSS 4 + shadcn/ui components
- Recharts for charts, Sonner for toasts, Phosphor icons for system glyphs

## Run it locally
1. Install deps: `npm install`
2. Start dev server: `npm run dev` (Vite, typically on http://localhost:5173)
3. Lint: `npm run lint`
4. Build: `npm run build`, preview: `npm run preview`

Requires Node 18+.

## Project map
- `src/App.tsx`: Main application layout and routing.
- `src/components/SettingsDashboard.tsx`: Configuration management interface.
- `src/components/ChatInterface.tsx`: RAG search and chat interface.
- `src/components/FileManager.tsx`: Document indexing management.
- `src/components/MetricsDashboard.tsx`: System metrics visualization.
- `src/components/LogsViewer.tsx`: Log file viewer.

## Integration
The UI communicates with the backend services via REST API:
- **Configuration**: Reads/writes to `config/settings.json` via `/api/config` endpoints.
- **Indexing**: Triggers document processing via `/api/index_path`.
- **Search**: Performs RAG search via `/api/search`.

## License
MIT (see `LICENSE`).
