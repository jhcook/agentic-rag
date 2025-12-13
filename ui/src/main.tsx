import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ErrorBoundary } from "react-error-boundary";

import App from './App.tsx'
import { ErrorFallback } from './ErrorFallback.tsx'
import { TooltipProvider } from './components/ui/tooltip.tsx'
import { useTheme } from './hooks/use-theme.ts'

import "./index.css"

/**
 * Theme provider component that applies system theme preference.
 */
function ThemeProvider({ children }: { children: React.ReactNode }) {
  useTheme()
  return <>{children}</>
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <ThemeProvider>
        <TooltipProvider>
          <App />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  </StrictMode>
)
