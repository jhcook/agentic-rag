import { useEffect, useState } from 'react'
import { Loader2, CheckCircle2, Trash2 } from 'lucide-react'
import { Progress } from '@/components/ui/progress'

type DeletionStatus = {
  queue_size: number
  processing: boolean
  last_completed: number | null
  total_processed: number
}

export function DeletionStatusBar({ config }: { config: any }) {
  const [status, setStatus] = useState<DeletionStatus>({
    queue_size: 0,
    processing: false,
    last_completed: null,
    total_processed: 0
  })
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    const url = `http://${host}:${port}/${base}/documents/delete/status`

    const pollStatus = async () => {
      try {
        const res = await fetch(url)
        if (res.ok) {
          const data: DeletionStatus = await res.json()
          setStatus(data)
          
          // Show status bar when there's activity
          if (data.queue_size > 0 || data.processing) {
            setVisible(true)
          } else if (data.last_completed) {
            // Keep visible for 3 seconds after completion
            const timeSinceCompletion = Date.now() / 1000 - data.last_completed
            if (timeSinceCompletion < 3) {
              setVisible(true)
            } else {
              setVisible(false)
            }
          } else {
            setVisible(false)
          }
        }
      } catch (e) {
        console.error('Failed to fetch deletion status', e)
      }
    }

    // Poll every 500ms for responsive updates
    const interval = setInterval(pollStatus, 500)
    pollStatus() // Initial poll

    return () => clearInterval(interval)
  }, [config])

  if (!visible) return null

  const isComplete = !status.processing && status.queue_size === 0 && status.last_completed
  const Icon = isComplete ? CheckCircle2 : status.processing ? Loader2 : Trash2

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-t z-50">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center gap-4">
          <Icon className={`h-5 w-5 shrink-0 ${status.processing ? 'animate-spin' : ''} ${isComplete ? 'text-green-500' : 'text-primary'}`} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-4 mb-1">
              <p className="text-sm font-medium">
                {isComplete ? (
                  'Deletion Complete'
                ) : status.processing ? (
                  'Processing Deletions...'
                ) : (
                  `${status.queue_size} file${status.queue_size !== 1 ? 's' : ''} queued for deletion`
                )}
              </p>
              <p className="text-xs text-muted-foreground">
                {status.total_processed} processed
              </p>
            </div>
            {(status.processing || status.queue_size > 0) && (
              <Progress 
                value={status.processing ? 50 : 0} 
                className="h-1.5"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
