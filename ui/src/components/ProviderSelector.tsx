import { useEffect, useState } from 'react'
import { Check, ChevronsUpDown, Cloud, Server } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { toast } from 'sonner'

type ConfigMode = {
  mode: string
  available_modes: string[]
}

interface ProviderSelectorProps {
  config: any
  activeMode: string
  onModeChange: (mode: string) => void
}

export function ProviderSelector({ config, activeMode, onModeChange }: ProviderSelectorProps) {
  const [open, setOpen] = useState(false)
  const [availableModes, setAvailableModes] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const fetchMode = async () => {
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
    try {
      const res = await fetch(`http://${host}:${port}/${base}/config/mode`)
      if (res.ok) {
        const data = await res.json()
        setAvailableModes(data.available_modes || [])
        // Sync if backend reports different mode
        if (data.mode && data.mode !== activeMode) {
          onModeChange(data.mode)
        }
      }
    } catch (e) {
      console.error("Failed to fetch mode", e)
    }
  }

  useEffect(() => {
    // Fetch immediately on mount
    fetchMode()
    // Poll for mode changes more frequently initially, then less frequently
    const interval = setInterval(fetchMode, 2000) // Poll every 2 seconds
    return () => clearInterval(interval)
  }, [config, activeMode])

  const handleSelect = async (newMode: string) => {
    setLoading(true)
    const host = config?.ragHost || '127.0.0.1'
    const port = config?.ragPort || '8001'
    const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')

    try {
      const res = await fetch(`http://${host}:${port}/${base}/config/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode })
      })

      if (!res.ok) throw new Error('Failed to switch mode')

      onModeChange(newMode)
      toast.success(`Switched to ${newMode} mode`)
      setOpen(false)

    } catch (e) {
      toast.error('Failed to switch provider')
    } finally {
      setLoading(false)
    }
  }

  const providers = [
    { value: 'ollama', label: 'Ollama', icon: Server },
    { value: 'openai_assistants', label: 'OpenAI Assistants', icon: Cloud },
    { value: 'google_gemini', label: 'Gemini + Drive', icon: Cloud },
    { value: 'vertex_ai_search', label: 'Vertex AI Agent', icon: Cloud },
  ]

  const currentProvider = providers.find(p => p.value === activeMode) ||
    { value: activeMode || 'none', label: activeMode === 'none' ? 'No Provider' : (activeMode || 'No Provider'), icon: Server }

  // Filter providers based on available modes from backend
  const availableProviderList = providers.filter(p => availableModes.includes(p.value))

  // Check if current mode is actually available
  const isCurrentModeAvailable = availableModes.includes(activeMode)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn(
            "w-full justify-between",
            !isCurrentModeAvailable && "border-destructive/50 bg-destructive/10"
          )}
          disabled={loading || availableProviderList.length === 0}
        >
          <div className="flex items-center gap-2 truncate">
            <currentProvider.icon className={cn(
              "h-4 w-4",
              !isCurrentModeAvailable && "text-destructive"
            )} />
            <span className={cn(!isCurrentModeAvailable && "text-destructive")}>
              {availableProviderList.length === 0 ? "No Providers Available" : currentProvider.label}
              {!isCurrentModeAvailable && availableProviderList.length > 0 && " (Unavailable)"}
            </span>
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search provider..." />
          <CommandList>
            <CommandEmpty>No provider found.</CommandEmpty>
            <CommandGroup>
              {availableProviderList.map((provider) => (
                <CommandItem
                  key={provider.value}
                  value={provider.value}
                  onSelect={() => handleSelect(provider.value)}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      activeMode === provider.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                  <div className="flex items-center gap-2">
                    <provider.icon className="h-4 w-4" />
                    {provider.label}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
