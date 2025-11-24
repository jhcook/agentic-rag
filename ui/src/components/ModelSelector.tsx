import { useEffect, useState } from 'react'
import { Check, ChevronsUpDown, Box } from 'lucide-react'
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

export function ModelSelector({ config, onModelSelect }: { config: any, onModelSelect: (model: string) => void }) {
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState<string>("local")

  // Fetch current mode to decide if we should show the selector
  useEffect(() => {
    const fetchMode = async () => {
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/mode`)
        if (res.ok) {
          const data = await res.json()
          setMode(data.mode)
        }
      } catch (e) {
        // ignore
      }
    }
    fetchMode()
    // Poll for mode changes every few seconds
    const interval = setInterval(fetchMode, 2000)
    return () => clearInterval(interval)
  }, [config])

  // Fetch models if mode is manual (Google)
  useEffect(() => {
    if (mode !== 'manual') {
        setModels([])
        return
    }

    const fetchModels = async () => {
      setLoading(true)
      const host = config?.ragHost || '127.0.0.1'
      const port = config?.ragPort || '8001'
      const base = (config?.ragPath || 'api').replace(/^\/+|\/+$/g, '')
      try {
        const res = await fetch(`http://${host}:${port}/${base}/config/models`)
        if (res.ok) {
          const data = await res.json()
          if (Array.isArray(data.models)) {
            setModels(data.models)
            
            // Try to restore from localStorage
            const savedModel = localStorage.getItem("gemini_model")
            
            if (savedModel && data.models.includes(savedModel)) {
                setSelectedModel(savedModel)
                onModelSelect(savedModel)
            } else if (data.models.length > 0) {
                // Default to first model (which is sorted to be latest pro)
                const defaultModel = data.models[0]
                setSelectedModel(defaultModel)
                onModelSelect(defaultModel)
            }
          }
        }
      } catch (e) {
        console.error("Failed to fetch models", e)
      } finally {
        setLoading(false)
      }
    }
    fetchModels()
  }, [config, mode])

  const handleSelect = (currentValue: string) => {
    setSelectedModel(currentValue)
    onModelSelect(currentValue)
    localStorage.setItem("gemini_model", currentValue)
    setOpen(false)
  }

  if (mode !== 'manual' || models.length === 0) {
    return null
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[200px] justify-between"
          disabled={loading}
        >
          <div className="flex items-center gap-2 truncate">
            <Box className="h-4 w-4" />
            {selectedModel || "Select model..."}
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search model..." />
          <CommandList>
            <CommandEmpty>No model found.</CommandEmpty>
            <CommandGroup>
              {models.map((model) => (
                <CommandItem
                  key={model}
                  value={model}
                  onSelect={() => handleSelect(model)}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedModel === model ? "opacity-100" : "opacity-0"
                    )}
                  />
                  {model}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
