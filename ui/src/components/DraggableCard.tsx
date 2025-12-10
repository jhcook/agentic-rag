import { ReactNode, useRef, useEffect, useState, useMemo } from 'react'
import { Card } from '@/components/ui/card'
import { useDraggable } from '@/hooks/use-draggable'
import { cn } from '@/lib/utils'
import { GripVertical } from 'lucide-react'

interface DraggableCardProps {
  id: string
  children: ReactNode
  className?: string
  defaultPosition?: { x: number; y: number }
  onPositionChange?: (id: string, x: number, y: number) => void
  storageKey?: string
  index?: number
  columns?: number
  gap?: number
  cardWidth?: number
  cardHeight?: number
}

/**
 * A draggable card component that can be moved around within its container.
 * Position is persisted to localStorage if storageKey is provided.
 * Supports initial grid positioning based on index and columns.
 */
export function DraggableCard({
  id,
  children,
  className,
  defaultPosition,
  onPositionChange,
  storageKey,
  index = 0,
  columns = 3,
  gap = 16,
  cardWidth = 300,
  cardHeight = 150
}: DraggableCardProps) {
  const [savedPosition, setSavedPosition] = useState<{ x: number; y: number } | null>(null)
  const [initialized, setInitialized] = useState(false)

  // Calculate grid position
  const gridPosition = useMemo(() => {
    if (defaultPosition) return defaultPosition
    const row = Math.floor(index / columns)
    const col = index % columns
    return {
      x: col * (cardWidth + gap) + gap,
      y: row * (cardHeight + gap) + gap
    }
  }, [defaultPosition, index, columns, cardWidth, gap, cardHeight])

  // Load saved position from localStorage
  useEffect(() => {
    if (storageKey && !initialized) {
      try {
        const saved = localStorage.getItem(storageKey)
        if (saved) {
          const positions = JSON.parse(saved)
          if (positions[id]) {
            setSavedPosition(positions[id])
          }
        }
      } catch (e) {
        console.error('Failed to load saved position:', e)
      }
      setInitialized(true)
    }
  }, [id, storageKey, initialized])

  const handlePositionChange = (x: number, y: number) => {
    if (onPositionChange) {
      onPositionChange(id, x, y)
    }

    // Save to localStorage
    if (storageKey) {
      try {
        const saved = localStorage.getItem(storageKey)
        const positions = saved ? JSON.parse(saved) : {}
        positions[id] = { x, y }
        localStorage.setItem(storageKey, JSON.stringify(positions))
        setSavedPosition({ x, y })
      } catch (e) {
        console.error('Failed to save position:', e)
      }
    }
  }

  const handleRef = useRef<HTMLDivElement>(null)
  const { elementRef, containerRef, position, isDragging, handlers } = useDraggable(handlePositionChange, handleRef as React.RefObject<HTMLElement>)
  const [calculatedWidth, setCalculatedWidth] = useState(cardWidth)

  // Calculate responsive width based on container
  useEffect(() => {
    const updateWidth = () => {
      // Find the DraggableCardContainer - it has 'relative' class and contains this card
      const element = elementRef.current
      if (!element) return

      // Traverse up to find the container with 'relative' class (DraggableCardContainer)
      let container: HTMLElement | null = element.parentElement
      while (container && !container.classList.contains('relative')) {
        container = container.parentElement
      }

      if (container) {
        // Set container ref for drag constraints
        if (containerRef) {
          containerRef.current = container as HTMLDivElement
        }

        // Calculate card width based on container's actual width
        const containerWidth = container.getBoundingClientRect().width
        const containerPadding = gap * 2
        const availableWidth = containerWidth - containerPadding
        const calculated = Math.floor((availableWidth - (gap * (columns - 1))) / columns)
        const newWidth = Math.max(calculated, 200) // Minimum 200px

        setCalculatedWidth(newWidth)
      }
    }

    // Use ResizeObserver to watch for container size changes
    const element = elementRef.current
    if (element) {
      let container: HTMLElement | null = element.parentElement
      while (container && !container.classList.contains('relative')) {
        container = container.parentElement
      }

      if (container) {
        const resizeObserver = new ResizeObserver(() => {
          updateWidth()
        })
        resizeObserver.observe(container)

        // Initial update with a small delay to ensure layout is complete
        const timeoutId = setTimeout(updateWidth, 50)

        // Also listen to window resize
        window.addEventListener('resize', updateWidth)

        return () => {
          clearTimeout(timeoutId)
          resizeObserver.disconnect()
          window.removeEventListener('resize', updateWidth)
        }
      }
    }

    // Fallback to window resize listener
    const timeoutId = setTimeout(updateWidth, 50)
    window.addEventListener('resize', updateWidth)
    return () => {
      clearTimeout(timeoutId)
      window.removeEventListener('resize', updateWidth)
    }
  }, [columns, gap, elementRef, containerRef])

  // Use saved position, default position, or current position
  // Recalculate grid position when width changes to maintain alignment
  const currentPosition = useMemo(() => {
    if (savedPosition && initialized) {
      // User has manually positioned this card - keep saved position
      return savedPosition
    }
    if (isDragging) return position
    if (defaultPosition) return defaultPosition
    // Recalculate grid position with current calculated width
    const row = Math.floor(index / columns)
    const col = index % columns
    return {
      x: col * (calculatedWidth + gap) + gap,
      y: row * (cardHeight + gap) + gap
    }
  }, [savedPosition, initialized, isDragging, position, defaultPosition, index, columns, calculatedWidth, gap, cardHeight])

  return (
    <div
      ref={elementRef}
      className={cn(
        'absolute transition-all duration-200',
        isDragging && 'shadow-2xl scale-105 z-50',
        !isDragging && 'shadow-sm hover:shadow-md'
      )}
      style={{
        left: `${currentPosition.x}px`,
        top: `${currentPosition.y}px`,
        width: `${calculatedWidth}px`,
        height: `${cardHeight}px`,
        willChange: isDragging ? 'transform' : 'auto'
      }}
    >
      <Card className={cn('w-full h-full flex flex-col relative group', className)}>
        <div
          ref={handleRef}
          className={cn(
            'absolute top-2 right-2 p-1.5 rounded-md cursor-grab active:cursor-grabbing',
            'opacity-0 group-hover:opacity-100 transition-opacity duration-200',
            'bg-background/90 backdrop-blur-sm border border-border/50',
            'hover:bg-background hover:border-border shadow-sm',
            'z-10 touch-none select-none'
          )}
          style={{
            opacity: isDragging ? 1 : undefined
          }}
          {...handlers}
          onMouseEnter={(e) => {
            if (!isDragging) {
              e.currentTarget.style.opacity = '1'
            }
          }}
          onMouseLeave={(e) => {
            if (!isDragging) {
              e.currentTarget.style.opacity = ''
            }
          }}
        >
          <GripVertical className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="flex-1 overflow-auto">
          {children}
        </div>
      </Card>
    </div>
  )
}

