import { ReactNode, useRef, useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface DraggableCardContainerProps {
  children: ReactNode
  className?: string
  columns?: number
  gap?: number
}

/**
 * Container for draggable cards that maintains grid-like initial layout
 * but allows cards to be positioned absolutely for dragging.
 */
export function DraggableCardContainer({
  children,
  className,
  columns = 3,
  gap = 16
}: DraggableCardContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  return (
    <div
      ref={containerRef}
      className={cn('relative min-h-[400px] w-full', className)}
      style={{ 
        padding: `${gap}px`,
        boxSizing: 'border-box',
        width: '100%'
      }}
    >
      {children}
    </div>
  )
}

