import { useRef, useState, useCallback } from 'react'

/**
 * Hook to make an element draggable within its parent container.
 * Uses Pointer Events API for better touch and mouse support.
 * Supports handle-based dragging where only a handle element triggers dragging.
 * 
 * @param onPositionChange - Optional callback when position changes
 * @param handleRef - Optional ref to a handle element. If provided, only the handle will trigger dragging.
 * @returns Object with ref, position, and drag state
 */
export function useDraggable(
  onPositionChange?: (x: number, y: number) => void,
  handleRef?: React.RefObject<HTMLElement>
) {
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [initialPos, setInitialPos] = useState({ x: 0, y: 0 })
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const elementRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    // Only allow left mouse button or touch
    if (e.button !== 0 && e.pointerType === 'mouse') return

    // If handleRef is provided, only allow dragging from the handle
    if (handleRef?.current && e.target !== handleRef.current && !handleRef.current.contains(e.target as Node)) {
      return
    }

    const element = elementRef.current
    // Find the container - it should be the parent of the wrapper div
    const wrapper = element?.parentElement
    const container = containerRef.current || wrapper?.parentElement
    if (!element || !container) return

    // Get initial positions
    const rect = element.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    
    const elementPos = {
      x: rect.left - containerRect.left,
      y: rect.top - containerRect.top
    }
    
    setInitialPos(elementPos)
    setPosition(elementPos)
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    })
    setIsDragging(true)
    
    // Capture pointer for smooth dragging
    element.setPointerCapture(e.pointerId)
    if (handleRef?.current) {
      handleRef.current.style.cursor = 'grabbing'
    } else {
      element.style.cursor = 'grabbing'
    }
  }, [handleRef])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return

    const element = elementRef.current
    // Find the container - it should be the parent of the wrapper div
    const wrapper = element?.parentElement
    const container = containerRef.current || wrapper?.parentElement
    if (!element || !container) return

    const containerRect = container.getBoundingClientRect()
    const elementRect = element.getBoundingClientRect()
    
    // Calculate new position
    let newX = e.clientX - containerRect.left - dragOffset.x
    let newY = e.clientY - containerRect.top - dragOffset.y

    // Constrain to container bounds - allow full width movement
    // Use container padding as the constraint boundary
    const padding = parseFloat(getComputedStyle(container).padding || '0')
    const maxX = containerRect.width - elementRect.width - padding
    const maxY = containerRect.height - elementRect.height - padding
    
    newX = Math.max(padding, Math.min(newX, maxX))
    newY = Math.max(padding, Math.min(newY, maxY))

    setPosition({ x: newX, y: newY })
    
    if (onPositionChange) {
      onPositionChange(newX, newY)
    }
  }, [isDragging, dragOffset, onPositionChange])

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    const element = elementRef.current
    if (element) {
      element.releasePointerCapture(e.pointerId)
      if (handleRef?.current) {
        handleRef.current.style.cursor = 'grab'
      } else {
        element.style.cursor = 'grab'
      }
    }
    setIsDragging(false)
  }, [handleRef])

  return {
    elementRef,
    containerRef,
    position,
    isDragging,
    handlers: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerUp,
      onPointerCancel: handlePointerUp,
    }
  }
}

