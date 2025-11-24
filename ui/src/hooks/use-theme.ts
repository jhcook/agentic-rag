import { useEffect, useState } from 'react'

/**
 * Hook to detect and apply system theme preference (dark mode).
 * Automatically applies the 'dark' class to the document element
 * based on the system's color scheme preference.
 */
export function useTheme() {
  const [isDark, setIsDark] = useState(() => {
    // Initialize from existing class or system preference
    if (typeof window === 'undefined') return false
    const html = document.documentElement
    if (html.classList.contains('dark')) return true
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    /**
     * Get the current system preference.
     */
    const getSystemPreference = (): boolean => {
      if (typeof window === 'undefined') return false
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }

    /**
     * Apply or remove the dark class from the document element.
     */
    const applyTheme = (dark: boolean) => {
      const html = document.documentElement
      if (dark) {
        html.classList.add('dark')
        // Also ensure body gets the background
        document.body.style.backgroundColor = ''
        document.body.style.color = ''
      } else {
        html.classList.remove('dark')
        // Reset body styles
        document.body.style.backgroundColor = ''
        document.body.style.color = ''
      }
      setIsDark(dark)
    }

    // Set initial theme based on system preference (only if not already set by inline script)
    const html = document.documentElement
    const hasDarkClass = html.classList.contains('dark')
    if (!hasDarkClass) {
      const initialDark = getSystemPreference()
      applyTheme(initialDark)
    } else {
      setIsDark(true)
    }

    // Listen for changes to system preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      applyTheme(e.matches)
    }

    // Modern browsers support addEventListener
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange)
      return () => {
        mediaQuery.removeEventListener('change', handleChange)
      }
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleChange)
      return () => {
        mediaQuery.removeListener(handleChange)
      }
    }
  }, [])

  return { isDark }
}

