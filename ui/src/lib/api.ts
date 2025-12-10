
/**
 * Configuration for the API connection.
 * Can be sourced from localStorage or defaults.
 */
export const getApiBase = () => {
    // In a real app, you might use a context or global state store.
    // For now, mirroring the logic in App.tsx or providing safe defaults.
    // If App.tsx updates global config, this might desync unless we use shared state.
    // For the AutotuneCard, we can simpler read from localStorage or default.

    // Attempting to read from window if available (passed from App) or just defaults
    const host = '127.0.0.1'
    const port = '8001'
    const base = 'api'

    // If we have a saved config in localStorage, use it
    try {
        const saved = localStorage.getItem('ollamaConfig')
        if (saved) {
            const config = JSON.parse(saved)
            return {
                host: config.ragHost || host,
                port: config.ragPort || port,
                base: (config.ragPath || base).replace(/^\/+|\/+$/g, '')
            }
        }
    } catch (e) {
        console.warn("Failed to read config from storage", e)
    }

    return { host, port, base }
}
