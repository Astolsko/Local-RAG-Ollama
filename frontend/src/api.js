const API = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = data.detail
    const msg =
      typeof d === 'string'
        ? d
        : Array.isArray(d)
          ? d.map((x) => x.msg).join(', ')
          : res.statusText
    throw new Error(msg)
  }
  return data
}

export const getHealth = () => request('/health')
export const getSources = () => request('/sources')
export const addSource = (name, text) =>
  request('/sources', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, text }),
  })
export const uploadSource = (file, name) => {
  const form = new FormData()
  form.append('file', file)
  if (name) form.append('name', name)
  return request('/sources/upload', { method: 'POST', body: form })
}
export const deleteSource = (id) => request(`/sources/${id}`, { method: 'DELETE' })

export const getSystemPrompt = () => request('/settings/system-prompt')
export const setSystemPrompt = (text) =>
  request('/settings/system-prompt', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

export const startChat = () => request('/chats/start', { method: 'POST' })
export const askQuestion = (question, sessionId, sourceIds) =>
  request('/chats/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, source_ids: sourceIds }),
  })
export const endChat = (sessionId, title) =>
  request('/chats/end', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, title }),
  })
export const clearChatSession = (sessionId) =>
  request(`/chats/clear/${sessionId}`, { method: 'POST' })

export const getChatHistory = () => request('/chats/history')
export const getChat = (id) => request(`/chats/history/${id}`)
export const deleteChat = (id) => request(`/chats/history/${id}`, { method: 'DELETE' })

// Dynamic Settings API
export const getSettings = () => request('/settings')
export const setSettings = (settings) =>
  request('/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
export const getOllamaModels = () => request('/settings/ollama-models')

// Source chunks details
export const getSource = (id) => request(`/sources/${id}`)

// Streaming ask question
export async function askQuestionStream(question, sessionId, sourceIds, onCitations, onChunk, onError) {
  try {
    const res = await fetch(`${API}/chats/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId, source_ids: sourceIds }),
    })
    
    if (!res.ok) {
      const data = await res.json().catch(() => ({}))
      const d = data.detail
      const msg =
        typeof d === 'string'
          ? d
          : Array.isArray(d)
            ? d.map((x) => x.msg).join(', ')
            : res.statusText
      throw new Error(msg)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      // Store the last partial line back in buffer
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const parsed = JSON.parse(line)
          if (parsed.citations !== undefined) {
            onCitations({ 
              citations: parsed.citations, 
              cached: parsed.cached, 
              confidence: parsed.confidence,
              prompt_tokens: parsed.prompt_tokens,
              response_tokens: parsed.response_tokens,
              request_id: parsed.request_id
            })
          } else if (parsed.text !== undefined) {
            onChunk(parsed.text)
          }
        } catch (e) {
          console.error("Failed to parse JSON stream line:", line, e)
        }
      }
    }
  } catch (err) {
    onError(err)
  }
}

// Submit user feedback (1 for thumbs up, -1 for thumbs down)
export async function submitFeedback(requestId, feedback) {
  const res = await fetch(`${API}/observability/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, feedback }),
  })
  if (!res.ok) throw new Error("Failed to submit feedback")
  return res.json()
}

// Fetch aggregated metrics
export async function fetchAggregatedMetrics() {
  const res = await fetch(`${API}/observability/metrics`)
  if (!res.ok) throw new Error("Failed to fetch metrics")
  return res.json()
}

// Fetch evaluation history
export async function fetchEvalHistory() {
  const res = await fetch(`${API}/observability/eval-history`)
  if (!res.ok) throw new Error("Failed to fetch evaluation history")
  return res.json()
}
