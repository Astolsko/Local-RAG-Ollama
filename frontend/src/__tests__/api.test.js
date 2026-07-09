import { describe, it, expect, vi, beforeEach } from 'vitest'
import { getHealth, askQuestion } from '../api'

global.fetch = vi.fn()

describe('api requests', () => {
  beforeEach(() => {
    fetch.mockReset()
  })

  it('getHealth fetches health endpoint', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'ok' }),
    })

    const data = await getHealth()
    expect(data).toEqual({ status: 'ok' })
    expect(fetch).toHaveBeenCalledWith('/api/health', expect.any(Object))
  })

  it('askQuestion sends request with source_ids', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ answer: 'test answer' }),
    })

    const data = await askQuestion('my question', 'session-123', ['src-1'])
    expect(data).toEqual({ answer: 'test answer' })
    expect(fetch).toHaveBeenCalledWith('/api/chats/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: 'my question',
        session_id: 'session-123',
        source_ids: ['src-1'],
      }),
    })
  })
})
