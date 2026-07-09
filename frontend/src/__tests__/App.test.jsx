import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import App from '../App'

// Mock the api module functions
vi.mock('../api', () => ({
  getHealth: vi.fn(() => Promise.resolve({ llm_model: 'qwen2.5', sources: 0, redis: true })),
  getSources: vi.fn(() => Promise.resolve([])),
  getChatHistory: vi.fn(() => Promise.resolve([])),
  startChat: vi.fn(() => Promise.resolve({ session_id: 'mock-session-123' })),
}))

describe('App component', () => {
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    
    // Mock localStorage
    const localStorageMock = (() => {
      let store = {}
      return {
        getItem: (key) => store[key] || null,
        setItem: (key, value) => { store[key] = value.toString() },
        removeItem: (key) => { delete store[key] },
        clear: () => { store = {} }
      }
    })()
    Object.defineProperty(global, 'localStorage', { value: localStorageMock })
  })

  it('renders sidebars and empty chat states', async () => {
    render(<App />)
    
    // Verify panels render
    expect(await screen.findByText('History')).not.toBeNull()
    expect(await screen.findByText('Sources')).not.toBeNull()
    
    // Verify no title header is present
    expect(screen.queryByText('RAG LLM')).toBeNull()
  })
})
