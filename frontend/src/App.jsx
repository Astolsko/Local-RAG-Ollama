import React, { useEffect, useRef, useState } from 'react'
import {
  addSource,
  askQuestion,
  askQuestionStream,
  clearChatSession,
  deleteChat,
  deleteSource,
  endChat,
  getChat,
  getChatHistory,
  getHealth,
  getSources,
  startChat,
  uploadSource,
  getSettings,
  setSettings,
  getOllamaModels,
  getSource,
  getSystemPrompt,
  setSystemPrompt,
  submitFeedback,
} from './api'
import MetricsDashboard from './pages/MetricsDashboard'
import './App.css'

function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function toUiMessages(stored) {
  return (stored || []).map((m) => ({
    role: m.role,
    text: m.text,
    citations: m.citations,
    prompt_tokens: m.prompt_tokens,
    response_tokens: m.response_tokens,
    request_id: m.request_id,
  }))
}

function renderMessageContent(text) {
  if (!text) return null

  // Extract math blocks so we don't parse markdown inside LaTeX
  const regex = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^\$\n]+?\$|\\\([\s\S]+?\\\))/g
  const parts = text.split(regex)

  return parts.map((part, index) => {
    if (!part) return null

    if (part.startsWith('$$') && part.endsWith('$$')) {
      const math = part.slice(2, -2)
      return renderMathBlock(math, true, index)
    }
    if (part.startsWith('\\[') && part.endsWith('\\]')) {
      const math = part.slice(2, -2)
      return renderMathBlock(math, true, index)
    }
    if (part.startsWith('$') && part.endsWith('$')) {
      const math = part.slice(1, -1)
      return renderMathBlock(math, false, index)
    }
    if (part.startsWith('\\(') && part.endsWith('\\)')) {
      const math = part.slice(2, -2)
      return renderMathBlock(math, false, index)
    }

    return <span key={index}>{parseMarkdown(part)}</span>
  })
}

function renderMathBlock(math, displayMode, index) {
  if (window.katex) {
    try {
      const html = window.katex.renderToString(math, {
        displayMode: displayMode,
        throwOnError: false,
        trust: true,
      })
      if (displayMode) {
        return (
          <div 
            key={index} 
            className="katex-display-container" 
            style={{ overflowX: 'auto', overflowY: 'hidden', width: '100%', maxWidth: '100%', margin: '0.5rem 0' }}
            dangerouslySetInnerHTML={{ __html: html }} 
          />
        )
      } else {
        return <span key={index} dangerouslySetInnerHTML={{ __html: html }} />
      }
    } catch (e) {
      // Fallback below
    }
  }
  return displayMode ? <pre key={index}>{math}</pre> : <code key={index}>{math}</code>
}

function parseMarkdown(text) {
  const codeBlockRegex = /(```[\s\S]*?```)/g
  const blocks = text.split(codeBlockRegex)

  return blocks.map((block, idx) => {
    if (block.startsWith('```') && block.endsWith('```')) {
      const lines = block.slice(3, -3).split('\n')
      let lang = ''
      let codeText = block.slice(3, -3)
      if (lines[0] && !lines[0].includes(' ') && lines[0].length < 15) {
        lang = lines[0]
        codeText = lines.slice(1).join('\n')
      }
      return (
        <pre key={idx} className="md-code-block" style={{ background: 'rgba(139, 117, 91, 0.05)', padding: '0.75rem', borderRadius: '8px', overflowX: 'auto', border: '1px solid rgba(139, 117, 91, 0.15)', margin: '0.5rem 0' }}>
          {lang && <div className="code-lang" style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#8e6843', fontWeight: 'bold', marginBottom: '0.25rem' }}>{lang}</div>}
          <code style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>{codeText.trim()}</code>
        </pre>
      )
    }

    const lines = block.split('\n')
    return lines.map((line, lIdx) => {
      // Headers
      const headerMatch = line.match(/^(#{1,6})\s+(.*)$/)
      if (headerMatch) {
        const level = headerMatch[1].length
        const content = headerMatch[2]
        const Tag = `h${Math.min(level + 2, 6)}`
        return <Tag key={lIdx} className="md-header" style={{ margin: '0.75rem 0 0.35rem 0', color: '#5c4d3c', fontWeight: '700' }}>{parseInlineMarkdown(content)}</Tag>
      }

      // Lists
      const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/)
      if (listMatch) {
        return (
          <div key={lIdx} style={{ paddingLeft: '1rem', display: 'flex', gap: '0.5rem', margin: '0.2rem 0' }}>
            <span>&bull;</span>
            <span>{parseInlineMarkdown(listMatch[3])}</span>
          </div>
        )
      }

      return (
        <span key={lIdx} style={{ display: 'block', minHeight: '1.2em' }}>
          {parseInlineMarkdown(line)}
        </span>
      )
    })
  })
}

function parseInlineMarkdown(text) {
  if (!text) return ''
  const inlineRegex = /(\*\*.*?\*\*|`.*?`)/g
  const parts = text.split(inlineRegex)

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={index} style={{ background: 'rgba(139, 117, 91, 0.08)', padding: '0.1rem 0.3rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.9em' }}>{part.slice(1, -1)}</code>
    }
    return part
  })
}

function App() {
  const [health, setHealth] = useState(null)
  const [sources, setSources] = useState([])
  const [name, setName] = useState('')
  const [text, setText] = useState('')
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [history, setHistory] = useState([])
  const [viewingId, setViewingId] = useState(null)
  
  // Selected sources for search filtering
  const [selectedSourceIds, setSelectedSourceIds] = useState([])
  const [showAddSourceModal, setShowAddSourceModal] = useState(false)

  // Settings Panel/Modal states
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [settingsData, setSettingsData] = useState({
    OLLAMA_BASE: '',
    EMBED_MODEL: '',
    LLM_MODEL: '',
    REDIS_URL: '',
    CHUNK_SIZE: 500,
    CHUNK_OVERLAP: 50,
    TOP_K: 4,
  })
  const [ollamaModels, setOllamaModels] = useState([])
  const [systemPromptText, setSystemPromptText] = useState('')
  const [settingsTab, setSettingsTab] = useState('general') // 'general', 'rag', 'prompt'
  const [loadingSettings, setLoadingSettings] = useState(false)

  // Source Preview Modal states
  const [showSourcePreviewModal, setShowSourcePreviewModal] = useState(false)
  const [previewSource, setPreviewSource] = useState(null)
  const [highlightedChunkIndex, setHighlightedChunkIndex] = useState(null)
  const [expandedCitationId, setExpandedCitationId] = useState(null)
  
  const [autoSaveHistory, setAutoSaveHistory] = useState(() => {
    const stored = localStorage.getItem('autoSaveHistory')
    return stored === null ? true : stored === 'true'
  })

  const [currentView, setCurrentView] = useState('chat') // 'chat' | 'metrics'
  
  const chatEnd = useRef(null)
  const fileRef = useRef(null)

  const isReadOnly = Boolean(viewingId)

  const refresh = async () => {
    const [h, s, hist] = await Promise.all([
      getHealth(),
      getSources(),
      getChatHistory(),
    ])
    setHealth(h)
    setSources(s)
    setHistory(hist)

    // Select all sources by default if they are loaded and not already tracked
    setSelectedSourceIds((prev) => {
      const sourceIds = s.map((src) => src.id)
      // Keep only selected IDs that still exist in the current sources list
      const validPrev = prev.filter((id) => sourceIds.includes(id))
      
      // If we have new sources, auto-select them
      const newIds = sourceIds.filter((id) => !prev.includes(id))
      return [...validPrev, ...newIds]
    })
  }

  const beginSession = async () => {
    const { session_id } = await startChat()
    setSessionId(session_id)
    setMessages([])
    setViewingId(null)
    setQuestion('')
    return session_id
  }

  useEffect(() => {
    localStorage.setItem('autoSaveHistory', autoSaveHistory)
  }, [autoSaveHistory])

  // Load active session from localStorage on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem('activeSessionId')
    const storedMessages = localStorage.getItem('activeMessages')
    if (storedSessionId && storedMessages) {
      setSessionId(storedSessionId)
      try {
        setMessages(JSON.parse(storedMessages))
      } catch (e) {
        setMessages([])
      }
      setViewingId(null)
      refresh().catch((e) => setError(e.message))
    } else {
      beginSession()
        .then(() => refresh())
        .catch((e) => setError(e.message))
    }
  }, [])

  // Persist active session messages and sessionId to localStorage
  useEffect(() => {
    if (!viewingId) {
      if (sessionId) {
        localStorage.setItem('activeSessionId', sessionId)
        localStorage.setItem('activeMessages', JSON.stringify(messages))
      } else {
        localStorage.removeItem('activeSessionId')
        localStorage.removeItem('activeMessages')
      }
    }
  }, [messages, sessionId, viewingId])

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  const handleAddSource = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await addSource(name.trim(), text.trim())
      setName('')
      setText('')
      setShowAddSourceModal(false)
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleUploadSourceFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setBusy(true)
    try {
      await uploadSource(file, name.trim() || file.name)
      setName('')
      setShowAddSourceModal(false)
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      e.target.value = ''
    }
  }

  const handleDeleteSource = async (id) => {
    setError('')
    setBusy(true)
    try {
      await deleteSource(id)
      setSelectedSourceIds((prev) => prev.filter((x) => x !== id))
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleNewChat = async () => {
    setError('')
    setBusy(true)
    try {
      if (sessionId && messages.length > 0) {
        if (autoSaveHistory) {
          await endChat(sessionId)
        } else {
          await clearChatSession(sessionId)
        }
        await refresh()
      }
      await beginSession()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleEndChat = async () => {
    if (!sessionId || messages.length === 0) return
    setError('')
    setBusy(true)
    try {
      if (autoSaveHistory) {
        await endChat(sessionId)
      } else {
        await clearChatSession(sessionId)
      }
      await refresh()
      await beginSession()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleOpenHistory = async (id) => {
    setError('')
    setBusy(true)
    try {
      const chat = await getChat(id)
      setViewingId(id)
      setMessages(toUiMessages(chat.messages))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleDeleteHistory = async (id) => {
    setError('')
    setBusy(true)
    try {
      await deleteChat(id)
      if (viewingId === id) {
        setViewingId(null)
        setMessages([])
        await beginSession()
      }
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleResume = async () => {
    setViewingId(null)
    setMessages([])
    await beginSession()
  }

  const handleAsk = async (e) => {
    e.preventDefault()
    const q = question.trim()
    if (!q || !sessionId || isReadOnly) return

    setError('')
    setQuestion('')
    
    // Append user query and empty assistant bubble with streaming flag
    setMessages((m) => [
      ...m,
      { role: 'user', text: q },
      { role: 'assistant', text: '', citations: [], cached: false, streaming: true }
    ])
    setBusy(true)
    
    try {
      await askQuestionStream(
        q,
        sessionId,
        selectedSourceIds,
        (citData) => {
          setMessages((m) => {
            const next = [...m]
            const lastIdx = next.length - 1
            if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
              next[lastIdx] = {
                ...next[lastIdx],
                citations: citData.citations,
                cached: citData.cached,
                confidence: citData.confidence,
                prompt_tokens: citData.prompt_tokens,
                response_tokens: citData.response_tokens,
                request_id: citData.request_id
              }
            }
            return next
          })
        },
        (chunkText) => {
          setMessages((m) => {
            const next = [...m]
            const lastIdx = next.length - 1
            if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
              next[lastIdx] = {
                ...next[lastIdx],
                text: next[lastIdx].text + chunkText
              }
            }
            return next
          })
        },
        (err) => {
          setError(err.message)
          // rollback
          setMessages((m) => {
            const userMsg = m[m.length - 2]
            setQuestion(userMsg ? userMsg.text : q)
            return m.slice(0, -2)
          })
        }
      )
    } finally {
      setMessages((m) => {
        const next = [...m]
        const lastIdx = next.length - 1
        if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
          next[lastIdx] = {
            ...next[lastIdx],
            streaming: false
          }
        }
        return next
      })
      setBusy(false)
    }
  }

  const handleOpenSettings = async () => {
    setError('')
    setShowSettingsModal(true)
    setLoadingSettings(true)
    try {
      const [currSettings, models, prompt] = await Promise.all([
        getSettings(),
        getOllamaModels(),
        getSystemPrompt()
      ])
      setSettingsData(currSettings)
      setOllamaModels(models.models || [])
      setSystemPromptText(prompt.text || '')
      setSettingsTab('general')
    } catch (err) {
      setError('Failed to load settings: ' + err.message)
    } finally {
      setLoadingSettings(false)
    }
  }

  const handleSaveSettings = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await Promise.all([
        setSettings(settingsData),
        setSystemPrompt(systemPromptText)
      ])
      setShowSettingsModal(false)
      await refresh()
    } catch (err) {
      setError('Failed to save settings: ' + err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleOpenSourcePreview = async (sourceId, highlightChunkIdx = null) => {
    setError('')
    try {
      const detail = await getSource(sourceId)
      setPreviewSource(detail)
      setHighlightedChunkIndex(highlightChunkIdx)
      setShowSourcePreviewModal(true)
      
      if (highlightChunkIdx !== null) {
        setTimeout(() => {
          const el = document.getElementById(`chunk-${highlightChunkIdx}`)
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }
        }, 300)
      }
    } catch (err) {
      setError('Failed to load source details: ' + err.message)
    }
  }

  const toggleSourceSelect = (id) => {
    setSelectedSourceIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((x) => x !== id)
      } else {
        return [...prev, id]
      }
    })
  }

  const handleFeedback = async (messageIndex, requestId, feedbackType) => {
    try {
      await submitFeedback(requestId, feedbackType)
      setMessages((prev) => {
        const next = [...prev]
        next[messageIndex] = {
          ...next[messageIndex],
          user_feedback: feedbackType
        }
        return next
      })
    } catch (err) {
      console.error(err)
      setError('Failed to submit feedback: ' + err.message)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <button type="button" className="ghost settings-toggle" onClick={handleOpenSettings} title="Settings">
            ⚙️ Settings
          </button>
          <div className="view-switch" style={{ marginLeft: '1rem', display: 'inline-flex', gap: '0.25rem', background: 'rgba(255,255,255,0.04)', padding: '0.25rem', borderRadius: '8px' }}>
            <button 
              type="button" 
              className={`ghost ${currentView === 'chat' ? 'active-view' : ''}`}
              style={{ padding: '0.25rem 0.75rem', fontSize: '0.85rem', borderRadius: '6px', cursor: 'pointer', background: currentView === 'chat' ? 'rgba(52, 152, 219, 0.2)' : 'none', color: currentView === 'chat' ? '#3498db' : '#94a3b8', border: 'none' }}
              onClick={() => setCurrentView('chat')}
            >
              💬 Chat
            </button>
            <button 
              type="button" 
              className={`ghost ${currentView === 'metrics' ? 'active-view' : ''}`}
              style={{ padding: '0.25rem 0.75rem', fontSize: '0.85rem', borderRadius: '6px', cursor: 'pointer', background: currentView === 'metrics' ? 'rgba(52, 152, 219, 0.2)' : 'none', color: currentView === 'metrics' ? '#3498db' : '#94a3b8', border: 'none' }}
              onClick={() => setCurrentView('metrics')}
            >
              📊 Metrics
            </button>
          </div>
        </div>
        {health && (
          <div className="meta">
            <span>{health.llm_model}</span>
            <span>{health.sources} source{health.sources !== 1 ? 's' : ''}</span>
            <span className={health.redis ? 'ok' : 'bad'}>
              redis {health.redis ? 'on' : 'off'}
            </span>
          </div>
        )}
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      {currentView === 'metrics' ? (
        <MetricsDashboard />
      ) : (
        <main className="layout">
        {/* Left Column: History Panel */}
        <aside className="panel history-panel">
          <div className="panel-header">
            <h2>History</h2>
            <label className="toggle-label" title="Keep previous chat history stored">
              <input
                type="checkbox"
                checked={autoSaveHistory}
                onChange={(e) => setAutoSaveHistory(e.target.checked)}
              />
              Auto-Save
            </label>
          </div>
          <ul className="history-list">
            {history.length === 0 && (
              <li className="empty">Ended chats appear here.</li>
            )}
            {history.map((h) => (
              <li key={h.id} className={`history-item ${viewingId === h.id ? 'active' : ''}`}>
                <button
                  type="button"
                  className="history-open"
                  disabled={busy}
                  onClick={() => handleOpenHistory(h.id)}
                >
                  <strong>{h.title}</strong>
                  <span>
                    {formatDate(h.created_at)} · {h.message_count} msg
                  </span>
                </button>
                <button
                  type="button"
                  className="ghost danger"
                  disabled={busy}
                  onClick={() => handleDeleteHistory(h.id)}
                  aria-label={`Delete ${h.title}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </aside>

        {/* Middle Column: Chat Panel */}
        <section className="panel chat-panel">
          <div className="chat-toolbar">
            <h2>{isReadOnly ? 'Saved chat' : 'Active chat'}</h2>
            <div className="row">
              {!isReadOnly && (
                <>
                  <button type="button" className="ghost" disabled={busy} onClick={handleNewChat}>
                    New chat
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={busy || messages.length === 0}
                    onClick={handleEndChat}
                  >
                    {autoSaveHistory ? 'Save & end' : 'End session'}
                  </button>
                </>
              )}
              {isReadOnly && (
                <button type="button" className="ghost" disabled={busy} onClick={handleResume}>
                  Back to active
                </button>
              )}
            </div>
          </div>

          <div className="chat">
            {messages.length === 0 && (
              <p className="empty-chat">
                {isReadOnly
                  ? 'This saved chat has no messages.'
                  : 'Ask about your sources. Use (+) to add documents.'}
              </p>
            )}
            {messages.map((msg, i) => (
              <article key={i} className={`bubble ${msg.role} ${(msg.role === 'assistant' && msg.streaming && !msg.text) ? 'loading' : ''}`}>
                <div className="message-content">
                  {(msg.role === 'assistant' && msg.streaming && !msg.text) ? 'Thinking…' : renderMessageContent(msg.text)}
                </div>
                {msg.cached && <span className="tag">cached context</span>}
                {msg.confidence !== undefined && msg.role === 'assistant' && (
                  <span className="tag confidence-tag" style={{
                    backgroundColor: msg.confidence >= 0.8 ? 'rgba(46, 204, 113, 0.15)' : msg.confidence >= 0.5 ? 'rgba(241, 196, 15, 0.15)' : 'rgba(231, 76, 60, 0.15)',
                    color: msg.confidence >= 0.8 ? '#2ecc71' : msg.confidence >= 0.5 ? '#e1b10c' : '#e74c3c',
                    marginLeft: '0.5rem',
                    fontWeight: 'bold',
                    border: '1px solid currentColor'
                  }}>
                    Confidence: {Math.round(msg.confidence * 100)}%
                  </span>
                )}
                {msg.prompt_tokens !== undefined && msg.role === 'assistant' && (
                  <span className="tag token-tag" style={{
                    backgroundColor: 'rgba(52, 152, 219, 0.15)',
                    color: '#3498db',
                    marginLeft: '0.5rem',
                    fontWeight: 'bold',
                    border: '1px solid currentColor'
                  }}>
                    {msg.prompt_tokens === 0 && msg.response_tokens === 0 ? "Tokens: 0 (Cached)" : `Tokens: ${msg.prompt_tokens} prompt / ${msg.response_tokens} response`}
                  </span>
                )}
                {msg.role === 'assistant' && msg.request_id && (
                  <div className="feedback-buttons" style={{ display: 'inline-flex', gap: '0.25rem', marginLeft: '0.5rem', verticalAlign: 'middle' }}>
                    <button
                      type="button"
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0 0.2rem',
                        fontSize: '0.85rem',
                        opacity: msg.user_feedback === 1 ? 1 : 0.4,
                        transition: 'opacity 0.2s',
                        filter: msg.user_feedback === 1 ? 'drop-shadow(0 0 2px #2ecc71)' : 'none'
                      }}
                      onClick={() => handleFeedback(i, msg.request_id, 1)}
                      title="Thumbs Up"
                    >
                      👍
                    </button>
                    <button
                      type="button"
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0 0.2rem',
                        fontSize: '0.85rem',
                        opacity: msg.user_feedback === -1 ? 1 : 0.4,
                        transition: 'opacity 0.2s',
                        filter: msg.user_feedback === -1 ? 'drop-shadow(0 0 2px #e74c3c)' : 'none'
                      }}
                      onClick={() => handleFeedback(i, msg.request_id, -1)}
                      title="Thumbs Down"
                    >
                      👎
                    </button>
                  </div>
                )}
                {msg.citations?.length > 0 && (
                  <div className="citations-chips" style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {msg.citations.map((c, cIdx) => {
                      const chipId = `${i}-${cIdx}`
                      const isExpanded = expandedCitationId === chipId
                      return (
                        <div key={cIdx} style={{ display: 'flex', flexDirection: 'column', width: isExpanded ? '100%' : 'auto' }}>
                          <button
                            type="button"
                            className="citation-chip"
                            style={{
                              background: isExpanded ? 'rgba(52, 152, 219, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                              color: isExpanded ? '#3498db' : '#cbd5e1',
                              border: isExpanded ? '1px solid #3498db' : '1px solid rgba(255, 255, 255, 0.1)',
                              borderRadius: '16px',
                              padding: '0.2rem 0.5rem',
                              fontSize: '0.75rem',
                              cursor: 'pointer',
                              fontWeight: '600',
                              transition: 'all 0.15s ease',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                              alignSelf: 'flex-start'
                            }}
                            onClick={() => setExpandedCitationId(isExpanded ? null : chipId)}
                          >
                            <span>📎 [{cIdx + 1}]</span>
                            <span style={{ fontSize: '0.7rem', fontWeight: 'normal', opacity: 0.8, maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {c.source_name}
                            </span>
                          </button>
                          {isExpanded && (
                            <div className="citation-expand-box" style={{
                              marginTop: '0.4rem',
                              padding: '0.75rem',
                              background: 'rgba(30, 41, 59, 0.45)',
                              border: '1px solid rgba(255, 255, 255, 0.08)',
                              borderRadius: '8px',
                              fontSize: '0.8rem',
                              color: '#94a3b8',
                              width: '100%',
                              boxSizing: 'border-box',
                              lineHeight: '1.4'
                            }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.25rem' }}>
                                <span style={{ fontWeight: '700', color: '#f8fafc' }}>
                                  {c.source_file || c.source_name} {c.page_number !== undefined ? `(Page ${c.page_number})` : ''}
                                </span>
                                <button 
                                  type="button" 
                                  onClick={() => handleOpenSourcePreview(c.source_id, c.chunk_index)}
                                  style={{
                                    background: 'rgba(52, 152, 219, 0.15)',
                                    border: 'none',
                                    borderRadius: '4px',
                                    color: '#3498db',
                                    padding: '0.15rem 0.4rem',
                                    fontSize: '0.7rem',
                                    cursor: 'pointer',
                                    fontWeight: 'bold'
                                  }}
                                >
                                  View Source
                                </button>
                              </div>
                              <div style={{ color: '#cbd5e1', fontStyle: 'italic', wordBreak: 'break-word' }}>
                                "{c.snippet}"
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </article>
            ))}
            {busy && messages[messages.length - 1]?.role === 'user' && (
              <article className="bubble assistant loading">Thinking…</article>
            )}
            <div ref={chatEnd} />
          </div>

          <form className="ask-form" onSubmit={handleAsk}>
            <button
              type="button"
              className="ghost add-btn"
              disabled={busy || isReadOnly}
              onClick={() => setShowAddSourceModal(true)}
              title="Add source (.txt, .md, .pdf)"
            >
              +
            </button>
            <input
              type="text"
              placeholder={isReadOnly ? 'Read-only — open active chat' : 'Your question…'}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={busy || isReadOnly}
            />
            <button
              type="submit"
              disabled={busy || !question.trim() || isReadOnly}
            >
              Send
            </button>
          </form>
        </section>

        {/* Right Column: Sources Panel */}
        <aside className="panel sources-panel">
          <h2>Sources</h2>
          <ul className="source-list">
            {sources.length === 0 && (
              <li className="empty">No sources yet. Use (+) to add.</li>
            )}
            {sources.map((s) => (
              <li key={s.id} className="source-item">
                <div className="source-meta">
                  <input
                    type="checkbox"
                    className="source-checkbox"
                    checked={selectedSourceIds.includes(s.id)}
                    onChange={() => toggleSourceSelect(s.id)}
                  />
                  <div className="source-info clickable" onClick={() => handleOpenSourcePreview(s.id)}>
                    <strong>{s.name}</strong>
                    <span>{s.chunks} chunk{s.chunks !== 1 ? 's' : ''}</span>
                  </div>
                </div>
                <button
                  type="button"
                  className="ghost danger"
                  disabled={busy}
                  onClick={() => handleDeleteSource(s.id)}
                  aria-label={`Delete ${s.name}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </aside>
      </main>
      )}

      {/* Add Source Modal Overlay */}
      {showAddSourceModal && (
        <div className="modal-overlay" onClick={() => setShowAddSourceModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Add Source</h3>
              <button type="button" className="close-btn" onClick={() => setShowAddSourceModal(false)}>×</button>
            </div>
            <form onSubmit={handleAddSource}>
              <div className="modal-body">
                <input
                  type="text"
                  placeholder="Source name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={busy}
                />
                <textarea
                  placeholder="Paste text content here..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={6}
                  disabled={busy}
                />
                <div className="modal-divider">or upload a file</div>
                <div className="modal-file-picker">
                  <button
                    type="button"
                    className="ghost file-btn"
                    onClick={() => fileRef.current?.click()}
                    disabled={busy}
                  >
                    Select File (.txt, .md, .pdf)
                  </button>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".txt,.md,.pdf,text/plain,application/pdf"
                    hidden
                    onChange={handleUploadSourceFile}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setShowAddSourceModal(false)}
                  disabled={busy}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={busy || !name.trim() || !text.trim()}
                >
                  Add
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Settings Modal Overlay */}
      {showSettingsModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Settings</h3>
              <button type="button" className="close-btn" onClick={() => setShowSettingsModal(false)}>×</button>
            </div>
            
            {loadingSettings ? (
              <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 2rem' }}>
                <div className="spinner" style={{ border: '3px solid rgba(139, 117, 91, 0.1)', width: '32px', height: '32px', borderRadius: '50%', borderLeftColor: '#8b755b', animation: 'spin 1s linear infinite', marginBottom: '1rem' }}></div>
                <div style={{ color: '#6e6151', fontSize: '0.85rem', fontWeight: '600' }}>Loading RAG configuration...</div>
              </div>
            ) : (
              <>
                <div className="modal-tabs">
                  <button 
                    type="button" 
                    className={`tab-btn ${settingsTab === 'general' ? 'active' : ''}`}
                    onClick={() => setSettingsTab('general')}
                  >
                    General Settings
                  </button>
                  <button 
                    type="button" 
                    className={`tab-btn ${settingsTab === 'rag' ? 'active' : ''}`}
                    onClick={() => setSettingsTab('rag')}
                  >
                    RAG Parameters
                  </button>
                  <button 
                    type="button" 
                    className={`tab-btn ${settingsTab === 'prompt' ? 'active' : ''}`}
                    onClick={() => setSettingsTab('prompt')}
                  >
                    System Prompt
                  </button>
                </div>

                <form onSubmit={handleSaveSettings}>
                  <div className="modal-body">
                    {settingsTab === 'general' && (
                      <>
                        <div className="form-group">
                          <label>Ollama Base URL</label>
                          <input 
                            type="text" 
                            value={settingsData.OLLAMA_BASE || ''} 
                            onChange={(e) => setSettingsData({ ...settingsData, OLLAMA_BASE: e.target.value })} 
                            disabled={busy}
                            required
                          />
                        </div>
                        <div className="form-row">
                          <div className="form-group">
                            <label>LLM Model</label>
                            <select 
                              value={settingsData.LLM_MODEL || ''} 
                              onChange={(e) => setSettingsData({ ...settingsData, LLM_MODEL: e.target.value })} 
                              disabled={busy}
                            >
                              {ollamaModels.map((m) => (
                                <option key={m} value={m}>{m}</option>
                              ))}
                              {!ollamaModels.includes(settingsData.LLM_MODEL) && settingsData.LLM_MODEL && (
                                <option value={settingsData.LLM_MODEL}>{settingsData.LLM_MODEL}</option>
                              )}
                            </select>
                          </div>
                          <div className="form-group">
                            <label>Embedding Model</label>
                            <select 
                              value={settingsData.EMBED_MODEL || ''} 
                              onChange={(e) => setSettingsData({ ...settingsData, EMBED_MODEL: e.target.value })} 
                              disabled={busy}
                            >
                              {ollamaModels.map((m) => (
                                <option key={m} value={m}>{m}</option>
                              ))}
                              {!ollamaModels.includes(settingsData.EMBED_MODEL) && settingsData.EMBED_MODEL && (
                                <option value={settingsData.EMBED_MODEL}>{settingsData.EMBED_MODEL}</option>
                              )}
                            </select>
                          </div>
                        </div>
                        <div className="form-group">
                          <label>Redis Connection URL</label>
                          <input 
                            type="text" 
                            value={settingsData.REDIS_URL || ''} 
                            onChange={(e) => setSettingsData({ ...settingsData, REDIS_URL: e.target.value })} 
                            disabled={busy}
                            required
                          />
                        </div>
                      </>
                    )}

                    {settingsTab === 'rag' && (
                      <>
                        <div className="form-row">
                          <div className="form-group">
                            <label>Chunk Size (characters)</label>
                            <input 
                              type="number" 
                              min="10"
                              max="10000"
                              value={settingsData.CHUNK_SIZE || 500} 
                              onChange={(e) => setSettingsData({ ...settingsData, CHUNK_SIZE: parseInt(e.target.value) || 0 })} 
                              disabled={busy}
                              required
                            />
                          </div>
                          <div className="form-group">
                            <label>Chunk Overlap (characters)</label>
                            <input 
                              type="number" 
                              min="0"
                              max="5000"
                              value={settingsData.CHUNK_OVERLAP || 50} 
                              onChange={(e) => setSettingsData({ ...settingsData, CHUNK_OVERLAP: parseInt(e.target.value) || 0 })} 
                              disabled={busy}
                              required
                            />
                          </div>
                        </div>
                        <div className="form-group">
                          <label>Top K Chunks Retrieved</label>
                          <input 
                            type="number" 
                            min="1"
                            max="50"
                            value={settingsData.TOP_K || 4} 
                            onChange={(e) => setSettingsData({ ...settingsData, TOP_K: parseInt(e.target.value) || 0 })} 
                            disabled={busy}
                            required
                          />
                        </div>
                      </>
                    )}

                    {settingsTab === 'prompt' && (
                      <div className="form-group">
                        <label>Editable System Prompt</label>
                        <textarea 
                          rows={10}
                          value={systemPromptText} 
                          onChange={(e) => setSystemPromptText(e.target.value)} 
                          disabled={busy}
                          placeholder="Enter prompt guidelines..."
                          required
                        />
                      </div>
                    )}
                  </div>
                  
                  <div className="modal-footer">
                    <button 
                      type="button" 
                      className="ghost" 
                      onClick={() => setShowSettingsModal(false)}
                      disabled={busy}
                    >
                      Cancel
                    </button>
                    <button 
                      type="submit" 
                      disabled={busy}
                    >
                      Save Settings
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}

      {/* Source Preview Modal Overlay */}
      {showSourcePreviewModal && previewSource && (
        <div className="modal-overlay" onClick={() => setShowSourcePreviewModal(false)}>
          <div className="modal-content" style={{ maxWidth: '800px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Preview: {previewSource.name}</h3>
              <button type="button" className="close-btn" onClick={() => setShowSourcePreviewModal(false)}>×</button>
            </div>
            
            <div className="modal-body">
              <p style={{ fontSize: '0.85rem', color: '#6e6151', marginBottom: '1rem' }}>
                Total chunks: <strong>{previewSource.chunks?.length || 0}</strong> &middot; Click any citation in chat history to highlight a matched chunk.
              </p>
              
              <div className="source-chunks-list">
                {previewSource.chunks?.map((chunk) => (
                  <div 
                    key={chunk.chunk_index} 
                    id={`chunk-${chunk.chunk_index}`}
                    className={`chunk-card ${highlightedChunkIndex === chunk.chunk_index ? 'highlight-chunk' : ''}`}
                  >
                    <div className="chunk-header">
                      <span>Chunk #{chunk.chunk_index + 1}</span>
                      <span>Page {chunk.page_number} &middot; {chunk.topic}</span>
                    </div>
                    <div className="chunk-text">{chunk.text}</div>
                  </div>
                ))}
                {(!previewSource.chunks || previewSource.chunks.length === 0) && (
                  <p style={{ textAlign: 'center', padding: '2rem', color: '#6e6151' }}>No chunks available.</p>
                )}
              </div>
            </div>
            
            <div className="modal-footer">
              <button 
                type="button" 
                onClick={() => setShowSourcePreviewModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
