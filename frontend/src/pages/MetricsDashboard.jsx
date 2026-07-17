import React, { useState, useEffect } from 'react'
import { fetchAggregatedMetrics, fetchEvalHistory } from '../api'

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)
  const [evalHistory, setEvalHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('performance') // 'performance' | 'retrieval' | 'usage'
  const [selectedStage, setSelectedStage] = useState('embed')

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [metRes, evalRes] = await Promise.all([
        fetchAggregatedMetrics(),
        fetchEvalHistory()
      ])
      setMetrics(metRes)
      setEvalHistory(evalRes)
    } catch (err) {
      console.error(err)
      setError(err.message || 'Failed to load observability data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  if (loading) {
    return (
      <div className="observability-loading">
        <style>{`
          .observability-loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            color: #94a3b8;
            font-family: inherit;
          }
          .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(52, 152, 219, 0.1);
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
        <div className="spinner"></div>
        <p>Analyzing system logs...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="observability-error">
        <style>{`
          .observability-error {
            padding: 2rem;
            text-align: center;
            color: #ef4444;
          }
          .retry-btn {
            margin-top: 1rem;
            padding: 0.5rem 1.5rem;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
          }
          .retry-btn:hover { background: #dc2626; }
        `}</style>
        <h3>Observability Layer Unavailable</h3>
        <p>{error}</p>
        <button className="retry-btn" onClick={loadData}>Retry Connection</button>
      </div>
    )
  }

  const { summary, daily } = metrics || { summary: {}, daily: [] }

  // Simple SVG Line Chart generator
  const renderLineChart = (data, valueKey, labelKey, color = '#3498db', strokeWidth = 3, yMin = null, yMax = null) => {
    if (!data || data.length === 0) {
      return (
        <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
          No logged data points found
        </div>
      )
    }

    const width = 500
    const height = 180
    const padding = 25

    const xValues = data.map((_, i) => i)
    const yValues = data.map(d => Number(d[valueKey] || 0))

    const calculatedMin = yMin !== null ? yMin : Math.min(...yValues, 0)
    const calculatedMax = yMax !== null ? yMax : Math.max(...yValues, 1)
    const yRange = calculatedMax - calculatedMin === 0 ? 1 : calculatedMax - calculatedMin

    // Map data indices and values to screen coordinates
    const points = data.map((d, i) => {
      const x = padding + (i / (data.length - 1 || 1)) * (width - padding * 2)
      const y = height - padding - ((Number(d[valueKey] || 0) - calculatedMin) / yRange) * (height - padding * 2)
      return { x, y, val: d[valueKey], label: d[labelKey] }
    })

    // Construct path commands
    let pathD = ''
    let fillD = ''
    if (points.length > 0) {
      pathD = `M ${points[0].x} ${points[0].y} `
      for (let i = 1; i < points.length; i++) {
        pathD += `L ${points[i].x} ${points[i].y} `
      }
      fillD = `${pathD} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`
    }

    const gradId = `area-grad-${valueKey}`
    return (
      <div className="svg-container">
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%">
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.25" />
              <stop offset="100%" stopColor={color} stopOpacity="0.00" />
            </linearGradient>
          </defs>
          
          {/* Grid lines */}
          <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="rgba(184, 135, 85, 0.1)" strokeDasharray="3" />
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(184, 135, 85, 0.1)" strokeDasharray="3" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(184, 135, 85, 0.2)" />

          {/* Area fill */}
          {points.length > 0 && <path d={fillD} fill={`url(#${gradId})`} />}

          {/* Line stroke */}
          {points.length > 0 && <path d={pathD} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />}

          {/* Markers */}
          {points.map((pt, idx) => (
            <g key={idx} className="chart-marker-group">
              <circle cx={pt.x} cy={pt.y} r="4" fill="#ffffff" stroke={color} strokeWidth="2" className="chart-circle" />
              <text x={pt.x} y={pt.y - 8} textAnchor="middle" fontSize="9" fill="#8d7d6f" className="marker-text" style={{ display: 'none' }}>
                {typeof pt.val === 'number' && pt.val % 1 !== 0 ? pt.val.toFixed(2) : pt.val}
              </text>
            </g>
          ))}
          
          {/* X axis labels (first, middle, last) */}
          {points.length > 0 && (
            <>
              <text x={points[0].x} y={height - 8} fontSize="9" fill="#64748b" textAnchor="start">
                {points[0].label}
              </text>
              {points.length > 2 && (
                <text x={points[Math.floor(points.length / 2)].x} y={height - 8} fontSize="9" fill="#64748b" textAnchor="middle">
                  {points[Math.floor(points.length / 2)].label}
                </text>
              )}
              {points.length > 1 && (
                <text x={points[points.length - 1].x} y={height - 8} fontSize="9" fill="#64748b" textAnchor="end">
                  {points[points.length - 1].label}
                </text>
              )}
            </>
          )}
        </svg>
      </div>
    )
  }

  // Calculate stats
  const totalReq = summary.total_requests || 0
  const hitRate = summary.cache_hit_rate || 0.0
  const p50Lat = summary.p50_total_latency || 0.0
  const p95Lat = summary.p95_total_latency || 0.0
  const refuseRate = summary.refusal_rate || 0.0
  const faithAvg = summary.avg_faithfulness || 0.0
  const thumbsUp = summary.thumbs_up || 0
  const thumbsDown = summary.thumbs_down || 0
  const feedbackRatio = thumbsUp + thumbsDown > 0 ? (thumbsUp / (thumbsUp + thumbsDown)) * 100 : null

  return (
    <div className="dashboard">
      <style>{`
        .dashboard {
          padding: 1.5rem;
          color: #3c3024;
          font-family: 'Inter', system-ui, sans-serif;
          max-width: 1200px;
          margin: 0 auto;
          overflow-y: auto;
          flex: 1;
          width: 100%;
          box-sizing: border-box;
        }
        .header-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 2rem;
          border-bottom: 1px solid rgba(184, 135, 85, 0.15);
          padding-bottom: 1rem;
        }
        .header-title {
          font-size: 1.5rem;
          font-weight: 700;
          color: #2c2520;
          letter-spacing: -0.025em;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .header-desc {
          font-size: 0.875rem;
          color: #7f6f5f;
          margin-top: 0.25rem;
        }
        .refresh-button {
          background: rgba(255, 255, 255, 0.65);
          border: 1px solid rgba(184, 135, 85, 0.2);
          color: #3c3024;
          padding: 0.5rem 1rem;
          border-radius: 8px;
          cursor: pointer;
          font-size: 0.875rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: all 0.2s ease;
        }
        .refresh-button:hover {
          background: #ffffff;
          border-color: rgba(184, 135, 85, 0.4);
          color: #b88755;
        }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }
        .metric-card {
          background: rgba(255, 255, 255, 0.65);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(184, 135, 85, 0.12);
          border-radius: 12px;
          padding: 1.25rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        }
        .metric-label {
          color: #8d7d6f;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.5rem;
        }
        .metric-value {
          font-size: 1.75rem;
          font-weight: 700;
          color: #2c2520;
          letter-spacing: -0.02em;
        }
        .metric-subtext {
          font-size: 0.75rem;
          color: #a39587;
          margin-top: 0.25rem;
        }
        .charts-container {
          background: rgba(255, 255, 255, 0.55);
          border: 1px solid rgba(184, 135, 85, 0.12);
          border-radius: 16px;
          padding: 1.5rem;
          margin-bottom: 2rem;
        }
        .tab-bar {
          display: flex;
          gap: 0.5rem;
          border-bottom: 1px solid rgba(184, 135, 85, 0.12);
          margin-bottom: 1.5rem;
          padding-bottom: 0.5rem;
        }
        .tab-btn {
          background: none;
          border: none;
          color: #8d7d6f;
          padding: 0.5rem 1rem;
          cursor: pointer;
          font-size: 0.875rem;
          font-weight: 500;
          border-radius: 6px;
          transition: all 0.2s ease;
        }
        .tab-btn.active {
          background: rgba(184, 135, 85, 0.15);
          color: #b88755;
          font-weight: 600;
        }
        .tab-btn:hover:not(.active) {
          color: #3c3024;
          background: rgba(255,255,255,0.4);
        }
        .charts-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }
        .chart-box {
          background: rgba(255, 255, 255, 0.65);
          border: 1px solid rgba(184, 135, 85, 0.1);
          border-radius: 12px;
          padding: 1.25rem;
        }
        .chart-title {
          font-size: 0.875rem;
          font-weight: 600;
          color: #3c3024;
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .svg-container {
          width: 100%;
          height: 180px;
        }
        .chart-circle {
          transition: r 0.15s ease, fill 0.15s ease;
        }
        .chart-marker-group:hover .chart-circle {
          r: 6;
          fill: #b88755;
        }
        .chart-marker-group:hover .marker-text {
          display: block;
        }
        .eval-section {
          background: rgba(255, 255, 255, 0.45);
          border: 1px solid rgba(184, 135, 85, 0.1);
          border-radius: 12px;
          padding: 1rem;
        }
        .eval-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        .eval-row {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem;
          border-bottom: 1px solid rgba(184, 135, 85, 0.08);
          font-size: 0.8125rem;
          color: #8d7d6f;
        }
        .eval-row:last-child {
          border-bottom: none;
        }
        .eval-val {
          font-weight: 600;
          color: #3c3024;
        }
      `}</style>

      <div className="header-bar">
        <div>
          <h2 className="header-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3498db" strokeWidth="2.5">
              <path d="M18 20V10M12 20V4M6 20v-6"/>
            </svg>
            Observability Metrics
          </h2>
          <div className="header-desc">RAG analytics, generation quality, and stage-wise performance dashboard.</div>
        </div>
        <button className="refresh-button" onClick={loadData}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
          </svg>
          Refresh Data
        </button>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Total Requests</div>
          <div className="metric-value">{totalReq}</div>
          <div className="metric-subtext">Requests processed</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Semantic Cache Hits</div>
          <div className="metric-value">{Math.round(hitRate * 100)}%</div>
          <div className="metric-subtext">{summary.total_requests ? Math.round(hitRate * totalReq) : 0} cache hits</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">p50 / p95 Latency</div>
          <div className="metric-value">
            {p50Lat >= 1000 ? `${(p50Lat / 1000).toFixed(1)}s` : `${Math.round(p50Lat)}ms`}
            <span style={{ fontSize: '0.9rem', color: '#64748b', fontWeight: 'normal', margin: '0 0.25rem' }}>/</span>
            {p95Lat >= 1000 ? `${(p95Lat / 1000).toFixed(1)}s` : `${Math.round(p95Lat)}ms`}
          </div>
          <div className="metric-subtext">Stage-to-stage total</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Refusal Rate</div>
          <div className="metric-value">{Math.round(refuseRate * 100)}%</div>
          <div className="metric-subtext">Fallback triggers</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Faithfulness Judge</div>
          <div className="metric-value">{faithAvg > 0 ? `${faithAvg.toFixed(1)}/5` : '--'}</div>
          <div className="metric-subtext">LLM-as-judge score</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Thumbs-Up Ratio</div>
          <div className="metric-value">{feedbackRatio !== null ? `${Math.round(feedbackRatio)}%` : '--'}</div>
          <div className="metric-subtext">{thumbsUp} 👍 vs {thumbsDown} 👎</div>
        </div>
      </div>

      <div className="charts-container">
        <div className="tab-bar">
          <button className={`tab-btn ${activeTab === 'performance' ? 'active' : ''}`} onClick={() => setActiveTab('performance')}>
            Performance & Latency
          </button>
          <button className={`tab-btn ${activeTab === 'retrieval' ? 'active' : ''}`} onClick={() => setActiveTab('retrieval')}>
            Retrieval Accuracy (Eval Set)
          </button>
          <button className={`tab-btn ${activeTab === 'usage' ? 'active' : ''}`} onClick={() => setActiveTab('usage')}>
            Usage & Feedback
          </button>
        </div>

        {activeTab === 'performance' && (
          <div className="charts-grid">
            <div className="chart-box">
              <div className="chart-title">
                <span>p50 / p95 Latency Over Time (ms)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Lower is better</span>
              </div>
              {renderLineChart(daily, 'p50_latency', 'date', '#3498db')}
            </div>

            <div className="chart-box">
              <div className="chart-title">
                <span>Cache Hit Rate Over Time (%)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Higher is better</span>
              </div>
              {renderLineChart(daily.map(d => ({ ...d, chr_pct: Math.round(d.cache_hit_rate * 100) })), 'chr_pct', 'date', '#2ecc71', 3, 0, 100)}
            </div>

            <div className="chart-box">
              <div className="chart-title">
                <span>Faithfulness Judge Trend (Avg 1-5)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Higher is better</span>
              </div>
              {renderLineChart(daily, 'avg_faithfulness', 'date', '#e67e22', 3, 1, 5)}
            </div>

            <div className="chart-box" style={{ gridColumn: 'span 2' }}>
              <div className="chart-title" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
                <span>Stage-wise Average Latency (ms)</span>
                <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                  {[
                    { key: 'cache_check', label: 'Cache Check' },
                    { key: 'embed', label: 'Embedding' },
                    { key: 'bm25', label: 'BM25' },
                    { key: 'vector', label: 'Vector DB' },
                    { key: 'rrf', label: 'RRF Fusion' },
                    { key: 'rerank', label: 'Reranker' },
                    { key: 'ttft', label: 'TTFT' },
                    { key: 'generate', label: 'LLM Total' }
                  ].map(stage => (
                    <button
                      key={stage.key}
                      onClick={() => setSelectedStage(stage.key)}
                      style={{
                        padding: '0.25rem 0.5rem',
                        fontSize: '0.7rem',
                        borderRadius: '4px',
                        border: '1px solid rgba(184, 135, 85, 0.2)',
                        background: selectedStage === stage.key ? 'rgba(184, 135, 85, 0.15)' : 'rgba(255, 255, 255, 0.5)',
                        color: selectedStage === stage.key ? '#b88755' : '#64748b',
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      {stage.label}
                    </button>
                  ))}
                </div>
              </div>
              {renderLineChart(daily, `${selectedStage}_latency`, 'date', '#9b59b6')}
            </div>
          </div>
        )}

        {activeTab === 'retrieval' && (
          <div className="charts-grid">
            <div className="chart-box">
              <div className="chart-title">
                <span>Recall@5 and Recall@10 (Golden Set)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>From run_eval.py</span>
              </div>
              {evalHistory.length > 0 ? (
                renderLineChart(evalHistory.map(h => ({ ...h, rec5_pct: Math.round(h.recall_5 * 100) })), 'rec5_pct', 'timestamp', '#e67e22', 3, 0, 100)
              ) : (
                <div className="eval-section">
                  <p style={{ textAlign: 'center', color: '#94a3b8', fontSize: '0.875rem', margin: '2rem 0' }}>
                    No evaluation runs logged in eval/history.csv. 
                    <br />
                    Run <code>python eval/run_eval.py</code> to populate.
                  </p>
                </div>
              )}
            </div>

            <div className="chart-box">
              <div className="chart-title">
                <span>Mean Reciprocal Rank (MRR)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Retrieval quality</span>
              </div>
              {evalHistory.length > 0 ? (
                renderLineChart(evalHistory, 'mrr', 'timestamp', '#9b59b6', 3, 0, 1)
              ) : (
                <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
                  No evaluation runs found
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'usage' && (
          <div className="charts-grid">
            <div className="chart-box">
              <div className="chart-title">
                <span>Daily Requests</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Traffic count</span>
              </div>
              {renderLineChart(daily, 'requests', 'date', '#1abc9c')}
            </div>

            <div className="chart-box">
              <div className="chart-title">
                <span>Daily Tokens Consumed</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>LLM output weight</span>
              </div>
              {renderLineChart(daily, 'tokens', 'date', '#f1c40f')}
            </div>

            <div className="chart-box">
              <div className="chart-title">
                <span>Thumbs-Up Rate Over Time (%)</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Real-world quality signal</span>
              </div>
              {renderLineChart(
                daily.map(d => ({ 
                  ...d, 
                  thumbs_up_rate: (d.thumbs_up + d.thumbs_down) > 0 
                    ? Math.round((d.thumbs_up / (d.thumbs_up + d.thumbs_down)) * 100) 
                    : 0 
                })), 
                'thumbs_up_rate', 
                'date', 
                '#e74c3c', 
                3, 
                0, 
                100
              )}
            </div>
          </div>
        )}
      </div>

      {evalHistory.length > 0 && (
        <div className="eval-section" style={{ marginTop: '1.5rem' }}>
          <h4 style={{ margin: '0 0 1rem 0', color: '#cbd5e1', fontSize: '0.9rem' }}>Latest Evaluation Run Details</h4>
          <div className="eval-list">
            <div className="eval-row">
              <span>Timestamp</span>
              <span className="eval-val">{new Date(evalHistory[evalHistory.length - 1].timestamp).toLocaleString()}</span>
            </div>
            <div className="eval-row">
              <span>Git Commit Reference</span>
              <span className="eval-val"><code>{evalHistory[evalHistory.length - 1].commit}</code></span>
            </div>
            <div className="eval-row">
              <span>Recall@5 / Recall@10</span>
              <span className="eval-val">
                {Math.round(evalHistory[evalHistory.length - 1].recall_5 * 100)}% / {Math.round(evalHistory[evalHistory.length - 1].recall_10 * 100)}%
              </span>
            </div>
            <div className="eval-row">
              <span>Mean Reciprocal Rank (MRR)</span>
              <span className="eval-val">{(evalHistory[evalHistory.length - 1].mrr).toFixed(3)}</span>
            </div>
            <div className="eval-row">
              <span>Avg Faithfulness Score (Judge)</span>
              <span className="eval-val">{evalHistory[evalHistory.length - 1].avg_faithfulness} / 5</span>
            </div>
            <div className="eval-row">
              <span>Sentence-level Citation Accuracy</span>
              <span className="eval-val">{Math.round(evalHistory[evalHistory.length - 1].citation_accuracy * 100)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
