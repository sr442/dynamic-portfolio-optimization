
import React, { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  BarChart2, PieChart, Activity, TrendingUp, DollarSign, Download, Play, RefreshCw, AlertCircle
} from 'lucide-react';

const API_URL = 'http://localhost:8000/api';

function App() {
  const [activeTab, setActiveTab] = useState('live'); // 'live' or 'backtest'

  // Configuration State
  const [tickers, setTickers] = useState('AAPL, MSFT, GOOG, AMZN, SPY, NVDA');
  const [capital, setCapital] = useState(100000);
  const [optimizer, setOptimizer] = useState('Mean-Variance');
  const [riskAversion, setRiskAversion] = useState(2.5);

  // Results State
  const [loading, setLoading] = useState(false);
  const [liveResults, setLiveResults] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/optimize`, {
        tickers: tickers.split(',').map(t => t.trim()),
        capital: Number(capital),
        optimizer,
        risk_aversion: Number(riskAversion),
        lookback_days: 365
      });
      setLiveResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Optimization failed');
    } finally {
      setLoading(false);
    }
  };

  const handleBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/backtest`, {
        tickers: tickers.split(',').map(t => t.trim()),
        capital: Number(capital),
        start_date: '2020-01-01',
        end_date: '2023-01-01',
        optimizer,
        risk_aversion: Number(riskAversion)
      });
      setBacktestResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="container">
        <div className="flex-row" style={{ alignItems: 'center' }}>
          <Activity className="text-accent" size={32} />
          <h1>Antigravity AI Asset Manager</h1>
        </div>
        <div className="status-badge">
          <span className="blink" style={{ marginRight: '0.5rem' }}>●</span>
          System Online
        </div>
      </header>

      <main className="container dashboard-grid">

        {/* Sidebar Configuration */}
        <aside className="panel">
          <div className="panel-title flex-row" style={{ justifyContent: 'space-between' }}>
            <span>Control Panel</span>
            <RefreshCw size={18} className="text-muted" style={{ cursor: 'pointer' }} onClick={() => window.location.reload()} />
          </div>

          <div className="form-group">
            <label>Investment Universe (Tickers)</label>
            <textarea
              value={tickers}
              onChange={e => setTickers(e.target.value)}
              placeholder="e.g. AAPL, MSFT, GOOG"
            />
          </div>

          <div className="form-group">
            <label>Initial Capital ($)</label>
            <input
              type="number"
              value={capital}
              onChange={e => setCapital(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Optimizer Model</label>
            <select value={optimizer} onChange={e => setOptimizer(e.target.value)}>
              <option value="Mean-Variance">Mean-Variance (Markowitz)</option>
              <option value="Risk Parity">Risk Parity (Equal Risk)</option>
            </select>
          </div>

          {optimizer === 'Mean-Variance' && (
            <div className="form-group">
              <label>Risk Aversion (Lambda): {riskAversion}</label>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.1"
                value={riskAversion}
                onChange={e => setRiskAversion(e.target.value)}
              />
            </div>
          )}

          <div style={{ marginTop: '2rem' }}>
            <div className="flex-row" style={{ marginBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
              <button
                className={`btn ${activeTab === 'live' ? 'btn-primary' : ''}`}
                style={{ borderRadius: '0', background: activeTab === 'live' ? '' : 'transparent' }}
                onClick={() => setActiveTab('live')}
              >
                Live Optimization
              </button>
              <button
                className={`btn ${activeTab === 'backtest' ? 'btn-primary' : ''}`}
                style={{ borderRadius: '0', background: activeTab === 'backtest' ? '' : 'transparent' }}
                onClick={() => setActiveTab('backtest')}
              >
                Backtest
              </button>
            </div>

            <button
              className="btn btn-primary"
              onClick={activeTab === 'live' ? handleOptimize : handleBacktest}
              disabled={loading}
            >
              {loading ? (
                <RefreshCw className="blink" size={20} />
              ) : (
                <div className="flex-row" style={{ gap: '0.5rem', justifyContent: 'center' }}>
                  <Play size={20} />
                  Run {activeTab === 'live' ? 'Optimization' : 'Simulation'}
                </div>
              )}
            </button>

            {error && (
              <div style={{ marginTop: '1rem', color: 'var(--danger)', fontSize: '0.9rem', display: 'flex', gap: '0.5rem' }}>
                <AlertCircle size={16} />
                {error}
              </div>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <section className="content-area">

          {/* LIVE MODE */}
          {activeTab === 'live' && (
            <div className="fade-in">
              {!liveResults ? (
                <div className="panel" style={{ textAlign: 'center', padding: '4rem' }}>
                  <BarChart2 size={64} className="text-muted" style={{ marginBottom: '1rem' }} />
                  <h3>Ready to Optimize</h3>
                  <p className="text-muted">Configure your universe and click "Run Optimization" to generate real-time signals.</p>
                </div>
              ) : (
                <>
                  <div className="panel" style={{ marginBottom: '2rem' }}>
                    <div className="panel-title">
                      Current Market Allocation ({liveResults.market_date})
                    </div>
                    <div className="table-container">
                      <table>
                        <thead>
                          <tr>
                            <th>Asset</th>
                            <th>Target Weight</th>
                            <th>Target Value</th>
                            <th>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {liveResults.allocation.map((item) => (
                            <tr key={item.ticker}>
                              <td style={{ fontWeight: 'bold' }}>{item.ticker}</td>
                              <td>{(item.weight * 100).toFixed(2)}%</td>
                              <td style={{ color: 'var(--success)' }}>${item.value.toLocaleString()}</td>
                              <td>
                                <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>REBALANCE</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
                      <button className="btn" style={{ width: 'auto', background: 'var(--border)' }}>
                        <Download size={16} style={{ marginRight: '0.5rem' }} />
                        Export Trade Sheet
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* BACKTEST MODE */}
          {activeTab === 'backtest' && (
            <div className="fade-in">
              {!backtestResults ? (
                <div className="panel" style={{ textAlign: 'center', padding: '4rem' }}>
                  <TrendingUp size={64} className="text-muted" style={{ marginBottom: '1rem' }} />
                  <h3>Strategy Verification</h3>
                  <p className="text-muted">Run historical simulations to validate your AI models before deploying capital.</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

                  {/* Metrics */}
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Total Return</div>
                      <div className="metric-value" style={{ color: 'var(--success)' }}>
                        {(backtestResults.metrics["Total Return"] * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">CAGR</div>
                      <div className="metric-value">
                        {(backtestResults.metrics["CAGR"] * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Sharpe Ratio</div>
                      <div className="metric-value text-accent">
                        {backtestResults.metrics["Sharpe Ratio"].toFixed(2)}
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Max Drawdown</div>
                      <div className="metric-value" style={{ color: 'var(--danger)' }}>
                        {(backtestResults.metrics["Max Drawdown"] * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {/* Charts */}
                  <div className="panel" style={{ height: '400px' }}>
                    <div className="panel-title">Equity Curve</div>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={backtestResults.chart_data}>
                        <defs>
                          <linearGradient id="colorPv" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="Date" tick={{ fill: '#94a3b8' }} />
                        <YAxis tick={{ fill: '#94a3b8' }} />
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                          labelStyle={{ color: '#f8fafc' }}
                        />
                        <Area
                          type="monotone"
                          dataKey="PortfolioValue"
                          stroke="#3b82f6"
                          fillOpacity={1}
                          fill="url(#colorPv)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                </div>
              )}
            </div>
          )}

        </section>
      </main>
    </div>
  );
}

export default App;
