import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { 
  Play, 
  Pause,
  RefreshCw, 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  BarChart3,
  Calendar,
  Clock,
  Target,
  Award,
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  Activity,
  Settings,
  Zap
} from 'lucide-react'

const Backtesting = () => {
  const [backtestConfig, setBacktestConfig] = useState({
    pairs: ['BTC/USD', 'ETH/USD', 'XRP/USD'],
    timeframe: '1h',
    startDate: '2024-01-01',
    endDate: '2024-06-01',
    initialCapital: 10000,
    strategy: 'combined',
    minOpportunityScore: 0.6,
    positionSize: 5.0,
    maxConcurrentPairs: 5
  })

  const [backtestState, setBacktestState] = useState({
    isRunning: false,
    progress: 0,
    currentStep: 0,
    totalSteps: 0,
    status: 'idle', // idle, running, completed, error
    startTime: null,
    endTime: null
  })

  const [backtestResults, setBacktestResults] = useState(null)
  const [backtestHistory, setBacktestHistory] = useState([])
  const [selectedResult, setSelectedResult] = useState(null)

  // Mock backtest results for demonstration
  const mockResults = {
    run_details: {
      start_timestamp: "2024-01-01T00:00:00+00:00",
      end_timestamp: "2024-06-01T00:00:00+00:00",
      duration_days: 152,
      initial_capital: 10000.0,
      final_capital: 12847.32,
      quote_currency: "USD"
    },
    performance: {
      total_pnl: 2847.32,
      total_pnl_pct: 28.47,
      total_fees: 234.56,
      max_drawdown_pct: 8.23,
      sharpe_ratio: 1.84,
      sortino_ratio: 2.31
    },
    trade_stats: {
      total_trades_executed: 324,
      total_round_trips: 162,
      wins: 98,
      losses: 64,
      neutral: 0,
      win_rate_pct: 60.49,
      profit_factor: 1.67,
      avg_win_pnl: 45.23,
      avg_loss_pnl: -27.14,
      max_win_pnl: 234.56,
      max_loss_pnl: -123.45,
      avg_trade_duration_seconds: 86400,
      avg_trade_duration_human: "24:00:00"
    }
  }

  // Mock portfolio performance data
  const mockPortfolioData = [
    { date: '2024-01-01', value: 10000, benchmark: 10000, drawdown: 0 },
    { date: '2024-01-15', value: 10234, benchmark: 10120, drawdown: -1.2 },
    { date: '2024-02-01', value: 10456, benchmark: 10180, drawdown: -2.1 },
    { date: '2024-02-15', value: 10123, benchmark: 10250, drawdown: -3.4 },
    { date: '2024-03-01', value: 10678, benchmark: 10320, drawdown: -1.8 },
    { date: '2024-03-15', value: 10890, benchmark: 10450, drawdown: -0.9 },
    { date: '2024-04-01', value: 11234, benchmark: 10580, drawdown: -0.3 },
    { date: '2024-04-15', value: 11456, benchmark: 10720, drawdown: 0 },
    { date: '2024-05-01', value: 11789, benchmark: 10850, drawdown: 0 },
    { date: '2024-05-15', value: 12123, benchmark: 10980, drawdown: 0 },
    { date: '2024-06-01', value: 12847, benchmark: 11100, drawdown: 0 }
  ]

  // Mock trade distribution data
  const mockTradeData = [
    { range: '-100 to -50', count: 8, pnl: -612 },
    { range: '-50 to -25', count: 15, pnl: -487 },
    { range: '-25 to 0', count: 41, pnl: -523 },
    { range: '0 to 25', count: 52, pnl: 678 },
    { range: '25 to 50', count: 28, pnl: 1045 },
    { range: '50 to 100', count: 14, pnl: 934 },
    { range: '100+', count: 4, pnl: 812 }
  ]

  const updateConfig = (key, value) => {
    setBacktestConfig(prev => ({ ...prev, [key]: value }))
  }

  const togglePair = (pair) => {
    const current = backtestConfig.pairs
    const updated = current.includes(pair)
      ? current.filter(p => p !== pair)
      : [...current, pair]
    updateConfig('pairs', updated)
  }

  const startBacktest = async () => {
    setBacktestState({
      isRunning: true,
      progress: 0,
      currentStep: 0,
      totalSteps: 1000, // Mock total steps
      status: 'running',
      startTime: new Date(),
      endTime: null
    })

    // Simulate backtest progress
    for (let i = 0; i <= 100; i += 2) {
      await new Promise(resolve => setTimeout(resolve, 100))
      setBacktestState(prev => ({
        ...prev,
        progress: i,
        currentStep: Math.floor((i / 100) * 1000)
      }))
    }

    // Complete backtest
    setBacktestState(prev => ({
      ...prev,
      isRunning: false,
      progress: 100,
      status: 'completed',
      endTime: new Date()
    }))

    setBacktestResults(mockResults)
    
    // Add to history
    const newResult = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      config: { ...backtestConfig },
      results: mockResults,
      duration: '2.3s'
    }
    setBacktestHistory(prev => [newResult, ...prev])
  }

  const stopBacktest = () => {
    setBacktestState(prev => ({
      ...prev,
      isRunning: false,
      status: 'idle',
      progress: 0
    }))
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value)
  }

  const formatPercent = (value) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6 min-h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center">
            <BarChart3 className="h-6 w-6 mr-3 text-blue-400" />
            Backtesting
          </h2>
          <p className="text-slate-400 mt-1">Test your trading strategies with historical data</p>
        </div>
        <div className="flex items-center space-x-3">
          {backtestState.status === 'completed' && (
            <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-800">
              <Download className="h-4 w-4 mr-2" />
              Export Results
            </Button>
          )}
          <Button
            onClick={backtestState.isRunning ? stopBacktest : startBacktest}
            disabled={backtestConfig.pairs.length === 0}
            className={`${
              backtestState.isRunning 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-blue-600 hover:bg-blue-700'
            } text-white`}
          >
            {backtestState.isRunning ? (
              <>
                <Pause className="h-4 w-4 mr-2" />
                Stop Backtest
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Backtest
              </>
            )}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="configure" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">
          <TabsTrigger value="configure" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </TabsTrigger>
          <TabsTrigger value="results" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <BarChart3 className="h-4 w-4 mr-2" />
            Results
          </TabsTrigger>
          <TabsTrigger value="analysis" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Activity className="h-4 w-4 mr-2" />
            Analysis
          </TabsTrigger>
          <TabsTrigger value="history" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <FileText className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Configuration Tab */}
        <TabsContent value="configure" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Backtest Parameters</CardTitle>
                <CardDescription className="text-slate-400">Configure your backtest settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-slate-300">Start Date</Label>
                    <Input
                      type="date"
                      value={backtestConfig.startDate}
                      onChange={(e) => updateConfig('startDate', e.target.value)}
                      className="bg-slate-800/50 border-slate-700 text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-slate-300">End Date</Label>
                    <Input
                      type="date"
                      value={backtestConfig.endDate}
                      onChange={(e) => updateConfig('endDate', e.target.value)}
                      className="bg-slate-800/50 border-slate-700 text-white"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Timeframe</Label>
                  <select 
                    value={backtestConfig.timeframe} 
                    onChange={(e) => updateConfig('timeframe', e.target.value)}
                    className="w-full p-2 bg-slate-800/50 border border-slate-700 rounded-md text-white"
                  >
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="1h">1 Hour</option>
                    <option value="4h">4 Hours</option>
                    <option value="1d">1 Day</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Initial Capital</Label>
                  <Input
                    type="number"
                    value={backtestConfig.initialCapital}
                    onChange={(e) => updateConfig('initialCapital', parseFloat(e.target.value))}
                    className="bg-slate-800/50 border-slate-700 text-white"
                    placeholder="10000"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Strategy</Label>
                  <select 
                    value={backtestConfig.strategy} 
                    onChange={(e) => updateConfig('strategy', e.target.value)}
                    className="w-full p-2 bg-slate-800/50 border border-slate-700 rounded-md text-white"
                  >
                    <option value="combined">Combined Strategy</option>
                    <option value="technical">Technical Only</option>
                    <option value="sentiment">Sentiment Only</option>
                    <option value="ai">AI Only</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Trading Pairs</CardTitle>
                <CardDescription className="text-slate-400">Select pairs to backtest</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Available Pairs</Label>
                  <div className="flex flex-wrap gap-2">
                    {["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "DOT/USD", "DOGE/USD", "ADA/USD", "SOL/USD"].map((pair) => (
                      <Badge
                        key={pair}
                        variant={backtestConfig.pairs.includes(pair) ? "default" : "outline"}
                        className={`cursor-pointer transition-colors ${
                          backtestConfig.pairs.includes(pair)
                            ? 'bg-blue-600 text-white hover:bg-blue-700'
                            : 'border-slate-600 text-slate-400 hover:bg-slate-800'
                        }`}
                        onClick={() => togglePair(pair)}
                      >
                        {pair}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Min Opportunity Score: {backtestConfig.minOpportunityScore.toFixed(2)}
                  </Label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={backtestConfig.minOpportunityScore}
                    onChange={(e) => updateConfig('minOpportunityScore', parseFloat(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Position Size: {backtestConfig.positionSize.toFixed(1)}%
                  </Label>
                  <input
                    type="range"
                    min="1.0"
                    max="20.0"
                    step="0.5"
                    value={backtestConfig.positionSize}
                    onChange={(e) => updateConfig('positionSize', parseFloat(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Max Concurrent Pairs: {backtestConfig.maxConcurrentPairs}
                  </Label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={backtestConfig.maxConcurrentPairs}
                    onChange={(e) => updateConfig('maxConcurrentPairs', parseInt(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Progress Section */}
          {backtestState.status !== 'idle' && (
            <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  {backtestState.isRunning ? (
                    <RefreshCw className="h-5 w-5 mr-2 animate-spin text-blue-400" />
                  ) : backtestState.status === 'completed' ? (
                    <CheckCircle2 className="h-5 w-5 mr-2 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 mr-2 text-yellow-400" />
                  )}
                  Backtest {backtestState.status === 'running' ? 'Running' : backtestState.status === 'completed' ? 'Completed' : 'Status'}
                </CardTitle>
                <CardDescription className="text-slate-400">
                  {backtestState.isRunning && `Step ${backtestState.currentStep.toLocaleString()} of ${backtestState.totalSteps.toLocaleString()}`}
                  {backtestState.status === 'completed' && `Completed in ${((backtestState.endTime - backtestState.startTime) / 1000).toFixed(1)}s`}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Progress</span>
                    <span className="text-white">{backtestState.progress}%</span>
                  </div>
                  <Progress value={backtestState.progress} className="h-2" />
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-6">
          {backtestResults ? (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <DollarSign className="h-5 w-5 text-blue-400" />
                      <div>
                        <p className="text-xs text-slate-400">Total Return</p>
                        <p className="text-lg font-bold text-white">
                          {formatCurrency(backtestResults.performance.total_pnl)}
                        </p>
                        <p className={`text-xs ${backtestResults.performance.total_pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {formatPercent(backtestResults.performance.total_pnl_pct)}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <TrendingDown className="h-5 w-5 text-red-400" />
                      <div>
                        <p className="text-xs text-slate-400">Max Drawdown</p>
                        <p className="text-lg font-bold text-white">
                          {backtestResults.performance.max_drawdown_pct.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Award className="h-5 w-5 text-yellow-400" />
                      <div>
                        <p className="text-xs text-slate-400">Win Rate</p>
                        <p className="text-lg font-bold text-white">
                          {backtestResults.trade_stats.win_rate_pct.toFixed(1)}%
                        </p>
                        <p className="text-xs text-slate-400">
                          {backtestResults.trade_stats.wins}W / {backtestResults.trade_stats.losses}L
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5 text-emerald-400" />
                      <div>
                        <p className="text-xs text-slate-400">Sharpe Ratio</p>
                        <p className="text-lg font-bold text-white">
                          {backtestResults.performance.sharpe_ratio.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Portfolio Performance Chart */}
              <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white">Portfolio Performance</CardTitle>
                  <CardDescription className="text-slate-400">Strategy vs Benchmark comparison</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={mockPortfolioData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#0F1419', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                        labelStyle={{ color: '#E2E8F0' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        name="Strategy"
                        dot={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="benchmark" 
                        stroke="#64748B" 
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        name="Benchmark"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-white">Performance Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Initial Capital</span>
                      <span className="text-white font-mono">{formatCurrency(backtestResults.run_details.initial_capital)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Final Capital</span>
                      <span className="text-white font-mono">{formatCurrency(backtestResults.run_details.final_capital)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Total Fees</span>
                      <span className="text-white font-mono">{formatCurrency(backtestResults.performance.total_fees)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sortino Ratio</span>
                      <span className="text-white font-mono">{backtestResults.performance.sortino_ratio.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Profit Factor</span>
                      <span className="text-white font-mono">{backtestResults.trade_stats.profit_factor.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Duration</span>
                      <span className="text-white font-mono">{backtestResults.run_details.duration_days} days</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-white">Trade Statistics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Total Trades</span>
                      <span className="text-white font-mono">{backtestResults.trade_stats.total_trades_executed}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Round Trips</span>
                      <span className="text-white font-mono">{backtestResults.trade_stats.total_round_trips}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Avg Win</span>
                      <span className="text-emerald-400 font-mono">{formatCurrency(backtestResults.trade_stats.avg_win_pnl)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Avg Loss</span>
                      <span className="text-red-400 font-mono">{formatCurrency(backtestResults.trade_stats.avg_loss_pnl)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Max Win</span>
                      <span className="text-emerald-400 font-mono">{formatCurrency(backtestResults.trade_stats.max_win_pnl)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Max Loss</span>
                      <span className="text-red-400 font-mono">{formatCurrency(backtestResults.trade_stats.max_loss_pnl)}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
              <CardContent className="p-12 text-center">
                <BarChart3 className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-400 mb-2">No Results Available</h3>
                <p className="text-slate-500">Run a backtest to see detailed performance metrics and analysis.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          {backtestResults ? (
            <>
              {/* Drawdown Chart */}
              <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white">Drawdown Analysis</CardTitle>
                  <CardDescription className="text-slate-400">Portfolio drawdown over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={mockPortfolioData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#0F1419', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="drawdown" 
                        stroke="#EF4444" 
                        fill="#EF4444"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Trade Distribution */}
              <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white">Trade P&L Distribution</CardTitle>
                  <CardDescription className="text-slate-400">Distribution of trade outcomes</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={mockTradeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="range" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#0F1419', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar 
                        dataKey="count" 
                        fill="#3B82F6"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
              <CardContent className="p-12 text-center">
                <Activity className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-400 mb-2">No Analysis Available</h3>
                <p className="text-slate-500">Run a backtest to see detailed trade analysis and performance breakdowns.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card className="border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white">Backtest History</CardTitle>
              <CardDescription className="text-slate-400">Previous backtest runs and results</CardDescription>
            </CardHeader>
            <CardContent>
              {backtestHistory.length > 0 ? (
                <ScrollArea className="h-96">
                  <div className="space-y-3">
                    {backtestHistory.map((result) => (
                      <div
                        key={result.id}
                        className="p-4 border border-slate-700/50 rounded-lg hover:bg-slate-800/30 cursor-pointer transition-colors"
                        onClick={() => setSelectedResult(result)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-white font-medium">
                              {result.config.pairs.join(', ')} - {result.config.timeframe}
                            </p>
                            <p className="text-sm text-slate-400">
                              {new Date(result.timestamp).toLocaleString()} • Duration: {result.duration}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className={`font-mono ${
                              result.results.performance.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                              {formatPercent(result.results.performance.total_pnl_pct)}
                            </p>
                            <p className="text-sm text-slate-400">
                              {result.results.trade_stats.win_rate_pct.toFixed(1)}% Win Rate
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-12">
                  <FileText className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-400 mb-2">No History Available</h3>
                  <p className="text-slate-500">Your completed backtests will appear here.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Backtesting
