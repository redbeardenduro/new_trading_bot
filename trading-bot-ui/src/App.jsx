import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  BarChart3, 
  Play, 
  Pause,
  Terminal,
  Maximize2,
  Minimize2,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle2,
  Brain,
  MessageSquare,
  Newspaper,
  Target,
  Zap,
  Settings as SettingsIcon,
  PieChart,
  LineChart,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { LineChart as RechartsLineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RechartsPieChart, Cell, Pie } from 'recharts'
import Settings from './components/Settings.jsx'
import Backtesting from './components/Backtesting.jsx'
import './App.css'

// Mock data for demonstration
const mockPortfolioData = [
  { name: 'Jan', value: 10000, benchmark: 10000 },
  { name: 'Feb', value: 10500, benchmark: 10200 },
  { name: 'Mar', value: 11200, benchmark: 10400 },
  { name: 'Apr', value: 10800, benchmark: 10600 },
  { name: 'May', value: 12400, benchmark: 10800 },
  { name: 'Jun', value: 16000, benchmark: 11000 }
]

const mockAssetData = [
  { name: 'BTC', value: 35, color: '#F7931A' },
  { name: 'ETH', value: 25, color: '#627EEA' },
  { name: 'XRP', value: 15, color: '#23292F' },
  { name: 'LTC', value: 10, color: '#BFBBBB' },
  { name: 'DOT', value: 10, color: '#E6007A' },
  { name: 'DOGE', value: 5, color: '#C2A633' }
]

const mockTradingData = [
  { time: '00:00', BTC: 47800, ETH: 3450, XRP: 0.95 },
  { time: '04:00', BTC: 48200, ETH: 3480, XRP: 0.97 },
  { time: '08:00', BTC: 47900, ETH: 3420, XRP: 0.94 },
  { time: '12:00', BTC: 48500, ETH: 3510, XRP: 0.98 },
  { time: '16:00', BTC: 48800, ETH: 3540, XRP: 0.99 },
  { time: '20:00', BTC: 49200, ETH: 3580, XRP: 1.01 }
]

const mockTradingPairs = [
  { symbol: 'BTC', pair: 'BTC/USD', price: 47800.00, change: 2.3, volume: '1.2B', rsi: 45.2, macd: 0.15, signal: 'BUY', confidence: 85 },
  { symbol: 'ETH', pair: 'ETH/USD', price: 3450.00, change: 1.8, volume: '800M', rsi: 52.1, macd: -0.05, signal: 'HOLD', confidence: 72 },
  { symbol: 'XRP', pair: 'XRP/USD', price: 0.9500, change: -0.5, volume: '450M', rsi: 65.8, macd: -0.12, signal: 'SELL', confidence: 68 },
  { symbol: 'LTC', pair: 'LTC/USD', price: 185.00, change: 3.2, volume: '200M', rsi: 42.5, macd: 0.08, signal: 'BUY', confidence: 78 },
  { symbol: 'DOT', pair: 'DOT/USD', price: 28.50, change: -1.2, volume: '150M', rsi: 48.9, macd: 0.02, signal: 'HOLD', confidence: 65 },
  { symbol: 'DOGE', pair: 'DOGE/USD', price: 0.3200, change: 5.8, volume: '300M', rsi: 38.7, macd: 0.18, signal: 'BUY', confidence: 82 }
]

const mockConsoleMessages = [
  { timestamp: '18:45:23.456', level: 'INFO', message: 'Trading bot initialized successfully' },
  { timestamp: '18:45:24.123', level: 'INFO', message: 'Kraken API connection established' },
  { timestamp: '18:45:25.789', level: 'DEBUG', message: 'RSI: 45.2, MACD: 0.15, BB Position: 0.7' },
  { timestamp: '18:45:26.234', level: 'INFO', message: 'Generating BUY signal for BTC/USD' },
  { timestamp: '18:45:27.567', level: 'INFO', message: 'Executing paper trade: BUY 0.05 BTC at $47,800' },
  { timestamp: '18:45:28.890', level: 'WARNING', message: 'Reddit API rate limit approaching' }
]

function App() {
  const [botRunning, setBotRunning] = useState(false)
  const [paperTrading, setPaperTrading] = useState(true)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [showConsole, setShowConsole] = useState(true)
  const [consoleHeight, setConsoleHeight] = useState(200)

  // Mock portfolio values
  const portfolioValue = 16000
  const dailyPnL = 320
  const totalTrades = 47
  const winRate = 68.5

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value)
  }

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'BUY': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
      case 'SELL': return 'text-red-400 bg-red-500/10 border-red-500/20'
      case 'HOLD': return 'text-amber-400 bg-amber-500/10 border-amber-500/20'
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/20'
    }
  }

  const getLogLevelColor = (level) => {
    switch (level) {
      case 'INFO': return 'text-emerald-400'
      case 'DEBUG': return 'text-slate-400'
      case 'WARNING': return 'text-amber-400'
      case 'ERROR': return 'text-red-400'
      default: return 'text-slate-300'
    }
  }

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'trading', label: 'Trading', icon: LineChart },
    { id: 'backtesting', label: 'Backtesting', icon: Activity },
    { id: 'portfolio', label: 'Portfolio', icon: PieChart },
    { id: 'settings', label: 'Settings', icon: SettingsIcon }
  ]

  const renderDashboard = () => (
    <div className="space-y-8">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400 mb-1">Portfolio Value</p>
                <p className="text-2xl font-bold text-white">{formatCurrency(portfolioValue)}</p>
                <p className="text-sm text-emerald-400 mt-1 flex items-center">
                  <TrendingUp className="h-4 w-4 mr-1" />
                  +12.5% (30d)
                </p>
              </div>
              <div className="h-12 w-12 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                <DollarSign className="h-6 w-6 text-emerald-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400 mb-1">Daily P&L</p>
                <p className="text-2xl font-bold text-white">{formatCurrency(dailyPnL)}</p>
                <p className="text-sm text-emerald-400 mt-1 flex items-center">
                  <TrendingUp className="h-4 w-4 mr-1" />
                  +2.1% today
                </p>
              </div>
              <div className="h-12 w-12 rounded-xl bg-blue-500/10 flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-blue-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400 mb-1">Total Trades</p>
                <p className="text-2xl font-bold text-white">{totalTrades}</p>
                <p className="text-sm text-slate-400 mt-1 flex items-center">
                  <Activity className="h-4 w-4 mr-1" />
                  +3 today
                </p>
              </div>
              <div className="h-12 w-12 rounded-xl bg-purple-500/10 flex items-center justify-center">
                <Activity className="h-6 w-6 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400 mb-1">Win Rate</p>
                <p className="text-2xl font-bold text-white">{winRate}%</p>
                <p className="text-sm text-amber-400 mt-1 flex items-center">
                  <Target className="h-4 w-4 mr-1" />
                  Above avg
                </p>
              </div>
              <div className="h-12 w-12 rounded-xl bg-amber-500/10 flex items-center justify-center">
                <Target className="h-6 w-6 text-amber-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardHeader className="pb-4">
            <CardTitle className="text-white text-lg">Portfolio Performance</CardTitle>
            <CardDescription className="text-slate-400">6-month performance vs benchmark</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={mockPortfolioData}>
                <defs>
                  <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151', 
                    borderRadius: '8px',
                    color: '#F9FAFB'
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  fill="url(#portfolioGradient)" 
                />
                <Line 
                  type="monotone" 
                  dataKey="benchmark" 
                  stroke="#6B7280" 
                  strokeWidth={1}
                  strokeDasharray="5 5"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-slate-700/50 bg-slate-800/30">
          <CardHeader className="pb-4">
            <CardTitle className="text-white text-lg">Asset Allocation</CardTitle>
            <CardDescription className="text-slate-400">Current portfolio distribution</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <ResponsiveContainer width="100%" height={200}>
                <RechartsPieChart>
                  <Pie
                    data={mockAssetData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {mockAssetData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RechartsPieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-3 mt-6">
              {mockAssetData.map((asset) => (
                <div key={asset.name} className="flex items-center space-x-3">
                  <div className="h-3 w-3 rounded-full" style={{ backgroundColor: asset.color }}></div>
                  <span className="text-sm text-slate-300 font-medium">{asset.name}</span>
                  <span className="text-sm text-slate-400">{asset.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Trading Pairs */}
      <Card className="border-slate-700/50 bg-slate-800/30">
        <CardHeader>
          <CardTitle className="text-white text-lg">Active Trading Pairs</CardTitle>
          <CardDescription className="text-slate-400">Real-time market data and signals</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Asset</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Price</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Change</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Volume</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Indicators</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Signal</th>
                </tr>
              </thead>
              <tbody>
                {mockTradingPairs.map((pair) => (
                  <tr key={pair.symbol} className="border-b border-slate-700/50 hover:bg-slate-700/20 transition-colors">
                    <td className="py-4 px-4">
                      <div className="flex items-center space-x-3">
                        <div className="h-10 w-10 rounded-full bg-slate-700 flex items-center justify-center">
                          <span className="text-sm font-bold text-white">{pair.symbol}</span>
                        </div>
                        <div>
                          <p className="text-white font-medium">{pair.pair}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      <p className="text-white font-mono text-lg">{formatCurrency(pair.price)}</p>
                    </td>
                    <td className="py-4 px-4">
                      <p className={`font-medium ${pair.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pair.change >= 0 ? '+' : ''}{pair.change}%
                      </p>
                    </td>
                    <td className="py-4 px-4">
                      <p className="text-slate-300">{pair.volume}</p>
                    </td>
                    <td className="py-4 px-4">
                      <div className="text-sm space-y-1">
                        <p className="text-slate-400">RSI: {pair.rsi}</p>
                        <p className="text-slate-400">MACD: {pair.macd}</p>
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center space-x-3">
                        <Badge className={`text-sm font-medium border ${getSignalColor(pair.signal)}`}>
                          {pair.signal}
                        </Badge>
                        <span className="text-sm text-slate-400">{pair.confidence}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const renderTrading = () => (
    <div className="space-y-8">
      <Card className="border-slate-700/50 bg-slate-800/30">
        <CardHeader>
          <CardTitle className="text-white text-lg">Price Charts</CardTitle>
          <CardDescription className="text-slate-400">Real-time price movements</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <RechartsLineChart data={mockTradingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151', 
                  borderRadius: '8px',
                  color: '#F9FAFB'
                }}
              />
              <Line type="monotone" dataKey="BTC" stroke="#F7931A" strokeWidth={2} />
              <Line type="monotone" dataKey="ETH" stroke="#627EEA" strokeWidth={2} />
              <Line type="monotone" dataKey="XRP" stroke="#23292F" strokeWidth={2} />
            </RechartsLineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )

  const renderPortfolio = () => (
    <div className="space-y-8">
      <Card className="border-slate-700/50 bg-slate-800/30">
        <CardContent className="p-8 text-center">
          <PieChart className="h-16 w-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-medium text-slate-300 mb-2">Portfolio Management</h3>
          <p className="text-slate-500">Advanced portfolio analytics and management tools coming soon.</p>
        </CardContent>
      </Card>
    </div>
  )

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      {/* Header */}
      <div className="border-b border-slate-700/50 bg-slate-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-between px-8 py-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-emerald-500 to-blue-600 flex items-center justify-center">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">TradingBot Pro</h1>
                <p className="text-sm text-slate-400">Enhanced Crypto Trading Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 ml-8">
              <div className={`h-2 w-2 rounded-full ${botRunning ? 'bg-emerald-400' : 'bg-slate-500'}`}></div>
              <Badge variant={botRunning ? "default" : "secondary"} className={`text-sm font-medium ${botRunning ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-slate-500/20 text-slate-400 border-slate-500/30'}`}>
                {botRunning ? "LIVE" : "OFFLINE"}
              </Badge>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3 text-sm">
              <span className="text-slate-400">Paper Trading</span>
              <Switch 
                checked={paperTrading} 
                onCheckedChange={setPaperTrading}
                className="data-[state=checked]:bg-emerald-600"
              />
            </div>
            
            <Button 
              onClick={() => setBotRunning(!botRunning)}
              className={`font-medium px-6 ${botRunning 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-emerald-600 hover:bg-emerald-700 text-white'
              }`}
            >
              {botRunning ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
              {botRunning ? 'Stop Bot' : 'Start Bot'}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Tab Navigation */}
        <div className="border-b border-slate-700/50 bg-slate-800/30">
          <div className="px-8">
            <div className="flex space-x-1">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.id
                        ? 'border-emerald-500 text-emerald-400 bg-slate-700/30'
                        : 'border-transparent text-slate-400 hover:text-slate-300 hover:bg-slate-700/20'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className={`flex-1 overflow-auto ${showConsole ? `pb-${consoleHeight}` : ''}`}>
          <div className="p-8">
            {activeTab === 'dashboard' && renderDashboard()}
            {activeTab === 'trading' && renderTrading()}
            {activeTab === 'backtesting' && <Backtesting />}
            {activeTab === 'portfolio' && renderPortfolio()}
            {activeTab === 'settings' && <Settings />}
          </div>
        </div>

        {/* Bottom Console */}
        {showConsole && (
          <div 
            className="border-t border-slate-700/50 bg-slate-800/50 backdrop-blur-sm"
            style={{ height: `${consoleHeight}px` }}
          >
            <div className="flex items-center justify-between px-6 py-3 border-b border-slate-700/50">
              <div className="flex items-center space-x-3">
                <Terminal className="h-4 w-4 text-slate-400" />
                <h3 className="font-medium text-white">Console</h3>
                <Badge variant="outline" className="text-xs border-emerald-500/30 text-emerald-400">
                  Connected
                </Badge>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setConsoleHeight(consoleHeight === 200 ? 400 : 200)}
                  className="h-8 w-8 p-0 text-slate-400 hover:text-white"
                >
                  {consoleHeight === 200 ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowConsole(false)}
                  className="h-8 w-8 p-0 text-slate-400 hover:text-white"
                >
                  ×
                </Button>
              </div>
            </div>
            
            <ScrollArea className="h-full p-4">
              <div className="space-y-1 font-mono text-sm">
                {mockConsoleMessages.map((msg, index) => (
                  <div key={index} className="flex space-x-3">
                    <span className="text-slate-500 text-xs">{msg.timestamp}</span>
                    <span className={`font-medium text-xs ${getLogLevelColor(msg.level)}`}>{msg.level}</span>
                    <span className="text-slate-300">{msg.message}</span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* Console Toggle (when hidden) */}
        {!showConsole && (
          <div className="border-t border-slate-700/50 bg-slate-800/30">
            <div className="px-6 py-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowConsole(true)}
                className="text-slate-400 hover:text-white"
              >
                <Terminal className="h-4 w-4 mr-2" />
                Show Console
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
