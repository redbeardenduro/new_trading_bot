import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Slider } from '@/components/ui/slider.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { 
  Save, 
  RefreshCw, 
  Key, 
  Settings as SettingsIcon, 
  TrendingUp, 
  Shield, 
  Database, 
  Bell,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react'

const Settings = () => {
  const [config, setConfig] = useState({
    // Bot Configuration
    strategy: 'combined',
    timeframe: '1h',
    paperTrading: true,
    tradingInterval: 60,
    maxConcurrentPairs: 5,
    baseCurrencies: ['BTC', 'ETH', 'XRP', 'LTC', 'DOT'],
    initialCapital: 10000,
    
    // Trading Parameters
    minOpportunityScore: 0.6,
    positionSize: 5.0,
    volatilityLookback: 14,
    performanceLookback: 30,
    dynamicThreshold: true,
    rebalanceThreshold: 5.0,
    
    // Technical Indicators
    rsiOversold: 30,
    rsiOverbought: 70,
    macdThreshold: 0,
    bbThreshold: 5.0,
    
    // Risk Management
    maxAllocation: 7.5,
    minAllocation: 1.0,
    varConfidence: 0.95,
    varTimeHorizon: 1,
    monteCarloVar: true,
    feeRate: 0.1,
    slippage: 0.2,
    
    // API Keys
    krakenApiKey: '',
    krakenApiSecret: '',
    openaiApiKey: '',
    redditClientId: '',
    redditClientSecret: '',
    newsApiKey: '',
    
    // System Settings
    logLevel: 'INFO',
    logToFile: true,
    logFilename: 'trading_bot.log'
  })

  const [showApiKeys, setShowApiKeys] = useState({})
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const toggleCurrency = (currency) => {
    const current = config.baseCurrencies
    const updated = current.includes(currency)
      ? current.filter(c => c !== currency)
      : [...current, currency]
    updateConfig('baseCurrencies', updated)
  }

  const toggleApiKeyVisibility = (key) => {
    setShowApiKeys(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const saveConfiguration = async () => {
    setSaving(true)
    setSaveStatus(null)
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      setSaveStatus('success')
      setTimeout(() => setSaveStatus(null), 3000)
    } catch (error) {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus(null), 3000)
    } finally {
      setSaving(false)
    }
  }

  const renderApiKeyField = (key, label, placeholder) => {
    const isVisible = showApiKeys[key]
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-sm font-medium text-slate-300">{label}</Label>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => toggleApiKeyVisibility(key)}
            className="h-6 w-6 p-0 text-slate-400 hover:text-white"
          >
            {isVisible ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          </Button>
        </div>
        <Input
          type={isVisible ? "text" : "password"}
          value={config[key]}
          onChange={(e) => updateConfig(key, e.target.value)}
          placeholder={placeholder}
          className="bg-slate-800 border-slate-700 text-white"
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Settings</h2>
          <p className="text-slate-400">Configure your trading bot parameters and API credentials</p>
        </div>
        <div className="flex items-center space-x-2">
          {saveStatus && (
            <div className={`flex items-center space-x-1 text-sm ${
              saveStatus === 'success' ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {saveStatus === 'success' ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertTriangle className="h-4 w-4" />
              )}
              <span>{saveStatus === 'success' ? 'Settings saved!' : 'Save failed!'}</span>
            </div>
          )}
          <Button
            onClick={saveConfiguration}
            disabled={saving}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            {saving ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {saving ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="general" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 bg-[#0F1419] border border-slate-800">
          <TabsTrigger value="general" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <SettingsIcon className="h-4 w-4 mr-2" />
            General
          </TabsTrigger>
          <TabsTrigger value="trading" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <TrendingUp className="h-4 w-4 mr-2" />
            Trading
          </TabsTrigger>
          <TabsTrigger value="risk" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Shield className="h-4 w-4 mr-2" />
            Risk
          </TabsTrigger>
          <TabsTrigger value="apis" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Key className="h-4 w-4 mr-2" />
            APIs
          </TabsTrigger>
          <TabsTrigger value="system" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Bell className="h-4 w-4 mr-2" />
            System
          </TabsTrigger>
        </TabsList>

        {/* General Settings */}
        <TabsContent value="general" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Bot Configuration</CardTitle>
                <CardDescription className="text-slate-400">Basic bot operation settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Trading Strategy</Label>
                  <select 
                    value={config.strategy} 
                    onChange={(e) => updateConfig('strategy', e.target.value)}
                    className="w-full p-2 bg-slate-800 border border-slate-700 rounded-md text-white"
                  >
                    <option value="combined">Combined Strategy</option>
                    <option value="technical">Technical Only</option>
                    <option value="sentiment">Sentiment Only</option>
                    <option value="ai">AI Only</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Timeframe</Label>
                  <select 
                    value={config.timeframe} 
                    onChange={(e) => updateConfig('timeframe', e.target.value)}
                    className="w-full p-2 bg-slate-800 border border-slate-700 rounded-md text-white"
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
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium text-slate-300">Paper Trading</Label>
                    <Switch
                      checked={config.paperTrading}
                      onCheckedChange={(checked) => updateConfig('paperTrading', checked)}
                      className="data-[state=checked]:bg-blue-600"
                    />
                  </div>
                  <p className="text-xs text-slate-500">Enable paper trading for risk-free testing</p>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Trading Interval: {config.tradingInterval}s
                  </Label>
                  <Slider
                    value={[config.tradingInterval]}
                    onValueChange={([value]) => updateConfig('tradingInterval', value)}
                    min={30}
                    max={300}
                    step={30}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Max Concurrent Pairs: {config.maxConcurrentPairs}
                  </Label>
                  <Slider
                    value={[config.maxConcurrentPairs]}
                    onValueChange={([value]) => updateConfig('maxConcurrentPairs', value)}
                    min={1}
                    max={10}
                    step={1}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Trading Pairs</CardTitle>
                <CardDescription className="text-slate-400">Select cryptocurrencies to trade</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Base Currencies</Label>
                  <div className="flex flex-wrap gap-2">
                    {["BTC", "ETH", "XRP", "LTC", "DOT", "DOGE", "ADA", "SOL", "MATIC", "LINK"].map((currency) => (
                      <Badge
                        key={currency}
                        variant={config.baseCurrencies.includes(currency) ? "default" : "outline"}
                        className={`cursor-pointer ${
                          config.baseCurrencies.includes(currency)
                            ? 'bg-blue-600 text-white hover:bg-blue-700'
                            : 'border-slate-600 text-slate-400 hover:bg-slate-800'
                        }`}
                        onClick={() => toggleCurrency(currency)}
                      >
                        {currency}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Initial Capital (Paper Trading)</Label>
                  <Input
                    type="number"
                    value={config.initialCapital}
                    onChange={(e) => updateConfig('initialCapital', parseFloat(e.target.value))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="10000"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Trading Settings */}
        <TabsContent value="trading" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Trading Parameters</CardTitle>
                <CardDescription className="text-slate-400">Core trading algorithm settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Minimum Opportunity Score: {config.minOpportunityScore.toFixed(2)}
                  </Label>
                  <Slider
                    value={[config.minOpportunityScore]}
                    onValueChange={([value]) => updateConfig('minOpportunityScore', value)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Position Size: {config.positionSize.toFixed(1)}%
                  </Label>
                  <Slider
                    value={[config.positionSize]}
                    onValueChange={([value]) => updateConfig('positionSize', value)}
                    min={1.0}
                    max={20.0}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Volatility Lookback: {config.volatilityLookback} days
                  </Label>
                  <Slider
                    value={[config.volatilityLookback]}
                    onValueChange={([value]) => updateConfig('volatilityLookback', value)}
                    min={7}
                    max={30}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium text-slate-300">Dynamic Threshold</Label>
                    <Switch
                      checked={config.dynamicThreshold}
                      onCheckedChange={(checked) => updateConfig('dynamicThreshold', checked)}
                      className="data-[state=checked]:bg-blue-600"
                    />
                  </div>
                  <p className="text-xs text-slate-500">Automatically adjust thresholds based on market conditions</p>
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Technical Indicators</CardTitle>
                <CardDescription className="text-slate-400">Configure technical analysis thresholds</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    RSI Oversold: {config.rsiOversold}
                  </Label>
                  <Slider
                    value={[config.rsiOversold]}
                    onValueChange={([value]) => updateConfig('rsiOversold', value)}
                    min={20}
                    max={40}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    RSI Overbought: {config.rsiOverbought}
                  </Label>
                  <Slider
                    value={[config.rsiOverbought]}
                    onValueChange={([value]) => updateConfig('rsiOverbought', value)}
                    min={60}
                    max={80}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    MACD Threshold: {config.macdThreshold}
                  </Label>
                  <Slider
                    value={[config.macdThreshold]}
                    onValueChange={([value]) => updateConfig('macdThreshold', value)}
                    min={-0.1}
                    max={0.1}
                    step={0.01}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Bollinger Bands: {config.bbThreshold.toFixed(1)}%
                  </Label>
                  <Slider
                    value={[config.bbThreshold]}
                    onValueChange={([value]) => updateConfig('bbThreshold', value)}
                    min={1.0}
                    max={10.0}
                    step={0.5}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Risk Management */}
        <TabsContent value="risk" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Portfolio Allocation</CardTitle>
                <CardDescription className="text-slate-400">Asset allocation and rebalancing settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Max Allocation per Asset: {config.maxAllocation.toFixed(1)}%
                  </Label>
                  <Slider
                    value={[config.maxAllocation]}
                    onValueChange={([value]) => updateConfig('maxAllocation', value)}
                    min={5.0}
                    max={25.0}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Min Allocation per Asset: {config.minAllocation.toFixed(1)}%
                  </Label>
                  <Slider
                    value={[config.minAllocation]}
                    onValueChange={([value]) => updateConfig('minAllocation', value)}
                    min={0.5}
                    max={5.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Rebalance Threshold: {config.rebalanceThreshold.toFixed(1)}%
                  </Label>
                  <Slider
                    value={[config.rebalanceThreshold]}
                    onValueChange={([value]) => updateConfig('rebalanceThreshold', value)}
                    min={1.0}
                    max={15.0}
                    step={0.5}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white">Risk Management</CardTitle>
                <CardDescription className="text-slate-400">Value at Risk and trading costs</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    VaR Confidence Level: {(config.varConfidence * 100).toFixed(0)}%
                  </Label>
                  <Slider
                    value={[config.varConfidence]}
                    onValueChange={([value]) => updateConfig('varConfidence', value)}
                    min={0.90}
                    max={0.99}
                    step={0.01}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Trading Fees: {config.feeRate.toFixed(2)}%
                  </Label>
                  <Slider
                    value={[config.feeRate]}
                    onValueChange={([value]) => updateConfig('feeRate', value)}
                    min={0.05}
                    max={0.5}
                    step={0.01}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">
                    Slippage: {config.slippage.toFixed(2)}%
                  </Label>
                  <Slider
                    value={[config.slippage]}
                    onValueChange={([value]) => updateConfig('slippage', value)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium text-slate-300">Monte Carlo VaR</Label>
                    <Switch
                      checked={config.monteCarloVar}
                      onCheckedChange={(checked) => updateConfig('monteCarloVar', checked)}
                      className="data-[state=checked]:bg-blue-600"
                    />
                  </div>
                  <p className="text-xs text-slate-500">Use Monte Carlo simulation for VaR calculation</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* API Credentials */}
        <TabsContent value="apis" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Key className="h-5 w-5 mr-2" />
                  Exchange APIs
                </CardTitle>
                <CardDescription className="text-slate-400">Configure exchange API credentials</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {renderApiKeyField('krakenApiKey', 'Kraken API Key', 'Enter your Kraken API key')}
                {renderApiKeyField('krakenApiSecret', 'Kraken API Secret', 'Enter your Kraken API secret')}
              </CardContent>
            </Card>

            <Card className="border-slate-800 bg-[#0F1419]">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Key className="h-5 w-5 mr-2" />
                  AI & Data APIs
                </CardTitle>
                <CardDescription className="text-slate-400">Configure AI and data source APIs</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {renderApiKeyField('openaiApiKey', 'OpenAI API Key', 'Enter your OpenAI API key')}
                {renderApiKeyField('newsApiKey', 'News API Key', 'Enter your News API key')}
                {renderApiKeyField('redditClientId', 'Reddit Client ID', 'Enter your Reddit client ID')}
                {renderApiKeyField('redditClientSecret', 'Reddit Client Secret', 'Enter your Reddit client secret')}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Settings */}
        <TabsContent value="system" className="space-y-6">
          <Card className="border-slate-800 bg-[#0F1419]">
            <CardHeader>
              <CardTitle className="text-white">System Configuration</CardTitle>
              <CardDescription className="text-slate-400">Configure logging and system settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Log Level</Label>
                  <select 
                    value={config.logLevel} 
                    onChange={(e) => updateConfig('logLevel', e.target.value)}
                    className="w-full p-2 bg-slate-800 border border-slate-700 rounded-md text-white"
                  >
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO">INFO</option>
                    <option value="WARNING">WARNING</option>
                    <option value="ERROR">ERROR</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium text-slate-300">Log to File</Label>
                    <Switch
                      checked={config.logToFile}
                      onCheckedChange={(checked) => updateConfig('logToFile', checked)}
                      className="data-[state=checked]:bg-blue-600"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-300">Log Filename</Label>
                  <Input
                    value={config.logFilename}
                    onChange={(e) => updateConfig('logFilename', e.target.value)}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="trading_bot.log"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Settings
