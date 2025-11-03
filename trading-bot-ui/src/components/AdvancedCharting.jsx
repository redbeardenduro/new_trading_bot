import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Checkbox } from './ui/checkbox';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area, ComposedChart, ReferenceLine, ReferenceArea,
  ScatterChart, Scatter, Cell
} from 'recharts';
import { 
  TrendingUp, TrendingDown, BarChart3, Target, Zap, Settings, 
  PenTool, MousePointer, Type, Square, ArrowRight, Minus,
  Activity, Eye, Layers, RefreshCw, Download, Upload
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

const AdvancedCharting = () => {
  const [chartData, setChartData] = useState(null);
  const [indicators, setIndicators] = useState({});
  const [patterns, setPatterns] = useState([]);
  const [volumeProfile, setVolumeProfile] = useState(null);
  const [multiTimeframe, setMultiTimeframe] = useState(null);
  const [drawings, setDrawings] = useState([]);
  const [annotations, setAnnotations] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('chart');
  
  const [selectedAsset, setSelectedAsset] = useState('BTC');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [selectedIndicators, setSelectedIndicators] = useState(['RSI', 'MACD', 'BB']);
  const [drawingMode, setDrawingMode] = useState(null);
  const [showPatterns, setShowPatterns] = useState(true);
  const [showVolumeProfile, setShowVolumeProfile] = useState(false);
  
  const chartRef = useRef(null);

  const fetchData = async (endpoint, params = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const queryString = new URLSearchParams(params).toString();
      const url = `http://localhost:5000/api/charting/${endpoint}${queryString ? '?' + queryString : ''}`;
      
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return null;
      }
      
      return data;
    } catch (err) {
      setError(`Failed to fetch data: ${err.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const loadChartData = async () => {
    const data = await fetchData('price-data', {
      asset: selectedAsset,
      timeframe: selectedTimeframe,
      limit: 500
    });
    
    if (data) {
      setChartData(data.data);
    }
  };

  const loadIndicators = async () => {
    const data = await fetchData('indicators', {
      asset: selectedAsset,
      timeframe: selectedTimeframe,
      indicators: selectedIndicators.join(',')
    });
    
    if (data) {
      setIndicators(data.indicators);
    }
  };

  const loadPatterns = async () => {
    const data = await fetchData('patterns', {
      asset: selectedAsset,
      timeframe: selectedTimeframe
    });
    
    if (data) {
      setPatterns(data.patterns);
    }
  };

  const loadVolumeProfile = async () => {
    const data = await fetchData('volume-profile', {
      asset: selectedAsset,
      timeframe: selectedTimeframe,
      bins: 50
    });
    
    if (data) {
      setVolumeProfile(data.volume_profile);
    }
  };

  const loadMultiTimeframe = async () => {
    const data = await fetchData('multi-timeframe', {
      asset: selectedAsset,
      timeframes: '1h,4h,1d'
    });
    
    if (data) {
      setMultiTimeframe(data.analysis);
    }
  };

  const loadDrawings = async () => {
    const data = await fetchData('drawings', {
      chart_id: `${selectedAsset}_${selectedTimeframe}`
    });
    
    if (data) {
      setDrawings(data.drawings);
    }
  };

  const loadAnnotations = async () => {
    const data = await fetchData('annotations', {
      chart_id: `${selectedAsset}_${selectedTimeframe}`
    });
    
    if (data) {
      setAnnotations(data.annotations);
    }
  };

  useEffect(() => {
    loadChartData();
    loadIndicators();
    if (showPatterns) loadPatterns();
    if (showVolumeProfile) loadVolumeProfile();
    loadMultiTimeframe();
    loadDrawings();
    loadAnnotations();
  }, [selectedAsset, selectedTimeframe, selectedIndicators]);

  const formatPrice = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(value);
  };

  const formatVolume = (value) => {
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
  };

  const getPatternColor = (patternType) => {
    const colors = {
      'double_top': '#f44336',
      'double_bottom': '#4caf50',
      'head_shoulders': '#ff9800',
      'triangle': '#9c27b0',
      'flag': '#2196f3',
      'pennant': '#00bcd4'
    };
    return colors[patternType] || '#757575';
  };

  const MainChartCard = () => {
    if (!chartData || chartData.length === 0) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Price Chart</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-64">
              <div className="text-gray-500">No chart data available</div>
            </div>
          </CardContent>
        </Card>
      );
    }

    // Prepare chart data with indicators
    const enrichedData = chartData.map((candle, index) => {
      const enriched = { ...candle };
      
      // Add indicator data
      Object.entries(indicators).forEach(([name, indicator]) => {
        if (indicator.data) {
          Object.entries(indicator.data).forEach(([key, values]) => {
            const timestamps = Object.keys(values);
            const timestamp = candle.timestamp;
            if (values[timestamp] !== undefined) {
              enriched[`${name}_${key}`] = values[timestamp];
            }
          });
        }
      });
      
      return enriched;
    });

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{selectedAsset} - {selectedTimeframe.toUpperCase()}</span>
            <div className="flex items-center gap-2">
              <Button
                variant={drawingMode === 'trend_line' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setDrawingMode(drawingMode === 'trend_line' ? null : 'trend_line')}
              >
                <PenTool className="h-4 w-4" />
              </Button>
              <Button
                variant={drawingMode === 'horizontal_line' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setDrawingMode(drawingMode === 'horizontal_line' ? null : 'horizontal_line')}
              >
                <Minus className="h-4 w-4" />
              </Button>
              <Button
                variant={drawingMode === 'text_annotation' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setDrawingMode(drawingMode === 'text_annotation' ? null : 'text_annotation')}
              >
                <Type className="h-4 w-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={enrichedData} ref={chartRef}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis yAxisId="price" orientation="right" />
                <YAxis yAxisId="volume" orientation="left" />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value, name) => {
                    if (name.includes('volume')) return [formatVolume(value), name];
                    if (typeof value === 'number') return [formatPrice(value), name];
                    return [value, name];
                  }}
                />
                <Legend />
                
                {/* Candlestick representation using Area charts */}
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="high"
                  stroke="#82ca9d"
                  fill="transparent"
                  strokeWidth={1}
                  dot={false}
                />
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="low"
                  stroke="#82ca9d"
                  fill="transparent"
                  strokeWidth={1}
                  dot={false}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="close"
                  stroke="#8884d8"
                  strokeWidth={2}
                  dot={false}
                  name="Close Price"
                />
                
                {/* Volume bars */}
                <Bar
                  yAxisId="volume"
                  dataKey="volume"
                  fill="#ffc658"
                  opacity={0.3}
                  name="Volume"
                />
                
                {/* Technical Indicators */}
                {indicators.SMA && (
                  <>
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="SMA_SMA_20"
                      stroke="#ff7300"
                      strokeWidth={1}
                      dot={false}
                      name="SMA 20"
                    />
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="SMA_SMA_50"
                      stroke="#387908"
                      strokeWidth={1}
                      dot={false}
                      name="SMA 50"
                    />
                  </>
                )}
                
                {indicators.BB && (
                  <>
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="BB_upper"
                      stroke="#ff0000"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      name="BB Upper"
                    />
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="BB_lower"
                      stroke="#ff0000"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      name="BB Lower"
                    />
                  </>
                )}
                
                {/* Pattern overlays */}
                {showPatterns && patterns.map((pattern, index) => (
                  <ReferenceArea
                    key={index}
                    x1={pattern.start_time}
                    x2={pattern.end_time}
                    fill={getPatternColor(pattern.type)}
                    fillOpacity={0.1}
                    stroke={getPatternColor(pattern.type)}
                    strokeWidth={2}
                    strokeDasharray="3 3"
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Chart Controls */}
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={showPatterns}
                  onCheckedChange={setShowPatterns}
                />
                <Label>Show Patterns</Label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={showVolumeProfile}
                  onCheckedChange={setShowVolumeProfile}
                />
                <Label>Volume Profile</Label>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline">
                <Download className="h-4 w-4 mr-1" />
                Export
              </Button>
              <Button size="sm" variant="outline">
                <Upload className="h-4 w-4 mr-1" />
                Import
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const IndicatorsCard = () => {
    const rsiData = indicators.RSI?.data ? Object.entries(indicators.RSI.data).map(([timestamp, value]) => ({
      timestamp,
      rsi: value
    })) : [];

    const macdData = indicators.MACD?.data ? 
      Object.keys(indicators.MACD.data.macd || {}).map(timestamp => ({
        timestamp,
        macd: indicators.MACD.data.macd[timestamp],
        signal: indicators.MACD.data.signal[timestamp],
        histogram: indicators.MACD.data.histogram[timestamp]
      })) : [];

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* RSI */}
        {rsiData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">RSI (14)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rsiData.slice(-100)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <ReferenceLine y={70} stroke="red" strokeDasharray="3 3" />
                    <ReferenceLine y={30} stroke="green" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="rsi" stroke="#8884d8" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-between text-sm text-gray-500 mt-2">
                <span>Oversold (30)</span>
                <span>Current: {rsiData[rsiData.length - 1]?.rsi?.toFixed(2) || 'N/A'}</span>
                <span>Overbought (70)</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* MACD */}
        {macdData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">MACD (12,26,9)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={macdData.slice(-100)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="gray" />
                    <Line type="monotone" dataKey="macd" stroke="#8884d8" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="signal" stroke="#82ca9d" strokeWidth={2} dot={false} />
                    <Bar dataKey="histogram" fill="#ffc658" opacity={0.6} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  };

  const PatternsCard = () => {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Detected Patterns
          </CardTitle>
          <CardDescription>Chart pattern recognition results</CardDescription>
        </CardHeader>
        <CardContent>
          {patterns.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Target className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No patterns detected</p>
            </div>
          ) : (
            <div className="space-y-4">
              {patterns.map((pattern, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge 
                        style={{ backgroundColor: getPatternColor(pattern.type) }}
                        className="text-white"
                      >
                        {pattern.type.replace('_', ' ').toUpperCase()}
                      </Badge>
                      <span className="text-sm text-gray-500">
                        Confidence: {(pattern.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {new Date(pattern.start_time).toLocaleDateString()} - {new Date(pattern.end_time).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-700 mb-2">{pattern.description}</p>
                  
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>Key Points: {pattern.key_points.length}</span>
                    {pattern.metadata && Object.keys(pattern.metadata).length > 0 && (
                      <span>
                        {Object.entries(pattern.metadata).map(([key, value]) => 
                          `${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`
                        ).join(', ')}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const VolumeProfileCard = () => {
    if (!volumeProfile) return null;

    const profileData = volumeProfile.volume_profile?.slice(0, 20).map(vp => ({
      price: vp.price_level,
      volume: vp.volume,
      isPOC: Math.abs(vp.price_level - volumeProfile.poc_level?.price_level) < 0.01
    })) || [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Volume Profile
          </CardTitle>
          <CardDescription>Volume distribution by price level</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Key Levels</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-2 bg-blue-50 rounded">
                  <span className="text-sm font-medium">Point of Control</span>
                  <span className="font-bold text-blue-600">
                    {formatPrice(volumeProfile.poc_level?.price_level || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 bg-green-50 rounded">
                  <span className="text-sm font-medium">Value Area High</span>
                  <span className="font-bold text-green-600">
                    {formatPrice(volumeProfile.value_area_high || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 bg-red-50 rounded">
                  <span className="text-sm font-medium">Value Area Low</span>
                  <span className="font-bold text-red-600">
                    {formatPrice(volumeProfile.value_area_low || 0)}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Volume Distribution</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={profileData} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="price" type="category" tickFormatter={(value) => formatPrice(value)} />
                    <Tooltip 
                      formatter={(value, name) => [formatVolume(value), 'Volume']}
                      labelFormatter={(value) => formatPrice(value)}
                    />
                    <Bar dataKey="volume" fill="#8884d8">
                      {profileData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.isPOC ? '#ff7300' : '#8884d8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const MultiTimeframeCard = () => {
    if (!multiTimeframe) return null;

    const timeframes = multiTimeframe.timeframes || {};
    const alignment = multiTimeframe.alignment || {};
    const signals = multiTimeframe.signals || {};

    const getTrendColor = (trend) => {
      const colors = {
        'bullish': '#4caf50',
        'bearish': '#f44336',
        'neutral': '#ff9800'
      };
      return colors[trend] || '#757575';
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Multi-Timeframe Analysis
          </CardTitle>
          <CardDescription>Cross-timeframe trend and signal analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-4">Timeframe Analysis</h4>
              <div className="space-y-3">
                {Object.entries(timeframes).map(([tf, data]) => (
                  <div key={tf} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{tf.toUpperCase()}</span>
                      <Badge 
                        style={{ backgroundColor: getTrendColor(data.trend) }}
                        className="text-white"
                      >
                        {data.trend.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {data.momentum && (
                        <>
                          <div>
                            <span className="text-gray-500">ROC 5:</span>
                            <span className={`ml-1 ${data.momentum.roc_5 >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {(data.momentum.roc_5 * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">ROC 20:</span>
                            <span className={`ml-1 ${data.momentum.roc_20 >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {(data.momentum.roc_20 * 100).toFixed(2)}%
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Signal Analysis</h4>
              
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Overall Alignment</span>
                    <Badge variant="outline">
                      {alignment.overall_alignment?.replace('_', ' ').toUpperCase()}
                    </Badge>
                  </div>
                  <div className="text-sm text-gray-600">
                    Strength: {((alignment.alignment_strength || 0) * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Primary Signal</span>
                    <Badge 
                      variant={signals.primary_signal === 'buy' ? 'default' : 
                               signals.primary_signal === 'sell' ? 'destructive' : 'secondary'}
                    >
                      {signals.primary_signal?.toUpperCase() || 'NEUTRAL'}
                    </Badge>
                  </div>
                  <div className="text-sm text-gray-600 mb-2">
                    Confidence: {((signals.confidence || 0) * 100).toFixed(1)}%
                  </div>
                  
                  {signals.supporting_factors && signals.supporting_factors.length > 0 && (
                    <div className="text-sm">
                      <div className="text-green-600 font-medium">Supporting:</div>
                      <ul className="list-disc list-inside text-green-600">
                        {signals.supporting_factors.map((factor, index) => (
                          <li key={index}>{factor}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {signals.conflicting_factors && signals.conflicting_factors.length > 0 && (
                    <div className="text-sm mt-2">
                      <div className="text-red-600 font-medium">Conflicting:</div>
                      <ul className="list-disc list-inside text-red-600">
                        {signals.conflicting_factors.map((factor, index) => (
                          <li key={index}>{factor}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading && !chartData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          {error}
          <Button onClick={() => window.location.reload()} className="ml-2" size="sm">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Advanced Charting</h1>
          <p className="text-gray-600">Professional trading charts with advanced analysis tools</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedAsset} onValueChange={setSelectedAsset}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="BTC">BTC</SelectItem>
              <SelectItem value="ETH">ETH</SelectItem>
              <SelectItem value="LTC">LTC</SelectItem>
              <SelectItem value="XRP">XRP</SelectItem>
              <SelectItem value="DOGE">DOGE</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1m">1m</SelectItem>
              <SelectItem value="5m">5m</SelectItem>
              <SelectItem value="1h">1h</SelectItem>
              <SelectItem value="4h">4h</SelectItem>
              <SelectItem value="1d">1d</SelectItem>
            </SelectContent>
          </Select>
          
          <Button onClick={() => {
            loadChartData();
            loadIndicators();
            if (showPatterns) loadPatterns();
            if (showVolumeProfile) loadVolumeProfile();
          }} className="flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="chart">Chart</TabsTrigger>
          <TabsTrigger value="indicators">Indicators</TabsTrigger>
          <TabsTrigger value="patterns">Patterns</TabsTrigger>
          <TabsTrigger value="volume">Volume Profile</TabsTrigger>
          <TabsTrigger value="timeframes">Multi-TF</TabsTrigger>
        </TabsList>

        <TabsContent value="chart" className="space-y-6">
          <MainChartCard />
        </TabsContent>

        <TabsContent value="indicators" className="space-y-6">
          <div className="flex items-center gap-4 mb-4">
            <Label>Select Indicators:</Label>
            <div className="flex flex-wrap gap-2">
              {['RSI', 'MACD', 'BB', 'SMA', 'EMA', 'STOCH', 'ATR'].map(indicator => (
                <div key={indicator} className="flex items-center gap-1">
                  <Checkbox
                    checked={selectedIndicators.includes(indicator)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedIndicators([...selectedIndicators, indicator]);
                      } else {
                        setSelectedIndicators(selectedIndicators.filter(i => i !== indicator));
                      }
                    }}
                  />
                  <Label className="text-sm">{indicator}</Label>
                </div>
              ))}
            </div>
          </div>
          <IndicatorsCard />
        </TabsContent>

        <TabsContent value="patterns" className="space-y-6">
          <PatternsCard />
        </TabsContent>

        <TabsContent value="volume" className="space-y-6">
          <VolumeProfileCard />
        </TabsContent>

        <TabsContent value="timeframes" className="space-y-6">
          <MultiTimeframeCard />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedCharting;
