import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, ScatterChart, Scatter,
  Heatmap, HeatmapChart
} from 'recharts';
import { 
  TrendingUp, TrendingDown, BarChart3, Target, Zap, Settings, 
  Calendar, Users, Activity, AlertTriangle, CheckCircle, Search
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

const EnhancedBacktesting = () => {
  const [backtestData, setBacktestData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedRunId, setSelectedRunId] = useState('');
  const [availableResults, setAvailableResults] = useState([]);

  const fetchData = async (endpoint, params = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const queryString = new URLSearchParams(params).toString();
      const url = `http://localhost:5000/api/backtesting/enhanced/${endpoint}${queryString ? '?' + queryString : ''}`;
      
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

  const fetchAvailableResults = async () => {
    const data = await fetchData('results');
    if (data) {
      setAvailableResults(data.results || []);
    }
  };

  const fetchSummaryData = async () => {
    const data = await fetchData('summary');
    if (data) {
      setBacktestData(prev => ({ ...prev, summary: data }));
    }
  };

  useEffect(() => {
    fetchAvailableResults();
    fetchSummaryData();
  }, []);

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const WalkForwardCard = () => {
    const [wfData, setWfData] = useState(null);
    const [wfParams, setWfParams] = useState({
      train_period_days: 180,
      test_period_days: 60,
      step_days: 30
    });

    const runWalkForward = async () => {
      const data = await fetchData('walk-forward', wfParams);
      if (data) {
        setWfData(data);
      }
    };

    const chartData = wfData?.windows?.map((window, index) => ({
      window: `W${window.window_id}`,
      train_return: (window.train_metrics?.avg_total_return || 0) * 100,
      test_return: (window.test_metrics?.avg_total_return || 0) * 100,
      degradation: window.out_of_sample_performance * 100
    })) || [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Walk-Forward Analysis
          </CardTitle>
          <CardDescription>Progressive backtesting methodology</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <Label>Train Period (Days)</Label>
              <Input
                type="number"
                value={wfParams.train_period_days}
                onChange={(e) => setWfParams(prev => ({ ...prev, train_period_days: parseInt(e.target.value) }))}
              />
            </div>
            <div>
              <Label>Test Period (Days)</Label>
              <Input
                type="number"
                value={wfParams.test_period_days}
                onChange={(e) => setWfParams(prev => ({ ...prev, test_period_days: parseInt(e.target.value) }))}
              />
            </div>
            <div>
              <Label>Step Size (Days)</Label>
              <Input
                type="number"
                value={wfParams.step_days}
                onChange={(e) => setWfParams(prev => ({ ...prev, step_days: parseInt(e.target.value) }))}
              />
            </div>
          </div>
          
          <Button onClick={runWalkForward} className="mb-4" disabled={loading}>
            Run Walk-Forward Analysis
          </Button>

          {wfData && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-sm text-gray-500">Total Windows</div>
                  <div className="text-2xl font-bold">{wfData.summary?.total_windows || 0}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500">Avg OOS Return</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {formatPercentage(wfData.summary?.avg_oos_return || 0)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500">OOS Consistency</div>
                  <div className="text-2xl font-bold text-green-600">
                    {formatPercentage(wfData.summary?.oos_consistency || 0)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500">Degradation</div>
                  <div className={`text-2xl font-bold ${(wfData.summary?.degradation || 0) < 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {formatPercentage(wfData.summary?.degradation || 0)}
                  </div>
                </div>
              </div>

              {chartData.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">In-Sample vs Out-of-Sample Performance</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="window" />
                      <YAxis />
                      <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                      <Legend />
                      <Bar dataKey="train_return" fill="#8884d8" name="In-Sample Return" />
                      <Bar dataKey="test_return" fill="#82ca9d" name="Out-of-Sample Return" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const StrategyComparisonCard = () => {
    const [comparisonData, setComparisonData] = useState(null);
    const [selectedMetrics, setSelectedMetrics] = useState(['total_return', 'sharpe_ratio', 'max_drawdown']);

    const runComparison = async () => {
      const data = await fetchData('strategy-comparison', { metrics: selectedMetrics.join(',') });
      if (data) {
        setComparisonData(data);
      }
    };

    useEffect(() => {
      if (availableResults.length > 0) {
        runComparison();
      }
    }, [availableResults, selectedMetrics]);

    const chartData = comparisonData?.comparison_matrix?.slice(0, 10).map(item => ({
      run_id: item.run_id.substring(0, 8),
      total_return: (item.total_return || 0) * 100,
      sharpe_ratio: item.sharpe_ratio || 0,
      max_drawdown: Math.abs(item.max_drawdown || 0) * 100,
      composite_score: (item.composite_score || 0) * 100
    })) || [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Strategy Comparison Matrix
          </CardTitle>
          <CardDescription>Side-by-side strategy performance comparison</CardDescription>
        </CardHeader>
        <CardContent>
          {comparisonData && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-sm text-gray-500">Best Overall Strategy</div>
                  <div className="text-lg font-bold text-green-600">
                    {comparisonData.best_overall?.substring(0, 12) || 'N/A'}
                  </div>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-sm text-gray-500">Total Strategies</div>
                  <div className="text-lg font-bold text-blue-600">
                    {comparisonData.total_strategies || 0}
                  </div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm text-gray-500">Metrics Analyzed</div>
                  <div className="text-lg font-bold text-purple-600">
                    {comparisonData.metrics_analyzed?.length || 0}
                  </div>
                </div>
              </div>

              {chartData.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Top 10 Strategies Performance</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={chartData} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="run_id" type="category" width={80} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="composite_score" fill="#8884d8" name="Composite Score %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {comparisonData.rankings && (
                <div>
                  <h4 className="font-semibold mb-2">Metric Rankings</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(comparisonData.rankings).map(([metric, ranking]) => (
                      <div key={metric} className="p-3 border rounded-lg">
                        <div className="font-medium mb-2 capitalize">{metric.replace('_', ' ')}</div>
                        <div className="text-sm">
                          <div className="flex justify-between">
                            <span>Best:</span>
                            <span className="font-semibold text-green-600">
                              {ranking.best_run?.substring(0, 8)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Value:</span>
                            <span className="font-semibold">
                              {typeof ranking.best_value === 'number' ? 
                                (metric.includes('return') || metric.includes('drawdown') ? 
                                  formatPercentage(ranking.best_value) : 
                                  ranking.best_value.toFixed(3)
                                ) : 
                                ranking.best_value
                              }
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const TradeAnalysisCard = () => {
    const [tradeData, setTradeData] = useState(null);

    const runTradeAnalysis = async () => {
      const data = await fetchData('trade-analysis', selectedRunId ? { run_id: selectedRunId } : {});
      if (data) {
        setTradeData(data);
      }
    };

    useEffect(() => {
      runTradeAnalysis();
    }, [selectedRunId]);

    const pnlHistogramData = tradeData?.trade_distribution?.pnl_histogram ? 
      tradeData.trade_distribution.pnl_histogram.bins?.slice(0, -1).map((bin, index) => ({
        range: `${bin.toFixed(0)} - ${tradeData.trade_distribution.pnl_histogram.bins[index + 1]?.toFixed(0)}`,
        count: tradeData.trade_distribution.pnl_histogram.counts[index] || 0
      })) || [] : [];

    const assetData = tradeData?.asset_analysis ? 
      Object.entries(tradeData.asset_analysis).map(([asset, stats]) => ({
        asset,
        trades: stats.trades,
        win_rate: stats.win_rate * 100,
        total_pnl: stats.total_pnl,
        avg_pnl: stats.avg_pnl
      })) : [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Trade Analysis Dashboard
          </CardTitle>
          <CardDescription>Individual trade breakdown with statistics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <Label>Select Strategy (Optional)</Label>
            <Select value={selectedRunId} onValueChange={setSelectedRunId}>
              <SelectTrigger>
                <SelectValue placeholder="All strategies" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All strategies</SelectItem>
                {availableResults.map(result => (
                  <SelectItem key={result.run_id} value={result.run_id}>
                    {result.run_id.substring(0, 12)}... ({result.trade_count} trades)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {tradeData && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-sm text-gray-500">Total Trades</div>
                  <div className="text-2xl font-bold">{tradeData.trade_summary?.total_trades || 0}</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-sm text-gray-500">Win Rate</div>
                  <div className="text-2xl font-bold text-green-600">
                    {formatPercentage(tradeData.trade_summary?.win_rate || 0)}
                  </div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <div className="text-sm text-gray-500">Profit Factor</div>
                  <div className="text-2xl font-bold text-purple-600">
                    {(tradeData.trade_summary?.profit_factor || 0).toFixed(2)}
                  </div>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <div className="text-sm text-gray-500">Avg Duration (hrs)</div>
                  <div className="text-2xl font-bold text-orange-600">
                    {(tradeData.trade_summary?.avg_duration_hours || 0).toFixed(1)}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {pnlHistogramData.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2">PnL Distribution</h4>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={pnlHistogramData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {assetData.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2">Performance by Asset</h4>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={assetData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="asset" />
                        <YAxis />
                        <Tooltip formatter={(value, name) => 
                          name === 'win_rate' ? `${value.toFixed(1)}%` : 
                          name === 'trades' ? value :
                          formatCurrency(value)
                        } />
                        <Legend />
                        <Bar dataKey="win_rate" fill="#82ca9d" name="Win Rate %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {tradeData.time_analysis && (
                <div>
                  <h4 className="font-semibold mb-2">Time-Based Analysis</h4>
                  <Tabs defaultValue="hourly">
                    <TabsList>
                      <TabsTrigger value="hourly">Hourly</TabsTrigger>
                      <TabsTrigger value="daily">Daily</TabsTrigger>
                      <TabsTrigger value="monthly">Monthly</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="hourly">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(tradeData.time_analysis.hourly || {}).map(([hour, stats]) => ({
                          hour: `${hour}:00`,
                          trades: stats.trades,
                          pnl: stats.total_pnl
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="trades" fill="#8884d8" name="Trades" />
                        </BarChart>
                      </ResponsiveContainer>
                    </TabsContent>
                    
                    <TabsContent value="daily">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(tradeData.time_analysis.daily || {}).map(([day, stats]) => ({
                          day,
                          trades: stats.trades,
                          pnl: stats.total_pnl
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="day" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="trades" fill="#82ca9d" name="Trades" />
                        </BarChart>
                      </ResponsiveContainer>
                    </TabsContent>
                    
                    <TabsContent value="monthly">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(tradeData.time_analysis.monthly || {}).map(([month, stats]) => ({
                          month,
                          trades: stats.trades,
                          pnl: stats.total_pnl
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="month" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="trades" fill="#ffc658" name="Trades" />
                        </BarChart>
                      </ResponsiveContainer>
                    </TabsContent>
                  </Tabs>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const PerformanceHeatmapsCard = () => {
    const [heatmapData, setHeatmapData] = useState(null);

    const fetchHeatmaps = async () => {
      const data = await fetchData('performance-heatmaps');
      if (data) {
        setHeatmapData(data);
      }
    };

    useEffect(() => {
      fetchHeatmaps();
    }, []);

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Performance Heatmaps
          </CardTitle>
          <CardDescription>Monthly and yearly performance visualization</CardDescription>
        </CardHeader>
        <CardContent>
          {heatmapData && (
            <div className="space-y-6">
              {heatmapData.yearly_summary && (
                <div>
                  <h4 className="font-semibold mb-2">Yearly Performance Summary</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(heatmapData.yearly_summary).map(([year, return_val]) => (
                      <div key={year} className="text-center p-3 border rounded-lg">
                        <div className="text-sm text-gray-500">{year}</div>
                        <div className={`text-lg font-bold ${return_val >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatPercentage(return_val)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {heatmapData.parameter_heatmap && (
                <div>
                  <h4 className="font-semibold mb-2">Parameter Performance Analysis</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Object.entries(heatmapData.parameter_heatmap)
                      .sort(([,a], [,b]) => b.avg_return - a.avg_return)
                      .slice(0, 10)
                      .map(([paramKey, stats], index) => (
                      <div key={index} className="flex justify-between items-center p-2 border rounded">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Config #{index + 1}</div>
                          <div className="text-xs text-gray-500">
                            {Object.entries(stats.parameters).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(', ')}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`font-semibold ${stats.avg_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatPercentage(stats.avg_return)}
                          </div>
                          <div className="text-xs text-gray-500">
                            Consistency: {formatPercentage(stats.consistency)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const CorrelationAnalysisCard = () => {
    const [correlationData, setCorrelationData] = useState(null);

    const fetchCorrelation = async () => {
      const data = await fetchData('correlation-analysis');
      if (data) {
        setCorrelationData(data);
      }
    };

    useEffect(() => {
      fetchCorrelation();
    }, []);

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Correlation Analysis
          </CardTitle>
          <CardDescription>Strategy correlation matrices and diversification analysis</CardDescription>
        </CardHeader>
        <CardContent>
          {correlationData && (
            <div className="space-y-6">
              {correlationData.diversification_analysis && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-sm text-gray-500">Avg Correlation</div>
                    <div className="text-lg font-bold text-blue-600">
                      {(correlationData.diversification_analysis.avg_correlation || 0).toFixed(3)}
                    </div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-sm text-gray-500">Diversification Ratio</div>
                    <div className="text-lg font-bold text-green-600">
                      {(correlationData.diversification_analysis.diversification_ratio || 0).toFixed(3)}
                    </div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <div className="text-sm text-gray-500">Portfolio Vol</div>
                    <div className="text-lg font-bold text-purple-600">
                      {formatPercentage(correlationData.diversification_analysis.portfolio_volatility || 0)}
                    </div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded-lg">
                    <div className="text-sm text-gray-500">Diversification Benefit</div>
                    <div className="text-lg font-bold text-orange-600">
                      {(correlationData.diversification_analysis.diversification_benefit || 0).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {correlationData.high_correlations && correlationData.high_correlations.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">High Correlations (>70%)</h4>
                  <div className="space-y-2">
                    {correlationData.high_correlations.map((corr, index) => (
                      <div key={index} className="flex justify-between items-center p-2 border rounded">
                        <div>
                          <span className="font-medium">{corr.strategy_1.substring(0, 8)}</span>
                          <span className="mx-2">↔</span>
                          <span className="font-medium">{corr.strategy_2.substring(0, 8)}</span>
                        </div>
                        <Badge variant={Math.abs(corr.correlation) > 0.9 ? "destructive" : "secondary"}>
                          {(corr.correlation * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {correlationData.clustering_suggestions && correlationData.clustering_suggestions.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Strategy Clustering Suggestions</h4>
                  <div className="space-y-2">
                    {correlationData.clustering_suggestions.map((cluster, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="font-medium mb-1">Cluster {cluster.cluster_id + 1}</div>
                        <div className="text-sm text-gray-600 mb-2">
                          Strategies: {cluster.strategies?.map(s => s.substring(0, 8)).join(', ')}
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Avg Correlation: {(cluster.avg_correlation * 100).toFixed(1)}%</span>
                          <Badge variant={cluster.avg_correlation > 0.7 ? "destructive" : "default"}>
                            {cluster.recommendation}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  if (loading && !backtestData) {
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
          <h1 className="text-3xl font-bold">Enhanced Backtesting Analytics</h1>
          <p className="text-gray-600">Advanced backtesting analysis and optimization tools</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            {availableResults.length} Results Available
          </Badge>
          <Button onClick={() => window.location.reload()} className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Refresh Data
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="walkforward">Walk-Forward</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="trades">Trade Analysis</TabsTrigger>
          <TabsTrigger value="heatmaps">Heatmaps</TabsTrigger>
          <TabsTrigger value="correlation">Correlation</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <StrategyComparisonCard />
            <CorrelationAnalysisCard />
          </div>
        </TabsContent>

        <TabsContent value="walkforward" className="space-y-6">
          <WalkForwardCard />
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <StrategyComparisonCard />
        </TabsContent>

        <TabsContent value="trades" className="space-y-6">
          <TradeAnalysisCard />
        </TabsContent>

        <TabsContent value="heatmaps" className="space-y-6">
          <PerformanceHeatmapsCard />
        </TabsContent>

        <TabsContent value="correlation" className="space-y-6">
          <CorrelationAnalysisCard />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EnhancedBacktesting;
