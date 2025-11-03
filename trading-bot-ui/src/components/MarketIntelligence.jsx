import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter
} from 'recharts';
import { 
  Brain, TrendingUp, TrendingDown, AlertTriangle, Radio, Newspaper,
  Activity, Target, Zap, Bell, Eye, Settings, RefreshCw
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

const MarketIntelligence = () => {
  const [intelligenceData, setIntelligenceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchData = async (endpoint = 'comprehensive') => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5000/api/market-intelligence/${endpoint}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setIntelligenceData(data);
      }
    } catch (err) {
      setError(`Failed to fetch intelligence data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchData();
      }, 60000); // Refresh every minute
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getSentimentColor = (score) => {
    if (score <= -0.6) return '#d32f2f';
    if (score <= -0.2) return '#f57c00';
    if (score >= 0.6) return '#388e3c';
    if (score >= 0.2) return '#689f38';
    return '#fbc02d';
  };

  const getRegimeColor = (regime) => {
    const colors = {
      'bull': '#4caf50',
      'bear': '#f44336',
      'sideways': '#ff9800',
      'volatile': '#9c27b0'
    };
    return colors[regime] || '#757575';
  };

  const SentimentRadarCard = ({ data }) => {
    if (!data || data.error) return null;

    const radarData = data.radar_data || [];
    const overallSentiment = data.overall_sentiment || {};

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Radio className="h-5 w-5" />
            Sentiment Radar
          </CardTitle>
          <CardDescription>Real-time social media sentiment aggregation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <div className="text-center mb-4">
                <div className="text-3xl font-bold mb-2" style={{ color: getSentimentColor(overallSentiment.score) }}>
                  {overallSentiment.level?.replace('_', ' ').toUpperCase() || 'NEUTRAL'}
                </div>
                <div className="text-lg text-gray-600">
                  Score: {(overallSentiment.score || 0).toFixed(3)}
                </div>
                <div className="text-sm text-gray-500">
                  Confidence: {formatPercentage(overallSentiment.confidence || 0)}
                </div>
              </div>

              {data.trending_assets && data.trending_assets.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Trending Assets</h4>
                  <div className="space-y-2">
                    {data.trending_assets.slice(0, 5).map((asset, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="font-medium">{asset.asset}</span>
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant={asset.sentiment_score > 0 ? "default" : "destructive"}
                            style={{ backgroundColor: getSentimentColor(asset.sentiment_score) }}
                          >
                            {(asset.sentiment_score * 100).toFixed(0)}%
                          </Badge>
                          <span className="text-sm text-gray-500">
                            Vol: {asset.volume}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {radarData.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Asset Sentiment Radar</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="asset" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Sentiment"
                      dataKey="sentiment"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.3}
                    />
                    <Radar
                      name="Confidence"
                      dataKey="confidence"
                      stroke="#82ca9d"
                      fill="#82ca9d"
                      fillOpacity={0.2}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {data.sentiment_distribution && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Sentiment Distribution</h4>
              <div className="grid grid-cols-5 gap-2">
                {Object.entries(data.sentiment_distribution).map(([level, count]) => (
                  <div key={level} className="text-center p-2 border rounded">
                    <div className="text-sm font-medium capitalize">{level.replace('_', ' ')}</div>
                    <div className="text-lg font-bold">{count}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const NewsImpactCard = ({ data }) => {
    if (!data || data.error) return null;

    const sentimentDist = data.sentiment_distribution || {};
    const topArticles = data.top_articles || [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Newspaper className="h-5 w-5" />
            News Impact Analyzer
          </CardTitle>
          <CardDescription>News sentiment scoring and trend analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-sm text-gray-500">Total Articles</div>
              <div className="text-2xl font-bold">{data.total_articles || 0}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">High Impact</div>
              <div className="text-2xl font-bold text-red-600">{data.high_impact_articles || 0}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Avg Sentiment</div>
              <div className={`text-2xl font-bold ${(data.average_sentiment || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {(data.average_sentiment || 0).toFixed(3)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Keywords</div>
              <div className="text-2xl font-bold text-blue-600">{data.top_keywords?.length || 0}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Sentiment Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={Object.entries(sentimentDist).map(([key, value]) => ({ name: key, value }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label
                  >
                    {Object.entries(sentimentDist).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {data.top_keywords && (
              <div>
                <h4 className="font-semibold mb-2">Top Keywords</h4>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {data.top_keywords.slice(0, 10).map(([keyword, count], index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm">{keyword}</span>
                      <Badge variant="outline">{count}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {topArticles.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold mb-2">High Impact Articles</h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {topArticles.slice(0, 5).map((article, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="font-medium text-sm mb-1">{article.headline}</div>
                    <div className="flex justify-between items-center text-xs text-gray-500">
                      <span>{article.source}</span>
                      <div className="flex gap-2">
                        <Badge 
                          variant={article.sentiment_score > 0 ? "default" : "destructive"}
                          className="text-xs"
                        >
                          Sentiment: {(article.sentiment_score * 100).toFixed(0)}%
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          Impact: {(article.impact_score * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const MarketRegimeCard = ({ data }) => {
    if (!data || data.error) return null;

    const regimeDistribution = data.regime_distribution || {};
    const assetRegimes = data.asset_regimes || {};

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Market Regime Detection
          </CardTitle>
          <CardDescription>Bull/bear/sideways market identification</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-3 rounded-lg" style={{ backgroundColor: getRegimeColor(data.overall_regime) + '20' }}>
              <div className="text-sm text-gray-500">Overall Regime</div>
              <div className="text-xl font-bold capitalize" style={{ color: getRegimeColor(data.overall_regime) }}>
                {data.overall_regime || 'Unknown'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Confidence</div>
              <div className="text-xl font-bold text-blue-600">
                {formatPercentage(data.regime_confidence || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Regime Changes</div>
              <div className="text-xl font-bold text-orange-600">
                {data.regime_changes?.length || 0}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Overall Stress</div>
              <div className="text-xl font-bold text-red-600">
                {formatPercentage(data.market_stress_indicators?.overall_stress || 0)}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Regime Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Object.entries(regimeDistribution).map(([regime, count]) => ({ regime, count }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="regime" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Asset Regimes</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {Object.entries(assetRegimes).map(([asset, regime]) => (
                  <div key={asset} className="flex justify-between items-center p-2 border rounded">
                    <span className="font-medium">{asset}</span>
                    <div className="flex items-center gap-2">
                      <Badge 
                        style={{ backgroundColor: getRegimeColor(regime.regime) }}
                        className="text-white"
                      >
                        {regime.regime}
                      </Badge>
                      <span className="text-sm text-gray-500">
                        Score: {regime.regime_score?.toFixed(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {data.regime_changes && data.regime_changes.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold mb-2">Recent Regime Changes</h4>
              <div className="space-y-2">
                {data.regime_changes.map((change, index) => (
                  <div key={index} className="p-2 bg-yellow-50 border border-yellow-200 rounded">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{change.asset}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{change.previous_regime}</Badge>
                        <span>→</span>
                        <Badge style={{ backgroundColor: getRegimeColor(change.current_regime) }} className="text-white">
                          {change.current_regime}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(change.change_timestamp).toLocaleString()} 
                      (Confidence: {formatPercentage(change.confidence)})
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const FearGreedIndexCard = ({ data }) => {
    if (!data || data.error) return null;

    const components = data.components || {};

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Fear & Greed Index
          </CardTitle>
          <CardDescription>Custom crypto market sentiment indicator</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center mb-6">
            <div className="relative w-48 h-48 mx-auto mb-4">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="#e5e7eb"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke={data.color || '#fbc02d'}
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${(data.index_value || 0) * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-3xl font-bold" style={{ color: data.color }}>
                  {Math.round(data.index_value || 0)}
                </div>
                <div className="text-sm text-gray-500">Fear & Greed</div>
              </div>
            </div>
            
            <div className="text-xl font-bold mb-2" style={{ color: data.color }}>
              {data.level || 'Neutral'}
            </div>
            
            <div className="text-sm text-gray-600 max-w-md mx-auto">
              {data.interpretation || 'Market sentiment is balanced.'}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-500">Historical Average</div>
              <div className="text-lg font-bold">
                {Math.round(data.historical_average || 50)}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-500">Deviation</div>
              <div className={`text-lg font-bold ${(data.deviation_from_average || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {(data.deviation_from_average || 0) > 0 ? '+' : ''}{Math.round(data.deviation_from_average || 0)}
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Index Components</h4>
            <div className="space-y-2">
              {Object.entries(components).map(([component, data]) => (
                <div key={component} className="flex justify-between items-center p-2 border rounded">
                  <span className="capitalize">{component.replace('_', ' ')}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${data.score || 0}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium w-12 text-right">
                      {Math.round(data.score || 0)}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {Math.round((data.weight || 0) * 100)}%
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const CorrelationMatrixCard = ({ data }) => {
    if (!data || data.error) return null;

    const assets = data.assets || [];
    const matrix = data.correlation_matrix || [];
    const strongestCorr = data.strongest_correlations || [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Dynamic Correlation Matrix
          </CardTitle>
          <CardDescription>Asset correlation with heatmaps</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-sm text-gray-500">Avg Correlation</div>
              <div className="text-lg font-bold text-blue-600">
                {(data.average_correlation || 0).toFixed(3)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Correlation Stress</div>
              <div className="text-lg font-bold text-red-600">
                {formatPercentage(data.correlation_stress || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Assets</div>
              <div className="text-lg font-bold">{assets.length}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Clusters</div>
              <div className="text-lg font-bold text-purple-600">
                {data.clusters?.length || 0}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Correlation Matrix</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="p-1"></th>
                      {assets.map(asset => (
                        <th key={asset} className="p-1 text-center">{asset}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {assets.map((asset, i) => (
                      <tr key={asset}>
                        <td className="p-1 font-medium">{asset}</td>
                        {assets.map((_, j) => {
                          const corr = matrix[i]?.[j] || 0;
                          const intensity = Math.abs(corr);
                          const color = corr > 0 ? 
                            `rgba(34, 197, 94, ${intensity})` : 
                            `rgba(239, 68, 68, ${intensity})`;
                          
                          return (
                            <td 
                              key={j} 
                              className="p-1 text-center text-xs"
                              style={{ backgroundColor: color }}
                            >
                              {corr.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Strongest Correlations</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {strongestCorr.slice(0, 8).map((corr, index) => (
                  <div key={index} className="flex justify-between items-center p-2 border rounded">
                    <div className="text-sm">
                      <span className="font-medium">{corr.asset_1}</span>
                      <span className="mx-1">↔</span>
                      <span className="font-medium">{corr.asset_2}</span>
                    </div>
                    <Badge 
                      variant={Math.abs(corr.correlation) > 0.7 ? "destructive" : "default"}
                      className="text-xs"
                    >
                      {(corr.correlation * 100).toFixed(0)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {data.clusters && data.clusters.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold mb-2">Correlation Clusters</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {data.clusters.map((cluster, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="font-medium mb-2">Cluster {cluster.cluster_id + 1}</div>
                    <div className="text-sm text-gray-600 mb-2">
                      Assets: {cluster.assets.join(', ')}
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Avg Correlation:</span>
                      <Badge variant="outline">
                        {(cluster.avg_correlation * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const AlertsCard = ({ data }) => {
    if (!data || data.error) return null;

    const alerts = data.alerts || [];
    const alertSummary = data.alert_summary || {};

    const getSeverityColor = (severity) => {
      const colors = {
        'critical': '#d32f2f',
        'high': '#f57c00',
        'medium': '#fbc02d',
        'low': '#689f38'
      };
      return colors[severity] || '#757575';
    };

    const getSeverityIcon = (severity) => {
      switch (severity) {
        case 'critical':
        case 'high':
          return <AlertTriangle className="h-4 w-4" />;
        case 'medium':
          return <Bell className="h-4 w-4" />;
        case 'low':
          return <Eye className="h-4 w-4" />;
        default:
          return <Bell className="h-4 w-4" />;
      }
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Intelligent Alert System
          </CardTitle>
          <CardDescription>Smart notifications based on market conditions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 mb-6">
            {Object.entries(alertSummary).map(([severity, count]) => (
              <div key={severity} className="text-center p-3 border rounded-lg">
                <div className="flex items-center justify-center mb-1" style={{ color: getSeverityColor(severity) }}>
                  {getSeverityIcon(severity)}
                </div>
                <div className="text-sm text-gray-500 capitalize">{severity}</div>
                <div className="text-xl font-bold" style={{ color: getSeverityColor(severity) }}>
                  {count}
                </div>
              </div>
            ))}
          </div>

          <div>
            <h4 className="font-semibold mb-2">Recent Alerts</h4>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {alerts.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Bell className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No alerts at this time</p>
                </div>
              ) : (
                alerts.slice(0, 20).map((alert, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div style={{ color: getSeverityColor(alert.severity) }}>
                          {getSeverityIcon(alert.severity)}
                        </div>
                        <Badge 
                          style={{ backgroundColor: getSeverityColor(alert.severity) }}
                          className="text-white text-xs"
                        >
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {alert.alert_type.replace('_', ' ')}
                        </Badge>
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <div className="text-sm font-medium mb-1">
                      {alert.message}
                    </div>
                    
                    <div className="text-xs text-gray-500">
                      Triggered by: {alert.triggered_by}
                    </div>
                    
                    {alert.data && Object.keys(alert.data).length > 0 && (
                      <div className="mt-2 text-xs">
                        <details className="cursor-pointer">
                          <summary className="text-blue-600 hover:text-blue-800">
                            View details
                          </summary>
                          <div className="mt-1 p-2 bg-gray-50 rounded text-xs">
                            <pre className="whitespace-pre-wrap">
                              {JSON.stringify(alert.data, null, 2)}
                            </pre>
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading && !intelligenceData) {
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
          <Button onClick={() => fetchData()} className="ml-2" size="sm">
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
          <h1 className="text-3xl font-bold">Market Intelligence Dashboard</h1>
          <p className="text-gray-600">Real-time market analysis and sentiment monitoring</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            onClick={() => setAutoRefresh(!autoRefresh)}
            className="flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button onClick={() => fetchData()} className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Refresh Data
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
          <TabsTrigger value="news">News</TabsTrigger>
          <TabsTrigger value="regime">Regime</TabsTrigger>
          <TabsTrigger value="feargreed">Fear & Greed</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SentimentRadarCard data={intelligenceData?.sentiment_radar} />
            <FearGreedIndexCard data={intelligenceData?.fear_greed_index} />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <MarketRegimeCard data={intelligenceData?.market_regime} />
            <AlertsCard data={intelligenceData?.alerts} />
          </div>
        </TabsContent>

        <TabsContent value="sentiment" className="space-y-6">
          <SentimentRadarCard data={intelligenceData?.sentiment_radar} />
          <CorrelationMatrixCard data={intelligenceData?.correlation_matrix} />
        </TabsContent>

        <TabsContent value="news" className="space-y-6">
          <NewsImpactCard data={intelligenceData?.news_impact} />
        </TabsContent>

        <TabsContent value="regime" className="space-y-6">
          <MarketRegimeCard data={intelligenceData?.market_regime} />
        </TabsContent>

        <TabsContent value="feargreed" className="space-y-6">
          <FearGreedIndexCard data={intelligenceData?.fear_greed_index} />
        </TabsContent>

        <TabsContent value="alerts" className="space-y-6">
          <AlertsCard data={intelligenceData?.alerts} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MarketIntelligence;
