import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, ScatterChart, Scatter
} from 'recharts';
import { 
  TrendingUp, TrendingDown, AlertTriangle, Target, BarChart3, 
  PieChart as PieChartIcon, Activity, Shield, Zap, Calculator
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const PortfolioAnalytics = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchAnalytics = async (endpoint = 'comprehensive') => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5000/api/portfolio/analytics/${endpoint}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setAnalyticsData(data);
      }
    } catch (err) {
      setError(`Failed to fetch analytics: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const RiskAttributionCard = ({ data }) => {
    if (!data || data.error) return null;

    const riskData = [
      { name: 'Systematic Risk', value: data.risk_decomposition?.systematic_pct || 0, color: '#FF8042' },
      { name: 'Idiosyncratic Risk', value: data.risk_decomposition?.idiosyncratic_pct || 0, color: '#0088FE' }
    ];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Risk Attribution Analysis
          </CardTitle>
          <CardDescription>Factor-based risk decomposition</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Portfolio Volatility</h4>
              <div className="text-2xl font-bold text-blue-600">
                {formatPercentage(data.portfolio_volatility || 0)}
              </div>
            </div>
            <div>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={riskData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {riskData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {data.factor_exposures && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Factor Exposures</h4>
              <div className="space-y-2">
                {Object.entries(data.factor_exposures).map(([asset, exposure]) => (
                  <div key={asset} className="flex justify-between items-center">
                    <span>{asset}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">β: {exposure.beta?.toFixed(3)}</Badge>
                      <Badge variant="outline">ρ: {exposure.correlation?.toFixed(3)}</Badge>
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

  const PerformanceAttributionCard = ({ data }) => {
    if (!data || data.error) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Performance Attribution
          </CardTitle>
          <CardDescription>Alpha vs Beta analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-sm text-gray-500">Alpha</div>
              <div className={`text-xl font-bold ${data.alpha >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(data.alpha || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Beta</div>
              <div className="text-xl font-bold text-blue-600">
                {(data.beta || 0).toFixed(3)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Information Ratio</div>
              <div className="text-xl font-bold text-purple-600">
                {(data.information_ratio || 0).toFixed(3)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Tracking Error</div>
              <div className="text-xl font-bold text-orange-600">
                {formatPercentage(data.tracking_error || 0)}
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex justify-between items-center">
              <span>Portfolio Return</span>
              <span className="font-semibold">{formatPercentage(data.portfolio_return || 0)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Benchmark Return ({data.benchmark_name})</span>
              <span className="font-semibold">{formatPercentage(data.benchmark_return || 0)}</span>
            </div>
            <div className="flex justify-between items-center font-bold">
              <span>Excess Return</span>
              <span className={data.excess_return >= 0 ? 'text-green-600' : 'text-red-600'}>
                {formatPercentage(data.excess_return || 0)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const SharpeOptimizationCard = ({ data }) => {
    if (!data || data.error) return null;

    const weightsData = data.optimal_weights ? Object.entries(data.optimal_weights).map(([asset, weight]) => ({
      asset,
      weight: weight * 100
    })) : [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Sharpe Ratio Optimization
          </CardTitle>
          <CardDescription>Optimal portfolio allocation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Optimization Results</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Current Sharpe Ratio</span>
                  <span className="font-semibold">{(data.current_sharpe_ratio || 0).toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Optimal Sharpe Ratio</span>
                  <span className="font-semibold text-green-600">{(data.optimal_sharpe_ratio || 0).toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Improvement</span>
                  <span className={`font-semibold ${data.improvement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {(data.improvement || 0).toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Optimal Return</span>
                  <span className="font-semibold">{formatPercentage(data.optimal_return || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Optimal Volatility</span>
                  <span className="font-semibold">{formatPercentage(data.optimal_volatility || 0)}</span>
                </div>
              </div>
            </div>
            
            {weightsData.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Optimal Weights</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={weightsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="asset" />
                    <YAxis />
                    <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    <Bar dataKey="weight" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  const VaRAnalysisCard = ({ data }) => {
    if (!data || data.error) return null;

    const varData = [
      { method: 'Historical', daily: data.historical_var?.daily || 0, monthly: data.historical_var?.monthly || 0 },
      { method: 'Parametric', daily: data.parametric_var?.daily || 0, monthly: data.parametric_var?.monthly || 0 },
      { method: 'Monte Carlo', daily: data.monte_carlo_var?.daily || 0, monthly: data.monte_carlo_var?.monthly || 0 }
    ].filter(item => item.daily !== 0 || item.monthly !== 0);

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Value at Risk Analysis
          </CardTitle>
          <CardDescription>Risk quantification at {((1 - data.confidence_level) * 100).toFixed(0)}% confidence</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {varData.map((item, index) => (
              <div key={index} className="p-3 border rounded-lg">
                <div className="font-semibold mb-2">{item.method} VaR</div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-gray-500">Daily</span>
                    <div className="text-lg font-bold text-red-600">
                      {formatPercentage(Math.abs(item.daily))}
                    </div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">Monthly</span>
                    <div className="text-lg font-bold text-red-600">
                      {formatPercentage(Math.abs(item.monthly))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {data.expected_shortfall && (
              <div className="p-3 border rounded-lg bg-red-50">
                <div className="font-semibold mb-2">Expected Shortfall (CVaR)</div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-gray-500">Daily</span>
                    <div className="text-lg font-bold text-red-700">
                      {formatPercentage(Math.abs(data.expected_shortfall.daily))}
                    </div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">Monthly</span>
                    <div className="text-lg font-bold text-red-700">
                      {formatPercentage(Math.abs(data.expected_shortfall.monthly))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  const DrawdownAnalysisCard = ({ data }) => {
    if (!data || data.error) return null;

    const drawdownData = data.underwater_curve ? 
      Object.entries(data.underwater_curve).map(([date, value]) => ({
        date: new Date(date).toLocaleDateString(),
        drawdown: value
      })).slice(-50) : []; // Show last 50 points

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="h-5 w-5" />
            Drawdown Analysis
          </CardTitle>
          <CardDescription>Portfolio drawdown metrics and underwater curve</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <div className="text-sm text-gray-500">Max Drawdown</div>
              <div className="text-xl font-bold text-red-600">
                {formatPercentage(Math.abs(data.max_drawdown || 0))}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Current Drawdown</div>
              <div className="text-xl font-bold text-orange-600">
                {formatPercentage(Math.abs(data.current_drawdown || 0))}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Avg Recovery (Days)</div>
              <div className="text-xl font-bold text-blue-600">
                {Math.round(data.avg_recovery_time_days || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Calmar Ratio</div>
              <div className="text-xl font-bold text-purple-600">
                {(data.calmar_ratio || 0).toFixed(3)}
              </div>
            </div>
          </div>
          
          {drawdownData.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Underwater Curve</h4>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={drawdownData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                  <Area type="monotone" dataKey="drawdown" stroke="#ff7300" fill="#ff7300" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const MonteCarloCard = ({ data }) => {
    if (!data || data.error) return null;

    const percentileData = data.percentiles ? Object.entries(data.percentiles).map(([key, value]) => ({
      percentile: key.replace('p', '') + '%',
      value: value
    })) : [];

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Monte Carlo Simulation
          </CardTitle>
          <CardDescription>{data.simulations} simulations over {data.days} days</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-sm text-gray-500">Current Value</div>
              <div className="text-xl font-bold">
                {formatCurrency(data.current_value || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Expected Value</div>
              <div className="text-xl font-bold text-blue-600">
                {formatCurrency(data.expected_final_value || 0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500">Probability of Loss</div>
              <div className="text-xl font-bold text-red-600">
                {formatPercentage(data.probability_of_loss || 0)}
              </div>
            </div>
          </div>
          
          {percentileData.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Value Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={percentileData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="percentile" />
                  <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value) => formatCurrency(value)} />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="p-3 bg-red-50 rounded-lg">
              <div className="text-sm text-gray-500">Worst Case (5%)</div>
              <div className="text-lg font-bold text-red-600">
                {formatCurrency(data.worst_case_5pct || 0)}
              </div>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <div className="text-sm text-gray-500">Best Case (95%)</div>
              <div className="text-lg font-bold text-green-600">
                {formatCurrency(data.best_case_5pct || 0)}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
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
          <Button onClick={() => fetchAnalytics()} className="ml-2" size="sm">
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
          <h1 className="text-3xl font-bold">Portfolio Analytics</h1>
          <p className="text-gray-600">Advanced portfolio analysis and risk metrics</p>
        </div>
        <Button onClick={() => fetchAnalytics()} className="flex items-center gap-2">
          <Calculator className="h-4 w-4" />
          Refresh Analytics
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
          <TabsTrigger value="var">VaR</TabsTrigger>
          <TabsTrigger value="simulation">Simulation</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <RiskAttributionCard data={analyticsData?.risk_attribution} />
            <PerformanceAttributionCard data={analyticsData?.performance_attribution} />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DrawdownAnalysisCard data={analyticsData?.drawdown_analysis} />
            <VaRAnalysisCard data={analyticsData?.var_analysis} />
          </div>
        </TabsContent>

        <TabsContent value="risk" className="space-y-6">
          <RiskAttributionCard data={analyticsData?.risk_attribution} />
          <DrawdownAnalysisCard data={analyticsData?.drawdown_analysis} />
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <PerformanceAttributionCard data={analyticsData?.performance_attribution} />
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          <SharpeOptimizationCard data={analyticsData?.sharpe_optimization} />
        </TabsContent>

        <TabsContent value="var" className="space-y-6">
          <VaRAnalysisCard data={analyticsData?.var_analysis} />
        </TabsContent>

        <TabsContent value="simulation" className="space-y-6">
          <MonteCarloCard data={analyticsData?.monte_carlo_simulation} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PortfolioAnalytics;
