import React from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  BarChart3, 
  LineChart, 
  Activity, 
  PieChart, 
  Settings as SettingsIcon,
  TrendingUp,
  Brain,
  Target,
  Layers
} from 'lucide-react';

const Navigation = ({ activeTab, setActiveTab, botRunning, paperTrading }) => {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'trading', label: 'Trading', icon: LineChart },
    { id: 'backtesting', label: 'Backtesting', icon: Activity },
    { id: 'portfolio', label: 'Portfolio', icon: PieChart },
    { id: 'portfolio-analytics', label: 'Portfolio Analytics', icon: TrendingUp },
    { id: 'enhanced-backtesting', label: 'Enhanced Backtesting', icon: Target },
    { id: 'market-intelligence', label: 'Market Intelligence', icon: Brain },
    { id: 'advanced-charting', label: 'Advanced Charting', icon: Layers },
    { id: 'settings', label: 'Settings', icon: SettingsIcon }
  ];

  return (
    <div className="border-b border-slate-700/50 bg-slate-800/30">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Enhanced Trading Bot</h1>
              <p className="text-sm text-slate-400">Advanced AI-Powered Trading Platform</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`h-2 w-2 rounded-full ${botRunning ? 'bg-emerald-400' : 'bg-red-400'}`}></div>
              <span className="text-sm text-slate-300">{botRunning ? 'Running' : 'Stopped'}</span>
            </div>
            
            {paperTrading && (
              <Badge variant="outline" className="border-amber-500/30 text-amber-400">
                Paper Trading
              </Badge>
            )}
          </div>
        </div>

        <nav className="flex space-x-1 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? "default" : "ghost"}
                size="sm"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 whitespace-nowrap ${
                  activeTab === tab.id 
                    ? 'bg-blue-600 text-white' 
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </Button>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default Navigation;
