"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { cn } from "@/lib/utils";
import { urlService, Activity as ActivityType, DashboardStats } from '@/app/api/urlService';
import axios from 'axios';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

const actionColors = {
  blocked: 'destructive',
  allowed: 'default',
  visited: 'secondary',
  checking: 'outline'
} as const;

const Dashboard = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [youtubeActivityEnabled, setYoutubeActivityEnabled] = useState(false);

  const RISK_COLORS = {
    high: '#ef4444',
    medium: '#f97316',
    low: '#22c55e',
    unknown: '#94a3b8'
  };

  const calculateRiskDistribution = (activities: ActivityType[]) => {
    const distribution: { [key: string]: number } = {
      high: 0,
      medium: 0,
      low: 0,
      unknown: 0
    };

    activities?.forEach(activity => {
      const risk = activity.risk_level?.toLowerCase() || 'unknown';
      distribution[risk]++;
    });

    return Object.entries(distribution).map(([risk, count]) => ({
      name: risk,
      value: count
    }));
  };

  const formatActionData = (activities: ActivityType[]) => {
    const timeGroups: { [key: string]: { blocked: number, allowed: number, visited: number } } = {};

    activities?.forEach(activity => {
      const hour = new Date(activity.timestamp).getHours();
      const timeKey = `${hour.toString().padStart(2, '0')}:00`;

      if (!timeGroups[timeKey]) {
        timeGroups[timeKey] = { blocked: 0, allowed: 0, visited: 0 };
      }

      timeGroups[timeKey][activity.action as keyof typeof timeGroups[string]]++;
    });

    return Object.entries(timeGroups).map(([time, data]) => ({
      time,
      ...data
    }));
  };

  const fetchAndUpdateStats = async () => {
    setIsLoading(true);
    try {
      const data = await urlService.getDashboardStats();
      setStats(data);
      setError(null);
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Failed to fetch dashboard stats');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSettings = async () => {
    try {
      const response = await axios.get('/api/settings');
      setAlertsEnabled(response.data.alertsEnabled);
      setYoutubeActivityEnabled(response.data.youtubeActivityEnabled);
    } catch (err) {
      console.error('Error fetching settings:', err);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchAndUpdateStats();
    fetchSettings();

    // Cleanup
    return () => {};
  }, []);

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={() => window.location.reload()} className="mt-4">
          Retry
        </Button>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-6">
        <Alert variant="default">
          <AlertDescription>No data available.</AlertDescription>
        </Alert>
      </div>
    );
  }

  const chartData = formatActionData(stats?.recent_activities ?? []);

  return (
    <div className="container">
      <div className="p-6 space-y-6">
        {/* Protection Stats Card */}
        <Card>
          <CardHeader>
            <CardTitle>Protection Statistics</CardTitle>
            <CardDescription>Real-time protection overview</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Total Sites</span>
              <Badge variant="secondary">{stats?.total_sites ?? 0}</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span>Blocked Sites</span>
              <Badge variant="destructive">{stats?.blocked_sites ?? 0}</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span>Allowed Sites</span>
              <Badge variant="default">{stats?.allowed_sites ?? 0}</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span>Visited Sites</span>
              <Badge variant="secondary">{stats?.visited_sites ?? 0}</Badge>
            </div>
            <Progress
              value={stats ? (stats.blocked_sites / Math.max(stats.total_sites, 1)) * 100 : 0}
              className="h-2"
            />
          </CardContent>
        </Card>

        {/* Recent Activity Card */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest browsing activities</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="blocked" stroke="#ef4444" />
                <Line type="monotone" dataKey="allowed" stroke="#3b82f6" />
                <Line type="monotone" dataKey="visited" stroke="#22c55e" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Distribution and Alerts Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Risk Distribution Card */}
          <Card>
            <CardHeader>
              <CardTitle>Risk Distribution</CardTitle>
              <CardDescription>Distribution of risk levels across activities</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={calculateRiskDistribution(stats?.recent_activities ?? [])}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, value, percent }) =>
                      `${name} (${value}): ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {calculateRiskDistribution(stats?.recent_activities ?? []).map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={RISK_COLORS[entry.name as keyof typeof RISK_COLORS]}
                      />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => [`${value} sites`, 'Count']} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Alerts Card */}
          <Card>
            <CardHeader>
              <CardTitle>Alerts</CardTitle>
              <CardDescription>Important notifications and warnings</CardDescription>
            </CardHeader>
            <CardContent>
              {stats.alerts?.length > 0 ? (
                <div className="space-y-4">
                  {stats.alerts.map((alert) => (
                    <div key={alert.id} className="flex items-center justify-between">
                      <span>{alert.message}</span>
                      <Badge
                        variant={
                          alert.priority === 'high'
                            ? 'destructive'
                            : alert.priority === 'medium'
                            ? 'secondary'
                            : 'default'
                        }
                      >
                        {alert.priority}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  No alerts to display.
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Settings Card */}
        <Card>
          <CardHeader>
            <CardTitle>Settings</CardTitle>
            <CardDescription>Manage your settings</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <Label htmlFor="alerts-toggle" className="text-sm font-medium">
                Enable Alerts
              </Label>
              <Switch
                id="alerts-toggle"
                checked={alertsEnabled}
                onChange={() => setAlertsEnabled(!alertsEnabled)}
              />
            </div>
            <div className="flex items-center space-x-4 mt-4">
              <Label htmlFor="youtube-activity-toggle" className="text-sm font-medium">
                Enable YouTube Activity
              </Label>
              <Switch
                id="youtube-activity-toggle"
                checked={youtubeActivityEnabled}
                onChange={() => setYoutubeActivityEnabled(!youtubeActivityEnabled)}
              />
            </div>
          </CardContent>
        </Card>

        {/* ML Model Updates Card */}
        <Card>
          <CardHeader>
            <CardTitle>ML Model Updates</CardTitle>
            <CardDescription>Updates related to the ML model used in the Chrome extension</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={fetchAndUpdateStats} className="mt-4">
              Refresh ML Model Updates
            </Button>
            {stats.ml_model_updates?.length > 0 ? (
              <div className="space-y-4 mt-4">
                {stats.ml_model_updates.map((update) => (
                  <div key={update.id} className="flex items-center justify-between">
                    <span>{update.message}</span>
                    <Badge
                      variant={
                        update.priority === 'high'
                          ? 'destructive'
                          : update.priority === 'medium'
                          ? 'secondary'
                          : 'default'
                      }
                    >
                      {update.priority}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                No ML model updates to display.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
