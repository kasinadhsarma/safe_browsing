"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { urlService, Activity as ActivityType } from '@/app/api/urlService';
import { Spinner } from '@/components/ui/spinner'; // Assuming you have a Spinner component
import { Alert, AlertDescription } from '@/components/ui/alert'; // Assuming you have an Alert component
import axios from 'axios';

const AlertsPage = () => {
  const [alerts, setAlerts] = useState<ActivityType[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [youtubeActivityEnabled, setYoutubeActivityEnabled] = useState(false);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      const data = await urlService.getAlerts();
      setAlerts(data.map(alert => {
        return {
          ...alert,
          url: alert.url || 'N/A',
          category: alert.category || 'Unknown',
          risk_level: alert.risk_level || 'Unknown',
          age_group: alert.age_group || 'N/A',
          block_reason: alert.block_reason || '-',
          ml_scores: alert.ml_scores || {}
        };
      }).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
      setError(null);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      setError('Failed to fetch alerts. Please try again later.');
    } finally {
      setLoading(false);
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
    fetchAlerts();
    fetchSettings();
    // Refresh alerts every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const getBadgeVariant = (risk_level?: string) => {
    switch (risk_level?.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': return 'default';
      case 'unknown': return 'outline';
      default: return 'outline';
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto py-8 flex justify-center">
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto py-8">
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Alerts</CardTitle>
          <CardDescription>Recent blocked activities</CardDescription>
        </CardHeader>
        <CardContent>
          {alerts.length === 0 ? (
            <Alert>
              <AlertDescription>No blocked activities found.</AlertDescription>
            </Alert>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Website</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Risk Level</TableHead>
                  <TableHead>Age Group</TableHead>
                  <TableHead>Block Reason</TableHead>
                  <TableHead>ML Score</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {alerts.map((alert, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium max-w-md truncate">
                      <a
                        href={alert.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:underline"
                      >
                        {alert.url || 'N/A'}
                      </a>
                    </TableCell>
                    <TableCell>
                      <Badge variant="destructive">{alert.action}</Badge>
                    </TableCell>
                    <TableCell>{alert.category || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge variant={getBadgeVariant(alert.risk_level)}>
                        {alert.risk_level || 'N/A'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">
                        {alert.age_group || 'N/A'}
                      </Badge>
                    </TableCell>
                    <TableCell className="max-w-xs truncate">
                      {alert.block_reason || '-'}
                    </TableCell>
                    <TableCell>
                      {alert.ml_scores && Object.keys(alert.ml_scores).length > 0 ? 
                        (() => {
                          if (typeof alert.ml_scores === 'object') {
                            const modelScores = {
                              knn: typeof alert.ml_scores.knn === 'object' ? (alert.ml_scores.knn as any).probability || 0 : alert.ml_scores.knn || 0,
                              svm: typeof alert.ml_scores.svm === 'object' ? (alert.ml_scores.svm as any).probability || 0 : alert.ml_scores.svm || 0,
                              nb: typeof alert.ml_scores.nb === 'object' ? (alert.ml_scores.nb as any).probability || 0 : alert.ml_scores.nb || 0
                            };
                            return (
                              <div>
                                <p>KNN: {(modelScores.knn * 100).toFixed(1)}%</p>
                                <p>SVM: {(modelScores.svm * 100).toFixed(1)}%</p>
                                <p>NB: {(modelScores.nb * 100).toFixed(1)}%</p>
                              </div>
                            );
                          }
                          return 'N/A';
                        })()
                        : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {new Date(alert.timestamp).toLocaleString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default AlertsPage;
