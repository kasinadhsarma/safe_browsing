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
      setAlerts(data);
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
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {alerts.map((alert, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{alert.url || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge variant="destructive">{alert.action}</Badge>
                    </TableCell>
                    <TableCell>{alert.category || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          alert.risk_level?.toLowerCase() === 'high' ? 'destructive' :
                          alert.risk_level?.toLowerCase() === 'medium' ? 'secondary' :
                          'default'
                        }
                      >
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
