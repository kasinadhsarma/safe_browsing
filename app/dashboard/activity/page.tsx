"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { urlService, Activity as ActivityType } from '@/app/api/urlService';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { RefreshCw, Search } from 'lucide-react';
import axios from 'axios';

const ActivityPage = () => {
  const [activities, setActivities] = useState<ActivityType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [youtubeActivityEnabled, setYoutubeActivityEnabled] = useState(false);

const fetchActivities = async () => {
  setIsLoading(true);
  setError(null);
  try {
    console.log('Fetching activities...');
    const data = await urlService.getRecentActivities();
    console.log('Activities fetched:', data);
    console.log('Activities length:', data.length);
    setActivities(data.map(activity => ({
      ...activity,
      timestamp: activity.timestamp || new Date().toISOString(),
      url: activity.url || 'N/A'
    })));
  } catch (error) {
    console.error('Error fetching activities:', error);
    setError('Failed to fetch activities. Please try again.');
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
    fetchActivities();
    fetchSettings();
  }, []);

  const filteredActivities = activities.filter(activity =>
    activity.url?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getBadgeVariant = (risk_level?: string) => {
    switch (risk_level?.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      default: return 'default';
    }
  };

  const getActionBadgeVariant = (action: string) => {
    switch (action.toLowerCase()) {
      case 'blocked': return 'destructive';
      case 'allowed': return 'outline';
      case 'visited': return 'secondary';
      default: return 'default';
    }
  };

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Latest browsing activities</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-6">
            <div className="relative w-full max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search activities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button variant="outline" onClick={fetchActivities}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isLoading ? (
            <div className="space-y-4">
              {[...Array(5)].map((_, index) => (
                <Skeleton key={index} className="h-12 w-full" />
              ))}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Website</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Risk Level</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredActivities.length > 0 ? (
                  filteredActivities.map((activity, index) => (
                    <TableRow key={`${activity.url}-${index}`}>
                      <TableCell className="font-medium max-w-md truncate">
                        <a
                          href={activity.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:underline"
                        >
                          {activity.url || 'N/A'}
                        </a>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getActionBadgeVariant(activity.action)}>
                          {activity.action}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {activity.category || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getBadgeVariant(activity.risk_level)}>
                          {activity.risk_level || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {new Date(activity.timestamp).toLocaleString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                      No activities found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ActivityPage;
