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
    setActivities(data.map(activity => {
      // Ensure valid timestamp
      let timestamp = activity.timestamp;
      try {
        // Test if timestamp is valid
        new Date(timestamp).toISOString();
      } catch {
        timestamp = new Date().toISOString();
      }

      return {
        ...activity,
        timestamp,
        url: activity.url || 'N/A',
        category: activity.category || 'Unknown',
        risk_level: activity.risk_level || 'Unknown',
        age_group: activity.age_group || 'N/A',
        block_reason: activity.block_reason || '-',
        ml_scores: activity.ml_scores || {}
      };
    }).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
  } catch (error) {
    console.error('Error fetching activities:', error);
    setError('Failed to fetch activities. Please try again.');
  } finally {
    setIsLoading(false);
  }
};

const checkUrl = async (url: string, age_group: string) => {
  try {
    console.log(`Checking URL: ${url} for age group: ${age_group}`);
    const result = await urlService.checkUrl(url, age_group);
    console.log('URL check result:', result);
    // Handle the result as needed
  } catch (error) {
    console.error('Error checking URL:', error);
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
      case 'low': return 'default';
      case 'unknown': return 'outline';
      default: return 'outline';
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
                  <TableHead>Age Group</TableHead>
                  <TableHead>Block Reason</TableHead>
                  <TableHead>ML Score</TableHead>
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
                          {activity.category === 'video' ? 'Video' : activity.category === 'audio' ? 'Audio' : activity.category === 'information' ? 'Information' : activity.category === 'adult' ? 'Adult Content' : activity.category || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getBadgeVariant(activity.risk_level)}>
                          {activity.risk_level || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {activity.age_group || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate">
                        {activity.block_reason || '-'}
                      </TableCell>
                      <TableCell>
                        {activity.ml_scores && Object.keys(activity.ml_scores).length > 0 ? 
                          (() => {
                            if (typeof activity.ml_scores === 'object') {
                              const modelScores = {
                                knn: typeof activity.ml_scores.knn === 'object' ? (activity.ml_scores.knn as any).probability || 0 : activity.ml_scores.knn || 0,
                                svm: typeof activity.ml_scores.svm === 'object' ? (activity.ml_scores.svm as any).probability || 0 : activity.ml_scores.svm || 0,
                                nb: typeof activity.ml_scores.nb === 'object' ? (activity.ml_scores.nb as any).probability || 0 : activity.ml_scores.nb || 0
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
                        {(() => {
                          try {
                            return new Date(activity.timestamp).toLocaleString('en-US', {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            });
                          } catch {
                            return 'Invalid Date';
                          }
                        })()}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
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
