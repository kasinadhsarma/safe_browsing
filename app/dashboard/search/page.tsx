"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import axios from 'axios';
import { urlService, Activity as ActivityType } from '@/app/api/urlService';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Spinner } from '@/components/ui/spinner';
import { Alert, AlertDescription } from '@/components/ui/alert';

const SearchPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<ActivityType[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [youtubeActivityEnabled, setYoutubeActivityEnabled] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await urlService.getRecentActivities();
      const filteredResults = data.filter(activity =>
        activity.url?.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setResults(filteredResults);
    } catch (error) {
      console.error('Error searching activities:', error);
      setError('Failed to search activities. Please try again.');
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
    fetchSettings();
  }, []);

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Search</CardTitle>
          <CardDescription>Search for activities and alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4 mb-6">
            <Input
              type="text"
              placeholder="Enter search query"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <Button onClick={handleSearch}>Search</Button>
          </div>

          {loading && (
            <div className="flex justify-center">
              <Spinner />
            </div>
          )}

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {results.length > 0 ? (
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
                {results.map((activity, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{activity.url || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge variant="destructive">{activity.action}</Badge>
                    </TableCell>
                    <TableCell>{activity.category || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          activity.risk_level?.toLowerCase() === 'high' ? 'destructive' :
                          activity.risk_level?.toLowerCase() === 'medium' ? 'secondary' :
                          'default'
                        }
                      >
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
                ))}
              </TableBody>
            </Table>
          ) : (
            <Alert>
              <AlertDescription>No results found.</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SearchPage;
