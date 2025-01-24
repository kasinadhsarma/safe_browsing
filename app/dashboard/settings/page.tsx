"use client";

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

export default function SettingsPage() {
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [youtubeActivityEnabled, setYoutubeActivityEnabled] = useState(false);

  const handleAlertToggle = () => {
    setAlertsEnabled(!alertsEnabled);
    console.log('Alerts enabled:', !alertsEnabled);
  };

  const handleYoutubeActivityToggle = () => {
    setYoutubeActivityEnabled(!youtubeActivityEnabled);
    console.log('YouTube activity enabled:', !youtubeActivityEnabled);
  };

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Activity Alerts</CardTitle>
          <CardDescription>Manage your activity alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <Label htmlFor="alerts-toggle" className="text-sm font-medium">
              Enable Alerts
            </Label>
            <Switch
              id="alerts-toggle"
              checked={alertsEnabled}
              onChange={handleAlertToggle}
            />
          </div>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>YouTube Activity</CardTitle>
          <CardDescription>Manage your YouTube activity settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <Label htmlFor="youtube-activity-toggle" className="text-sm font-medium">
              Enable YouTube Activity
            </Label>
            <Switch
              id="youtube-activity-toggle"
              checked={youtubeActivityEnabled}
              onChange={handleYoutubeActivityToggle}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}