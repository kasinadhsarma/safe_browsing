"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import axios, { AxiosResponse, AxiosError } from 'axios';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    alertsEnabled: false,
    youtubeActivityEnabled: false,
    contentFilterLevel: 'strict', // strict, moderate, light
    timeLimit: 120, // minutes per day
    schedulingEnabled: false,
    scheduledStartTime: '08:00',
    scheduledEndTime: '20:00',
    passwordProtection: true,
    blockAds: true,
    blockUnknownSites: true,
    safeSearch: true
  });

  const [whitelist, setWhitelist] = useState('');
  const [blacklist, setBlacklist] = useState('');
  const [adminPassword, setAdminPassword] = useState('');

  useEffect(() => {
    // Fetch current settings from backend
    axios.get('/api/settings')
      .then((response: AxiosResponse) => {
        setSettings(response.data);
      })
      .catch((error: AxiosError) => {
        console.error('Error fetching settings:', error);
      });
  }, []);

  const handleSettingChange = (setting: string, value: any) => {
    const newSettings = { ...settings, [setting]: value };
    setSettings(newSettings);
    axios.post('/api/settings', { [setting]: value })
      .catch((error: AxiosError) => {
        console.error(`Error updating ${setting}:`, error);
      });
  };

  const handleListUpdate = (type: 'whitelist' | 'blacklist', urls: string) => {
    axios.post(`/api/settings/${type}`, { urls })
      .catch((error: AxiosError) => {
        console.error(`Error updating ${type}:`, error);
      });
  };

  const updatePassword = () => {
    if (adminPassword) {
      axios.post('/api/settings/password', { password: adminPassword })
        .catch((error: AxiosError) => {
          console.error('Error updating password:', error);
        });
    }
  };

  return (
    <div className="container mx-auto py-8 space-y-6">
      {/* Content Filtering */}
      <Card>
        <CardHeader>
          <CardTitle>Content Filtering</CardTitle>
          <CardDescription>Configure content filtering settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Filtering Level</Label>
            <Select 
              value={settings.contentFilterLevel}
              onValueChange={(value) => handleSettingChange('contentFilterLevel', value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select filtering level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="strict">Strict</SelectItem>
                <SelectItem value="moderate">Moderate</SelectItem>
                <SelectItem value="light">Light</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex items-center space-x-4">
            <Label htmlFor="block-ads">Block Advertisements</Label>
            <Switch
              id="block-ads"
              checked={settings.blockAds}
              onChange={() => handleSettingChange('blockAds', !settings.blockAds)}
            />
          </div>

          <div className="flex items-center space-x-4">
            <Label htmlFor="safe-search">Enable Safe Search</Label>
            <Switch
              id="safe-search"
              checked={settings.safeSearch}
              onChange={() => handleSettingChange('safeSearch', !settings.safeSearch)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Time Management */}
      <Card>
        <CardHeader>
          <CardTitle>Time Management</CardTitle>
          <CardDescription>Set usage limits and schedules</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Daily Time Limit (minutes)</Label>
            <Slider
              value={[settings.timeLimit]}
              onValueChange={([value]) => handleSettingChange('timeLimit', value)}
              min={30}
              max={480}
              step={30}
            />
            <p className="text-sm text-gray-500">{settings.timeLimit} minutes per day</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center space-x-4">
              <Label htmlFor="scheduling">Enable Scheduling</Label>
              <Switch
                id="scheduling"
                checked={settings.schedulingEnabled}
                onChange={() => handleSettingChange('schedulingEnabled', !settings.schedulingEnabled)}
              />
            </div>

            {settings.schedulingEnabled && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Start Time</Label>
                  <Input
                    type="time"
                    value={settings.scheduledStartTime}
                    onChange={(e) => handleSettingChange('scheduledStartTime', e.target.value)}
                  />
                </div>
                <div>
                  <Label>End Time</Label>
                  <Input
                    type="time"
                    value={settings.scheduledEndTime}
                    onChange={(e) => handleSettingChange('scheduledEndTime', e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Website Lists */}
      <Card>
        <CardHeader>
          <CardTitle>Website Management</CardTitle>
          <CardDescription>Manage allowed and blocked websites</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Allowed Websites (Whitelist)</Label>
            <Textarea
              placeholder="Enter URLs (one per line)"
              value={whitelist}
              onChange={(e) => setWhitelist(e.target.value)}
              onBlur={() => handleListUpdate('whitelist', whitelist)}
              rows={4}
            />
          </div>

          <div className="space-y-2">
            <Label>Blocked Websites (Blacklist)</Label>
            <Textarea
              placeholder="Enter URLs (one per line)"
              value={blacklist}
              onChange={(e) => setBlacklist(e.target.value)}
              onBlur={() => handleListUpdate('blacklist', blacklist)}
              rows={4}
            />
          </div>

          <div className="flex items-center space-x-4">
            <Label htmlFor="block-unknown">Block Unknown Sites</Label>
            <Switch
              id="block-unknown"
              checked={settings.blockUnknownSites}
              onChange={() => handleSettingChange('blockUnknownSites', !settings.blockUnknownSites)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Alerts & Monitoring */}
      <Card>
        <CardHeader>
          <CardTitle>Alerts & Monitoring</CardTitle>
          <CardDescription>Configure monitoring and notification settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Label htmlFor="alerts-toggle">Activity Alerts</Label>
              <Switch
                id="alerts-toggle"
                checked={settings.alertsEnabled}
                onChange={() => handleSettingChange('alertsEnabled', !settings.alertsEnabled)}
              />
            </div>

            <div className="flex items-center space-x-4">
              <Label htmlFor="youtube-activity-toggle">YouTube Monitoring</Label>
              <Switch
                id="youtube-activity-toggle"
                checked={settings.youtubeActivityEnabled}
                onChange={() => handleSettingChange('youtubeActivityEnabled', !settings.youtubeActivityEnabled)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Security */}
      <Card>
        <CardHeader>
          <CardTitle>Security</CardTitle>
          <CardDescription>Protect these settings with a password</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center space-x-4">
            <Label htmlFor="password-protection">Password Protection</Label>
            <Switch
              id="password-protection"
              checked={settings.passwordProtection}
              onChange={() => handleSettingChange('passwordProtection', !settings.passwordProtection)}
            />
          </div>

          {settings.passwordProtection && (
            <div className="space-y-2">
              <Label>Admin Password</Label>
              <div className="flex space-x-2">
                <Input
                  type="password"
                  value={adminPassword}
                  onChange={(e) => setAdminPassword(e.target.value)}
                  placeholder="Enter new password"
                />
                <Button onClick={updatePassword}>Update</Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
