"use client"

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Shield, Clock, AlertTriangle, ChevronLeft, LogOut, Settings, Home, Search, Activity } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { SafeNetLogo } from '@/components/safenet-logo'
import { cn } from "@/lib/utils"
import { urlService, Activity as ActivityType, DashboardStats } from '../api/urlService'

const sidebarItems = [
  { icon: Home, label: 'Overview', id: 'overview' },
  { icon: Clock, label: 'Activity', id: 'activity' },
  { icon: AlertTriangle, label: 'Alerts', id: 'alerts' },
  { icon: Settings, label: 'Settings', id: 'settings' }, // P37c2
]

const actionColors = {
  blocked: 'destructive',
  allowed: 'default',
  visited: 'secondary',
  checking: 'outline'
} as const

const Dashboard = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [activeSection, setActiveSection] = useState('overview')
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [alertsEnabled, setAlertsEnabled] = useState(false) // Pcb3a

  const formatActionData = (activities: ActivityType[]) => {
    const timeGroups: { [key: string]: { blocked: number, allowed: number, visited: number } } = {}
    
    activities.forEach(activity => {
      const hour = new Date(activity.timestamp).getHours()
      const timeKey = `${hour.toString().padStart(2, '0')}:00`
      
      if (!timeGroups[timeKey]) {
        timeGroups[timeKey] = { blocked: 0, allowed: 0, visited: 0 }
      }
      
      timeGroups[timeKey][activity.action as keyof typeof timeGroups[string]]++
    })

    return Object.entries(timeGroups).map(([time, data]) => ({
      time,
      ...data
    }))
  }

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await urlService.getDashboardStats()
        setStats(data)
      } catch (error) {
        console.error('Error fetching dashboard stats:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchStats()
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleLogout = () => {
    // Implement logout functionality here
    console.log('User logged out') // P920f
  }

  const handleAlertToggle = () => {
    setAlertsEnabled(!alertsEnabled)
    // Implement alert settings update logic here
    console.log('Alerts enabled:', !alertsEnabled) // Pcb3a
  }

  if (isLoading) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>
  }

  return (
    <div className="min-h-screen bg-background">
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-10 bg-background/80 backdrop-blur-sm md:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      <aside className={cn(
        "fixed left-0 top-0 z-20 h-full border-r bg-card transition-all duration-300",
        isSidebarOpen ? "w-64" : "w-20",
        "shadow-lg"
      )}>
        <div className="flex h-16 items-center justify-between border-b px-4">
          <div className={cn(
            "flex items-center gap-2",
            isSidebarOpen ? "opacity-100" : "opacity-0 w-0"
          )}>
            <SafeNetLogo className="h-8 w-8" />
            <span className="font-bold">SafeNet</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          >
            <ChevronLeft className={cn(
              "h-4 w-4 transition-transform",
              !isSidebarOpen && "rotate-180"
            )} />
          </Button>
        </div>
        
        <nav className="flex flex-col gap-2 p-4">
          {sidebarItems.map((item) => (
            <Button
              key={item.id}
              variant={activeSection === item.id ? "default" : "ghost"}
              className={cn(
                "flex items-center transition-all duration-200 group relative",
                isSidebarOpen ? "justify-start" : "justify-center"
              )}
              onClick={() => setActiveSection(item.id)}
            >
              <item.icon className={cn(
                "h-5 w-5",
                !isSidebarOpen && "mx-auto"
              )} />
              {isSidebarOpen && <span className="ml-2">{item.label}</span>}
              {!isSidebarOpen && (
                <div className="absolute left-full ml-2 hidden rounded-md bg-accent px-2 py-1 text-sm group-hover:block">
                  {item.label}
                </div>
              )}
            </Button>
          ))}
        </nav>
      </aside>

      <main className={cn(
        "transition-all duration-300",
        isSidebarOpen ? "ml-64" : "ml-20"
      )}>
        <header className="sticky top-0 z-10 h-16 border-b bg-card/80 backdrop-blur">
          <div className="flex h-full items-center justify-between px-4">
            <h1 className="text-lg font-semibold">Safe Browsing Dashboard</h1>
            <div className="flex items-center gap-4">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="/placeholder.svg" alt="@user" />
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>My Account</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem className="text-red-500" onClick={handleLogout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </header>

        <div className="p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-blue-50 dark:bg-blue-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Shield className="h-4 w-4" />
                  <span>Total Sites</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats?.total_sites || 0}</div>
                <Progress value={100} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-red-50 dark:bg-red-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Sites Blocked</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats?.blocked_sites || 0}</div>
                <Progress 
                  value={stats ? (stats.blocked_sites / stats.total_sites) * 100 : 0} 
                  className="mt-2" 
                />
              </CardContent>
            </Card>

            <Card className="bg-green-50 dark:bg-green-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Search className="h-4 w-4" />
                  <span>Sites Allowed</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats?.allowed_sites || 0}</div>
                <Progress 
                  value={stats ? (stats.allowed_sites / stats.total_sites) * 100 : 0} 
                  className="mt-2" 
                />
              </CardContent>
            </Card>

            <Card className="bg-purple-50 dark:bg-purple-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-4 w-4" />
                  <span>Sites Visited</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats?.visited_sites || 0}</div>
                <Progress 
                  value={stats ? (stats.visited_sites / stats.total_sites) * 100 : 0} 
                  className="mt-2" 
                />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Activity Overview</CardTitle>
              <CardDescription>24-hour monitoring summary</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats ? formatActionData(stats.recent_activities) : []}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--background)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius)'
                      }}
                    />
                    <Legend />
                    <Line type="monotone" name="Blocked" dataKey="blocked" stroke="var(--destructive)" strokeWidth={2} />
                    <Line type="monotone" name="Allowed" dataKey="allowed" stroke="var(--primary)" strokeWidth={2} />
                    <Line type="monotone" name="Visited" dataKey="visited" stroke="var(--muted-foreground)" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Latest browsing activities</CardDescription>
            </CardHeader>
            <CardContent>
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
                  {stats?.recent_activities.map((activity: ActivityType, index: number) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">{activity.url}</TableCell>
                      <TableCell>
                        <Badge variant={actionColors[activity.action as keyof typeof actionColors]}>
                          {activity.action}
                        </Badge>
                      </TableCell>
                      <TableCell>{activity.category || 'Unknown'}</TableCell>
                      <TableCell>{activity.risk_level || 'N/A'}</TableCell>
                      <TableCell>
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Alert>
            <AlertDescription className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>
                Active monitoring: Tracking site visits, safety checks, and blocking attempts in real-time
              </span>
            </AlertDescription>
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle>Alert Settings</CardTitle>
              <CardDescription>Manage your activity alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-4">
                <label htmlFor="alerts-toggle" className="text-sm font-medium">
                  Enable Alerts
                </label>
                <input
                  id="alerts-toggle"
                  type="checkbox"
                  checked={alertsEnabled}
                  onChange={handleAlertToggle}
                  className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}

export default Dashboard
