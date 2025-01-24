"use client"

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Shield, Clock, AlertTriangle, ChevronLeft, LogOut, Settings, Home, Search, Activity, Lock } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { SafeNetLogo } from '@/components/safenet-logo'
import { cn } from "@/lib/utils"
import Link from 'next/link'

const springConfig = "cubic-bezier(0.87, 0, 0.13, 1)"

const sidebarItems = [
  { icon: Home, label: 'Overview', id: 'overview' },
  { icon: Clock, label: 'Activity', id: 'activity' },
  { icon: AlertTriangle, label: 'Alerts', id: 'alerts' },
]

const Dashboard = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [activeSection, setActiveSection] = useState('overview')
  const [blockedSites, setBlockedSites] = useState([
    { url: 'example.com', method: 'ML Classification', date: '2024-01-24', risk: 'High' },
    { url: 'unsafe-site.com', method: 'ML Classification', date: '2024-01-24', risk: 'High' }
  ])
  const [newSite, setNewSite] = useState("")
  const [stats, setStats] = useState({
    sitesBlocked: 2,
    threatsDetected: 0,
    safeSearches: 0,
    lastUpdated: new Date().toLocaleTimeString()
  })
  const [mlStatus, setMlStatus] = useState({
    knn: 95,
    svm: 92,
    naiveBayes: 88,
    lastCheck: new Date().toLocaleTimeString()
  })

  const [activityData] = useState([
    { time: '00:00', threats: 4, searches: 12 },
    { time: '04:00', threats: 2, searches: 8 },
    { time: '08:00', threats: 8, searches: 15 },
    { time: '12:00', threats: 6, searches: 20 },
    { time: '16:00', threats: 5, searches: 18 },
    { time: '20:00', threats: 3, searches: 10 },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setMlStatus(prev => ({
        knn: Math.min(100, prev.knn + Math.random() * 2 - 1),
        svm: Math.min(100, prev.svm + Math.random() * 2 - 1),
        naiveBayes: Math.min(100, prev.naiveBayes + Math.random() * 2 - 1),
        lastCheck: new Date().toLocaleTimeString()
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const handleAddSite = () => {
    if (newSite && !blockedSites.find(site => site.url === newSite)) {
      setBlockedSites([...blockedSites, {
        url: newSite,
        method: 'ML Classification',
        date: new Date().toISOString().split('T')[0],
        risk: 'High'
      }])
      setNewSite("")
      setStats(prev => ({
        ...prev,
        sitesBlocked: prev.sitesBlocked + 1,
        lastUpdated: new Date().toLocaleTimeString()
      }))
    }
  }

  const handleRemoveSite = (url) => {
    setBlockedSites(blockedSites.filter(site => site.url !== url))
    setStats(prev => ({
      ...prev,
      sitesBlocked: prev.sitesBlocked - 1,
      lastUpdated: new Date().toLocaleTimeString()
    }))
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
                  <DropdownMenuItem className="text-red-500">
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
                  <span>Sites Blocked</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.sitesBlocked}</div>
                <Progress value={75} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-red-50 dark:bg-red-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Threats Detected</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.threatsDetected}</div>
                <Progress value={45} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-green-50 dark:bg-green-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Search className="h-4 w-4" />
                  <span>Safe Searches</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.safeSearches}</div>
                <Progress value={90} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="bg-purple-50 dark:bg-purple-900/20">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-4 w-4" />
                  <span>ML Model Health</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {((mlStatus.knn + mlStatus.svm + mlStatus.naiveBayes) / 3).toFixed(1)}%
                </div>
                <Progress value={92} className="mt-2" />
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
                  <LineChart data={activityData}>
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
                    <Line type="monotone" dataKey="threats" stroke="var(--destructive)" strokeWidth={2} />
                    <Line type="monotone" dataKey="searches" stroke="var(--primary)" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="sites" className="space-y-4">
            <TabsList className="grid w-[400px] grid-cols-2">
              <TabsTrigger value="sites">Blocked Sites</TabsTrigger>
              <TabsTrigger value="ml">ML Models</TabsTrigger>
            </TabsList>

            <TabsContent value="sites">
              <div className="grid gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Add Blocked Website</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex space-x-2">
                      <div className="flex-grow">
                        <Label htmlFor="newSite" className="sr-only">New site to block</Label>
                        <Input
                          id="newSite"
                          value={newSite}
                          onChange={(e) => setNewSite(e.target.value)}
                          placeholder="Enter website URL"
                          className="w-full"
                        />
                      </div>
                      <Button onClick={handleAddSite} className="flex items-center space-x-2">
                        <Lock className="h-4 w-4" />
                        <span>Block Site</span>
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Blocked Websites</CardTitle>
                    <CardDescription>
                      Currently blocked: {blockedSites.length} sites
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Website</TableHead>
                          <TableHead>Detection Method</TableHead>
                          <TableHead>Blocked Since</TableHead>
                          <TableHead>Risk Level</TableHead>
                          <TableHead className="text-right">Action</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {blockedSites.map((site) => (
                          <TableRow key={site.url}>
                            {/* Continuing from previous code */}
                            <TableCell>{site.method}</TableCell>
                            <TableCell>{site.date}</TableCell>
                            <TableCell>
                              <Badge variant="destructive">{site.risk}</Badge>
                            </TableCell>
                            <TableCell className="text-right">
                              <Button 
                                variant="destructive" 
                                size="sm"
                                onClick={() => handleRemoveSite(site.url)}
                              >
                                Remove
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="ml">
              <Card>
                <CardHeader>
                  <CardTitle>ML Model Performance</CardTitle>
                  <CardDescription>
                    Last check: {mlStatus.lastCheck}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label>K-Nearest Neighbor</Label>
                      <span className="text-sm font-medium">{mlStatus.knn.toFixed(1)}%</span>
                    </div>
                    <Progress value={mlStatus.knn} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label>Support Vector Machine</Label>
                      <span className="text-sm font-medium">{mlStatus.svm.toFixed(1)}%</span>
                    </div>
                    <Progress value={mlStatus.svm} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label>Naive Bayes Classifier</Label>
                      <span className="text-sm font-medium">{mlStatus.naiveBayes.toFixed(1)}%</span>
                    </div>
                    <Progress value={mlStatus.naiveBayes} className="h-2" />
                  </div>

                  <div className="mt-4 p-4 bg-muted rounded-lg">
                    <h4 className="font-semibold mb-2">Model Usage</h4>
                    <ul className="space-y-2 text-sm">
                      <li>• KNN: URL classification and pattern detection</li>
                      <li>• SVM: Content analysis and categorization</li>
                      <li>• Naive Bayes: Image and text classification</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <Alert>
            <AlertDescription className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>
                Active protection: KNN (URL classification), SVM (content analysis), 
                and Naive Bayes (image detection) working together to keep browsing safe
              </span>
            </AlertDescription>
          </Alert>
        </div>
      </main>
    </div>
  )
}

export default Dashboard