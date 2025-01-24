"use client";

import { useState } from 'react';
import { usePathname } from 'next/navigation';
import { Home, Activity, AlertTriangle, Settings, Search, ChevronLeft, LogOut, UserCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { SafeNetLogo } from '@/components/safenet-logo';
import { cn } from "@/lib/utils";
import { ModeToggle } from '@/components/mode-toggle';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const sidebarItems = [
  { icon: Home, label: 'Home', id: 'overview', path: '/dashboard' },
  { icon: Activity, label: 'Activity', id: 'activity', path: '/dashboard/activity' },
  { icon: AlertTriangle, label: 'Alerts', id: 'alerts', path: '/dashboard/alerts' },
  { icon: Settings, label: 'Settings', id: 'settings', path: '/dashboard/settings' },
  { icon: Search, label: 'Search', id: 'search', path: '/dashboard/search' },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const pathname = usePathname();

  const handleLogout = () => {
    // You would typically make an API call to logout and clear session
    window.location.href = '/auth';
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
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

        <nav className="flex flex-col gap-2 p-4 pb-20">
          {sidebarItems.map((item) => (
            <Button
              key={item.id}
              variant={pathname === item.path ? "default" : "ghost"}
              className={cn(
                "flex items-center transition-all duration-200 group relative",
                isSidebarOpen ? "justify-start" : "justify-center"
              )}
              onClick={() => setIsSidebarOpen(true)}
              asChild
            >
              <a href={item.path}>
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
              </a>
            </Button>
          ))}
        </nav>

        {/* Profile and Theme Section */}
        <div className={cn(
          "absolute bottom-0 left-0 right-0 p-4 border-t bg-card",
          "flex items-center gap-2",
          !isSidebarOpen && "justify-center"
        )}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="relative h-8 w-8 rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarImage src="/placeholder.svg" alt="@user" />
                  <AvatarFallback>U</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <UserCircle className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-red-500" onClick={handleLogout}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          
          {isSidebarOpen && <div className="h-8 w-px bg-border" />}
          <ModeToggle />
        </div>
      </aside>

      {/* Main Content */}
      <main className={cn(
        "transition-all duration-300",
        isSidebarOpen ? "ml-64" : "ml-20"
      )}>
        {children}
      </main>
    </div>
  );
}
