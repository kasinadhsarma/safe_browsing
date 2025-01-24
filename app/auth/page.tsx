"use client"

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SignupForm } from '@/components/auth/SignupForm'
import { LoginForm } from '@/components/auth/LoginForm'
import { ForgotPasswordForm } from '@/components/auth/ForgotPasswordForm'

export default function AuthPage() {
  const [activeTab, setActiveTab] = useState('login')
  const searchParams = useSearchParams()
  const router = useRouter()

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['signup', 'login', 'forgot-password'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  useEffect(() => {
    const fetchAdvancedMode = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/check`, {
          credentials: 'include',
          headers: {
            'X-Advanced-Mode': 'true'
          }
        })
        if (response.ok) {
          // const data = await response.json()
          // fetchAdvancedMode(data.advancedMode) // Removed unnecessary call
        }
      } catch (error) {
        console.error('Error fetching advanced mode:', error)
      }
    }

    fetchAdvancedMode()
  }, [])

  const handleTabChange = (value: string) => {
    setActiveTab(value)
    router.push(`/auth?tab=${value}`, { scroll: false })
  }

  return (
    <div className="flex justify-center items-center min-h-screen bg-background">
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full max-w-md">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="signup">Sign Up</TabsTrigger>
          <TabsTrigger value="login">Login</TabsTrigger>
          <TabsTrigger value="forgot-password">Forgot Password</TabsTrigger>
        </TabsList>
        <TabsContent value="signup">
          <SignupForm />
        </TabsContent>
        <TabsContent value="login">
          <LoginForm />
        </TabsContent>
        <TabsContent value="forgot-password">
          <ForgotPasswordForm />
        </TabsContent>
      </Tabs>
    </div>
  )
}
