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

  const handleTabChange = (value: string) => {
    setActiveTab(value)
    router.push(`/auth?tab=${value}`, { scroll: false })
  }

  const handleLoginSuccess = () => {
    router.push('/dashboard')
  }

  const handleSignupSuccess = () => {
    router.push('/dashboard')
  }

  const handleForgotPasswordSuccess = () => {
    router.push('/dashboard')
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
          <SignupForm onSuccess={handleSignupSuccess} onError={(error) => console.error(error)} />
        </TabsContent>
        <TabsContent value="login">
          <LoginForm onSuccess={handleLoginSuccess} onError={(error) => console.error(error)} />
        </TabsContent>
        <TabsContent value="forgot-password">
          <ForgotPasswordForm onSuccess={handleForgotPasswordSuccess} />
        </TabsContent>
      </Tabs>
    </div>
  )
}