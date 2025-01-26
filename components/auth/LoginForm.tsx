"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import Link from 'next/link'

interface LoginFormProps {
  onSuccess: () => void // Callback for successful login
  onError: (error: string) => void // Callback for login errors
}

export function LoginForm({ onSuccess, onError }: LoginFormProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null);

    // Mock API call
    try {
      // Simulate a delay for the API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Mock successful login
      onSuccess();
    } catch (err) {
      setError('An error occurred during login');
      onError('An error occurred during login');
    }
  }

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Login</CardTitle>
        <CardDescription>Enter your credentials to access your account</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="text-red-500 mb-4">{error}</div>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="your@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <Button type="submit" className="w-full">Login</Button>
        </form>
      </CardContent>
      <CardFooter>
        <Link href="/auth?tab=forgot-password" className="text-sm text-blue-500 hover:underline">
          Forgot your password?
        </Link>
      </CardFooter>
    </Card>
  )
}