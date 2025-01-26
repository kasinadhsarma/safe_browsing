"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import Link from 'next/link'
import axios from 'axios';

interface ForgotPasswordFormProps {
  onSuccess: () => void; // Callback for successful password reset
}

export function ForgotPasswordForm({ onSuccess }: ForgotPasswordFormProps) {
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null);
    setSuccess(null);
    try {
      const response = await axios.post('/api/forgot-password', { email });
      if (response.data.success) {
        setSuccess('Password reset email sent. Please check your inbox.');
        onSuccess(); // Call the onSuccess callback
      } else {
        setError(response.data.message || 'Failed to send password reset email.');
      }
    } catch (err) {
      setError('An error occurred during password reset.');
    }
  }

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Forgot Password</CardTitle>
        <CardDescription>Enter your email to reset your password</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="text-red-500 mb-4">{error}</div>
        )}
        {success && (
          <div className="text-green-500 mb-4">{success}</div>
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
          <Button type="submit" className="w-full">Reset Password</Button>
        </form>
      </CardContent>
      <CardFooter>
        <Link href="/auth?tab=login" className="text-sm text-blue-500 hover:underline">
          Remember your password? Log in
        </Link>
      </CardFooter>
    </Card>
  )
}