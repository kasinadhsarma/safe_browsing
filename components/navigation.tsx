"use client"

import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { ModeToggle } from '@/components/mode-toggle'

export function Navigation() {
  const pathname = usePathname()
  const searchParams = useSearchParams()

  if (pathname.startsWith('/dashboard')) {
    return null
  }

  return (
    <nav className="border-b">
      <div className="container mx-auto px-4 py-2 flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold">
          SafeNet
        </Link>
        <div className="flex items-center space-x-4">
          <Link href="/" passHref>
            <Button variant={pathname === '/' ? 'default' : 'ghost'}>Home</Button>
          </Link>
          <Link href="/technology" passHref>
            <Button variant={pathname === '/technology' ? 'default' : 'ghost'}>Technology</Button>
          </Link>
          <Link href="/auth?tab=login" passHref>
            <Button variant={pathname === '/auth' && searchParams.get('tab') === 'login' ? 'default' : 'ghost'}>Login</Button>
          </Link>
          <Link href="/auth?tab=signup" passHref>
            <Button variant={pathname === '/auth' && searchParams.get('tab') === 'signup' ? 'default' : 'ghost'}>Sign Up</Button>
          </Link>
          <ModeToggle />
        </div>
      </div>
    </nav>
  )
}
