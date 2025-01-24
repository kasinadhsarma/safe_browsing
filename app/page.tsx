"use client"

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import Link from 'next/link'
import { Navigation } from '@/components/navigation'
import { ChevronDown, Shield, Brain, Eye, ArrowRight } from 'lucide-react'

const sections = [
  {
    id: 'hero',
    title: 'Protect Your Family Online',
    subtitle: 'Advanced AI-powered content filtering for a safer internet experience'
  },
  {
    id: 'technology',
    title: 'Our Technology',
    content: [
      {
        title: 'K-Nearest Neighbour',
        description: 'Classifies content based on similarity patterns',
        icon: Brain
      },
      {
        title: 'Support Vector Machine',
        description: 'Creates optimal boundaries between safe and unsafe content',
        icon: Shield
      },
      {
        title: 'Deep Learning',
        description: 'Advanced image recognition and filtering',
        icon: Eye
      }
    ]
  },
  {
    id: 'plans',
    title: 'Choose Your Plan',
    plans: [
      {
        name: 'Free',
        price: '$0',
        features: ['Basic content filtering', 'Time limits', 'Email reports']
      },
      {
        name: 'Premium',
        price: '$9.99',
        features: ['AI-powered analysis', 'Real-time protection', '24/7 support']
      }
    ]
  }
]

export default function Home() {
  const [expandedSection, setExpandedSection] = useState<string | null>('hero')
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  const toggleSection = (sectionId: string) => {
    setExpandedSection(expandedSection === sectionId ? null : sectionId)
  }

  if (!isMounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      {/* Main content */}
      <main className="container mx-auto px-4 pt-24 pb-16 space-y-8">
        {/* Hero Section */}
        <section className="min-h-[80vh] flex flex-col justify-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center space-y-6"
          >
            <h1 className="text-5xl font-extrabold tracking-tight lg:text-6xl">
              Protect Your Family with
              <span className="text-primary"> SafeNet</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Advanced AI-powered content filtering for a safer online experience
            </p>
            <div className="flex justify-center gap-4">
              <Link href="/auth?tab=signup" passHref>
                <Button size="lg">
                  Get Started <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/technology" passHref>
                <Button size="lg" variant="outline">
                  Learn More <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </motion.div>
        </section>

        {/* Technology Section */}
        <AnimatePresence>
          {expandedSection === 'technology' && (
            <motion.section
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <Card className="w-full">
                <CardContent className="p-6 space-y-6">
                  <div className="text-center space-y-4">
                    <h2 className="text-3xl font-bold">Our Technology</h2>
                    <p className="text-muted-foreground">
                      Powered by advanced machine learning algorithms
                    </p>
                  </div>
                  <div className="grid md:grid-cols-3 gap-6">
                    {sections[1].content?.map((item) => (
                      <div
                        key={item.title}
                        className="flex flex-col items-center text-center space-y-2 p-4"
                      >
                        <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                          <item.icon className="h-6 w-6 text-primary" />
                        </div>
                        <h3 className="font-semibold">{item.title}</h3>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Plans Section */}
        <AnimatePresence>
          {expandedSection === 'plans' && (
            <motion.section
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <Card className="w-full">
                <CardContent className="p-6 space-y-6">
                  <div className="text-center space-y-4">
                    <h2 className="text-3xl font-bold">Choose Your Plan</h2>
                    <p className="text-muted-foreground">
                      Select the perfect plan for your family&apos;s needs
                    </p>
                  </div>
                  <div className="grid md:grid-cols-2 gap-6">
                    {sections[2].plans?.map((plan) => (
                      <div
                        key={plan.name}
                        className="flex flex-col items-center text-center space-y-4 p-6 rounded-lg border"
                      >
                        <h3 className="text-xl font-semibold">{plan.name}</h3>
                        <p className="text-3xl font-bold">{plan.price}</p>
                        <ul className="space-y-2">
                          {plan.features.map((feature) => (
                            <li key={feature} className="text-sm text-muted-foreground">
                              {feature}
                            </li>
                          ))}
                        </ul>
                        {plan.name === 'Free' ? (
                          <Link href="/auth?tab=signup" passHref>
                            <Button className="w-full">
                              Get Started <ArrowRight className="ml-2 h-4 w-4" />
                            </Button>
                          </Link>
                        ) : (
                          <Button className="w-full" disabled>
                            Coming Soon <span className="ml-2 text-xs opacity-70">(Premium)</span>
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Section navigation */}
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 flex gap-2 p-2 rounded-full bg-background/80 backdrop-blur-sm border shadow-lg">
          {sections.map((section) => (
            <Button
              key={section.id}
              variant={expandedSection === section.id ? "default" : "ghost"}
              size="icon"
              onClick={() => toggleSection(section.id)}
            >
              <ChevronDown
                className={`h-4 w-4 transition-transform ${
                  expandedSection === section.id ? "rotate-180" : ""
                }`}
              />
            </Button>
          ))}
        </div>
      </main>
    </div>
  )
}