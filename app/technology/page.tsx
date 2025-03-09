"use client"

import { useState, useEffect, Suspense } from 'react'
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Brain, Shield, Eye, Lock, Zap, ArrowRight, ChevronRight, ChevronUp } from 'lucide-react'
import { Navigation } from '@/components/navigation'
import Link from 'next/link'

const techData = [
  {
    id: 'knn',
    name: 'K-Nearest Neighbour',
    description: 'Classifies content based on similarity to known examples providing flexible and accurate content filtering.',
    icon: Brain,
    color: 'from-blue-500 to-blue-700'
  },
  {
    id: 'svm',
    name: 'Support Vector Machine',
    description: 'Creates optimal boundaries between safe and unsafe content ensuring robust classification.',
    icon: Shield,
    color: 'from-green-500 to-green-700'
  },
  {
    id: 'nbc',
    name: 'Naive Bayes Classifier',
    description: 'Utilizes probabilistic approaches to quickly categorize content based on its features.',
    icon: Zap,
    color: 'from-yellow-500 to-yellow-700'
  },
  {
    id: 'dl',
    name: 'Deep Learning Image Analysis',
    description: 'Employs neural networks to analyze and filter images protecting against inappropriate visual content.',
    icon: Eye,
    color: 'from-purple-500 to-purple-700'
  }
]

function TechnologyContent() {
  const [activeTab, setActiveTab] = useState('knn')
  const [showScrollTop, setShowScrollTop] = useState(false)

  const { scrollYProgress } = useScroll()
  const opacity = useTransform(scrollYProgress, [0, 0.05], [1, 0])

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    hover: { scale: 1.05, boxShadow: "0 10px 30px -10px rgba(0,0,0,0.2)" }
  }

  const renderContent = (content: string | number | object) => {
    if (typeof content === 'object' && content !== null) {
      return JSON.stringify(content)
    }
    return content
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background/95 to-background/90 transition-colors duration-300">
      <Navigation />
      <main className="container mx-auto px-4 py-12 space-y-24">
        {/* Hero Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-8 py-20 relative overflow-hidden"
        >
          <motion.div
            className="absolute inset-0 z-0"
            style={{
              backgroundImage: 'url("/placeholder.svg?height=1080&width=1920")',
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              opacity: 0.1
            }}
            animate={{
              scale: [1, 1.05, 1],
              rotate: [0, 1, 0]
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
          <motion.h1
            className="text-6xl font-extrabold tracking-tight lg:text-7xl bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60 relative z-10"
            style={{ opacity }}
          >
            Cutting-Edge Technology
          </motion.h1>
          <motion.p
            className="text-2xl text-muted-foreground max-w-3xl mx-auto relative z-10"
            style={{ opacity }}
          >
            Discover how our AI-powered protection keeps your family safe online
          </motion.p>
          <motion.div
            className="inline-block relative z-10"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link href="#tech-showcase">
              <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 text-xl px-8 py-6">
                Explore Our Tech <ChevronRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
          </motion.div>
        </motion.section>

        {/* Main Technology Card */}
        <Card className="w-full overflow-hidden border-2 shadow-xl" id="tech-showcase">
          <motion.div
            className="bg-gradient-to-r from-primary/20 to-primary/10 p-16 relative overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="absolute inset-0 bg-grid-white/10" />
            <CardHeader className="relative z-10">
              <CardTitle className="text-5xl font-bold mb-4">AI-Powered Protection</CardTitle>
              <CardDescription className="text-2xl">
                Multiple AI algorithms working in harmony for comprehensive safety
              </CardDescription>
            </CardHeader>
          </motion.div>
          <CardContent className="p-0">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="w-full justify-start rounded-none border-b bg-muted/50 p-0 h-auto flex-wrap">
                {techData.map((tech) => (
                  <TabsTrigger
                    key={tech.id}
                    value={tech.id}
                    className="data-[state=active]:bg-background rounded-none border-b-2 border-transparent px-8 py-6 data-[state=active]:border-primary transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <tech.icon className="h-6 w-6" />
                      <span className="font-semibold">{tech.name}</span>
                    </div>
                  </TabsTrigger>
                ))}
              </TabsList>
              <AnimatePresence mode="wait">
                {techData.map((tech) => (
                  <TabsContent key={tech.id} value={tech.id} className="p-16">
                    <motion.div
                      key={tech.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                      className="flex flex-col md:flex-row items-start gap-12"
                    >
                      <div className={`bg-gradient-to-br ${tech.color} p-8 rounded-3xl shadow-lg`}>
                        <tech.icon className="h-20 w-20 text-white" />
                      </div>
                      <div className="space-y-6 flex-1">
                        <h3 className="text-4xl font-bold">{tech.name}</h3>
                        <p className="text-muted-foreground text-xl leading-relaxed">
                          {renderContent(tech.description)}
                        </p>
                        <div className="flex flex-wrap gap-3 pt-4">
                          <Badge variant="secondary" className="px-4 py-2 text-lg">Advanced ML</Badge>
                          <Badge variant="secondary" className="px-4 py-2 text-lg">Real-time</Badge>
                          <Badge variant="secondary" className="px-4 py-2 text-lg">Adaptive</Badge>
                        </div>
                      </div>
                    </motion.div>
                  </TabsContent>
                ))}
              </AnimatePresence>
            </Tabs>
          </CardContent>
        </Card>

        {/* Feature Grid */}
        <section className="grid md:grid-cols-2 gap-16">
          {[
            {
              icon: Lock,
              title: "Real-time Protection",
              description: "Instant analysis and filtering for complete online safety",
              badges: ["Instant Analysis", "Continuous Learning", "Adaptive Filtering"],
              color: "from-cyan-500 to-blue-500"
            },
            {
              icon: Eye,
              title: "Visual Content Analysis",
              description: "Advanced image recognition for comprehensive protection",
              badges: ["Image Recognition", "Video Analysis", "Content Filtering"],
              color: "from-purple-500 to-pink-500"
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              whileHover="hover"
            >
              <Card className="h-full transition-all duration-300 shadow-md hover:shadow-2xl border-2">
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-6">
                    <div className={`bg-gradient-to-br ${feature.color} p-6 rounded-2xl shadow-lg`}>
                      <feature.icon className="h-10 w-10 text-white" />
                    </div>
                    <CardTitle className="text-3xl font-bold">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground text-xl mb-8 leading-relaxed">
                    {renderContent(feature.description)}
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {feature.badges.map((badge) => (
                      <Badge key={badge} variant="outline" className="px-4 py-2 text-lg">
                        {badge}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </section>

        {/* CTA Section */}
        <motion.section
          className="text-center space-y-10 py-20 bg-gradient-to-r from-primary/20 to-primary/10 rounded-3xl shadow-xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h2 className="text-5xl font-bold">Ready to Experience the Difference?</h2>
          <p className="text-2xl text-muted-foreground max-w-3xl mx-auto">
            Join thousands of families already protected by SafeNet&apos;s advanced technology
          </p>
          <motion.div
            className="inline-block"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link href="/auth?tab=signup">
              <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 text-xl px-10 py-8">
                Start Your Free Trial <ArrowRight className="ml-3 h-6 w-6" />
              </Button>
            </Link>
          </motion.div>
        </motion.section>
      </main>

      {/* Floating scroll to top button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            className="fixed bottom-8 right-8 bg-primary text-primary-foreground p-3 rounded-full shadow-lg z-50"
            onClick={scrollToTop}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <ChevronUp className="h-6 w-6" />
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function TechnologyPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <TechnologyContent />
    </Suspense>
  )
}
