'use client'

import { Card } from "@/components/ui/card"
import { useEffect, useState } from "react"

interface Activity {
  url: string
  timestamp: string
  action: 'blocked' | 'allowed' | 'visited' | 'checking'
  category?: string
}

const actionColors = {
  blocked: 'bg-red-100 text-red-700',
  allowed: 'bg-green-100 text-green-700',
  visited: 'bg-blue-100 text-blue-700',
  checking: 'bg-yellow-100 text-yellow-700'
} as const

export default function ActivityPage() {
  const [activities, setActivities] = useState<Activity[]>([])
  const [filter, setFilter] = useState<Activity['action'] | 'all'>('all')
  
  const filteredActivities = activities.filter(activity => 
    filter === 'all' ? true : activity.action === filter
  )

  useEffect(() => {
    // Function to fetch activities from chrome.storage
    const fetchActivities = async () => {
      if (typeof chrome !== 'undefined' && chrome.storage) {
        chrome.storage.local.get(['browsing_activity'], (result) => {
          if (result.browsing_activity) {
            setActivities(result.browsing_activity)
          }
        })
      }
    }

    fetchActivities()

    // Set up listener for new activities
    if (typeof chrome !== 'undefined' && chrome.storage) {
      chrome.storage.onChanged.addListener((changes) => {
        if (changes.browsing_activity) {
          setActivities(changes.browsing_activity.newValue)
        }
      })
    }
  }, [])

  return (
    <div className="container mx-auto py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Browsing Activity</h1>
        <div className="flex items-center space-x-2">
          <label htmlFor="activity-filter" className="text-sm text-gray-600">
            Show:
          </label>
          <select 
            id="activity-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value as Activity['action'] | 'all')}
            className="px-3 py-2 border rounded-md bg-white"
            aria-label="Filter activities"
          >
            <option value="all">All Activities</option>
            <option value="visited">Visited Sites</option>
            <option value="blocked">Blocked Sites</option>
            <option value="allowed">Allowed Sites</option>
            <option value="checking">Being Checked</option>
          </select>
        </div>
      </div>
      <div className="grid gap-4">
        {filteredActivities.length === 0 ? (
          <Card className="p-4">
            <p className="text-center text-gray-500">No activity recorded yet</p>  
          </Card>
        ) : (
          filteredActivities.map((activity, index) => (
            <Card key={index} className="p-4">
              <div className="flex items-start justify-between">
                <div>
                  <p className="font-medium truncate max-w-[500px]">{activity.url}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    {new Date(activity.timestamp).toLocaleString()}
                  </p>
                  {activity.category && (
                    <p className="text-sm text-gray-500">Category: {activity.category}</p>
                  )}
                </div>
                <span className={`px-2 py-1 rounded text-sm ${actionColors[activity.action]}`}>
                  {activity.action.charAt(0).toUpperCase() + activity.action.slice(1)}
                </span>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  )
}
