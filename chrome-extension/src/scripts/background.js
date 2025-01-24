// Store for tracking recent URLs and their check times
const recentUrlChecks = new Map();
const API_BASE_URL = 'http://localhost:8000/api';

// Function to log errors with retry mechanism
async function logErrorWithRetry(error, retryCount = 3) {
  for (let i = 0; i < retryCount; i++) {
    try {
      console.error(`Error attempt ${i + 1}:`, error);
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      return;
    } catch (e) {
      if (i === retryCount - 1) console.error('Failed to log error after retries:', e);
    }
  }
}

// Function to normalize URL
function normalizeUrl(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.origin + urlObj.pathname;
  } catch (e) {
    console.error('Error normalizing URL:', e);
    return url;
  }
}

// Function to get domain category based on content and URL patterns
function analyzeDomain(url) {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname;
    const path = urlObj.pathname;

    // Social Media
    if (hostname.match(/(facebook|twitter|instagram|linkedin|tiktok|reddit|snapchat)\.(com|org)/i)) {
      return { category: 'Social Media', riskLevel: 'Medium' };
    }

    // Video Platforms
    if (hostname.match(/(youtube|vimeo|twitch|netflix|disney|hulu)\.(com|tv)/i)) {
      return { category: 'Video', riskLevel: 'Medium' };
    }

    // Educational
    if (hostname.match(/(coursera|udemy|edx|khanacademy|mit|edu)\.(org|com|edu)/i)) {
      return { category: 'Educational', riskLevel: 'Low' };
    }

    // Gaming
    if (hostname.match(/(minecraft|roblox|fortnite|gaming|steam|epicgames)\.(com|net)/i)) {
      return { category: 'Gaming', riskLevel: 'Medium' };
    }

    // News & Media
    if (hostname.match(/(news|cnn|bbc|nytimes|reuters)\.(com|org|net)/i)) {
      return { category: 'News', riskLevel: 'Low' };
    }

    // AI & Chat
    if (hostname.match(/(chat\.openai|bard\.google|bing|claude)\.(com|ai)/i)) {
      return { category: 'AI Chat', riskLevel: 'Medium' };
    }

    // Search Engines
    if (hostname.match(/(google|bing|yahoo|duckduckgo)\.(com|org)/i)) {
      return { category: 'Search', riskLevel: 'Low' };
    }

    // Check for potential risks in URL
    const riskPatterns = {
      gambling: /(gambling|casino|bet|poker|lottery)\.(com|net)/i,
      adult: /(adult|xxx|porn|sex)\.(com|net)/i,
      violence: /(gore|violence|death|fight)\.(com|net)/i,
      malware: /(crack|warez|keygen|hack)\.(com|net)/i
    };

    for (const [risk, pattern] of Object.entries(riskPatterns)) {
      if (hostname.match(pattern)) {
        return { category: risk.charAt(0).toUpperCase() + risk.slice(1), riskLevel: 'High' };
      }
    }

    // Default categorization based on TLD
    if (hostname.endsWith('.edu')) {
      return { category: 'Educational', riskLevel: 'Low' };
    }
    if (hostname.endsWith('.gov')) {
      return { category: 'Government', riskLevel: 'Low' };
    }
    if (hostname.endsWith('.org')) {
      return { category: 'Organization', riskLevel: 'Low' };
    }

    return { category: 'General', riskLevel: 'Medium' };
  } catch (e) {
    console.error('Error analyzing domain:', e);
    return { category: 'Unknown', riskLevel: 'Medium' };
  }
}

// Record activity to backend and local storage
async function recordActivity(url, action, analysis = null) {
  if (url.startsWith('chrome-extension://') || url === 'about:blank') {
    return;
  }

  const domain = analysis || analyzeDomain(url);
  const activity = {
    url: normalizeUrl(url),
    timestamp: new Date().toISOString(),
    action,
    category: domain.category,
    risk_level: domain.riskLevel
  };

  try {
    // Send to backend
    const response = await fetch(`${API_BASE_URL}/activity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(activity)
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    // Store locally
    chrome.storage.local.get(['browsing_activity'], (result) => {
      const activities = result.browsing_activity || [];
      activities.unshift(activity);
      if (activities.length > 100) activities.length = 100;
      chrome.storage.local.set({ browsing_activity: activities });
    });
  } catch (error) {
    await logErrorWithRetry(error);
    // Store locally on error
    chrome.storage.local.get(['browsing_activity'], (result) => {
      const activities = result.browsing_activity || [];
      activities.unshift(activity);
      if (activities.length > 100) activities.length = 100;
      chrome.storage.local.set({ browsing_activity: activities });
    });
  }
}

// Check URL against backend and local rules
async function checkUrl(url, tabId) {
  try {
    const analysis = analyzeDomain(url);
    await recordActivity(url, 'checking', analysis);

    // Check high-risk URLs immediately
    if (analysis.riskLevel === 'High') {
      await recordActivity(url, 'blocked', analysis);
      return { blocked: true, analysis };
    }

    // Check with backend
    const formData = new FormData();
    formData.append('url', url);

    const response = await fetch(`${API_BASE_URL}/check-url`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();

    if (data.blocked) {
      await recordActivity(url, 'blocked', {
        category: data.category || analysis.category,
        riskLevel: data.risk_level || analysis.riskLevel
      });

      // Show blocked page
      const blockedUrl = chrome.runtime.getURL('src/blocked.html') +
        `?url=${encodeURIComponent(url)}` +
        `&category=${encodeURIComponent(data.category || analysis.category)}` +
        `&risk=${encodeURIComponent(data.risk_level || analysis.riskLevel)}`;

      await chrome.tabs.update(tabId, { url: blockedUrl });
      return { blocked: true, analysis: data };
    }

    await recordActivity(url, 'allowed', {
      category: data.category || analysis.category,
      riskLevel: data.risk_level || analysis.riskLevel
    });
    return { blocked: false, analysis: data };
  } catch (error) {
    await logErrorWithRetry(error);
    await recordActivity(url, 'error', { category: 'Error', riskLevel: 'Unknown' });
    return { blocked: false, analysis: analyzeDomain(url) };
  }
}

// Monitor tab updates
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.url) {
    console.log('Checking URL:', changeInfo.url);
    await checkUrl(changeInfo.url, tabId);
  }
});

// Monitor web navigation
chrome.webNavigation.onCommitted.addListener(async (details) => {
  if (details.frameId === 0 && details.url) {  // Main frame only
    await recordActivity(details.url, 'visited');
  }
});

// Monitor history
chrome.history.onVisited.addListener(async (historyItem) => {
  await recordActivity(historyItem.url, 'history');
});

// Sync with backend periodically
async function syncWithBackend() {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();
    chrome.storage.local.set({ dashboard_stats: data });
  } catch (error) {
    await logErrorWithRetry(error);
  }
}

// Set up periodic sync
chrome.alarms.create('syncAlarm', { periodInMinutes: 0.5 });  // Every 30 seconds
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'syncAlarm') {
    syncWithBackend();
  }
});

// Initial sync
syncWithBackend();

// Log startup
console.log('Safe Browsing extension initialized');
