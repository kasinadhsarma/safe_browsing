// Store for tracking recent URLs and their check times
const recentUrlChecks = new Map();
const API_BASE_URL = 'http://localhost:8000/api';

// List of known safe domains that should never be blocked
const KNOWN_SAFE_DOMAINS = {
  'youtube.com': true,
  'www.youtube.com': true,
  'vimeo.com': true,
  'www.vimeo.com': true,
  'netflix.com': true,
  'www.netflix.com': true,
  'disney.com': true,
  'www.disney.com': true,
  'google.com': true,
  'www.google.com': true,
  'facebook.com': true,
  'www.facebook.com': true,
  'twitter.com': true,
  'www.twitter.com': true
};

// Function to log errors with retry mechanism
async function logErrorWithRetry(error, retryCount = 3) {
  for (let i = 0; i < retryCount; i++) {
    try {
      // Simulate sending the error to a server (replace with actual implementation)
      await sendErrorToServer(error);
      console.error(`Error logged successfully on attempt ${i + 1}`);
      return;
    } catch (e) {
      console.error(`Error attempt ${i + 1}:`, e);
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
  console.error('Failed to log error after retries:', error);
}

// Send error to backend server
async function sendErrorToServer(error) {
  try {
    await fetch(`${API_BASE_URL}/log-error`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      mode: 'cors',
      credentials: 'include',
      body: JSON.stringify({
        error: error.message,
        timestamp: new Date().toISOString(),
        stack: error.stack
      })
    });
  } catch (e) {
    console.error('Failed to send error to server:', e);
  }
}

// Function to normalize URL
function normalizeUrl(url) {
  try {
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }
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
    const hostname = urlObj.hostname.toLowerCase();
    const path = urlObj.pathname.toLowerCase();
    const query = urlObj.search.toLowerCase();

    // Check if this is a known safe domain first
    if (KNOWN_SAFE_DOMAINS[hostname]) {
      return { category: 'Safe Site', riskLevel: 'Low' };
    }

    // Social Media - Safe
    if (hostname.match(/(facebook|twitter|instagram|linkedin|tiktok|reddit|snapchat)\.(com|org)/i)) {
      return { category: 'Social Media', riskLevel: 'Low' };
    }

    // Video Platforms - Safe
    if (hostname.match(/(youtube|vimeo|twitch|netflix|disney|hulu)\.(com|tv)/i)) {
      return { category: 'Video', riskLevel: 'Low' };
    }

    // Educational - Safe
    if (hostname.match(/(coursera|udemy|edx|khanacademy|mit|edu)\.(org|com|edu)/i)) {
      return { category: 'Educational', riskLevel: 'Low' };
    }

    // Gaming - Safe
    if (hostname.match(/(minecraft|roblox|fortnite|gaming|steam|epicgames)\.(com|net)/i)) {
      return { category: 'Gaming', riskLevel: 'Low' };
    }

    // News & Media - Safe
    if (hostname.match(/(news|cnn|bbc|nytimes|reuters)\.(com|org|net)/i)) {
      return { category: 'News', riskLevel: 'Low' };
    }

    // AI & Chat - Safe
    if (hostname.match(/(chat\.openai|bard\.google|bing|claude)\.(com|ai)/i)) {
      return { category: 'AI Chat', riskLevel: 'Low' };
    }

    // Search Engines - Safe
    if (hostname.match(/(google|bing|yahoo|duckduckgo)\.(com|org)/i)) {
      return { category: 'Search', riskLevel: 'Low' };
    }

    // Check for inappropriate content
    const riskPatterns = {
      gambling: {
        domains: /(gambling|casino|bet|poker|lottery|blackjack|roulette|slot|bingo)\.(com|net|org|xyz|app|bet|game)/i,
        paths: /(gambling|casino|betting|wager|poker)/i
      },
      adult: {
        domains: /(adult|xxx|porn|sex|nude|escort|dating|cam|strip|playboy|ass|boob|dick|pussy|milf|teen|mature|hentai|nsfw)\.(com|net|org|xyz|app|xxx|sex|adult|porn)/i,
        paths: /(porn|xxx|adult|nsfw|sex|nude|naked|escort|pussy|cock|boob|dick|ass|milf|teen|fuck|cum)/i
      },
      violence: {
        domains: /(gore|violence|death|fight|torture|weapon|drug|cartel)\.(com|net|org|xyz|app)/i,
        paths: /(gore|violence|death|blood|murder|kill|torture|fight)/i
      },
      malware: {
        domains: /(crack|warez|keygen|hack|torrent|pirate|stolen)\.(com|net|org|xyz|app)/i,
        paths: /(crack|warez|keygen|hack|torrent|pirate)/i
      },
      phishing: {
        domains: /(phish|scam|fake|fraud|spam)\.(com|net|org|xyz|app)/i,
        paths: /(phishing|scam|hack|spam|fraud)/i
      }
    };

    // Check both domain and path patterns
    for (const [risk, patterns] of Object.entries(riskPatterns)) {
      if (
        hostname.match(patterns.domains) ||
        path.match(patterns.paths) ||
        query.match(patterns.paths)
      ) {
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

// Record activity with local storage as backup
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

  // Store locally first as backup
  chrome.storage.local.get(['browsing_activity'], (result) => {
    const activities = result.browsing_activity || [];
    activities.unshift(activity);
    if (activities.length > 100) activities.length = 100;
    chrome.storage.local.set({ browsing_activity: activities });

    // Update local stats
    chrome.storage.local.get(['dashboard_stats'], (statsResult) => {
      const stats = statsResult.dashboard_stats || {
        total_sites: 0,
        blocked_sites: 0,
        allowed_sites: 0,
        visited_sites: 0
      };

      stats.total_sites++;
      if (action === 'blocked') stats.blocked_sites++;
      if (action === 'allowed') stats.allowed_sites++;
      if (action === 'visited') stats.visited_sites++;

      chrome.storage.local.set({ dashboard_stats: stats });
    });
  });

  // Try to send to backend
  try {
    await fetch(`${API_BASE_URL}/activity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      mode: 'cors',
      credentials: 'include',
      body: JSON.stringify(activity)
    });
  } catch (error) {
    console.log('Failed to sync activity with backend (using local storage):', error.message);
  }
}

// Update dashboard stats every minute
chrome.alarms.create('statsUpdate', { periodInMinutes: 1 });

// Check URL against local rules
async function checkUrl(url, tabId) {
  try {
    const analysis = analyzeDomain(url);
    
    // Debug logging
    console.log('URL Check:', {
      url,
      hostname: new URL(url).hostname,
      category: analysis.category,
      riskLevel: analysis.riskLevel
    });

    await recordActivity(url, 'checking', analysis);

    // Only block inappropriate content (high risk)
    if (analysis.riskLevel === 'High') {
      const blockedUrl = chrome.runtime.getURL('src/blocked.html') +
        `?url=${encodeURIComponent(url)}` +
        `&category=${encodeURIComponent(analysis.category)}` +
        `&risk=${encodeURIComponent(analysis.riskLevel)}`;

      await chrome.tabs.update(tabId, { url: blockedUrl });
      await recordActivity(url, 'blocked', analysis);
      return { blocked: true, analysis };
    }

    // Allow all other URLs
    await recordActivity(url, 'allowed', analysis);
    return { blocked: false, analysis };
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

// Handle alarms and sync with backend
chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === 'statsUpdate') {
    try {
      // Get backend stats
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        mode: 'cors',
        credentials: 'include'
      });

      if (response.ok) {
        const backendStats = await response.json();
        chrome.storage.local.set({ dashboard_stats: backendStats });
      } else {
        // If backend fails, update local stats only
        chrome.storage.local.get(['dashboard_stats', 'browsing_activity'], (result) => {
          const stats = result.dashboard_stats || {
            total_sites: 0,
            blocked_sites: 0,
            allowed_sites: 0,
            visited_sites: 0
          };
          
          const activities = result.browsing_activity || [];
          const now = new Date();
          const last24h = activities.filter(a => 
            (now - new Date(a.timestamp)) < 24 * 60 * 60 * 1000
          );

          stats.daily_visits = last24h.length;
          chrome.storage.local.set({ dashboard_stats: stats });
        });
      }
    } catch (error) {
      console.log('Failed to sync with backend:', error.message);
    }
  }
});

// Log startup
chrome.runtime.onStartup.addListener(() => {
  console.log('Extension started');
});
