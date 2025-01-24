const recentUrlChecks = new Map();
const API_BASE_URL = 'http://localhost:8000/api';

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

async function logErrorWithRetry(error, retryCount = 3) {
  for (let i = 0; i < retryCount; i++) {
    try {
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

function analyzeDomain(url) {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();
    const path = urlObj.pathname.toLowerCase();
    const query = urlObj.search.toLowerCase();

    if (KNOWN_SAFE_DOMAINS[hostname]) {
      return { category: 'Safe Site', riskLevel: 'Low' };
    }

    if (hostname.match(/(facebook|twitter|instagram|linkedin|tiktok|reddit|snapchat)\.(com|org)/i)) {
      return { category: 'Social Media', riskLevel: 'Low' };
    }

    if (hostname.match(/(youtube|vimeo|twitch|netflix|disney|hulu)\.(com|tv)/i)) {
      return { category: 'Video', riskLevel: 'Low' };
    }

    if (hostname.match(/(coursera|udemy|edx|khanacademy|mit|edu)\.(org|com|edu)/i)) {
      return { category: 'Educational', riskLevel: 'Low' };
    }

    if (hostname.match(/(minecraft|roblox|fortnite|gaming|steam|epicgames)\.(com|net)/i)) {
      return { category: 'Gaming', riskLevel: 'Low' };
    }

    if (hostname.match(/(news|cnn|bbc|nytimes|reuters)\.(com|org|net)/i)) {
      return { category: 'News', riskLevel: 'Low' };
    }

    if (hostname.match(/(chat\.openai|bard\.google|bing|claude)\.(com|ai)/i)) {
      return { category: 'AI Chat', riskLevel: 'Low' };
    }

    if (hostname.match(/(google|bing|yahoo|duckduckgo)\.(com|org)/i)) {
      return { category: 'Search', riskLevel: 'Low' };
    }

    const riskPatterns = {
      gambling: {
        domains: /(gambling|casino|bet|poker|lottery|blackjack|roulette|slot|bingo)\.(com|net|org|xyz|app|bet|game)/i,
        paths: /(gambling|casino|betting|wager|poker)/i
      },
      adult: {
        domains: /(adult|xxx|porn|sex|nude|escort|dating|cam|strip|playboy|ass|boob|dick|pussy|milf|teen|mature|hentai|nsfw)\.(com|net|org|xyz|app|xxx|sex|adult|porn)/i,
        paths: /(porn|xxx|adult|nsfw|sex|nude|naked|escort|pussy
