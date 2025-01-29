const API_ENDPOINT = 'http://localhost:8000/api';
const CHECK_URL_ENDPOINT = `${API_ENDPOINT}/check-url`;
const LOG_ACTIVITY_ENDPOINT = `${API_ENDPOINT}/activity`;

// Initialize settings
chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.sync.set({
        blockingEnabled: true,
        alertsEnabled: true,
        statsCollectionEnabled: true
    });
});

// Caches
const urlCache = new Map();
const statsCache = {
    totalSites: 0,
    blockedSites: 0,
    lastUpdate: 0
};

// Clear caches periodically (every hour)
setInterval(() => {
    urlCache.clear();
    statsCache.lastUpdate = 0;
}, 3600000);

// Function to update stats
async function updateStats() {
    try {
        const response = await fetch(`${API_ENDPOINT}/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        const stats = await response.json();
        
        statsCache.totalSites = stats.total_sites;
        statsCache.blockedSites = stats.blocked_sites;
        statsCache.lastUpdate = Date.now();
        
        // Notify popup of updated stats
        chrome.runtime.sendMessage({
            type: 'statsUpdate',
            stats: {
                total_sites: statsCache.totalSites,
                blocked_sites: statsCache.blockedSites
            }
        });
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

async function checkUrl(url) {
    try {
        // Check cache first
        if (urlCache.has(url)) {
            return urlCache.get(url);
        }

        // Get user settings
        const settings = await new Promise(resolve => {
            chrome.storage.sync.get(['ageGroup'], (result) => {
                resolve({
                    ageGroup: result.ageGroup || 'kid' // Default to most restrictive
                });
            });
        });

        // Prepare form data
        const formData = new FormData();
        formData.append('url', url);
        formData.append('age_group', settings.ageGroup);

        // Make API request
        const response = await fetch(CHECK_URL_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Cache the result
        urlCache.set(url, result);

        // Log activity
        await logActivity(url, result);

        return result;
    } catch (error) {
        console.error('Error checking URL:', error);
        return { blocked: false, error: true };
    }
}

// Update stats every 5 seconds
setInterval(updateStats, 5000);

async function logActivity(url, result) {
    // Update local stats immediately
    statsCache.totalSites++;
    if (result.blocked) statsCache.blockedSites++;
    try {
        const activity = {
            url,
            action: result.blocked ? 'blocked' : 'allowed',
            category: result.category || 'Unknown',
            risk_level: result.risk_level || 'Unknown',
            timestamp: new Date().toISOString(),
            age_group: result.age_group || 'kid',
            block_reason: result.block_reason || '',
            ml_scores: {
                knn: result.model_predictions?.knn || {},
                svm: result.model_predictions?.svm || {},
                nb: result.model_predictions?.nb || {}
            }
        };

        const formData = new FormData();
        formData.append('url', activity.url);
        formData.append('action', activity.action);
        formData.append('category', activity.category);
        formData.append('risk_level', activity.risk_level);
        formData.append('ml_scores', JSON.stringify(activity.ml_scores));
        formData.append('age_group', activity.age_group);
        formData.append('block_reason', activity.block_reason);

        await fetch(LOG_ACTIVITY_ENDPOINT, {
            method: 'POST',
            body: formData
        });
    } catch (error) {
        console.error('Error logging activity:', error);
    }
}

// Handle navigation events
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    // Skip non-main-frame navigations and extension pages
    if (details.frameId !== 0 || details.url.startsWith('chrome-extension://')) {
        return;
    }

    try {
        const settings = await chrome.storage.sync.get(['blockingEnabled']);
        if (!settings.blockingEnabled) return;

        const result = await checkUrl(details.url);

        if (result.blocked) {
            // Redirect to blocked page
            chrome.tabs.update(details.tabId, {
                url: chrome.runtime.getURL('src/blocked.html') +
                     `?url=${encodeURIComponent(details.url)}` +
                     `&category=${encodeURIComponent(result.category || 'Unknown')}` +
                     `&risk_level=${encodeURIComponent(result.risk_level || 'High')}` +
                     `&probability=${encodeURIComponent(result.probability || '0')}`
            });
        }
    } catch (error) {
        console.error('Error in navigation handler:', error);
    }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'checkUrl') {
        checkUrl(request.url)
            .then(result => sendResponse(result))
            .catch(error => sendResponse({ error: true }));
        return true; // Will respond asynchronously
    }
});

// Update badge when blocking is toggled
chrome.storage.onChanged.addListener((changes) => {
    if (changes.blockingEnabled) {
        const enabled = changes.blockingEnabled.newValue;
        chrome.action.setBadgeText({ text: enabled ? 'ON' : 'OFF' });
        chrome.action.setBadgeBackgroundColor({ 
            color: enabled ? '#1a73e8' : '#666666' 
        });
        
        // Force an immediate stats update when protection is toggled
        if (enabled) {
            updateStats();
        }
    }
});

// Initialize badge
chrome.storage.sync.get(['blockingEnabled'], (result) => {
    const enabled = result.blockingEnabled ?? true;
    chrome.action.setBadgeText({ text: enabled ? 'ON' : 'OFF' });
    chrome.action.setBadgeBackgroundColor({ 
        color: enabled ? '#1a73e8' : '#666666' 
    });
});
