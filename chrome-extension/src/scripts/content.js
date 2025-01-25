// Enhanced SafeBrowsing content script

// Configuration
const API_ENDPOINT = 'http://localhost:8000/api';
const CHECK_URL_ENDPOINT = `${API_ENDPOINT}/check-url`;
const LOG_ACTIVITY_ENDPOINT = `${API_ENDPOINT}/activity`;
const LOG_ERROR_ENDPOINT = `${API_ENDPOINT}/log-error`;

// Cache for URL check results
const urlCache = new Map();

class SafeBrowsingChecker {
    constructor() {
        this.blockingEnabled = true;
        this.setupListeners();
    }

    setupListeners() {
        // Listen for settings changes
        chrome.storage.onChanged.addListener((changes) => {
            if (changes.blockingEnabled) {
                this.blockingEnabled = changes.blockingEnabled.newValue;
            }
        });

        // Get initial settings
        chrome.storage.sync.get(['blockingEnabled'], (result) => {
            this.blockingEnabled = result.blockingEnabled ?? true;
        });
    }

    async checkUrl(url) {
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
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Cache the result
            urlCache.set(url, result);

            // Log activity
            await this.logActivity(url, result);

            return result;
        } catch (error) {
            console.error('Error checking URL:', error);
            await this.logError(error, url);
            return null;
        }
    }

    async logActivity(url, result) {
        try {
            const activity = {
                url,
                action: result.blocked ? 'blocked' : 'allowed',
                category: result.category || 'Unknown',
                risk_level: result.risk_level || 'Unknown',
                timestamp: new Date().toISOString(),
            };

            // Create a new FormData instance
            const formData = new FormData();
            formData.append('url', activity.url);
            formData.append('action', activity.action);
            formData.append('category', activity.category || 'Unknown');
            formData.append('risk_level', activity.risk_level || 'Unknown');
            formData.append('ml_scores', JSON.stringify(result.predictions || {}));

            await fetch(LOG_ACTIVITY_ENDPOINT, {
                method: 'POST',
                body: formData
            });
        } catch (error) {
            console.error('Error logging activity:', error);
        }
    }

    async logError(error, url) {
        try {
            const errorLog = {
                error: error.message,
                stack: error.stack,
                url,
                timestamp: new Date().toISOString(),
            };

            await fetch(LOG_ERROR_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(errorLog),
            });
        } catch (error) {
            console.error('Error logging error:', error);
        }
    }

    createBlockingOverlay(result) {
        // Create a shadow DOM to isolate the overlay
        const overlay = document.createElement('div');
        overlay.attachShadow({ mode: 'open' });

        // Add content to the shadow DOM
        overlay.shadowRoot.innerHTML = `
            <style>
                #safe-browsing-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.85);
                    z-index: 2147483647;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                }
                .content {
                    background-color: white;
                    padding: 30px;
                    border-radius: 12px;
                    text-align: center;
                    max-width: 500px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                h2 {
                    color: #d32f2f;
                    margin: 0 0 20px;
                    font-size: 24px;
                }
                .risk-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-weight: bold;
                    margin-bottom: 16px;
                }
                .risk-high {
                    background-color: #ffebee;
                    color: #d32f2f;
                }
                .risk-medium {
                    background-color: #fff3e0;
                    color: #ef6c00;
                }
                .info {
                    margin: 16px 0;
                    padding: 16px;
                    background-color: #f5f5f5;
                    border-radius: 8px;
                    text-align: left;
                }
                .reason {
                    color: #d32f2f;
                    font-weight: 500;
                    margin-bottom: 12px;
                }
                p {
                    margin: 8px 0;
                    line-height: 1.5;
                    color: #333;
                }
                button {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    background-color: #d32f2f;
                    color: white;
                    cursor: pointer;
                    font-size: 16px;
                    transition: background-color 0.2s;
                    margin-top: 16px;
                }
                button:hover {
                    background-color: #b71c1c;
                }
                .category {
                    font-weight: 500;
                    color: #666;
                }
            </style>
            <div id="safe-browsing-overlay">
                <div class="content">
                    <h2>Access Blocked</h2>
                    <div class="risk-badge risk-${result.risk_level?.toLowerCase()}">
                        Risk Level: ${result.risk_level || 'Unknown'}
                    </div>
                    <div class="info">
                        ${result.block_reason ? `<div class="reason">${result.block_reason}</div>` : ''}
                        <p class="category">Category: ${result.category || 'Unknown'}</p>
                        ${result.predictions ? `<p>ML Confidence: ${(result.probability * 100).toFixed(1)}%</p>` : ''}
                    </div>
                    <button id="continue-btn">Continue Anyway</button>
                </div>
            </div>
        `;

        // Add event listener for the "Continue Anyway" button
        overlay.shadowRoot.getElementById('continue-btn').addEventListener('click', () => {
            overlay.remove();
            this.logActivity(window.location.href, {
                ...result,
                action: 'override',
            });
        });

        return overlay;
    }
}

// Initialize the checker
const checker = new SafeBrowsingChecker();

// Debounced URL check
let debounceTimer;
const DEBOUNCE_TIME = 500; // 500ms

// Monitor navigation events with debounce
const observer = new MutationObserver(() => {
    if (!checker.blockingEnabled) return;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        try {
            const currentUrl = window.location.href;
            const result = await checker.checkUrl(currentUrl);
            if (result && result.blocked) {
                // Remove any existing overlay first
                const existingOverlay = document.querySelector('div[shadowroot]');
                if (existingOverlay) existingOverlay.remove();

                // Add the new overlay
                document.body.appendChild(checker.createBlockingOverlay(result));
            }
        } catch (error) {
            console.error('Error in URL check:', error);
        }
    }, DEBOUNCE_TIME);
});

// Start observing the document
observer.observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
});

// Check clicked links before navigation
document.addEventListener('click', async (event) => {
    if (!checker.blockingEnabled) return;

    const link = event.target.closest('a');
    if (link && link.href) {
        const result = await checker.checkUrl(link.href);
        if (result && result.blocked) {
            event.preventDefault();
            document.body.appendChild(checker.createBlockingOverlay(result));
        }
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    const overlay = document.querySelector('div[shadowroot]');
    if (overlay) overlay.remove();
});
