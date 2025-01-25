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

            // Prepare form data
            const formData = new FormData();
            formData.append('url', url);

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
                    background-color: rgba(0, 0, 0, 0.7);
                    z-index: 2147483647; /* Maximum z-index */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .content {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                h2 {
                    color: #d32f2f;
                    margin: 0 0 16px;
                }
                p {
                    margin: 0 0 16px;
                }
                button {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                    background-color: #d32f2f;
                    color: white;
                    cursor: pointer;
                }
            </style>
            <div id="safe-browsing-overlay">
                <div class="content">
                    <h2>Access Blocked</h2>
                    <p>
                        This website has been blocked for your safety.<br>
                        Category: ${result.category}<br>
                        Risk Level: ${result.risk_level}<br>
                        ${result.suspicious_features ? `Suspicious Features: ${Object.keys(result.suspicious_features).join(', ')}` : ''}
                    </p>
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
