// Popup script for Safe Browsing extension

class PopupManager {
    constructor() {
        this.statusValue = document.getElementById('status-value');
        this.protectionToggle = document.getElementById('protection-toggle');
        this.sitesChecked = document.getElementById('sites-checked');
        this.threatsBlocked = document.getElementById('threats-blocked');
        this.riskLevel = document.getElementById('risk-level');
        this.dashboardBtn = document.getElementById('dashboard-btn');
        this.settingsBtn = document.getElementById('settings-btn');

        this.initializeListeners();
        this.loadStats();
        this.checkCurrentPage();
        
        // Listen for stats updates from background script
        chrome.runtime.onMessage.addListener((message) => {
            if (message.type === 'statsUpdate') {
                this.updateStats(message.stats);
            }
        });

        // Initial settings check
        chrome.storage.sync.get(['blockingEnabled'], (result) => {
            const enabled = result.blockingEnabled ?? true;
            this.protectionToggle.checked = enabled;
            this.updateStatus(enabled ? 'Protected' : 'Protection Disabled', enabled);
        });
    }

    initializeListeners() {
        // Protection toggle
        this.protectionToggle.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            chrome.storage.sync.set({ blockingEnabled: enabled }, () => {
                this.updateStatus(enabled ? 'Checking...' : 'Protection Disabled', false);
                if (enabled) this.checkCurrentPage();
            });
        });

        // Dashboard button
        this.dashboardBtn.addEventListener('click', () => {
            chrome.tabs.create({ url: 'http://localhost:3000/dashboard' });
        });

        // Settings button
        this.settingsBtn.addEventListener('click', () => {
            chrome.tabs.create({ url: 'http://localhost:3000/dashboard/settings' });
        });
    }

    async loadStats() {
        try {
            const stats = await this.fetchStats();
            this.updateStats(stats);
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    async fetchStats() {
        try {
            const response = await fetch('http://localhost:8000/api/stats');
            if (!response.ok) throw new Error('Failed to fetch stats');
            return await response.json();
        } catch (error) {
            console.error('Error fetching stats:', error);
            return {
                total_sites: 0,
                blocked_sites: 0,
                risk_level: 'Unknown'
            };
        }
    }

    updateStats(stats) {
        this.sitesChecked.textContent = stats.total_sites || 0;
        this.threatsBlocked.textContent = stats.blocked_sites || 0;
        this.riskLevel.textContent = stats.risk_level || 'Unknown';
    }

    async checkCurrentPage() {
        try {
            // Get current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab?.url) return;

            // Prepare form data
            const formData = new FormData();
            formData.append('url', tab.url);

            // Check URL
            const response = await fetch('http://localhost:8000/api/check-url', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Failed to check URL');
            
            const result = await response.json();
            
            // Update status
            this.updateStatus(
                result.blocked ? 'Unsafe - Blocked' : 'Safe',
                !result.blocked
            );

            // Add details if available
            if (result.category || result.risk_level) {
                const details = document.createElement('div');
                details.style.fontSize = '12px';
                details.style.marginTop = '4px';
                details.innerHTML = `
                    ${result.category ? `Category: ${result.category}<br>` : ''}
                    ${result.risk_level ? `Risk Level: ${result.risk_level}<br>` : ''}
                    ${result.probability ? `Confidence: ${(result.probability * 100).toFixed(1)}%` : ''}
                `;
                this.statusValue.appendChild(details);
            }

        } catch (error) {
            console.error('Error checking current page:', error);
            this.updateStatus('Error checking page', false);
        }
    }

    updateStatus(message, isSafe = true) {
        this.statusValue.textContent = message;
        this.statusValue.className = `status-value ${isSafe ? '' : 'unsafe'}`;
    }
}

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
    new PopupManager();
});
