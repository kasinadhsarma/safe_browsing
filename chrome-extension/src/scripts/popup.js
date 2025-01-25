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
        this.ageGroupSelect = document.getElementById('age-group-select');

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
        chrome.storage.sync.get(['blockingEnabled', 'ageGroup'], (result) => {
            const enabled = result.blockingEnabled ?? true;
            this.protectionToggle.checked = enabled;
            this.updateStatus(enabled ? 'Protected' : 'Protection Disabled', enabled);
            
            // Set age group
            if (this.ageGroupSelect) {
                this.ageGroupSelect.value = result.ageGroup || 'kid';
            }
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

        // Age group selector
        if (this.ageGroupSelect) {
            this.ageGroupSelect.addEventListener('change', (e) => {
                const ageGroup = e.target.value;
                chrome.storage.sync.set({ ageGroup }, () => {
                    console.log('Age group updated:', ageGroup);
                    // Recheck current page with new age group
                    if (this.protectionToggle.checked) {
                        this.checkCurrentPage();
                    }
                });
            });
        }

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
            // Get current age group
            const { ageGroup } = await chrome.storage.sync.get(['ageGroup']);
            
            const formData = new FormData();
            formData.append('url', tab.url);
            formData.append('age_group', ageGroup || 'kid');

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

            // Always show details with default values
            const details = document.createElement('div');
            details.style.fontSize = '12px';
            details.style.marginTop = '4px';
            details.innerHTML = `
                Category: ${result.category || 'Unknown'}<br>
                Risk Level: ${result.risk_level || 'Unknown'}<br>
                ${result.probability ? `Confidence: ${(result.probability * 100).toFixed(1)}%` : ''}<br>
                Age Group: ${result.age_group || 'Kid'}<br>
                ${result.block_reason ? `Block Reason: ${result.block_reason}` : ''}
            `;
            this.statusValue.appendChild(details);

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
