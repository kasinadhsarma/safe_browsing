function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatRelativeTime(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  const diffMinutes = Math.floor((now - date) / (1000 * 60));

  if (diffMinutes < 1) return 'just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

function getRiskLevelClass(level) {
  switch (level?.toLowerCase()) {
    case 'high': return 'risk-high';
    case 'medium': return 'risk-medium';
    case 'low': return 'risk-low';
    default: return 'risk-unknown';
  }
}

function getActionClass(action) {
  switch (action) {
    case 'blocked': return 'bg-red-100 text-red-700';
    case 'allowed': return 'bg-green-100 text-green-700';
    case 'visited': return 'bg-blue-100 text-blue-700';
    case 'history': return 'bg-purple-100 text-purple-700';
    case 'checking': return 'bg-yellow-100 text-yellow-700';
    default: return 'bg-gray-100 text-gray-700';
  }
}

function getTrendIcon(value, threshold) {
  if (value > threshold) return '↑';
  if (value < threshold) return '↓';
  return '→';
}

function formatUrl(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname + (urlObj.pathname !== '/' ? urlObj.pathname : '');
  } catch (e) {
    return url;
  }
}

function updateStats() {
  chrome.storage.local.get(['dashboard_stats', 'browsing_activity'], (result) => {
    const stats = result.dashboard_stats;
    const activities = result.browsing_activity || [];

    if (stats) {
      // Update summary cards
      document.getElementById('total').textContent = formatNumber(stats.total_sites);
      document.getElementById('blocked').textContent = formatNumber(stats.blocked_sites);
      document.getElementById('allowed').textContent = formatNumber(stats.allowed_sites);
      document.getElementById('visited').textContent = formatNumber(stats.visited_sites);

      // Calculate trends
      const blockRate = stats.blocked_sites / stats.total_sites;
      document.getElementById('block-trend').textContent = getTrendIcon(blockRate, 0.1);

      // Update risk distribution if exists
      if (stats.risk_levels) {
        updateRiskDistribution(stats.risk_levels);
      }
    }

    // Update recent activity list
    const recentList = document.getElementById('recent-activity');
    if (recentList && activities.length > 0) {
      recentList.innerHTML = activities
        .slice(0, 10)
        .map(activity => `
          <div class="activity-item border-b border-gray-100 last:border-0 p-2">
            <div class="flex items-center justify-between">
              <div class="flex flex-col flex-grow mr-2">
                <span class="font-medium text-sm truncate max-w-[200px]" title="${activity.url}">
                  ${formatUrl(activity.url)}
                </span>
                <div class="flex items-center text-xs text-gray-500 space-x-2">
                  <span>${formatRelativeTime(activity.timestamp)}</span>
                  <span class="category-tag">${activity.category || 'Unknown'}</span>
                </div>
              </div>
              <div class="flex flex-col items-end">
                <span class="text-xs px-2 py-1 rounded ${getActionClass(activity.action)}">
                  ${activity.action}
                </span>
                <span class="text-xs mt-1 ${getRiskLevelClass(activity.risk_level)}">
                  ${activity.risk_level || 'Unknown'} Risk
                </span>
              </div>
            </div>
          </div>
        `)
        .join('');
    }
  });
}

function updateRiskDistribution(riskLevels) {
  const container = document.getElementById('risk-distribution');
  if (!container) return;

  const total = Object.values(riskLevels).reduce((a, b) => a + b, 0);

  container.innerHTML = Object.entries(riskLevels)
    .map(([level, count]) => {
      const percentage = ((count / total) * 100).toFixed(1);
      return `
        <div class="flex items-center justify-between mb-1">
          <span class="text-xs ${getRiskLevelClass(level)}">${level}</span>
          <span class="text-xs">${percentage}%</span>
        </div>
        <div class="h-1 bg-gray-200 rounded">
          <div class="h-full ${getRiskLevelClass(level)} rounded" 
               style="width: ${percentage}%"></div>
        </div>
      `;
    })
    .join('');
}

// Set up filter controls
document.addEventListener('DOMContentLoaded', () => {
  // Initialize stats
  updateStats();

  // Set up filter buttons if they exist
  const filterButtons = document.querySelectorAll('[data-filter]');
  filterButtons.forEach(button => {
    button.addEventListener('click', () => {
      const filter = button.dataset.filter;

      // Update UI
      filterButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      // Update activity list
      chrome.storage.local.get(['browsing_activity'], (result) => {
        const activities = result.browsing_activity || [];
        const filtered = filter === 'all'
          ? activities
          : activities.filter(a => a.action === filter);

        updateActivityList(filtered);
      });
    });
  });

  // Set up view all button
  const viewAllBtn = document.getElementById('view-all');
  if (viewAllBtn) {
    viewAllBtn.addEventListener('click', () => {
      chrome.tabs.create({ url: chrome.runtime.getURL('dashboard.html') });
    });
  }
});

// Listen for changes in stats
chrome.storage.onChanged.addListener((changes) => {
  if (changes.dashboard_stats || changes.browsing_activity) {
    updateStats();
  }
});

function updateActivityList(activities) {
  const recentList = document.getElementById('recent-activity');
  if (recentList && activities.length > 0) {
    recentList.innerHTML = activities
      .slice(0, 10)
      .map(activity => `
        <div class="activity-item border-b border-gray-100 last:border-0 p-2">
          <div class="flex items-center justify-between">
            <div class="flex flex-col flex-grow mr-2">
              <span class="font-medium text-sm truncate max-w-[200px]" title="${activity.url}">
                ${formatUrl(activity.url)}
              </span>
              <div class="flex items-center text-xs text-gray-500 space-x-2">
                <span>${formatRelativeTime(activity.timestamp)}</span>
                <span class="category-tag">${activity.category || 'Unknown'}</span>
              </div>
            </div>
            <div class="flex flex-col items-end">
              <span class="text-xs px-2 py-1 rounded ${getActionClass(activity.action)}">
                ${activity.action}
              </span>
              <span class="text-xs mt-1 ${getRiskLevelClass(activity.risk_level)}">
                ${activity.risk_level || 'Unknown'} Risk
              </span>
            </div>
          </div>
        </div>
      `)
      .join('');
  }
}
