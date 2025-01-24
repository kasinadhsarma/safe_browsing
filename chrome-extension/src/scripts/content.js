// Content analysis patterns
const contentPatterns = {
  adult: /(xxx|porn|sex|adult|nude)/i,
  violence: /(violence|gore|death|kill|fight)/i,
  gambling: /(gambling|casino|bet|poker|lottery)/i,
  drugs: /(drugs|cocaine|heroin|weed|marijuana)/i,
  hate: /(hate|racist|discrimination|nazi)/i
};

// Analyze page content for risks
function analyzeContent() {
  const pageText = document.body.innerText.toLowerCase();
  const metaDesc = document.querySelector('meta[name="description"]')?.content || '';
  const title = document.title.toLowerCase();
  
  const foundPatterns = {};
  let riskLevel = 'Low';
  let riskCount = 0;

  // Check for risky content
  for (const [category, pattern] of Object.entries(contentPatterns)) {
    const matches = (pageText.match(pattern) || []).length +
                   (metaDesc.match(pattern) || []).length +
                   (title.match(pattern) || []).length;
    
    if (matches > 0) {
      foundPatterns[category] = matches;
      riskCount++;
    }
  }

  // Determine risk level based on matches
  if (riskCount >= 3) {
    riskLevel = 'High';
  } else if (riskCount > 0) {
    riskLevel = 'Medium';
  }

  return {
    riskLevel,
    patterns: foundPatterns,
    title: document.title,
    url: window.location.href,
    timestamp: new Date().toISOString()
  };
}

// Create and show block overlay
function showBlockOverlay(details) {
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.98);
    z-index: 2147483647;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  `;

  const content = document.createElement('div');
  content.style.cssText = `
    max-width: 500px;
    text-align: center;
    padding: 2rem;
  `;

  const icon = document.createElement('img');
  icon.src = chrome.runtime.getURL('src/icons/icon64.png');
  icon.style.marginBottom = '1rem';

  const title = document.createElement('h1');
  title.textContent = '⚠️ Access Blocked';
  title.style.cssText = `
    color: #dc2626;
    font-size: 24px;
    margin-bottom: 1rem;
  `;

  const message = document.createElement('p');
  message.textContent = 'This website has been blocked for your safety.';
  message.style.cssText = `
    color: #374151;
    margin-bottom: 1rem;
  `;

  const info = document.createElement('div');
  info.style.cssText = `
    background: #f3f4f6;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    text-align: left;
  `;
  info.innerHTML = `
    <div style="margin-bottom: 0.5rem;"><strong>Category:</strong> ${details.category}</div>
    <div style="margin-bottom: 0.5rem;"><strong>Risk Level:</strong> 
      <span style="color: ${details.risk_level === 'High' ? '#dc2626' : '#d97706'}">${details.risk_level}</span>
    </div>
    <div><strong>URL:</strong> ${window.location.href}</div>
  `;

  content.appendChild(icon);
  content.appendChild(title);
  content.appendChild(message);
  content.appendChild(info);
  overlay.appendChild(content);
  document.body.appendChild(overlay);
}

// Monitor for dynamic content changes
const observer = new MutationObserver((mutations) => {
  if (document.body) {
    const analysis = analyzeContent();
    chrome.runtime.sendMessage({
      type: 'CONTENT_ANALYZED',
      analysis
    });
  }
});

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'PAGE_BLOCKED') {
    showBlockOverlay(message);
    return true;
  }
  
  if (message.type === 'ANALYZE_CONTENT') {
    const analysis = analyzeContent();
    sendResponse(analysis);
    return true;
  }
});

// Start monitoring when content is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Initial content analysis
  const analysis = analyzeContent();
  chrome.runtime.sendMessage({
    type: 'CONTENT_ANALYZED',
    analysis
  });

  // Start observing for changes
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true
  });
});

// Clean up observer when page is unloaded
window.addEventListener('unload', () => {
  observer.disconnect();
});
