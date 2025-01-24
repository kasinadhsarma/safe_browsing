chrome.webRequest.onBeforeRequest.addListener(
  function(details) {
    // Send the URL to our backend for classification
    fetch('https://our-backend-url.com/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: details.url }),
    })
    .then(response => response.json())
    .then(data => {
      if (data.isSafe) {
        return { cancel: false };
      } else {
        return { cancel: true };
      }
    })
    .catch(error => {
      console.error('Error:', error);
      return { cancel: false };
    });
  },
  { urls: ["<all_urls>"] },
  ["blocking"]
);

