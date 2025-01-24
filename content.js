// Function to check and block images
function checkAndBlockImages() {
  const images = document.getElementsByTagName('img');
  for (let img of images) {
    // Send image URL to backend for classification
    fetch('https://our-backend-url.com/classify-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageUrl: img.src }),
    })
    .then(response => response.json())
    .then(data => {
      if (!data.isSafe) {
        img.style.filter = 'blur(10px)';
      }
    })
    .catch(error => console.error('Error:', error));
  }
}

// Run the function when the page loads
window.addEventListener('load', checkAndBlockImages);

// Run the function when new content is loaded (e.g., infinite scrolling)
const observer = new MutationObserver(checkAndBlockImages);
observer.observe(document.body, { childList: true, subtree: true });

