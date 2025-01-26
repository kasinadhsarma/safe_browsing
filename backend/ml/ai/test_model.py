import logging
import os
from training import predict_url, calculate_age_based_risk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Test URLs
test_urls = [
    # Safe URLs
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.python.org/downloads/",
    "https://github.com/features",

    # Potentially unsafe URLs
    "http://suspicious-site.xyz/download.exe",
    "http://192.168.1.1/admin/hack.php",
    "http://free-casino-games.tk/poker",
]

import logging
import os
from training import predict_url, calculate_age_based_risk
from image_classification import load_image_from_url, classify_image, is_inappropriate, handle_inappropriate_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Test URLs
test_urls = [
    # Safe URLs
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.python.org/downloads/",
    "https://github.com/features",

    # Potentially unsafe URLs
    "http://suspicious-site.xyz/download.exe",
    "http://192.168.1.1/admin/hack.php",
    "http://free-casino-games.tk/poker",
]

def test_model():
    logging.info("Testing URL classifier model...")

    # First check if models exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found at {models_dir}. Please train the models first.")
        return

    for url in test_urls:
        try:
            is_unsafe, probability, risk_score = predict_url(url, models_dir=models_dir)
            logging.info(f"\nURL: {url}")
            logging.info(f"Is Unsafe: {is_unsafe}")
            logging.info(f"Probability: {probability:.4f}")
            logging.info(f"Risk Score: {risk_score:.4f}")

            # Call calculate_age_based_risk
            risk_level, risk_score = calculate_age_based_risk(
                predictions={'knn': {'probability': probability}, 'svm': {'probability': probability}, 'nb': {'probability': probability}},
                features={},
                age_group='kid'
            )
            logging.info(f"Risk Level: {risk_level}, Risk Score: {risk_score}")

            # Check for inappropriate images
            image = load_image_from_url(url)
            label = classify_image(image)
            if is_inappropriate(label):
                handle_inappropriate_image(image)
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")

if __name__ == "__main__":
    test_model()
