import logging
import os
import sys
import urllib3
from pathlib import Path
import warnings
import requests
from requests.exceptions import RequestException
from urllib3.exceptions import InsecureRequestWarning

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.ml.ai.training import predict_url, calculate_age_based_risk
from backend.ml.ai.image_classification import load_image_from_url, classify_image, is_inappropriate, handle_inappropriate_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

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
    "https://brazzers.com"
]

def is_url_accessible(url):
    """Check if a URL is accessible"""
    try:
        response = requests.head(url, verify=False, timeout=2)
        return response.status_code == 200
    except RequestException:
        return False

def test_model():
    logging.info("Testing URL classifier model...")

    # First check if models exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'latest')
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found at {models_dir}. Please train the models first.")
        return

    successes = 0
    failures = 0

    for url in test_urls:
        try:
            logging.info(f"\nTesting URL: {url}")
            
            # Check URL accessibility
            if not is_url_accessible(url):
                logging.warning(f"URL {url} is not accessible, but continuing with analysis...")
            
            # Predict URL safety
            is_unsafe, risk_score, risk_level = predict_url(url, models_dir=models_dir)
            logging.info(f"Is Unsafe: {is_unsafe}")
            logging.info(f"Risk Score: {risk_score:.4f}")
            logging.info(f"Risk Level: {risk_level}")

            # Calculate age-based risk for different age groups
            test_age_groups = ['kid', 'teen', 'adult']
            for age_group in test_age_groups:
                # Simulate model predictions with the risk score
                predictions = {
                    'knn': {'probability': risk_score},
                    'svm': {'probability': risk_score},
                    'nb': {'probability': risk_score}
                }
                
                # Get risk assessment for this age group
                age_risk_level, age_risk_score = calculate_age_based_risk(
                    predictions=predictions,
                    features={},  # Features already considered in predict_url
                    age_group=age_group
                )
                logging.info(f"Age Group: {age_group} - Risk Level: {age_risk_level}, Risk Score: {age_risk_score:.4f}")

            # Check for inappropriate images
            try:
                image = load_image_from_url(url)
                if image is not None:
                    label = classify_image(image)
                    if is_inappropriate(label):
                        logging.info("Inappropriate image detected")
                        handle_inappropriate_image(image)
                    else:
                        logging.info("Image is appropriate")
                else:
                    logging.info("No image found or failed to load image")
            except Exception as e:
                logging.error(f"Error loading image from URL {url}: {str(e)}")
                logging.info("No image found or failed to load image")

            successes += 1

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            failures += 1

    # Report summary
    total = successes + failures
    logging.info(f"\nTest Summary:")
    logging.info(f"Total URLs tested: {total}")
    logging.info(f"Successful: {successes}")
    logging.info(f"Failed: {failures}")
    logging.info(f"Success rate: {(successes/total)*100:.1f}%")

if __name__ == "__main__":
    test_model()
