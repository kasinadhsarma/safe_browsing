import pandas as pd
import numpy as np
import logging
import os
from urllib.parse import urlparse
import re
from datetime import datetime
import whois
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global TF-IDF vectorizer to ensure consistent features
global_tfidf_vectorizer = TfidfVectorizer(max_features=20)

# Define fixed feature columns to ensure consistency 
FIXED_FEATURE_COLUMNS = [
    'length', 'num_dots', 'num_digits', 'num_special', 'entropy', 'token_count',
    *[f'tfidf_{i}' for i in range(20)],
    'domain_age_days', 'is_new_domain', 'page_text_length', 'page_entropy',
    'page_suspicious_word_count', 'avg_token_length', 'max_token_length',
    'min_token_length', 'domain_length', 'has_subdomain', 'has_www',
    'domain_entropy', 'is_ip_address', 'domain_digit_ratio',  
    'domain_special_ratio', 'domain_uppercase_ratio', 'path_length',
    'num_directories', 'path_entropy', 'has_double_slash',
    'directory_length_mean', 'directory_length_max', 'directory_length_min',
    'path_special_ratio', 'num_params', 'query_length', 'has_suspicious_params',
    'max_param_length', 'mean_param_length', 'param_entropy',
    'param_special_ratio', 'has_https', 'has_port', 'suspicious_tld',
    'has_fragment', 'has_redirect', 'has_obfuscation', 'has_suspicious_words',
    'suspicious_word_count', 'suspicious_word_ratio', 'has_executable',
    'has_archive', 'kid_unsafe_words', 'teen_unsafe_words', 'kid_unsafe_ratio',
    'teen_unsafe_ratio', 'kid_unsafe_score', 'teen_unsafe_score'
]

def load_url_data():
    """Load URLs from categorized files and combine them."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    safe_files = ['education_urls.csv', 'news_urls.csv', 'social_urls.csv']
    unsafe_files = ['gambling_urls.csv', 'inappropriate_urls.csv', 'malware_urls.csv', 'adult_urls.csv']

    try:
        safe_urls = []
        for file in safe_files:
            file_path = os.path.join(base_dir, file)
            df = pd.read_csv(file_path)
            df['target'] = 0  # Label for safe URLs
            safe_urls.append(df)
            logging.info(f"Loaded {len(df)} safe URLs from {file}")

        unsafe_urls = []
        for file in unsafe_files:
            file_path = os.path.join(base_dir, file)
            df = pd.read_csv(file_path)
            df['target'] = 1  # Label for unsafe URLs
            unsafe_urls.append(df)
            logging.info(f"Loaded {len(df)} unsafe URLs from {file}")

        if not safe_urls and not unsafe_urls:
            logging.error("No URLs loaded from the provided files.")
            return pd.DataFrame(), pd.Series()

        all_urls = pd.concat(safe_urls + unsafe_urls, ignore_index=True)
        all_urls = all_urls.sample(frac=1, random_state=42)  # Shuffle the dataset

        logging.info(f"Combined dataset has {len(all_urls)} samples.")

        X = all_urls.drop('target', axis=1)
        y = all_urls['target']

        logging.info(f"Dataset features: {X.columns.tolist()}")
        logging.info(f"Dataset labels: {y.value_counts().to_dict()}")

        return X, y

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.Series()

def calculate_entropy(text):
    """Calculate Shannon entropy of text."""
    if not text:
        return 0
    freqs = {}
    for c in text:
        freqs[c] = freqs.get(c, 0) + 1
    entropy = 0
    for freq in freqs.values():
        prob = freq / len(text)
        entropy -= prob * np.log2(prob)
    return entropy

# Configure requests session with minimal retries
def create_session():
    session = requests.Session()
    retries = Retry(
        total=1,  # Minimal retries
        backoff_factor=0.1,  # Quick retry
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = create_session()

# Enhanced suspicious patterns
SUSPICIOUS_PATTERNS = {
    'malware': r'(malware|virus|trojan|spyware|ransomware|backdoor|exploit|worm)',
    'phishing': r'(login|verify|account|secure|banking|password|credential)',
    'scam': r'(prize|winner|lottery|casino|free\s*money|discount|deal)',
    'adult': r'(xxx|porn|adult|sex|mature|escort|brazzers|pornhub|xvideos|xnxx|nude|nsfw)',
    'drugs': r'(drug|pharma|pill|medication|prescription)',
    'weapons': r'(weapon|gun|ammo|explosive)',
    'hacking': r'(hack|crack|keygen|serial|warez|leaked|dump)'
}

@lru_cache(maxsize=10000)
def get_domain_age(domain):
    """Get domain age with minimal processing."""
    try:
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$', domain):
            return 0
        
        try:
            domain_info = whois.whois(domain)
        except:
            return 0

        if not domain_info.creation_date:
            return 0

        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date

        if isinstance(creation_date, datetime):
            age_days = (datetime.now() - creation_date).days
            return max(age_days, 0)
        return 0
    except:
        return 0

def get_page_content(url):
    """Fetch content with minimal processing."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 Chrome/91.0.4472.124'}
        response = session.get(
            url,
            headers=headers,
            timeout=(2, 5),  # Aggressive timeouts
            verify=False,  # Skip SSL verification
            allow_redirects=False
        )
        
        if not response.ok:
            return ""

        text = BeautifulSoup(response.text, 'html.parser').get_text()
        return text[:5000]  # Limit content size
    except:
        return ""

def analyze_url_safety(url, content=""):
    """Quick safety analysis with enhanced adult content detection"""
    scores = {}
    for category, pattern in SUSPICIOUS_PATTERNS.items():
        url_matches = len(re.findall(pattern, url.lower()))
        content_matches = len(re.findall(pattern, content.lower())) if content else 0
        
        # Increase weight for adult content matches
        multiplier = 2.0 if category == 'adult' else 1.0
        scores[category] = ((url_matches * 2 + content_matches) * multiplier) / (2 if content else 1)
    
    return scores

def extract_url_features(url):
    """Extract features with parallel processing support."""
    features = {col: 0.0 for col in FIXED_FEATURE_COLUMNS}
    try:
        if not isinstance(url, str) or not url.startswith('http'):
            return features

        parsed = urlparse(url)
        tokens = [t for t in re.split(r'[/\-._?=&]', url.lower()) if t]

        # Basic features
        features.update({
            'length': float(len(url)),
            'num_dots': float(url.count('.')),
            'num_digits': float(sum(c.isdigit() for c in url)),
            'num_special': float(len(re.findall(r'[^a-zA-Z0-9.]', url))),
            'entropy': float(calculate_entropy(url)),
            'token_count': float(len(tokens))
        })

        # TF-IDF features
        if tokens:
            token_text = ' '.join(tokens)
            tfidf_features = global_tfidf_vectorizer.fit_transform([token_text])
            for i in range(min(20, tfidf_features.shape[1])):
                features[f'tfidf_{i}'] = float(tfidf_features[0, i])

        # Domain features
        domain = parsed.netloc
        domain_age = get_domain_age(domain)
        features.update({
            'domain_age_days': float(domain_age),
            'is_new_domain': float(domain_age < 90),
            'domain_length': float(len(domain)),
            'has_subdomain': float(domain.count('.') > 1),
            'has_www': float(domain.startswith('www.')),
            'domain_entropy': float(calculate_entropy(domain)),
            'is_ip_address': float(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain)))
        })

        # URL content features
        path = parsed.path
        features.update({
            'path_length': float(len(path)),
            'num_directories': float(path.count('/')),
            'path_entropy': float(calculate_entropy(path)),
            'has_double_slash': float('//' in path)
        })

        # Safety analysis
        content = get_page_content(url)
        safety_scores = analyze_url_safety(url, content)

        features.update({
            'page_text_length': float(len(content)),
            'page_entropy': float(calculate_entropy(content)),
            'page_suspicious_word_count': float(sum(safety_scores.values()))
        })

        for category, score in safety_scores.items():
            feature_name = f'{category}_score'
            if feature_name in features:
                features[feature_name] = float(score)

        logging.info(f"Extracted features for {url}: {features}")

        return features
    except Exception as e:
        logging.warning(f"Error extracting features for {url}: {str(e)}")
        return features

# Process URLs in parallel
def process_urls_parallel(urls, max_workers=50):
    """Process URLs in parallel using ThreadPoolExecutor."""
    features = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_url_features, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                feature_dict = future.result()
                features.append(feature_dict)
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
                features.append({col: 0.0 for col in FIXED_FEATURE_COLUMNS})
    return features
