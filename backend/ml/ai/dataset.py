import pandas as pd
import numpy as np
import torch
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse, parse_qs
import re
import os
import tld
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global TF-IDF vectorizer to ensure consistent features
global_tfidf_vectorizer = TfidfVectorizer(max_features=20)

def load_url_data():
    """Load URLs from categorized files and combine them"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Load safe URLs with proper header and single column format
        education_urls = pd.read_csv(os.path.join(base_dir, 'education_urls.csv'), names=['url'], header=None)
        news_urls = pd.read_csv(os.path.join(base_dir, 'news_urls.csv'), names=['url'], header=None)

        # Load unsafe URLs with proper header and single column format
        gambling_urls = pd.read_csv(os.path.join(base_dir, 'gambling_urls.csv'), names=['url'], header=None)
        inappropriate_urls = pd.read_csv(os.path.join(base_dir, 'inappropriate_urls.csv'), names=['url'], header=None)
        malware_urls = pd.read_csv(os.path.join(base_dir, 'malware_urls.csv'), names=['url'], header=None)
        adult_urls = pd.read_csv(os.path.join(base_dir, 'adult_urls.csv'), names=['url'], header=None)

        # Combine and label safe URLs
        safe_urls = pd.concat([education_urls, news_urls])
        safe_urls['is_blocked'] = 0
        safe_urls['category'] = 'safe'

        # Combine and label unsafe URLs with specific categories
        gambling_urls['category'] = 'gambling'
        inappropriate_urls['category'] = 'inappropriate'
        malware_urls['category'] = 'malware'
        adult_urls['category'] = 'adult'

        unsafe_urls = pd.concat([gambling_urls, inappropriate_urls, malware_urls, adult_urls])
        unsafe_urls['is_blocked'] = 1

        # Combine all and shuffle
        all_urls = pd.concat([safe_urls, unsafe_urls]).reset_index(drop=True)

        # Ensure 'url' column exists
        if 'url' not in all_urls.columns:
            logging.error("The 'url' column is missing in the dataset.")
            return pd.DataFrame()

        return all_urls.sample(frac=1, random_state=42)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()

def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
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

def get_tokenized_url(url):
    """Get tokens from URL for feature extraction"""
    tokens = re.split(r'[/\-._?=&]', url.lower())
    return [t for t in tokens if t]

def get_domain_age(domain):
    """Get domain age in days using whois"""
    try:
        import whois
        from datetime import datetime

        # Get domain info
        domain_info = whois.whois(domain)

        # Get creation date (handle both single date and list cases)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        # Calculate age in days
        if creation_date:
            age_days = (datetime.now() - creation_date).days
            return max(age_days, 0)  # Ensure non-negative
        return 0
    except Exception as e:
        logging.warning(f"Could not get domain age for {domain}: {e}")
        return 0

def get_page_content(url):
    """Fetch and analyze page content"""
    try:
        import requests
        from bs4 import BeautifulSoup

        # Fetch page content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logging.warning(f"Could not get page content for {url}: {e}")
        return ""

# Define fixed feature columns to ensure consistency
FIXED_FEATURE_COLUMNS = [
    'length', 'num_dots', 'num_digits', 'num_special', 'entropy', 'token_count',
    *[f'tfidf_{i}' for i in range(20)],  # Fixed 20 TF-IDF features
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

def extract_url_features(url):
    """Extract enhanced features from URL for classification"""
    # Initialize all features with default values
    features = {col: 0.0 for col in FIXED_FEATURE_COLUMNS}

    try:
        if not isinstance(url, str):
            logging.warning(f"Invalid URL type: {type(url)}")
            return features

        # Check for empty or malformed URLs
        if not url or not url.startswith('http'):
            logging.warning(f"Malformed URL: {url}")
            return features

        # Parse URL
        parsed = urlparse(url)
        tokens = get_tokenized_url(url)

        # Basic features
        features['length'] = float(len(url))
        features['num_dots'] = float(url.count('.'))
        features['num_digits'] = float(sum(c.isdigit() for c in url))
        features['num_special'] = float(len(re.findall(r'[^a-zA-Z0-9.]', url)))
        features['entropy'] = float(calculate_entropy(url))
        features['token_count'] = float(len(tokens))

        # Add TF-IDF features for URL tokens using global vectorizer
        if tokens:
            try:
                token_text = ' '.join(tokens)
                tfidf_features = global_tfidf_vectorizer.fit_transform([token_text])
                for i in range(min(20, tfidf_features.shape[1])):
                    features[f'tfidf_{i}'] = float(tfidf_features[0, i])
            except Exception as e:
                logging.warning(f"Error extracting TF-IDF features: {e}")

        # Ensure TF-IDF vectorizer is fitted
        if not global_tfidf_vectorizer.vocabulary_:
            logging.warning("TF-IDF vectorizer is not fitted")
            return features

        # Add domain reputation features
        domain = parsed.netloc
        try:
            domain_age = get_domain_age(domain)
            features['domain_age_days'] = float(domain_age)
            features['is_new_domain'] = float(domain_age < 90)
        except Exception as e:
            logging.warning(f"Error getting domain age: {e}")

        # Add content-based features
        try:
            page_text = get_page_content(url)
            features['page_text_length'] = float(len(page_text))
            features['page_entropy'] = float(calculate_entropy(page_text))
            features['page_suspicious_word_count'] = float(len(re.findall(
                r'(adult|xxx|poker|casino|malware|virus|hack|crack|keygen|warez|free-download|win|lucky|prize|bet|gamble|drug|sex|porn|dating|violence|gore|death|hate|weapon)',
                page_text.lower()
            )))
        except Exception as e:
            logging.warning(f"Error extracting content features: {e}")

        # Handle token length features safely
        token_lengths = [len(t) for t in tokens] if tokens else [0]
        features['avg_token_length'] = float(np.mean(token_lengths))
        features['max_token_length'] = float(max(token_lengths))
        features['min_token_length'] = float(min(token_lengths))

        # Domain features
        domain = parsed.netloc
        features['domain_length'] = float(len(domain))
        features['has_subdomain'] = float(domain.count('.') > 1)
        features['has_www'] = float(domain.startswith('www.'))
        features['domain_entropy'] = float(calculate_entropy(domain))
        features['is_ip_address'] = float(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain)))

        # Safe division for ratios
        domain_len = len(domain) if domain else 1
        features['domain_digit_ratio'] = float(sum(c.isdigit() for c in domain) / domain_len)
        features['domain_special_ratio'] = float(len(re.findall(r'[^a-zA-Z0-9.]', domain)) / domain_len)
        features['domain_uppercase_ratio'] = float(sum(c.isupper() for c in domain) / domain_len)

        # Path features
        path = parsed.path
        features['path_length'] = float(len(path))
        features['num_directories'] = float(path.count('/'))
        features['path_entropy'] = float(calculate_entropy(path))
        features['has_double_slash'] = float('//' in path)

        # Handle directory length features safely
        dir_lengths = [len(d) for d in path.split('/') if d]
        features['directory_length_mean'] = float(np.mean(dir_lengths)) if dir_lengths else 0.0
        features['directory_length_max'] = float(max(dir_lengths)) if dir_lengths else 0.0
        features['directory_length_min'] = float(min(dir_lengths)) if dir_lengths else 0.0
        features['path_special_ratio'] = float(len(re.findall(r'[^a-zA-Z0-9/]', path)) / len(path)) if path else 0.0

        # Query parameters
        query = parsed.query
        query_params = parse_qs(query)
        features['num_params'] = float(len(query_params))
        features['query_length'] = float(len(query))
        features['has_suspicious_params'] = float(bool(re.search(r'(redirect|url|link|goto|ref|source)', query.lower())))

        # Handle query parameter length features safely
        param_lengths = [len(v[0]) for v in query_params.values()] if query_params else [0]
        features['max_param_length'] = float(max(param_lengths)) if param_lengths else 0.0
        features['mean_param_length'] = float(np.mean(param_lengths)) if param_lengths else 0.0
        features['param_entropy'] = float(calculate_entropy(query))
        features['param_special_ratio'] = float(len(re.findall(r'[^a-zA-Z0-9=&]', query)) / len(query)) if query else 0.0

        # Security features
        features['has_https'] = float(parsed.scheme == 'https')
        features['has_port'] = float(bool(parsed.port))
        features['suspicious_tld'] = float(bool(re.search(r'\.(xyz|tk|ml|ga|cf|gq|pw|top|info|online)$', domain.lower())))
        features['has_fragment'] = float(bool(parsed.fragment))
        features['has_redirect'] = float(bool(re.search(r'(redirect|url|link|goto)', url.lower())))
        features['has_obfuscation'] = float(bool(re.search(r'(%[0-9a-fA-F]{2}|&#x[0-9a-fA-F]{2,4};)', url.lower())))

        # Content indicators
        suspicious_pattern = r'(adult|xxx|poker|casino|malware|virus|hack|crack|keygen|warez|free-download|win|lucky|prize|bet|gamble|drug|sex|porn|dating|violence|gore|death|hate|weapon)'
        features['has_suspicious_words'] = float(bool(re.search(suspicious_pattern, url.lower())))
        features['suspicious_word_count'] = float(len(re.findall(suspicious_pattern, url.lower())))
        features['suspicious_word_ratio'] = features['suspicious_word_count'] / features['token_count'] if features['token_count'] > 0 else 0.0
        features['has_executable'] = float(bool(re.search(r'\.(exe|bat|cmd|dll|bin|sh|py|pl|php|jsp|asp|cgi|js|html|htm|xml|json)$', url.lower())))
        features['has_archive'] = float(bool(re.search(r'\.(zip|rar|7z|tar|gz|bz2|xz)$', url.lower())))

        # Age group risk features
        kid_unsafe_pattern = r'(sex|porn|adult|dating|drug|gamble|bet|violence|gore|death|hate|weapon|abuse|exploit)'
        teen_unsafe_pattern = r'(hack|crack|cheat|free-money|betting|smoking|vaping|alcohol|weapon|violence|drug|porn)'

        features['kid_unsafe_words'] = float(len(re.findall(kid_unsafe_pattern, url.lower())))
        features['teen_unsafe_words'] = float(len(re.findall(teen_unsafe_pattern, url.lower())))
        features['kid_unsafe_ratio'] = features['kid_unsafe_words'] / features['token_count'] if features['token_count'] > 0 else 0.0
        features['teen_unsafe_ratio'] = features['teen_unsafe_words'] / features['token_count'] if features['token_count'] > 0 else 0.0
        features['kid_unsafe_score'] = features['kid_unsafe_ratio'] * 2 if features['has_https'] == 0 else features['kid_unsafe_ratio']
        features['teen_unsafe_score'] = features['teen_unsafe_ratio'] * 1.5 if features['has_https'] == 0 else features['teen_unsafe_ratio']

        # Normalize numeric features to [0,1] range
        for key in features:
            if isinstance(features[key], (int, float)) and key not in ['has_https', 'has_port', 'has_fragment', 'has_suspicious_words', 'has_executable']:
                features[key] = float(features[key])

        # Ensure only keys present in FIXED_FEATURE_COLUMNS are returned
        features = {key: features[key] for key in FIXED_FEATURE_COLUMNS}

        return features

    except Exception as e:
        logging.error(f"Error extracting URL features: {e}")
        return features  # Return initialized features rather than empty dict

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def generate_dataset():
    """Generate enhanced dataset with URL features and labels using parallel processing"""
    try:
        # Load URLs
        df = load_url_data()
        if df.empty:
            logging.warning("No URL data loaded")
            return df

        logging.info(f"Processing {len(df)} URLs...")

        # Extract features in parallel with progress bar
        features_list = []
        failed_urls = []
        valid_feature_count = None
        feature_keys = None

        def process_url(url):
            try:
                features = extract_url_features(url)
                if features:
                    # Ensure consistent feature dictionary structure
                    if valid_feature_count is None:
                        nonlocal feature_keys
                        feature_keys = list(features.keys())

                    # Validate and convert feature values
                    processed_features = {}
                    for key in feature_keys:
                        try:
                            val = features.get(key, 0)
                            processed_features[key] = float(val) if isinstance(val, (int, float)) else 0.0
                        except (ValueError, TypeError):
                            processed_features[key] = 0.0

                    return processed_features, None
                else:
                    return None, url
            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                return None, url

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_url, url): url for url in df['url']}

            # Use tqdm for progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
                result, failed_url = future.result()
                if result:
                    features_list.append(result)
                elif failed_url:
                    failed_urls.append(failed_url)

        if not features_list:
            logging.error("No valid features extracted")
            return pd.DataFrame()

        # Convert to DataFrame with error handling
        try:
            features_df = pd.DataFrame(features_list)
        except Exception as e:
            logging.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()

        # Handle missing values and data types
        for col in features_df.columns:
            try:
                # Convert to numeric, coerce errors to NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            except Exception as e:
                logging.error(f"Error converting column {col}: {e}")
                features_df[col] = 0.0

        # Fill remaining NaN values with 0
        features_df = features_df.fillna(0)

        # Add metadata columns
        features_df['url'] = df['url']
        features_df['is_blocked'] = df['is_blocked'].astype(float)
        features_df['category'] = df['category']

        # Ensure columns are present
        if 'is_blocked' not in features_df.columns or 'category' not in features_df.columns:
            raise ValueError("Missing 'is_blocked' or 'category' columns in the dataset")

        # Validate final dataset
        if len(features_df) == 0:
            logging.error("Empty dataset generated")
            return pd.DataFrame()

        # Ensure only the fixed feature columns and target column are included
        features_df = features_df[FIXED_FEATURE_COLUMNS + ['is_blocked']]

        logging.info(f"Successfully generated dataset with {len(features_df)} samples")
        return features_df

    except Exception as e:
        logging.error(f"Error generating dataset: {e}")
        return pd.DataFrame()

class URLDataset(Dataset):
    def __init__(self, urls, labels):
        """
        Initialize URL dataset
        Args:
            urls: List of URLs
            labels: List of labels (0 for safe, 1 for unsafe)
        """
        if not urls or not labels:
            raise ValueError("URLs and labels cannot be empty")
        if len(urls) != len(labels):
            raise ValueError("Number of URLs must match number of labels")

        self.urls = urls
        self.labels = [float(label) for label in labels]  # Convert labels to float

        # First pass: determine feature structure
        feature_keys = None
        for url in urls:
            try:
                features = extract_url_features(url)
                if features:
                    feature_keys = list(features.keys())
                    break
            except Exception:
                continue

        if not feature_keys:
            raise ValueError("Could not determine feature structure from any URL")

        # Extract features with consistent structure
        features_list = []
        for url in urls:
            try:
                features = extract_url_features(url)
                if features and len(features) == len(feature_keys):
                    # Ensure features are in the same order
                    feature_values = [float(features.get(key, 0.0)) for key in feature_keys]
                    features_list.append(feature_values)
                else:
                    # Use zeros if feature extraction failed or has inconsistent structure
                    features_list.append([0.0] * len(feature_keys))
            except Exception as e:
                logging.error(f"Error processing URL in dataset: {e}")
                features_list.append([0.0] * len(feature_keys))

        # Convert to numpy array with explicit dtype
        self.features = np.array(features_list, dtype=np.float32)

        # Validate feature array
        if len(self.features.shape) < 2 or 0 in self.features.shape:
            raise ValueError("Failed to extract valid features")

        # Handle standardization safely
        self.scaler = StandardScaler()
        try:
            # Add small epsilon to avoid zero variance
            epsilon = 1e-10
            features_for_scaling = self.features + epsilon
            self.features = self.scaler.fit_transform(features_for_scaling)
        except Exception as e:
            logging.warning(f"Standard scaling failed: {e}")
            # Fallback to simple normalization
            features_min = self.features.min(axis=0)
            features_max = self.features.max(axis=0)
            features_range = features_max - features_min
            features_range[features_range == 0] = 1  # Avoid division by zero
            self.features = (self.features - features_min) / features_range

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        try:
            return {
                'features': torch.FloatTensor(self.features[idx]),
                'label': torch.FloatTensor([self.labels[idx]])
            }
        except Exception as e:
            logging.error(f"Error getting item at index {idx}: {e}")
            # Return zero tensor as fallback
            return {
                'features': torch.zeros(self.features.shape[1], dtype=torch.float32),
                'label': torch.FloatTensor([0.0])
            }

# Example usage
if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_dataset()

    # Save the dataset to a CSV file
    dataset.to_csv('enhanced_url_dataset.csv', index=False)

    # Example of creating a PyTorch dataset
    urls = dataset['url'].tolist()
    labels = dataset['is_blocked'].tolist()

    url_dataset = URLDataset(urls, labels)
    dataloader = DataLoader(url_dataset, batch_size=32, shuffle=True)

    # Print the first batch
    for batch in dataloader:
        print(batch['features'], batch['label'])
        break
