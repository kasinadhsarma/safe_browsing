import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse, parse_qs
import re
import os
import tld
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def load_url_data():
    """Load URLs from categorized files and combine them"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Load safe URLs
        education_urls = pd.read_csv(os.path.join(base_dir, 'education_urls.csv'))
        news_urls = pd.read_csv(os.path.join(base_dir, 'news_urls.csv'))

        # Load unsafe URLs
        gambling_urls = pd.read_csv(os.path.join(base_dir, 'gambling_urls.csv'))
        inappropriate_urls = pd.read_csv(os.path.join(base_dir, 'inappropriate_urls.csv'))
        malware_urls = pd.read_csv(os.path.join(base_dir, 'malware_urls.csv'))
        adult_urls = pd.read_csv(os.path.join(base_dir, 'adult_urls.csv'))

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
        return all_urls.sample(frac=1, random_state=42)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
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

def extract_url_features(url):
    """Extract enhanced features from URL for classification"""
    features = {}

    try:
        # Parse URL
        parsed = urlparse(url)
        tokens = get_tokenized_url(url)

        # Basic features
        features['length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special'] = len(re.findall(r'[^a-zA-Z0-9.]', url))
        features['entropy'] = calculate_entropy(url)
        features['token_count'] = len(tokens)
        features['avg_token_length'] = np.mean([len(t) for t in tokens]) if tokens else 0
        features['max_token_length'] = max([len(t) for t in tokens]) if tokens else 0
        features['min_token_length'] = min([len(t) for t in tokens]) if tokens else 0
        
        # Domain features
        domain = parsed.netloc
        features['domain_length'] = len(domain)
        features['has_subdomain'] = domain.count('.') > 1
        features['has_www'] = domain.startswith('www.')
        features['domain_entropy'] = calculate_entropy(domain)
        features['is_ip_address'] = int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain)))
        features['domain_digit_ratio'] = sum(c.isdigit() for c in domain) / len(domain) if domain else 0
        features['domain_special_ratio'] = len(re.findall(r'[^a-zA-Z0-9.]', domain)) / len(domain) if domain else 0
        features['domain_uppercase_ratio'] = sum(c.isupper() for c in domain) / len(domain) if domain else 0
        
        # Path features
        path = parsed.path
        features['path_length'] = len(path)
        features['num_directories'] = path.count('/')
        features['path_entropy'] = calculate_entropy(path)
        features['has_double_slash'] = int('//' in path)
        features['directory_length_mean'] = np.mean([len(d) for d in path.split('/') if d]) if path else 0
        features['directory_length_max'] = max([len(d) for d in path.split('/') if d]) if path else 0
        features['directory_length_min'] = min([len(d) for d in path.split('/') if d]) if path else 0
        features['path_special_ratio'] = len(re.findall(r'[^a-zA-Z0-9/]', path)) / len(path) if path else 0
        
        # Query parameters
        query = parsed.query
        query_params = parse_qs(query)
        features['num_params'] = len(query_params)
        features['query_length'] = len(query)
        features['has_suspicious_params'] = int(bool(re.search(r'(redirect|url|link|goto|ref|source)', query.lower())))
        features['max_param_length'] = max([len(v[0]) for v in query_params.values()]) if query_params else 0
        features['mean_param_length'] = np.mean([len(v[0]) for v in query_params.values()]) if query_params else 0
        features['param_entropy'] = calculate_entropy(query)
        features['param_special_ratio'] = len(re.findall(r'[^a-zA-Z0-9=&]', query)) / len(query) if query else 0
        
        # Security features
        features['has_https'] = int(parsed.scheme == 'https')
        features['has_port'] = int(bool(parsed.port))
        features['suspicious_tld'] = int(bool(re.search(r'\.(xyz|tk|ml|ga|cf|gq|pw|top|info|online)$', domain.lower())))
        features['has_fragment'] = int(bool(parsed.fragment))
        features['has_redirect'] = int(bool(re.search(r'(redirect|url|link|goto)', url.lower())))
        features['has_obfuscation'] = int(bool(re.search(r'(%[0-9a-fA-F]{2}|&#x[0-9a-fA-F]{2,4};)', url.lower())))
        
        # Content indicators
        suspicious_pattern = r'(adult|xxx|poker|casino|malware|virus|hack|crack|keygen|warez|free-download|win|lucky|prize|bet|gamble|drug|sex|porn|dating|violence|gore|death|hate|weapon)'
        features['has_suspicious_words'] = int(bool(re.search(suspicious_pattern, url.lower())))
        features['suspicious_word_count'] = len(re.findall(suspicious_pattern, url.lower()))
        features['suspicious_word_ratio'] = features['suspicious_word_count'] / features['token_count'] if features['token_count'] > 0 else 0
        features['has_executable'] = int(bool(re.search(
            r'\.(exe|bat|cmd|dll|bin|sh|py|pl|php|jsp|asp|cgi|js|html|htm|xml|json)$',
            url.lower()
        )))
        features['has_archive'] = int(bool(re.search(
            r'\.(zip|rar|7z|tar|gz|bz2|xz)$',
            url.lower()
        )))
        
        # Age group risk features
        kid_unsafe_pattern = r'(sex|porn|adult|dating|drug|gamble|bet|violence|gore|death|hate|weapon|abuse|exploit)'
        teen_unsafe_pattern = r'(hack|crack|cheat|free-money|betting|smoking|vaping|alcohol|weapon|violence|drug|porn)'
        
        features['kid_unsafe_words'] = len(re.findall(kid_unsafe_pattern, url.lower()))
        features['teen_unsafe_words'] = len(re.findall(teen_unsafe_pattern, url.lower()))
        features['kid_unsafe_ratio'] = features['kid_unsafe_words'] / features['token_count'] if features['token_count'] > 0 else 0
        features['teen_unsafe_ratio'] = features['teen_unsafe_words'] / features['token_count'] if features['token_count'] > 0 else 0
        features['kid_unsafe_score'] = features['kid_unsafe_ratio'] * 2 if features['has_https'] == 0 else features['kid_unsafe_ratio']
        features['teen_unsafe_score'] = features['teen_unsafe_ratio'] * 1.5 if features['has_https'] == 0 else features['teen_unsafe_ratio']

        # Normalize numeric features to [0,1] range
        for key in features:
            if isinstance(features[key], (int, float)) and key not in ['has_https', 'has_port', 'has_fragment', 'has_suspicious_words', 'has_executable']:
                features[key] = float(features[key])

        return features

    except Exception as e:
        print(f"Error extracting URL features: {e}")
        return {}

def generate_dataset():
    """Generate enhanced dataset with URL features and labels"""
    # Load URLs
    df = load_url_data()
    if df.empty:
        return df

    # Extract features for each URL
    features_list = []
    for url in df['url']:
        features = extract_url_features(url)
        features_list.append(features)

    # Convert to DataFrame and handle missing values
    features_df = pd.DataFrame(features_list)
    
    # Fill missing values appropriately
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    bool_cols = features_df.select_dtypes(include=[bool]).columns
    
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    features_df[bool_cols] = features_df[bool_cols].fillna(False)

    # Add URL and label columns
    features_df['url'] = df['url']
    features_df['is_blocked'] = df['is_blocked']
    features_df['category'] = df['category']

    return features_df

class URLDataset(Dataset):
    def __init__(self, urls, labels):
        """
        Initialize URL dataset
        Args:
            urls: List of URLs
            labels: List of labels (0 for safe, 1 for unsafe)
        """
        self.urls = urls
        self.labels = labels
        
        # Extract features
        features_list = []
        for url in urls:
            features = extract_url_features(url)
            features_list.append(list(features.values()))
            
        # Convert to numpy array and standardize
        self.features = np.array(features_list)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
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