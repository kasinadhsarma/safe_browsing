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
        if not isinstance(url, str):
            return {}

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
        features['max_param_length'] = float(max(param_lengths))
        features['mean_param_length'] = float(np.mean(param_lengths))
        features['param_entropy'] = float(calculate_entropy(query))
        features['param_special_ratio'] = float(len(re.findall(r'[^a-zA-Z0-9=&]', query)) / len(query)) if query else 0.0
        
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
    try:
        # Load URLs
        df = load_url_data()
        if df.empty:
            print("Warning: No URL data loaded")
            return df

        print(f"Processing {len(df)} URLs...")
        
        # Extract features for each URL with better error handling
        features_list = []
        failed_urls = []
        valid_feature_count = None
        
        for url in df['url']:
            try:
                features = extract_url_features(url)
                if features:
                    # Ensure consistent feature dictionary structure
                    if valid_feature_count is None:
                        valid_feature_count = len(features)
                        feature_keys = list(features.keys())
                    
                    # Validate and convert feature values
                    processed_features = {}
                    for key in feature_keys:
                        try:
                            val = features.get(key, 0)
                            processed_features[key] = float(val) if isinstance(val, (int, float)) else 0.0
                        except (ValueError, TypeError):
                            processed_features[key] = 0.0
                    
                    features_list.append(processed_features)
                else:
                    # Use zeros for failed feature extraction
                    features_list.append({key: 0.0 for key in feature_keys}) if valid_feature_count else failed_urls.append(url)
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                if valid_feature_count and feature_keys:
                    features_list.append({key: 0.0 for key in feature_keys})
                else:
                    failed_urls.append(url)

        if failed_urls:
            print(f"Warning: Failed to process {len(failed_urls)} URLs")
            
        if not features_list:
            print("Error: No valid features extracted")
            return pd.DataFrame()

        # Convert to DataFrame with error handling
        try:
            features_df = pd.DataFrame(features_list)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()

        # Handle missing values and data types
        for col in features_df.columns:
            try:
                # Convert to numeric, coerce errors to NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                features_df[col] = 0.0

        # Fill remaining NaN values with 0
        features_df = features_df.fillna(0)

        # Add metadata columns
        features_df['url'] = df['url']
        features_df['is_blocked'] = df['is_blocked'].astype(float)
        features_df['category'] = df['category']

        # Validate final dataset
        if len(features_df) == 0:
            print("Error: Empty dataset generated")
            return pd.DataFrame()

        print(f"Successfully generated dataset with {len(features_df)} samples")
        return features_df

    except Exception as e:
        print(f"Error generating dataset: {e}")
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
                print(f"Error processing URL in dataset: {e}")
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
            print(f"Warning: Standard scaling failed: {e}")
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
            print(f"Error getting item at index {idx}: {e}")
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
