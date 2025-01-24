import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import re
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class URLDataset(Dataset):
    def __init__(self, urls: List[str], labels: List[int]):
        """
        Initialize the dataset with URLs and corresponding labels.
        
        Args:
            urls (List[str]): List of URLs.
            labels (List[int]): List of labels (0 or 1).
        """
        self.urls = urls
        self.labels = labels
        
    def __len__(self) -> int:
        """
        Return the number of URLs in the dataset.
        """
        return len(self.urls)
    
    def extract_features(self, url: str) -> torch.Tensor:
        """
        Extract features from a URL.
        
        Args:
            url (str): The URL to extract features from.
        
        Returns:
            torch.Tensor: A tensor containing the extracted features.
        """
        try:
            # Split the URL into parts
            parts = url.split('/')
            domain = parts[2] if len(parts) > 2 else ''
            
            features = {
                'length': len(url),
                'num_dots': url.count('.'),
                'num_digits': sum(c.isdigit() for c in url),
                'num_special': len(re.findall('[^A-Za-z0-9.]', url)),
                'has_https': int(url.startswith('https')),
                'domain_length': len(domain),
                'subdomain_count': len(domain.split('.')) - 1 if domain else 0,
                'path_length': len(parts[-1]) if len(parts) > 3 else 0
            }
            return torch.tensor([v for v in features.values()], dtype=torch.float32)
        except Exception as e:
            logging.error(f"Error extracting features from URL: {url}. Error: {e}")
            return torch.zeros(8, dtype=torch.float32)  # Return a zero tensor if extraction fails
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the features and label.
        """
        return {
            'features': self.extract_features(self.urls[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def generate_dataset() -> pd.DataFrame:
    """
    Generate a dataset of URLs with categories and labels.
    
    Returns:
        pd.DataFrame: A DataFrame containing URLs, categories, labels, and metadata.
    """
    categories = {
        'education': [
            ('harvard.edu', 'education'), ('mit.edu', 'education'),
            ('stanford.edu', 'education'), ('coursera.org', 'education'),
            ('udemy.com', 'education')
        ],
        'news': [
            ('news.com', 'news'), ('reuters.com', 'news'),
            ('bbc.com', 'news'), ('cnn.com', 'news')
        ],
        'inappropriate': [
            ('dating123.com', 'inappropriate'), ('18plus.com', 'inappropriate'),
            ('adultcontent.com', 'inappropriate'), ('xxx-site.com', 'inappropriate'),
            ('adult-dating.com', 'inappropriate'), ('adult-content.com', 'inappropriate')
        ],
        'gambling': [
            ('casino.com', 'gambling'), ('betting.com', 'gambling'),
            ('poker.com', 'gambling'), ('slots.com', 'gambling')
        ],
        'malware': [
            ('free-hack.com', 'malware'), ('malware.xyz', 'malware'),
            ('virus.site', 'malware'), ('trojan.info', 'malware')
        ]
    }
    
    all_urls = []
    for category, domains in categories.items():
        for domain, label in domains:
            # Add the main domain URL
            url = f"https://www.{domain}"
            all_urls.append({
                'url': url,
                'category': category,
                'label': label
            })
            
            # Generate random subdomains and paths
            for _ in range(50):
                subdomain = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=5))
                path = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=8))
                url = f"https://{subdomain}.{domain}/{path}"
                all_urls.append({
                    'url': url,
                    'category': category,
                    'label': label
                })
    
    # Create a DataFrame
    df = pd.DataFrame(all_urls)
    df['timestamp'] = pd.Timestamp.now()
    df['is_blocked'] = df['category'].isin(['inappropriate', 'gambling', 'malware'])
    df['risk_level'] = df['category'].map({
        'education': 'low',
        'news': 'low',
        'inappropriate': 'high',
        'gambling': 'high',
        'malware': 'high'
    })
    
    logging.info(f"Generated dataset with {len(df)} URLs.")
    return df