#!/usr/bin/env python3
import pandas as pd
from dataset import generate_dataset, extract_url_features
from training import train_models, ensemble_predict
import joblib
import os

def main():
    # Test dataset generation
    print("Testing dataset generation...")
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"\nSample data:")
    print(df.head())
    print(f"\nFeature columns:")
    print(df.columns.tolist())

    # Test model training
    print("\nTesting model training...")
    models = train_models()
    if models:
        for model_name, data in models.items():
            print(f"\n{model_name.upper()} Model Performance:")
            metrics = data['metrics']
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")

    # Test URL feature extraction and prediction
    test_urls = [
        'https://www.educational-site.org/course/python',
        'http://malicious-site.com/download/virus.exe',
        'https://news-portal.com/technology/latest',
        'http://gambling888.net/poker/online',
        'https://adult-content-site.com/xxx',
        'https://legitimate-news.com/article/12345'
    ]

    print("\nTesting URL classification...")
    for url in test_urls:
        # Extract features
        features = extract_url_features(url)
        print(f"\nURL: {url}")
        print("Features:")
        for feature, value in features.items():
            print(f"{feature}: {value}")
        
        # Make prediction
        feature_values = list(features.values())
        is_unsafe, probability, model_predictions = ensemble_predict(feature_values)
        
        print("\nPrediction Results:")
        print(f"Overall Safety Assessment:")
        print(f"Is Unsafe: {is_unsafe}")
        print(f"Unsafe Probability: {probability:.4f}")
        
        if model_predictions:
            print("\nIndividual Model Predictions:")
            for model, pred in model_predictions.items():
                print(f"{model.upper()}: Unsafe Probability = {pred['probability']:.4f}")

if __name__ == "__main__":
    main()
