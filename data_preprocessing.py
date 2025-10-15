"""
Data Preprocessing Module for AI-Powered Fraud Detector
This module handles data collection, cleaning, and preprocessing for both
certificate and job posting datasets.
"""

import pandas as pd
import numpy as np
import cv2
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for handling both certificate images
    and job posting text data.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
    
    def preprocess_certificate_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess certificate images for CNN model training.
        
        Args:
            image_path (str): Path to the certificate image
            target_size (tuple): Target size for resizing the image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize the image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
        
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def preprocess_job_posting_text(self, text):
        """
        Preprocess job posting text for NLP model training.
        
        Args:
            text (str): Raw job posting text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Join tokens back to text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def create_synthetic_datasets(self):
        """
        Create synthetic datasets for demonstration purposes.
        In a real-world scenario, these would be replaced with actual datasets.
        """
        # Synthetic job posting data
        job_postings = [
            {
                'text': 'Software Engineer position at Google. Competitive salary, great benefits. Apply now!',
                'label': 'genuine'
            },
            {
                'text': 'URGENT! Make $5000 per week working from home! No experience needed! Send $100 processing fee!',
                'label': 'fake'
            },
            {
                'text': 'Data Scientist role at Microsoft. PhD in Computer Science required. Remote work available.',
                'label': 'genuine'
            },
            {
                'text': 'Easy money! Work 2 hours daily, earn $3000 monthly! Send personal details and bank info!',
                'label': 'fake'
            },
            {
                'text': 'Marketing Manager position at Apple. 5+ years experience required. Excellent growth opportunities.',
                'label': 'genuine'
            },
            {
                'text': 'Government job lottery winner! Claim your $50,000 prize! Send processing fee immediately!',
                'label': 'fake'
            }
        ]
        
        # Create DataFrame
        job_df = pd.DataFrame(job_postings)
        
        # Preprocess text
        job_df['preprocessed_text'] = job_df['text'].apply(self.preprocess_job_posting_text)
        
        # Encode labels
        job_df['label_encoded'] = self.label_encoder.fit_transform(job_df['label'])
        
        return job_df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Labels
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Create and preprocess synthetic job posting data
    job_data = preprocessor.create_synthetic_datasets()
    print("Job Posting Dataset:")
    print(job_data.head())
    print(f"\nDataset shape: {job_data.shape}")
    
    # Split the data
    X = job_data['preprocessed_text']
    y = job_data['label_encoded']
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

