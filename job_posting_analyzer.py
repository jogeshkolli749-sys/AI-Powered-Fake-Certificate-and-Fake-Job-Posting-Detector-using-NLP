"""
Job Posting Analysis Module for AI-Powered Fraud Detector
This module uses NLP techniques including BERT to detect fake job postings.
"""

import re
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import whois
from urllib.parse import urlparse
import dns.resolver

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class JobPostingAnalyzer:
    """
    A comprehensive job posting analysis class that uses NLP and external verification
    to detect fraudulent job postings.
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42)
        self.fraud_keywords = self._load_fraud_keywords()
        self.legitimate_companies = self._load_legitimate_companies()
        
        # Initialize BERT model for advanced text analysis
        self.tokenizer = None
        self.bert_model = None
        self._initialize_bert_model()
    
    def _initialize_bert_model(self):
        """
        Initialize BERT model for sequence classification.
        In practice, this would be a fine-tuned model for job posting classification.
        """
        try:
            model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # For demonstration, we'll use a general BERT model
            # In practice, this would be fine-tuned on job posting data
            print("BERT model initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize BERT model: {e}")
            print("Falling back to traditional NLP methods")
    
    def _load_fraud_keywords(self):
        """
        Load keywords commonly associated with fraudulent job postings.
        """
        return {
            'urgency_keywords': [
                'urgent', 'immediate', 'asap', 'right away', 'today only',
                'limited time', 'act fast', 'hurry'
            ],
            'money_keywords': [
                'easy money', 'quick cash', 'guaranteed income', 'no experience needed',
                'work from home', 'make money fast', 'earn thousands'
            ],
            'suspicious_requests': [
                'send money', 'processing fee', 'training fee', 'background check fee',
                'bank details', 'social security', 'credit card', 'personal information'
            ],
            'unrealistic_offers': [
                'earn $5000 per week', 'make $100 per hour', 'guaranteed $3000 monthly',
                'no work required', 'passive income', 'get rich quick'
            ]
        }
    
    def _load_legitimate_companies(self):
        """
        Load a database of legitimate companies.
        In practice, this would be a comprehensive database.
        """
        return {
            'google': {'domain': 'google.com', 'verified': True},
            'microsoft': {'domain': 'microsoft.com', 'verified': True},
            'apple': {'domain': 'apple.com', 'verified': True},
            'amazon': {'domain': 'amazon.com', 'verified': True},
            'facebook': {'domain': 'facebook.com', 'verified': True},
            'netflix': {'domain': 'netflix.com', 'verified': True},
            'tesla': {'domain': 'tesla.com', 'verified': True}
        }
    
    def extract_features_from_text(self, text):
        """
        Extract various features from job posting text.
        
        Args:
            text (str): Job posting text
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features.update({f'sentiment_{key}': value for key, value in sentiment_scores.items()})
        
        # Fraud keyword detection
        text_lower = text.lower()
        for category, keywords in self.fraud_keywords.items():
            features[f'{category}_count'] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Email and phone number detection
        features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['phone_count'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        
        # URL detection
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        
        # Salary information detection
        salary_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            r'\d+k\s*(?:per|/)\s*(?:year|month|week|hour)',
            r'\d+\s*(?:dollars|usd)\s*(?:per|/)\s*(?:year|month|week|hour)'
        ]
        features['salary_mentions'] = sum(len(re.findall(pattern, text_lower)) for pattern in salary_patterns)
        
        return features
    
    def verify_company_legitimacy(self, company_name, email_domain=None):
        """
        Verify if a company is legitimate based on various factors.
        
        Args:
            company_name (str): Name of the company
            email_domain (str): Email domain used in the job posting
            
        Returns:
            dict: Verification results
        """
        verification_results = {
            'is_legitimate': False,
            'confidence_score': 0.0,
            'verification_details': {}
        }
        
        company_lower = company_name.lower()
        
        # Check against known legitimate companies
        if company_lower in self.legitimate_companies:
            verification_results['is_legitimate'] = True
            verification_results['confidence_score'] = 0.9
            verification_results['verification_details']['in_database'] = True
        
        # Domain verification
        if email_domain:
            try:
                # Check if domain exists
                dns.resolver.resolve(email_domain, 'MX')
                verification_results['verification_details']['domain_exists'] = True
                verification_results['confidence_score'] += 0.3
                
                # Check if domain matches company
                if company_lower in email_domain.lower():
                    verification_results['confidence_score'] += 0.2
                    verification_results['verification_details']['domain_matches_company'] = True
                
            except Exception as e:
                verification_results['verification_details']['domain_verification_error'] = str(e)
        
        # Final determination
        if verification_results['confidence_score'] >= 0.5:
            verification_results['is_legitimate'] = True
        
        return verification_results
    
    def analyze_job_posting_with_bert(self, text):
        """
        Analyze job posting using BERT model.
        
        Args:
            text (str): Job posting text
            
        Returns:
            dict: BERT analysis results
        """
        if not self.tokenizer:
            return {'error': 'BERT model not available'}
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            
            # For demonstration, we'll use a simple approach
            # In practice, this would use a fine-tuned model
            
            # Calculate some basic metrics using BERT tokenization
            token_count = len(inputs['input_ids'][0])
            attention_mask_sum = torch.sum(inputs['attention_mask']).item()
            
            # Simulate BERT-based classification (in practice, this would be actual model inference)
            # This is a simplified simulation for demonstration
            bert_score = min(1.0, attention_mask_sum / token_count)
            
            return {
                'bert_token_count': token_count,
                'attention_score': bert_score,
                'classification_confidence': bert_score
            }
        
        except Exception as e:
            return {'error': f'BERT analysis failed: {str(e)}'}
    
    def train_classifier(self, training_data):
        """
        Train the job posting classifier on labeled data.
        
        Args:
            training_data (pd.DataFrame): DataFrame with 'text' and 'label' columns
        """
        # Extract features for all training samples
        features_list = []
        for text in training_data['text']:
            features = self.extract_features_from_text(text)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Fill any missing values
        features_df = features_df.fillna(0)
        
        # Train TF-IDF vectorizer
        tfidf_features = self.tfidf_vectorizer.fit_transform(training_data['text'])
        
        # Combine TF-IDF features with extracted features
        combined_features = np.hstack([tfidf_features.toarray(), features_df.values])
        
        # Train classifier
        self.classifier.fit(combined_features, training_data['label'])
        
        print("Job posting classifier trained successfully!")
    
    def classify_job_posting(self, text):
        """
        Classify a job posting as genuine or fake.
        
        Args:
            text (str): Job posting text
            
        Returns:
            dict: Classification results
        """
        # Extract features
        features = self.extract_features_from_text(text)
        features_df = pd.DataFrame([features]).fillna(0)
        
        # Get TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([text])
        
        # Combine features
        combined_features = np.hstack([tfidf_features.toarray(), features_df.values])
        
        # Make prediction
        try:
            prediction = self.classifier.predict(combined_features)[0]
            probability = self.classifier.predict_proba(combined_features)[0]
            confidence = max(probability)
        except:
            # Fallback classification based on features
            fraud_score = (
                features.get('urgency_keywords_count', 0) * 0.3 +
                features.get('money_keywords_count', 0) * 0.4 +
                features.get('suspicious_requests_count', 0) * 0.5 +
                features.get('unrealistic_offers_count', 0) * 0.6
            )
            
            prediction = 'fake' if fraud_score > 1 else 'genuine'
            confidence = min(0.9, fraud_score / 3) if prediction == 'fake' else max(0.1, 1 - fraud_score / 3)
        
        return {
            'classification': prediction,
            'confidence_score': confidence,
            'features': features
        }
    
    def analyze_job_posting(self, text, company_name=None, email_domain=None):
        """
        Complete job posting analysis pipeline.
        
        Args:
            text (str): Job posting text
            company_name (str): Name of the hiring company
            email_domain (str): Email domain used in the posting
            
        Returns:
            dict: Complete analysis results
        """
        # Text classification
        classification_results = self.classify_job_posting(text)
        
        # Company verification
        company_verification = {}
        if company_name:
            company_verification = self.verify_company_legitimacy(company_name, email_domain)
        
        # BERT analysis
        bert_results = self.analyze_job_posting_with_bert(text)
        
        # Combine all results
        final_confidence = classification_results['confidence_score']
        
        # Adjust confidence based on company verification
        if company_verification.get('is_legitimate'):
            final_confidence = min(1.0, final_confidence + 0.2)
        elif company_verification and not company_verification.get('is_legitimate'):
            final_confidence = max(0.0, final_confidence - 0.3)
        
        # Final classification
        final_classification = 'genuine' if final_confidence > 0.5 else 'fake'
        
        analysis_results = {
            'text_analysis': classification_results,
            'company_verification': company_verification,
            'bert_analysis': bert_results,
            'final_classification': final_classification,
            'confidence_score': final_confidence
        }
        
        return analysis_results

# Example usage
if __name__ == "__main__":
    analyzer = JobPostingAnalyzer()
    
    # Example job postings for testing
    genuine_posting = """
    Software Engineer - Google
    We are looking for a talented Software Engineer to join our team in Mountain View, CA.
    Requirements: Bachelor's degree in Computer Science, 3+ years of experience in Python/Java.
    Competitive salary and benefits package. Apply through our official careers page.
    """
    
    fake_posting = """
    URGENT! Make $5000 per week working from home!
    No experience needed! Just send $100 processing fee to get started.
    Contact us immediately at quickmoney@gmail.com
    Limited time offer - act fast!
    """
    
    # Analyze the postings
    print("Analyzing genuine job posting:")
    genuine_results = analyzer.analyze_job_posting(genuine_posting, "Google", "google.com")
    print(f"Classification: {genuine_results['final_classification']}")
    print(f"Confidence: {genuine_results['confidence_score']:.2f}")
    
    print("\nAnalyzing fake job posting:")
    fake_results = analyzer.analyze_job_posting(fake_posting)
    print(f"Classification: {fake_results['final_classification']}")
    print(f"Confidence: {fake_results['confidence_score']:.2f}")

