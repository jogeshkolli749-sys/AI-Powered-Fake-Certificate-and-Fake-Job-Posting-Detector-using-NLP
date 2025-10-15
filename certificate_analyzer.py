"""
Certificate Analysis Module for AI-Powered Fraud Detector
This module handles OCR text extraction and forgery detection for certificates.
"""

import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
from datetime import datetime
import json

class CertificateAnalyzer:
    """
    A comprehensive certificate analysis class that combines OCR and CNN-based
    forgery detection to verify certificate authenticity.
    """
    
    def __init__(self):
        self.ocr_config = r'--oem 3 --psm 6'
        self.forgery_model = None
        self.trusted_institutions = self._load_trusted_institutions()
    
    def _load_trusted_institutions(self):
        """
        Load a database of trusted educational institutions.
        In a real implementation, this would connect to an actual database.
        """
        return {
            'harvard university': {
                'established': 1636,
                'location': 'cambridge, ma',
                'valid_degrees': ['bachelor', 'master', 'phd', 'md', 'jd']
            },
            'stanford university': {
                'established': 1885,
                'location': 'stanford, ca',
                'valid_degrees': ['bachelor', 'master', 'phd', 'md']
            },
            'mit': {
                'established': 1861,
                'location': 'cambridge, ma',
                'valid_degrees': ['bachelor', 'master', 'phd']
            }
        }
    
    def extract_text_from_certificate(self, image_path):
        """
        Extract text from certificate image using OCR.
        
        Args:
            image_path (str): Path to the certificate image
            
        Returns:
            dict: Extracted information including text, name, institution, etc.
        """
        try:
            # Read and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancement techniques
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(thresh, config=self.ocr_config)
            
            # Parse the extracted text
            parsed_info = self._parse_certificate_text(extracted_text)
            
            return {
                'raw_text': extracted_text,
                'parsed_info': parsed_info,
                'image_path': image_path
            }
        
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return None
    
    def _parse_certificate_text(self, text):
        """
        Parse extracted text to identify key information.
        
        Args:
            text (str): Raw extracted text from OCR
            
        Returns:
            dict: Parsed information
        """
        parsed_info = {
            'student_name': None,
            'institution': None,
            'degree': None,
            'graduation_date': None,
            'gpa': None
        }
        
        # Clean the text
        text = text.strip().lower()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract student name (usually appears after "this certifies that" or similar)
        name_patterns = [
            r'this certifies that\s+([a-zA-Z\s]+)',
            r'awarded to\s+([a-zA-Z\s]+)',
            r'presented to\s+([a-zA-Z\s]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parsed_info['student_name'] = match.group(1).strip().title()
                break
        
        # Extract institution name
        for institution in self.trusted_institutions.keys():
            if institution in text:
                parsed_info['institution'] = institution.title()
                break
        
        # Extract degree information
        degree_patterns = [
            r'bachelor of\s+([a-zA-Z\s]+)',
            r'master of\s+([a-zA-Z\s]+)',
            r'doctor of\s+([a-zA-Z\s]+)',
            r'phd in\s+([a-zA-Z\s]+)'
        ]
        
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parsed_info['degree'] = match.group(0).strip().title()
                break
        
        # Extract graduation date
        date_patterns = [
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parsed_info['graduation_date'] = match.group(0).strip()
                break
        
        return parsed_info
    
    def build_forgery_detection_model(self, input_shape=(224, 224, 3)):
        """
        Build a CNN model for detecting forged certificates.
        
        Args:
            input_shape (tuple): Input image shape
            
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification: genuine (0) or fake (1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.forgery_model = model
        return model
    
    def verify_certificate_authenticity(self, extracted_info):
        """
        Verify certificate authenticity based on extracted information.
        
        Args:
            extracted_info (dict): Information extracted from the certificate
            
        Returns:
            dict: Verification results with confidence score
        """
        verification_results = {
            'is_genuine': True,
            'confidence_score': 1.0,
            'issues_found': [],
            'verification_details': {}
        }
        
        parsed_info = extracted_info.get('parsed_info', {})
        
        # Check if institution is in trusted database
        institution = parsed_info.get('institution', '').lower()
        if institution and institution in self.trusted_institutions:
            verification_results['verification_details']['institution_verified'] = True
        elif institution:
            verification_results['is_genuine'] = False
            verification_results['confidence_score'] -= 0.3
            verification_results['issues_found'].append('Institution not in trusted database')
        
        # Check date validity
        graduation_date = parsed_info.get('graduation_date')
        if graduation_date:
            try:
                # Simple date validation (this would be more sophisticated in practice)
                current_year = datetime.now().year
                if any(str(year) in graduation_date for year in range(current_year + 1, current_year + 10)):
                    verification_results['is_genuine'] = False
                    verification_results['confidence_score'] -= 0.4
                    verification_results['issues_found'].append('Future graduation date detected')
            except:
                verification_results['confidence_score'] -= 0.1
                verification_results['issues_found'].append('Invalid date format')
        
        # Check for missing critical information
        critical_fields = ['student_name', 'institution', 'degree']
        missing_fields = [field for field in critical_fields if not parsed_info.get(field)]
        
        if missing_fields:
            verification_results['confidence_score'] -= 0.2 * len(missing_fields)
            verification_results['issues_found'].append(f'Missing critical information: {", ".join(missing_fields)}')
        
        # Ensure confidence score doesn't go below 0
        verification_results['confidence_score'] = max(0, verification_results['confidence_score'])
        
        # Final determination
        if verification_results['confidence_score'] < 0.5:
            verification_results['is_genuine'] = False
        
        return verification_results
    
    def analyze_certificate(self, image_path):
        """
        Complete certificate analysis pipeline.
        
        Args:
            image_path (str): Path to the certificate image
            
        Returns:
            dict: Complete analysis results
        """
        # Extract text and information
        extracted_info = self.extract_text_from_certificate(image_path)
        
        if not extracted_info:
            return {
                'error': 'Failed to extract information from certificate',
                'is_genuine': False,
                'confidence_score': 0.0
            }
        
        # Verify authenticity
        verification_results = self.verify_certificate_authenticity(extracted_info)
        
        # Combine results
        analysis_results = {
            'extracted_info': extracted_info,
            'verification_results': verification_results,
            'final_classification': 'genuine' if verification_results['is_genuine'] else 'fake',
            'confidence_score': verification_results['confidence_score']
        }
        
        return analysis_results

# Example usage
if __name__ == "__main__":
    analyzer = CertificateAnalyzer()
    
    # Build the forgery detection model
    model = analyzer.build_forgery_detection_model()
    print("Forgery detection model built successfully!")
    print(f"Model summary:")
    model.summary()
    
    # Example of certificate analysis (would require actual certificate image)
    print("\nCertificate Analyzer initialized and ready for use.")
    print("To analyze a certificate, call: analyzer.analyze_certificate('path/to/certificate.jpg')")

