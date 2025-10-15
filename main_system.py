"""
Main System Integration Module for AI-Powered Fraud Detector
This module integrates all components and provides a unified interface.
"""

import os
import json
from datetime import datetime
from data_preprocessing import DataPreprocessor
from certificate_analyzer import CertificateAnalyzer
from job_posting_analyzer import JobPostingAnalyzer

class FraudDetectionSystem:
    """
    Main system class that integrates certificate and job posting analysis.
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.certificate_analyzer = CertificateAnalyzer()
        self.job_posting_analyzer = JobPostingAnalyzer()
        self.analysis_history = []
    
    def analyze_certificate(self, image_path):
        """
        Analyze a certificate for authenticity.
        
        Args:
            image_path (str): Path to the certificate image
            
        Returns:
            dict: Analysis results
        """
        try:
            # Perform certificate analysis
            results = self.certificate_analyzer.analyze_certificate(image_path)
            
            # Add metadata
            results['analysis_type'] = 'certificate'
            results['timestamp'] = datetime.now().isoformat()
            results['input_file'] = image_path
            
            # Store in history
            self.analysis_history.append(results)
            
            return results
        
        except Exception as e:
            error_result = {
                'error': str(e),
                'analysis_type': 'certificate',
                'timestamp': datetime.now().isoformat(),
                'input_file': image_path,
                'final_classification': 'error',
                'confidence_score': 0.0
            }
            self.analysis_history.append(error_result)
            return error_result
    
    def analyze_job_posting(self, text, company_name=None, email_domain=None, url=None):
        """
        Analyze a job posting for fraud.
        
        Args:
            text (str): Job posting text
            company_name (str): Name of the hiring company
            email_domain (str): Email domain used in the posting
            url (str): URL of the job posting
            
        Returns:
            dict: Analysis results
        """
        try:
            # Perform job posting analysis
            results = self.job_posting_analyzer.analyze_job_posting(
                text, company_name, email_domain
            )
            
            # Add metadata
            results['analysis_type'] = 'job_posting'
            results['timestamp'] = datetime.now().isoformat()
            results['input_text'] = text[:200] + '...' if len(text) > 200 else text
            results['company_name'] = company_name
            results['url'] = url
            
            # Store in history
            self.analysis_history.append(results)
            
            return results
        
        except Exception as e:
            error_result = {
                'error': str(e),
                'analysis_type': 'job_posting',
                'timestamp': datetime.now().isoformat(),
                'input_text': text[:200] + '...' if len(text) > 200 else text,
                'final_classification': 'error',
                'confidence_score': 0.0
            }
            self.analysis_history.append(error_result)
            return error_result
    
    def get_analysis_summary(self):
        """
        Get a summary of all analyses performed.
        
        Returns:
            dict: Summary statistics
        """
        if not self.analysis_history:
            return {'message': 'No analyses performed yet'}
        
        total_analyses = len(self.analysis_history)
        certificate_analyses = sum(1 for a in self.analysis_history if a.get('analysis_type') == 'certificate')
        job_posting_analyses = sum(1 for a in self.analysis_history if a.get('analysis_type') == 'job_posting')
        
        genuine_count = sum(1 for a in self.analysis_history if a.get('final_classification') == 'genuine')
        fake_count = sum(1 for a in self.analysis_history if a.get('final_classification') == 'fake')
        error_count = sum(1 for a in self.analysis_history if a.get('final_classification') == 'error')
        
        avg_confidence = sum(a.get('confidence_score', 0) for a in self.analysis_history) / total_analyses
        
        return {
            'total_analyses': total_analyses,
            'certificate_analyses': certificate_analyses,
            'job_posting_analyses': job_posting_analyses,
            'genuine_count': genuine_count,
            'fake_count': fake_count,
            'error_count': error_count,
            'average_confidence': avg_confidence,
            'fraud_detection_rate': fake_count / max(1, total_analyses - error_count)
        }
    
    def export_results(self, output_file):
        """
        Export analysis results to a JSON file.
        
        Args:
            output_file (str): Path to the output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'analysis_history': self.analysis_history,
                    'summary': self.get_analysis_summary()
                }, f, indent=2)
            print(f"Results exported to {output_file}")
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
    
    def train_system(self, training_data_path=None):
        """
        Train the system components with provided data.
        
        Args:
            training_data_path (str): Path to training data
        """
        try:
            # Create synthetic training data for demonstration
            job_posting_data = self.preprocessor.create_synthetic_datasets()
            
            # Train the job posting analyzer
            self.job_posting_analyzer.train_classifier(job_posting_data)
            
            # Build the certificate forgery detection model
            self.certificate_analyzer.build_forgery_detection_model()
            
            print("System training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")

def demonstrate_system():
    """
    Demonstrate the fraud detection system with example data.
    """
    print("=== AI-Powered Fraud Detection System Demo ===\n")
    
    # Initialize the system
    system = FraudDetectionSystem()
    
    # Train the system
    print("Training the system...")
    system.train_system()
    print()
    
    # Example job posting analyses
    print("Analyzing job postings:")
    print("-" * 40)
    
    # Genuine job posting
    genuine_job = """
    Senior Data Scientist - Microsoft
    Microsoft is seeking a Senior Data Scientist to join our AI research team in Seattle, WA.
    
    Requirements:
    - PhD in Computer Science, Statistics, or related field
    - 5+ years of experience in machine learning and data analysis
    - Proficiency in Python, R, and SQL
    - Experience with cloud platforms (Azure preferred)
    
    We offer competitive salary, comprehensive benefits, and opportunities for professional growth.
    Apply through our careers portal at careers.microsoft.com
    """
    
    result1 = system.analyze_job_posting(
        genuine_job, 
        company_name="Microsoft", 
        email_domain="microsoft.com"
    )
    
    print(f"Genuine Job Posting Analysis:")
    print(f"Classification: {result1['final_classification']}")
    print(f"Confidence: {result1['confidence_score']:.2f}")
    print()
    
    # Fake job posting
    fake_job = """
    URGENT HIRING! WORK FROM HOME!
    
    Earn $5000 per week with just 2 hours of work daily!
    No experience required! No interviews needed!
    
    Just send $200 processing fee to secure your position.
    Contact: easymoney123@gmail.com
    
    LIMITED TIME OFFER - ONLY 10 SPOTS LEFT!
    Send your bank details and social security number for immediate processing.
    """
    
    result2 = system.analyze_job_posting(fake_job)
    
    print(f"Fake Job Posting Analysis:")
    print(f"Classification: {result2['final_classification']}")
    print(f"Confidence: {result2['confidence_score']:.2f}")
    print()
    
    # Display system summary
    print("System Analysis Summary:")
    print("-" * 40)
    summary = system.get_analysis_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Export results
    system.export_results("analysis_results.json")
    print("\nResults exported to analysis_results.json")

if __name__ == "__main__":
    demonstrate_system()

