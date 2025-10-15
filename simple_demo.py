"""
Simplified Demo for AI-Powered Fraud Detection System
This version demonstrates the core functionality without heavy dependencies.
"""

import re
import json
from datetime import datetime

class SimpleFraudDetector:
    """
    Simplified fraud detection system for demonstration purposes.
    """
    
    def __init__(self):
        self.fraud_keywords = {
            'urgency': ['urgent', 'immediate', 'asap', 'limited time', 'act fast'],
            'money': ['easy money', 'quick cash', 'guaranteed income', 'make money fast'],
            'suspicious': ['send money', 'processing fee', 'bank details', 'personal information'],
            'unrealistic': ['earn $5000 per week', 'no work required', 'get rich quick']
        }
        
        self.legitimate_companies = ['google', 'microsoft', 'apple', 'amazon', 'facebook']
        self.analysis_results = []
    
    def analyze_job_posting(self, text, company_name=None):
        """
        Analyze job posting for fraud indicators.
        """
        text_lower = text.lower()
        fraud_score = 0
        detected_issues = []
        
        # Check for fraud keywords
        for category, keywords in self.fraud_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                fraud_score += count * 0.3
                detected_issues.append(f"{category.title()} keywords detected: {count}")
        
        # Check company legitimacy
        if company_name and company_name.lower() not in self.legitimate_companies:
            fraud_score += 0.4
            detected_issues.append("Company not in verified database")
        
        # Check for excessive punctuation (common in scams)
        exclamation_count = text.count('!')
        if exclamation_count > 5:
            fraud_score += 0.2
            detected_issues.append(f"Excessive exclamation marks: {exclamation_count}")
        
        # Check for email patterns
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        if email_count > 2:
            fraud_score += 0.2
            detected_issues.append(f"Multiple email addresses: {email_count}")
        
        # Determine classification
        is_fake = fraud_score > 1.0
        confidence = min(0.95, fraud_score / 2.0) if is_fake else max(0.05, 1.0 - fraud_score / 2.0)
        
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'company_name': company_name,
            'classification': 'fake' if is_fake else 'genuine',
            'confidence_score': confidence,
            'fraud_score': fraud_score,
            'detected_issues': detected_issues,
            'timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results.append(result)
        return result
    
    def analyze_certificate(self, certificate_info):
        """
        Simulate certificate analysis.
        """
        issues = []
        fraud_score = 0
        
        # Check for missing information
        required_fields = ['student_name', 'institution', 'degree', 'date']
        missing_fields = [field for field in required_fields if not certificate_info.get(field)]
        
        if missing_fields:
            fraud_score += len(missing_fields) * 0.3
            issues.append(f"Missing fields: {', '.join(missing_fields)}")
        
        # Check institution
        institution = certificate_info.get('institution', '').lower()
        if institution and institution not in ['harvard', 'stanford', 'mit', 'berkeley']:
            fraud_score += 0.4
            issues.append("Institution not in verified database")
        
        # Check date validity
        cert_date = certificate_info.get('date', '')
        if cert_date and '2025' in cert_date:  # Future date
            fraud_score += 0.5
            issues.append("Future graduation date detected")
        
        is_fake = fraud_score > 0.8
        confidence = min(0.95, fraud_score / 1.5) if is_fake else max(0.05, 1.0 - fraud_score / 1.5)
        
        result = {
            'certificate_info': certificate_info,
            'classification': 'fake' if is_fake else 'genuine',
            'confidence_score': confidence,
            'fraud_score': fraud_score,
            'detected_issues': issues,
            'timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results.append(result)
        return result
    
    def get_summary_statistics(self):
        """
        Generate summary statistics of all analyses.
        """
        if not self.analysis_results:
            return {'message': 'No analyses performed'}
        
        total = len(self.analysis_results)
        fake_count = sum(1 for r in self.analysis_results if r['classification'] == 'fake')
        genuine_count = total - fake_count
        avg_confidence = sum(r['confidence_score'] for r in self.analysis_results) / total
        
        return {
            'total_analyses': total,
            'genuine_count': genuine_count,
            'fake_count': fake_count,
            'fraud_detection_rate': fake_count / total,
            'average_confidence': avg_confidence
        }

def run_demonstration():
    """
    Run a comprehensive demonstration of the fraud detection system.
    """
    print("=== AI-Powered Fraud Detection System Demo ===\n")
    
    detector = SimpleFraudDetector()
    
    # Test job postings
    print("1. JOB POSTING ANALYSIS")
    print("=" * 50)
    
    # Genuine job posting
    genuine_job = """
    Senior Software Engineer - Google
    We are looking for an experienced software engineer to join our team in Mountain View.
    Requirements: Bachelor's degree in CS, 5+ years experience, proficiency in Python/Java.
    Competitive salary and comprehensive benefits package.
    Apply through our official careers portal.
    """
    
    result1 = detector.analyze_job_posting(genuine_job, "Google")
    print("GENUINE JOB POSTING:")
    print(f"Classification: {result1['classification']}")
    print(f"Confidence: {result1['confidence_score']:.2f}")
    print(f"Issues detected: {len(result1['detected_issues'])}")
    print()
    
    # Fake job posting
    fake_job = """
    URGENT! WORK FROM HOME OPPORTUNITY!
    Make $5000 per week with just 2 hours of work daily!
    No experience needed! No interviews required!
    Send $200 processing fee to secure your position immediately!
    Contact: quickmoney@gmail.com or easycash@yahoo.com
    LIMITED TIME OFFER! ACT FAST! ONLY 5 SPOTS LEFT!
    """
    
    result2 = detector.analyze_job_posting(fake_job, "QuickMoney Inc")
    print("FAKE JOB POSTING:")
    print(f"Classification: {result2['classification']}")
    print(f"Confidence: {result2['confidence_score']:.2f}")
    print(f"Issues detected: {len(result2['detected_issues'])}")
    for issue in result2['detected_issues']:
        print(f"  - {issue}")
    print()
    
    # Test certificates
    print("2. CERTIFICATE ANALYSIS")
    print("=" * 50)
    
    # Genuine certificate
    genuine_cert = {
        'student_name': 'John Smith',
        'institution': 'Harvard',
        'degree': 'Bachelor of Science in Computer Science',
        'date': 'May 2023'
    }
    
    result3 = detector.analyze_certificate(genuine_cert)
    print("GENUINE CERTIFICATE:")
    print(f"Classification: {result3['classification']}")
    print(f"Confidence: {result3['confidence_score']:.2f}")
    print(f"Issues detected: {len(result3['detected_issues'])}")
    print()
    
    # Fake certificate
    fake_cert = {
        'student_name': 'Jane Doe',
        'institution': 'Fake University',
        'degree': '',  # Missing degree
        'date': 'December 2025'  # Future date
    }
    
    result4 = detector.analyze_certificate(fake_cert)
    print("FAKE CERTIFICATE:")
    print(f"Classification: {result4['classification']}")
    print(f"Confidence: {result4['confidence_score']:.2f}")
    print(f"Issues detected: {len(result4['detected_issues'])}")
    for issue in result4['detected_issues']:
        print(f"  - {issue}")
    print()
    
    # Summary statistics
    print("3. SYSTEM PERFORMANCE SUMMARY")
    print("=" * 50)
    summary = detector.get_summary_statistics()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Export results
    with open('demo_results.json', 'w') as f:
        json.dump({
            'analysis_results': detector.analysis_results,
            'summary_statistics': summary
        }, f, indent=2)
    
    print(f"\nDetailed results exported to: demo_results.json")
    
    return detector.analysis_results, summary

if __name__ == "__main__":
    results, summary = run_demonstration()

