"""
Comprehensive Test Suite for AI-Powered Fraud Detection System
This module contains unit tests and integration tests for all system components.
"""

import unittest
import json
import os
from simple_demo import SimpleFraudDetector

class TestFraudDetectionSystem(unittest.TestCase):
    """
    Test cases for the fraud detection system.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.detector = SimpleFraudDetector()
    
    def test_genuine_job_posting_detection(self):
        """Test detection of genuine job postings."""
        genuine_jobs = [
            {
                'text': 'Software Engineer position at Google. Competitive salary and benefits.',
                'company': 'Google',
                'expected': 'genuine'
            },
            {
                'text': 'Data Scientist role at Microsoft. PhD required. Remote work available.',
                'company': 'Microsoft',
                'expected': 'genuine'
            },
            {
                'text': 'Marketing Manager at Apple. 5+ years experience required.',
                'company': 'Apple',
                'expected': 'genuine'
            }
        ]
        
        for job in genuine_jobs:
            with self.subTest(job=job['text'][:50]):
                result = self.detector.analyze_job_posting(job['text'], job['company'])
                self.assertEqual(result['classification'], job['expected'])
                self.assertGreater(result['confidence_score'], 0.5)
    
    def test_fake_job_posting_detection(self):
        """Test detection of fake job postings."""
        fake_jobs = [
            {
                'text': 'URGENT! Make $5000 per week! No experience needed! Send processing fee!',
                'company': 'ScamCorp',
                'expected': 'fake'
            },
            {
                'text': 'Easy money! Work 2 hours daily! Send bank details immediately!',
                'company': 'QuickCash',
                'expected': 'fake'
            },
            {
                'text': 'Government lottery winner! Claim $50000! Send fee now!',
                'company': 'FakeLottery',
                'expected': 'fake'
            }
        ]
        
        for job in fake_jobs:
            with self.subTest(job=job['text'][:50]):
                result = self.detector.analyze_job_posting(job['text'], job['company'])
                self.assertEqual(result['classification'], job['expected'])
                self.assertGreater(result['confidence_score'], 0.5)
    
    def test_genuine_certificate_detection(self):
        """Test detection of genuine certificates."""
        genuine_certs = [
            {
                'student_name': 'John Smith',
                'institution': 'Harvard',
                'degree': 'Bachelor of Science',
                'date': 'May 2023',
                'expected': 'genuine'
            },
            {
                'student_name': 'Jane Doe',
                'institution': 'Stanford',
                'degree': 'Master of Science',
                'date': 'June 2022',
                'expected': 'genuine'
            }
        ]
        
        for cert in genuine_certs:
            with self.subTest(cert=cert['student_name']):
                result = self.detector.analyze_certificate(cert)
                self.assertEqual(result['classification'], cert['expected'])
                self.assertGreater(result['confidence_score'], 0.5)
    
    def test_fake_certificate_detection(self):
        """Test detection of fake certificates."""
        fake_certs = [
            {
                'student_name': 'Fake Person',
                'institution': 'Fake University',
                'degree': 'Fake Degree',
                'date': 'December 2025',  # Future date
                'expected': 'fake'
            },
            {
                'student_name': '',  # Missing name
                'institution': 'Unknown College',
                'degree': '',  # Missing degree
                'date': 'Invalid Date',
                'expected': 'fake'
            }
        ]
        
        for cert in fake_certs:
            with self.subTest(cert=cert.get('student_name', 'Unknown')):
                result = self.detector.analyze_certificate(cert)
                self.assertEqual(result['classification'], cert['expected'])
                self.assertGreater(result['confidence_score'], 0.3)
    
    def test_confidence_score_range(self):
        """Test that confidence scores are within valid range [0, 1]."""
        test_cases = [
            ('Genuine job at Google', 'Google'),
            ('URGENT SCAM! Send money now!', 'ScamCorp')
        ]
        
        for text, company in test_cases:
            result = self.detector.analyze_job_posting(text, company)
            self.assertGreaterEqual(result['confidence_score'], 0.0)
            self.assertLessEqual(result['confidence_score'], 1.0)
    
    def test_system_statistics(self):
        """Test system statistics calculation."""
        # Perform some analyses
        self.detector.analyze_job_posting('Genuine job at Google', 'Google')
        self.detector.analyze_job_posting('SCAM! Send money!', 'ScamCorp')
        
        stats = self.detector.get_summary_statistics()
        
        self.assertIn('total_analyses', stats)
        self.assertIn('genuine_count', stats)
        self.assertIn('fake_count', stats)
        self.assertIn('fraud_detection_rate', stats)
        self.assertIn('average_confidence', stats)
        
        self.assertEqual(stats['total_analyses'], 2)
        self.assertEqual(stats['genuine_count'] + stats['fake_count'], stats['total_analyses'])

class TestPerformanceMetrics(unittest.TestCase):
    """
    Test cases for performance evaluation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SimpleFraudDetector()
    
    def test_processing_speed(self):
        """Test processing speed for job postings."""
        import time
        
        test_text = "Software Engineer position at Google. Competitive salary."
        
        start_time = time.time()
        for _ in range(100):
            self.detector.analyze_job_posting(test_text, 'Google')
        end_time = time.time()
        
        processing_time = (end_time - start_time) / 100
        
        # Should process each job posting in less than 0.1 seconds
        self.assertLess(processing_time, 0.1)
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple items
        for i in range(50):
            self.detector.analyze_job_posting(f'Test job posting {i}', 'TestCorp')
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        self.assertLess(memory_increase, 50 * 1024 * 1024)

def run_comprehensive_tests():
    """
    Run comprehensive tests and generate a detailed report.
    """
    print("=== Running Comprehensive Test Suite ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFraudDetectionSystem))
    test_suite.addTest(unittest.makeSuite(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate test report
    test_report = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        'test_details': {
            'failures': [{'test': str(test), 'error': error} for test, error in result.failures],
            'errors': [{'test': str(test), 'error': error} for test, error in result.errors]
        }
    }
    
    # Save test report
    with open('test_report.json', 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n=== Test Summary ===")
    print(f"Total Tests: {test_report['total_tests']}")
    print(f"Failures: {test_report['failures']}")
    print(f"Errors: {test_report['errors']}")
    print(f"Success Rate: {test_report['success_rate']:.2%}")
    print(f"Test report saved to: test_report.json")
    
    return test_report

if __name__ == "__main__":
    run_comprehensive_tests()

