"""
Performance Analysis Module for AI-Powered Fraud Detection System
This module generates performance metrics and visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from simple_demo import SimpleFraudDetector
import time

def generate_performance_metrics():
    """
    Generate comprehensive performance metrics for the fraud detection system.
    """
    detector = SimpleFraudDetector()
    
    # Test data for evaluation
    test_cases = [
        # Genuine job postings
        {'text': 'Software Engineer at Google. Competitive salary.', 'company': 'Google', 'label': 'genuine'},
        {'text': 'Data Scientist at Microsoft. PhD required.', 'company': 'Microsoft', 'label': 'genuine'},
        {'text': 'Product Manager at Apple. 5+ years experience.', 'company': 'Apple', 'label': 'genuine'},
        {'text': 'Research Scientist at Amazon. Machine learning focus.', 'company': 'Amazon', 'label': 'genuine'},
        {'text': 'Marketing Manager at Facebook. Digital marketing experience.', 'company': 'Facebook', 'label': 'genuine'},
        
        # Fake job postings
        {'text': 'URGENT! Make $5000 weekly! No experience! Send fee!', 'company': 'ScamCorp', 'label': 'fake'},
        {'text': 'Easy money! Work 2 hours daily! Send bank details!', 'company': 'QuickCash', 'label': 'fake'},
        {'text': 'Government lottery! Claim $50000! Processing fee required!', 'company': 'FakeLottery', 'label': 'fake'},
        {'text': 'Immediate hiring! $100/hour! Send personal information!', 'company': 'InstantMoney', 'label': 'fake'},
        {'text': 'Work from home! Guaranteed income! Pay training fee!', 'company': 'HomeScam', 'label': 'fake'},
    ]
    
    # Certificate test data
    cert_test_cases = [
        # Genuine certificates
        {'student_name': 'John Smith', 'institution': 'Harvard', 'degree': 'BS Computer Science', 'date': 'May 2023', 'label': 'genuine'},
        {'student_name': 'Jane Doe', 'institution': 'Stanford', 'degree': 'MS Data Science', 'date': 'June 2022', 'label': 'genuine'},
        {'student_name': 'Bob Johnson', 'institution': 'MIT', 'degree': 'PhD Physics', 'date': 'December 2021', 'label': 'genuine'},
        
        # Fake certificates
        {'student_name': 'Fake Person', 'institution': 'Fake University', 'degree': 'Fake Degree', 'date': 'December 2025', 'label': 'fake'},
        {'student_name': '', 'institution': 'Unknown College', 'degree': '', 'date': 'Invalid', 'label': 'fake'},
        {'student_name': 'Test User', 'institution': 'Diploma Mill', 'degree': 'Instant PhD', 'date': 'Tomorrow', 'label': 'fake'},
    ]
    
    # Analyze job postings
    job_results = []
    job_processing_times = []
    
    for case in test_cases:
        start_time = time.time()
        result = detector.analyze_job_posting(case['text'], case['company'])
        end_time = time.time()
        
        job_processing_times.append(end_time - start_time)
        job_results.append({
            'predicted': result['classification'],
            'actual': case['label'],
            'confidence': result['confidence_score'],
            'processing_time': end_time - start_time
        })
    
    # Analyze certificates
    cert_results = []
    cert_processing_times = []
    
    for case in cert_test_cases:
        start_time = time.time()
        result = detector.analyze_certificate(case)
        end_time = time.time()
        
        cert_processing_times.append(end_time - start_time)
        cert_results.append({
            'predicted': result['classification'],
            'actual': case['label'],
            'confidence': result['confidence_score'],
            'processing_time': end_time - start_time
        })
    
    # Calculate metrics
    def calculate_metrics(results):
        tp = sum(1 for r in results if r['predicted'] == 'fake' and r['actual'] == 'fake')
        tn = sum(1 for r in results if r['predicted'] == 'genuine' and r['actual'] == 'genuine')
        fp = sum(1 for r in results if r['predicted'] == 'fake' and r['actual'] == 'genuine')
        fn = sum(1 for r in results if r['predicted'] == 'genuine' and r['actual'] == 'fake')
        
        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    job_metrics = calculate_metrics(job_results)
    cert_metrics = calculate_metrics(cert_results)
    
    # Performance metrics
    performance_metrics = {
        'job_posting_analysis': {
            'metrics': job_metrics,
            'avg_processing_time': np.mean(job_processing_times),
            'max_processing_time': np.max(job_processing_times),
            'min_processing_time': np.min(job_processing_times),
            'avg_confidence': np.mean([r['confidence'] for r in job_results])
        },
        'certificate_analysis': {
            'metrics': cert_metrics,
            'avg_processing_time': np.mean(cert_processing_times),
            'max_processing_time': np.max(cert_processing_times),
            'min_processing_time': np.min(cert_processing_times),
            'avg_confidence': np.mean([r['confidence'] for r in cert_results])
        },
        'overall_performance': {
            'total_tests': len(job_results) + len(cert_results),
            'overall_accuracy': (job_metrics['accuracy'] * len(job_results) + cert_metrics['accuracy'] * len(cert_results)) / (len(job_results) + len(cert_results))
        }
    }
    
    return performance_metrics, job_results, cert_results

def create_performance_visualizations(performance_metrics, job_results, cert_results):
    """
    Create visualizations for performance analysis.
    """
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AI-Powered Fraud Detection System - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    categories = ['Job Postings', 'Certificates']
    accuracies = [
        performance_metrics['job_posting_analysis']['metrics']['accuracy'],
        performance_metrics['certificate_analysis']['metrics']['accuracy']
    ]
    
    axes[0, 0].bar(categories, accuracies, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[0, 0].set_title('Classification Accuracy by Category')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 2. Processing Time Comparison
    job_times = [performance_metrics['job_posting_analysis']['avg_processing_time']]
    cert_times = [performance_metrics['certificate_analysis']['avg_processing_time']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, [job_times[0], 0], width, label='Job Postings', color='#2E86AB', alpha=0.8)
    axes[0, 1].bar(x + width/2, [0, cert_times[0]], width, label='Certificates', color='#A23B72', alpha=0.8)
    axes[0, 1].set_title('Average Processing Time')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].legend()
    
    # 3. Confusion Matrix for Job Postings
    job_tp = performance_metrics['job_posting_analysis']['metrics']['true_positives']
    job_tn = performance_metrics['job_posting_analysis']['metrics']['true_negatives']
    job_fp = performance_metrics['job_posting_analysis']['metrics']['false_positives']
    job_fn = performance_metrics['job_posting_analysis']['metrics']['false_negatives']
    
    confusion_matrix = np.array([[job_tn, job_fp], [job_fn, job_tp]])
    im = axes[1, 0].imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    axes[1, 0].set_title('Job Posting Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, confusion_matrix[i, j], ha="center", va="center", fontweight='bold')
    
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Predicted Genuine', 'Predicted Fake'])
    axes[1, 0].set_yticklabels(['Actual Genuine', 'Actual Fake'])
    
    # 4. Confidence Score Distribution
    job_confidences = [r['confidence'] for r in job_results]
    cert_confidences = [r['confidence'] for r in cert_results]
    
    axes[1, 1].hist(job_confidences, bins=10, alpha=0.7, label='Job Postings', color='#2E86AB')
    axes[1, 1].hist(cert_confidences, bins=10, alpha=0.7, label='Certificates', color='#A23B72')
    axes[1, 1].set_title('Confidence Score Distribution')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure for detailed metrics
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    job_values = [
        performance_metrics['job_posting_analysis']['metrics']['accuracy'],
        performance_metrics['job_posting_analysis']['metrics']['precision'],
        performance_metrics['job_posting_analysis']['metrics']['recall'],
        performance_metrics['job_posting_analysis']['metrics']['f1_score']
    ]
    cert_values = [
        performance_metrics['certificate_analysis']['metrics']['accuracy'],
        performance_metrics['certificate_analysis']['metrics']['precision'],
        performance_metrics['certificate_analysis']['metrics']['recall'],
        performance_metrics['certificate_analysis']['metrics']['f1_score']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, job_values, width, label='Job Postings', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, cert_values, width, label='Certificates', color='#A23B72', alpha=0.8)
    
    ax.set_title('Detailed Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (job_val, cert_val) in enumerate(zip(job_values, cert_values)):
        ax.text(i - width/2, job_val + 0.02, f'{job_val:.2f}', ha='center', fontweight='bold')
        ax.text(i + width/2, cert_val + 0.02, f'{cert_val:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance visualizations saved:")
    print("- performance_analysis.png")
    print("- detailed_metrics.png")

def main():
    """
    Main function to run performance analysis.
    """
    print("=== AI-Powered Fraud Detection System - Performance Analysis ===\n")
    
    # Generate performance metrics
    print("Generating performance metrics...")
    performance_metrics, job_results, cert_results = generate_performance_metrics()
    
    # Create visualizations
    print("Creating performance visualizations...")
    create_performance_visualizations(performance_metrics, job_results, cert_results)
    
    # Save detailed results
    with open('performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print(f"Job Posting Analysis:")
    print(f"  Accuracy: {performance_metrics['job_posting_analysis']['metrics']['accuracy']:.2%}")
    print(f"  Precision: {performance_metrics['job_posting_analysis']['metrics']['precision']:.2%}")
    print(f"  Recall: {performance_metrics['job_posting_analysis']['metrics']['recall']:.2%}")
    print(f"  F1-Score: {performance_metrics['job_posting_analysis']['metrics']['f1_score']:.2%}")
    print(f"  Avg Processing Time: {performance_metrics['job_posting_analysis']['avg_processing_time']:.4f}s")
    
    print(f"\nCertificate Analysis:")
    print(f"  Accuracy: {performance_metrics['certificate_analysis']['metrics']['accuracy']:.2%}")
    print(f"  Precision: {performance_metrics['certificate_analysis']['metrics']['precision']:.2%}")
    print(f"  Recall: {performance_metrics['certificate_analysis']['metrics']['recall']:.2%}")
    print(f"  F1-Score: {performance_metrics['certificate_analysis']['metrics']['f1_score']:.2%}")
    print(f"  Avg Processing Time: {performance_metrics['certificate_analysis']['avg_processing_time']:.4f}s")
    
    print(f"\nOverall System Accuracy: {performance_metrics['overall_performance']['overall_accuracy']:.2%}")
    
    print(f"\nDetailed metrics saved to: performance_metrics.json")

if __name__ == "__main__":
    main()

