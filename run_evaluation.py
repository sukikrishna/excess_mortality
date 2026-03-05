#!/usr/bin/env python3
"""
Main execution script for comprehensive model evaluation and plotting.

This script orchestrates the entire evaluation pipeline:
1. Runs the final model evaluation with optimal hyperparameters
2. Extracts and organizes prediction data
3. Creates comprehensive plots and comparisons

Usage:
    python run_evaluation.py

Make sure you have the following files in your directory:
- data/state_month_overdose.xlsx
- All required Python packages installed (see requirements below)

Requirements:
- tensorflow
- keras-tcn
- statsmodels
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pickle
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {description}")
    print(f"Executing: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {description} completed successfully!")
        print(f"  Duration: {duration:.2f} seconds")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✗ {description} failed!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Error code: {e.returncode}")
        
        if e.stdout:
            print("Standard Output:")
            print(e.stdout)
            
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
            
        return False
    
    except Exception as e:
        print(f"✗ Unexpected error running {script_name}: {str(e)}")
        return False

def check_requirements():
    """Check if required files and directories exist"""
    print_header("CHECKING REQUIREMENTS")
    
    required_files = [
        'data/state_month_overdose.xlsx'
    ]
    
    required_packages = [
        'tensorflow',
        'statsmodels',
        'sklearn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    # Check files
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("✗ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✓ All required files found")
    
    # Check packages
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("✗ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✓ All required packages available")
    
    # Check for TCN specifically
    try:
        from tcn import TCN
        print("✓ TCN package available")
    except ImportError:
        print("✗ TCN package not found")
        print("Install with: pip install keras-tcn")
        return False
    
    return True

def create_script_files():
    """Create the evaluation script files if they don't exist"""
    
    scripts_info = {
        'final_evaluation.py': "Final model evaluation script",
        'prediction_extractor.py': "Prediction data extraction script", 
        'plotting_script.py': "Comprehensive plotting script"
    }
    
    missing_scripts = []
    for script in scripts_info.keys():
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print_header("MISSING SCRIPT FILES")
        print("The following script files are missing:")
        for script in missing_scripts:
            print(f"  - {script}: {scripts_info[script]}")
        print("\nPlease create these files using the provided code artifacts.")
        return False
    
    return True

def main():
    """Main execution function"""
    print_header("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    print("This script will run the complete evaluation pipeline:")
    print("1. Model evaluation with optimal hyperparameters")
    print("2. Prediction data extraction and organization")
    print("3. Comprehensive plotting and visualization")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n✗ Requirements check failed. Please fix the issues above.")
        return False
    
    # Check script files
    if not create_script_files():
        print("\n✗ Script files check failed. Please create the missing files.")
        return False
    
    print("✓ All requirements satisfied. Starting evaluation pipeline...")
    
    # Step 1: Run model evaluation
    success1 = run_script('final_evaluation.py', 
                         'Model Evaluation with Optimal Hyperparameters')
    
    if not success1:
        print("\n✗ Model evaluation failed. Cannot proceed.")
        return False
    
    # Step 2: Extract prediction data
    success2 = run_script('prediction_extractor.py',
                         'Prediction Data Extraction and Organization')
    
    if not success2:
        print("\n✗ Prediction extraction failed. Cannot proceed.")
        return False
    
    # Step 3: Create plots
    success3 = run_script('plotting_script.py',
                         'Comprehensive Plotting and Visualization')
    
    if not success3:
        print("\n✗ Plotting failed.")
        return False
    
    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nGenerated outputs:")
    print("├── final_evaluation_results/")
    print("│   ├── final_model_comparison.csv")
    print("│   ├── all_predictions.pkl")
    print("│   ├── data_splits.pkl")
    print("│   └── [model_name]/")
    print("│       └── seed_[seed]/")
    print("│           ├── all_trials_metrics.csv")
    print("│           └── trial_[n]_[train/test]_predictions.csv")
    print("├── extracted_predictions/")
    print("│   ├── all_models_predictions_long_format.csv")
    print("│   ├── model_performance_summary.csv")
    print("│   ├── pairwise_model_comparisons.csv")
    print("│   └── [ModelName]/")
    print("│       ├── training_predictions.csv")
    print("│       ├── test_predictions.csv")
    print("│       └── all_predictions.csv")
    print("└── model_comparison_plots/")
    print("    ├── sarima_vs_[model]_comparison.png (4 plots)")
    print("    ├── comprehensive_metrics_table.csv")
    print("    ├── metrics_comparison_table.png")
    print("    ├── pi_overlap_heatmap.png")
    print("    └── pi_overlap_summary.csv")
    
    print("\nNext steps:")
    print("1. Review the model comparison plots in 'model_comparison_plots/'")
    print("2. Examine the comprehensive metrics in 'extracted_predictions/model_performance_summary.csv'")
    print("3. Use the long-format dataset for further analysis: 'extracted_predictions/all_models_predictions_long_format.csv'")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)