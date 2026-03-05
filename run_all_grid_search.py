#!/usr/bin/env python3
"""
Comprehensive grid search runner for all models with updated data and resume capability.

This script runs hyperparameter grid search for all models:
- LSTM
- SARIMA
- TCN
- Seq2Seq (without attention)
- Seq2Seq with Attention
- Transformer

Features:
- Resume capability: continues from where it left off
- Comprehensive final statistics and model comparison
- Robust error handling and progress tracking

Usage:
    python run_all_grid_search.py [--resume] [--models model1,model2,...]

Arguments:
    --resume: Skip models that already have complete results
    --models: Comma-separated list of specific models to run (optional)
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import glob
import json

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def check_model_completion(model_type, results_dir='results'):
    """Check if a model has completed all its grid search configurations"""
    model_path = os.path.join(results_dir, model_type)
    
    if not os.path.exists(model_path):
        return False, 0, 0
    
    # Count expected vs actual configurations based on model type
    expected_configs = get_expected_config_count(model_type)
    
    # Count completed configurations (those with summary files)
    completed_configs = 0
    for root, dirs, files in os.walk(model_path):
        if 'summary_metrics_validation.csv' in files:
            completed_configs += 1
    
    is_complete = completed_configs >= expected_configs
    
    return is_complete, completed_configs, expected_configs

def get_expected_config_count(model_type):
    """Calculate expected number of configurations for each model type"""
    # Based on the hyperparameter grids in the updated script
    LOOKBACKS = list(range(1, 13)) + [18, 24]  # 16 values
    BATCH_SIZES = [8, 16, 32, 48]  # 4 values
    EPOCHS_LIST = [50, 75, 100, 200, 300]  # 5 values
    
    if model_type == 'sarima':
        SARIMA_ORDERS = 5  # [(1,1,1), (2,1,1), (1,1,0), (0,1,1), (2,1,2)]
        SARIMA_SEASONAL_ORDERS = 3  # [(1,1,1,12), (1,0,1,12), (2,1,1,12)]
        return SARIMA_ORDERS * SARIMA_SEASONAL_ORDERS
    
    elif model_type in ['seq2seq', 'seq2seq_attn']:
        ENCODER_UNITS = 2  # [64, 128]
        DECODER_UNITS = 2  # [64, 128]
        return len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST) * ENCODER_UNITS * DECODER_UNITS
    
    elif model_type == 'transformer':
        D_MODEL = 2  # [64, 128]
        N_HEADS = 3  # [1, 2, 4]
        return len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST) * D_MODEL * N_HEADS
    
    else:  # LSTM, TCN
        return len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST)

def create_model_specific_script(model_type, base_script_path):
    """Create a model-specific grid search script"""
    
    # Read the base script
    with open(base_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace the MODEL_TYPE configuration
    script_content = script_content.replace(
        "MODEL_TYPE = 'lstm'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer'",
        f"MODEL_TYPE = '{model_type}'"
    )
    
    # Create model-specific script
    model_script_path = f'grid_search_{model_type}.py'
    with open(model_script_path, 'w') as f:
        f.write(script_content)
    
    return model_script_path

def run_model_grid_search(model_type, script_path):
    """Run grid search for a specific model"""
    print_header(f"RUNNING GRID SEARCH FOR {model_type.upper()}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the grid search script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {model_type.upper()} grid search completed successfully!")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        if result.stdout:
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            print("Last few lines of output:")
            for line in output_lines[-10:]:
                print(f"  {line}")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✗ {model_type.upper()} grid search failed!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Error code: {e.returncode}")
        
        if e.stdout:
            print("Standard Output:")
            print(e.stdout[-2000:])  # Last 2000 chars
            
        if e.stderr:
            print("Error Output:")
            print(e.stderr[-2000:])  # Last 2000 chars
        
        return False, duration
    
    except Exception as e:
        print(f"✗ Unexpected error running {model_type}: {str(e)}")
        return False, 0

def analyze_model_results(model_type, results_dir='results'):
    """Analyze results for a specific model"""
    model_path = os.path.join(results_dir, model_type)
    
    if not os.path.exists(model_path):
        return None
    
    # Find all validation summary files
    summary_files = glob.glob(os.path.join(model_path, '**/summary_metrics_validation.csv'), recursive=True)
    
    if not summary_files:
        return None
    
    configs = []
    
    for file_path in summary_files:
        config_name = os.path.basename(os.path.dirname(file_path))
        
        try:
            metrics_df = pd.read_csv(file_path, index_col=0)
            
            if 'RMSE' in metrics_df.columns:
                config_data = {
                    'model': model_type,
                    'config': config_name,
                    'mean_rmse': metrics_df.loc['mean', 'RMSE'],
                    'median_rmse': metrics_df.loc['median', 'RMSE'],
                    'std_rmse': metrics_df.loc['std', 'RMSE'],
                    'mean_mae': metrics_df.loc['mean', 'MAE'],
                    'median_mae': metrics_df.loc['median', 'MAE'],
                    'std_mae': metrics_df.loc['std', 'MAE'],
                    'mean_mape': metrics_df.loc['mean', 'MAPE'],
                    'median_mape': metrics_df.loc['median', 'MAPE'],
                    'std_mape': metrics_df.loc['std', 'MAPE']
                }
                
                # Calculate coefficient of variation
                config_data['cv_rmse'] = config_data['std_rmse'] / config_data['mean_rmse'] if config_data['mean_rmse'] > 0 else np.inf
                
                configs.append(config_data)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if configs:
        model_df = pd.DataFrame(configs)
        return model_df
    
    return None

def create_comprehensive_analysis(results_dir='results'):
    """Create comprehensive analysis across all models"""
    print_header("COMPREHENSIVE MODEL ANALYSIS")
    
    all_models = ['lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer']
    all_results = []
    model_summaries = {}
    
    for model in all_models:
        print(f"\nAnalyzing {model.upper()}...")
        
        model_results = analyze_model_results(model, results_dir)
        
        if model_results is not None and len(model_results) > 0:
            all_results.append(model_results)
            
            # Best configurations by different metrics
            best_mean_rmse = model_results.loc[model_results['mean_rmse'].idxmin()]
            best_median_rmse = model_results.loc[model_results['median_rmse'].idxmin()]
            
            model_summaries[model] = {
                'total_configs': len(model_results),
                'best_mean_rmse': best_mean_rmse['mean_rmse'],
                'best_mean_config': best_mean_rmse['config'],
                'best_median_rmse': best_median_rmse['median_rmse'],
                'best_median_config': best_median_rmse['config'],
                'mean_cv': model_results['cv_rmse'].mean(),
                'configs_with_high_variance': len(model_results[model_results['cv_rmse'] > 0.1])
            }
            
            print(f"  Configurations: {len(model_results)}")
            print(f"  Best mean RMSE: {best_mean_rmse['mean_rmse']:.4f} ({best_mean_rmse['config']})")
            print(f"  Best median RMSE: {best_median_rmse['median_rmse']:.4f} ({best_median_rmse['config']})")
            print(f"  Average CV: {model_results['cv_rmse'].mean():.3f}")
        else:
            print(f"  No results found for {model}")
            model_summaries[model] = {
                'total_configs': 0,
                'best_mean_rmse': np.nan,
                'best_mean_config': 'N/A',
                'best_median_rmse': np.nan,
                'best_median_config': 'N/A',
                'mean_cv': np.nan,
                'configs_with_high_variance': 0
            }
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save detailed results
        combined_df.to_csv(os.path.join(results_dir, 'all_models_detailed_results.csv'), index=False)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(model_summaries).T
        summary_df.to_csv(os.path.join(results_dir, 'models_summary.csv'))
        
        # Analysis by metric preference
        print_header("BEST MODELS BY METRIC")
        
        # Best by mean RMSE
        best_mean_overall = combined_df.loc[combined_df['mean_rmse'].idxmin()]
        print(f"Best Overall (Mean RMSE): {best_mean_overall['model'].upper()}")
        print(f"  Config: {best_mean_overall['config']}")
        print(f"  Mean RMSE: {best_mean_overall['mean_rmse']:.4f}")
        print(f"  Median RMSE: {best_mean_overall['median_rmse']:.4f}")
        
        # Best by median RMSE (more robust)
        best_median_overall = combined_df.loc[combined_df['median_rmse'].idxmin()]
        print(f"\nBest Overall (Median RMSE - Recommended): {best_median_overall['model'].upper()}")
        print(f"  Config: {best_median_overall['config']}")
        print(f"  Mean RMSE: {best_median_overall['mean_rmse']:.4f}")
        print(f"  Median RMSE: {best_median_overall['median_rmse']:.4f}")
        
        # Top 5 configurations by median RMSE
        print(f"\nTOP 5 CONFIGURATIONS (by Median RMSE):")
        top_5 = combined_df.nsmallest(5, 'median_rmse')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. {row['model'].upper()}: {row['config']}")
            print(f"     Median RMSE: {row['median_rmse']:.4f}, Mean RMSE: {row['mean_rmse']:.4f}")
        
        # Model ranking by best configuration
        print(f"\nMODEL RANKING (by best median RMSE):")
        model_best = combined_df.groupby('model')['median_rmse'].min().sort_values()
        for i, (model, rmse) in enumerate(model_best.items(), 1):
            print(f"  {i}. {model.upper()}: {rmse:.4f}")
        
        # Save final recommendations
        recommendations = {
            'best_overall_mean': {
                'model': best_mean_overall['model'],
                'config': best_mean_overall['config'],
                'rmse': best_mean_overall['mean_rmse']
            },
            'best_overall_median': {
                'model': best_median_overall['model'],
                'config': best_median_overall['config'],
                'rmse': best_median_overall['median_rmse']
            },
            'top_5_configs': [
                {
                    'model': row['model'],
                    'config': row['config'],
                    'median_rmse': row['median_rmse'],
                    'mean_rmse': row['mean_rmse']
                }
                for _, row in top_5.iterrows()
            ],
            'model_ranking': [
                {'model': model, 'best_median_rmse': rmse}
                for model, rmse in model_best.items()
            ]
        }
        
        with open(os.path.join(results_dir, 'final_recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        return summary_df, combined_df, recommendations
    
    else:
        print("No results found for any model!")
        return None, None, None

def save_progress(models_status, results_dir='results'):
    """Save progress to a JSON file"""
    progress_file = os.path.join(results_dir, 'grid_search_progress.json')
    
    progress_data = {
        'last_updated': datetime.now().isoformat(),
        'models_status': models_status
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(results_dir='results'):
    """Load progress from JSON file"""
    progress_file = os.path.join(results_dir, 'grid_search_progress.json')
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    
    return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run comprehensive grid search for all models')
    parser.add_argument('--resume', action='store_true', 
                       help='Skip models that already have complete results')
    parser.add_argument('--models', type=str, 
                       help='Comma-separated list of specific models to run')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run analysis on existing results, skip grid search')
    
    args = parser.parse_args()
    
    print_header("COMPREHENSIVE GRID SEARCH FOR ALL MODELS")
    
    # Available models
    all_models = ['lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer']
    
    # Determine which models to run
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(',')]
        models_to_run = [m for m in models_to_run if m in all_models]
    else:
        models_to_run = all_models.copy()
    
    print("Models to consider:", ', '.join([m.upper() for m in models_to_run]))
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if we should only analyze
    if args.analyze_only:
        print("\n📊 ANALYSIS-ONLY MODE")
        summary_df, combined_df, recommendations = create_comprehensive_analysis(results_dir)
        return True
    
    # Check data file exists
    data_path = 'data_updated/state_month_overdose_2015_2023.xlsx'
    if not os.path.exists(data_path):
        print(f"\n✗ Data file not found: {data_path}")
        return False
    
    # Check base script exists
    base_script = 'updated_grid_search.py'
    if not os.path.exists(base_script):
        print(f"\n✗ Base grid search script not found: {base_script}")
        return False
    
    # Check completion status
    models_status = {}
    
    print_header("CHECKING EXISTING RESULTS")
    
    for model in models_to_run:
        is_complete, completed, expected = check_model_completion(model, results_dir)
        models_status[model] = {
            'complete': is_complete,
            'completed_configs': completed,
            'expected_configs': expected,
            'completion_rate': completed / expected if expected > 0 else 0
        }
        
        status_str = "✓ COMPLETE" if is_complete else f"⚠ INCOMPLETE ({completed}/{expected})"
        print(f"  {model.upper():15} {status_str}")
    
    # Filter models based on resume flag
    if args.resume:
        models_to_run = [m for m in models_to_run if not models_status[m]['complete']]
        print(f"\n🔄 RESUME MODE: Running {len(models_to_run)} incomplete models")
    
    if not models_to_run:
        print("\n✓ All specified models are complete! Running analysis...")
        summary_df, combined_df, recommendations = create_comprehensive_analysis(results_dir)
        return True
    
    print(f"\n🚀 RUNNING GRID SEARCH FOR: {', '.join([m.upper() for m in models_to_run])}")
    
    total_start_time = time.time()
    results = {}
    
    # Run grid search for each model
    for i, model in enumerate(models_to_run, 1):
        print(f"\n📋 PROGRESS: {i}/{len(models_to_run)} models")
        
        try:
            # Create model-specific script
            model_script = create_model_specific_script(model, base_script)
            
            # Run grid search
            success, duration = run_model_grid_search(model, model_script)
            
            results[model] = {
                'success': success,
                'duration': duration
            }
            
            # Update status
            if success:
                is_complete, completed, expected = check_model_completion(model, results_dir)
                models_status[model].update({
                    'complete': is_complete,
                    'completed_configs': completed,
                    'completion_rate': completed / expected if expected > 0 else 0
                })
            
            # Save progress
            save_progress(models_status, results_dir)
            
            # Clean up model-specific script
            if os.path.exists(model_script):
                os.remove(model_script)
                
        except Exception as e:
            print(f"Error processing {model}: {e}")
            results[model] = {
                'success': False,
                'duration': 0
            }
            continue
    
    # Total execution time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print execution summary
    print_header("GRID SEARCH EXECUTION SUMMARY")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/3600:.1f} hours)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nModel Execution Results:")
    for model, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        duration_str = f"{result['duration']:.1f}s ({result['duration']/60:.1f}m)"
        print(f"  {model.upper():15} {status:10} Duration: {duration_str}")
    
    # Final analysis
    successful_models = [model for model, result in results.items() if result['success']]
    
    if successful_models or args.resume:
        print(f"\n📊 RUNNING COMPREHENSIVE ANALYSIS")
        
        try:
            summary_df, combined_df, recommendations = create_comprehensive_analysis(results_dir)
            
            print_header("FILES CREATED")
            print(f"📁 {results_dir}/")
            print(f"  📄 all_models_detailed_results.csv - Detailed results for all configurations")
            print(f"  📄 models_summary.csv - Summary statistics by model")
            print(f"  📄 final_recommendations.json - Best configurations and rankings")
            print(f"  📄 grid_search_progress.json - Progress tracking")
            
            print_header("NEXT STEPS")
            print("1. Review final_recommendations.json for best configurations")
            print("2. Use the recommended model/config for final evaluation")
            print("3. Retrain on train+validation data with optimal hyperparameters")
            print("4. Evaluate on test set (2020+) for final performance")
            
            return True
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return False
    else:
        print("\n✗ No models completed successfully. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)