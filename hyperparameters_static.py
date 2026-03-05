#!/usr/bin/env python3
"""
Script to extract optimal hyperparameters from grid search results and update configuration.

This script reads your hyperparameter grid search results and extracts the best configuration
for each model based on validation RMSE, then updates the configuration in the evaluation scripts.

Usage:
    python extract_optimal_hyperparameters.py

The script expects your grid search results to be in a CSV format with columns:
- model: Model name (lstm, tcn, seq2seq, seq2seq_attn, transformer, sarima)
- split: train/validation
- best_rmse_mean: Mean RMSE value
- config_name: Configuration string with hyperparameters

Based on your provided results table.
"""

import pandas as pd
import re
import os

def parse_config_name(config_name, model_type):
    """Parse configuration name to extract hyperparameters"""
    params = {}
    
    if model_type == 'lstm':
        # Example: lookback_5_bs_8_epochs_50
        match = re.search(r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)', config_name)
        if match:
            params = {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3))
            }
    
    elif model_type == 'tcn':
        # Example: lookback_7_bs_8_epochs_100
        match = re.search(r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)', config_name)
        if match:
            params = {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3))
            }
    
    elif model_type == 'seq2seq':
        # Example: lookback_7_bs_16_epochs_100_enc_64_dec_64_att_False
        match = re.search(r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)_enc_(\d+)_dec_(\d+)_att_(True|False)', config_name)
        if match:
            params = {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'encoder_units': int(match.group(4)),
                'decoder_units': int(match.group(5))
            }
    
    elif model_type == 'seq2seq_attn':
        # Example: lookback_5_bs_16_epochs_50_enc_128_dec_64_att_True
        match = re.search(r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)_enc_(\d+)_dec_(\d+)_att_(True|False)', config_name)
        if match:
            params = {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'encoder_units': int(match.group(4)),
                'decoder_units': int(match.group(5))
            }
    
    elif model_type == 'transformer':
        # Example: lookback_7_bs_32_epochs_100_dmodel_64_heads_2
        match = re.search(r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)_dmodel_(\d+)_heads_(\d+)', config_name)
        if match:
            params = {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'd_model': int(match.group(4)),
                'n_heads': int(match.group(5))
            }
    
    elif model_type == 'sarima':
        # Example: order_(1, 0, 0)_seasonal_(1, 1, 1, 12)
        order_match = re.search(r'order_\((\d+),\s*(\d+),\s*(\d+)\)', config_name)
        seasonal_match = re.search(r'seasonal_\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', config_name)
        if order_match and seasonal_match:
            params = {
                'order': (int(order_match.group(1)), int(order_match.group(2)), int(order_match.group(3))),
                'seasonal_order': (int(seasonal_match.group(1)), int(seasonal_match.group(2)), 
                                 int(seasonal_match.group(3)), int(seasonal_match.group(4)))
            }
    
    return params

def create_results_from_user_data():
    """Create results DataFrame from the user's provided data"""
    
    # User's results data
    results_data = [
        {'model': 'lstm', 'split': 'validation', 'best_rmse_mean': 293.12899603362695, 'config_name': 'lookback_5_bs_8_epochs_50'},
        {'model': 'lstm', 'split': 'train', 'best_rmse_mean': 314.579993100924, 'config_name': 'lookback_3_bs_32_epochs_100'},
        {'model': 'seq2seq', 'split': 'validation', 'best_rmse_mean': 216.7038519611297, 'config_name': 'lookback_7_bs_16_epochs_100_enc_64_dec_64_att_False'},
        {'model': 'seq2seq', 'split': 'train', 'best_rmse_mean': 244.75687385317104, 'config_name': 'lookback_12_bs_16_epochs_100_enc_64_dec_128_att_False'},
        {'model': 'seq2seq_attn', 'split': 'validation', 'best_rmse_mean': 233.66519930187687, 'config_name': 'lookback_5_bs_16_epochs_50_enc_128_dec_64_att_True'},
        {'model': 'seq2seq_attn', 'split': 'train', 'best_rmse_mean': 244.54231404381332, 'config_name': 'lookback_12_bs_16_epochs_100_enc_64_dec_128_att_True'},
        {'model': 'tcn', 'split': 'validation', 'best_rmse_mean': 1314.1690735442437, 'config_name': 'lookback_7_bs_8_epochs_100'},
        {'model': 'tcn', 'split': 'train', 'best_rmse_mean': 483.67237906773454, 'config_name': 'lookback_7_bs_8_epochs_100'},
        {'model': 'transformer', 'split': 'validation', 'best_rmse_mean': 334.18138853567854, 'config_name': 'lookback_7_bs_32_epochs_100_dmodel_64_heads_2'},
        {'model': 'transformer', 'split': 'train', 'best_rmse_mean': 254.0423054839484, 'config_name': 'lookback_11_bs_8_epochs_100_dmodel_64_heads_2'},
        {'model': 'sarima', 'split': 'validation', 'best_rmse_mean': 402.2453536520759, 'config_name': 'order_(1, 0, 0)_seasonal_(1, 1, 1, 12)'},
        {'model': 'sarima', 'split': 'train', 'best_rmse_mean': 779.496768313246, 'config_name': 'order_(1, 1, 1)_seasonal_(1, 1, 1, 12)'}
    ]
    
    return pd.DataFrame(results_data)

def extract_optimal_hyperparameters():
    """Extract optimal hyperparameters from grid search results"""
    
    print("="*80)
    print("EXTRACTING OPTIMAL HYPERPARAMETERS FROM GRID SEARCH RESULTS")
    print("="*80)
    
    # Create results DataFrame
    results_df = create_results_from_user_data()
    
    print("Loaded grid search results:")
    print(results_df.to_string(index=False))
    
    # Extract best validation configurations for each model
    validation_results = results_df[results_df['split'] == 'validation']
    
    optimal_params = {}
    
    print(f"\n{'='*60}")
    print("OPTIMAL HYPERPARAMETERS (Based on Validation RMSE)")
    print(f"{'='*60}")
    
    for model in validation_results['model'].unique():
        model_data = validation_results[validation_results['model'] == model]
        best_config = model_data.loc[model_data['best_rmse_mean'].idxmin()]
        
        print(f"\n{model.upper()}:")
        print(f"  Validation RMSE: {best_config['best_rmse_mean']:.4f}")
        print(f"  Configuration: {best_config['config_name']}")
        
        # Parse configuration
        params = parse_config_name(best_config['config_name'], model)
        if params:
            print(f"  Parsed parameters: {params}")
            optimal_params[model] = params
        else:
            print(f"  Warning: Could not parse configuration for {model}")
    
    return optimal_params

def generate_config_code(optimal_params):
    """Generate Python code for the optimal parameters configuration"""
    
    print(f"\n{'='*60}")
    print("GENERATED CONFIGURATION CODE")
    print(f"{'='*60}")
    
    config_code = "# Optimal hyperparameters from grid search results\n"
    config_code += "OPTIMAL_PARAMS = {\n"
    
    for model, params in optimal_params.items():
        config_code += f"    '{model}': {params},\n"
    
    config_code += "}\n"
    
    print(config_code)
    
    # Save to file
    with open('optimal_hyperparameters_config.py', 'w') as f:
        f.write(config_code)
    
    print("Configuration saved to: optimal_hyperparameters_config.py")
    
    return config_code

def update_evaluation_script(optimal_params):
    """Update the evaluation script with new optimal parameters"""
    
    script_name = 'final_evaluation_variable_horizon.py'
    
    if not os.path.exists(script_name):
        print(f"\nWarning: {script_name} not found. Cannot update automatically.")
        return False
    
    print(f"\nUpdating {script_name} with optimal parameters...")
    
    # Read the script
    with open(script_name, 'r') as f:
        content = f.read()
    
    # Generate new OPTIMAL_PARAMS section
    new_params_section = "OPTIMAL_PARAMS = {\n"
    for model, params in optimal_params.items():
        new_params_section += f"    '{model}': {params},\n"
    new_params_section += "}"
    
    # Replace the OPTIMAL_PARAMS section
    import re
    pattern = r'OPTIMAL_PARAMS = \{[^}]*\}'
    updated_content = re.sub(pattern, new_params_section, content, flags=re.DOTALL)
    
    # Create backup
    backup_name = f'{script_name}.backup'
    with open(backup_name, 'w') as f:
        f.write(content)
    
    # Write updated content
    with open(script_name, 'w') as f:
        f.write(updated_content)
    
    print(f"  ✓ Updated {script_name}")
    print(f"  ✓ Backup saved as {backup_name}")
    
    return True

def print_performance_summary(results_df):
    """Print a summary of model performance"""
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE SUMMARY (Validation RMSE)")
    print(f"{'='*60}")
    
    validation_results = results_df[results_df['split'] == 'validation'].copy()
    validation_results = validation_results.sort_values('best_rmse_mean')
    
    print("Ranking (lower RMSE is better):")
    for idx, (_, row) in enumerate(validation_results.iterrows()):
        print(f"  {idx+1}. {row['model'].upper()}: {row['best_rmse_mean']:.2f}")
    
    # Identify concerning results
    high_rmse_threshold = 500
    concerning_models = validation_results[validation_results['best_rmse_mean'] > high_rmse_threshold]
    
    if not concerning_models.empty:
        print(f"\n⚠️  Models with high RMSE (>{high_rmse_threshold}):")
        for _, row in concerning_models.iterrows():
            print(f"    {row['model'].upper()}: {row['best_rmse_mean']:.2f}")
        print("    Consider reviewing these model configurations or training procedures.")

def main():
    """Main function"""
    print("Starting optimal hyperparameter extraction...")
    
    # Extract optimal hyperparameters
    optimal_params = extract_optimal_hyperparameters()
    
    if not optimal_params:
        print("\n✗ No optimal parameters extracted. Please check your results data.")
        return False
    
    # Generate configuration code
    config_code = generate_config_code(optimal_params)
    
    # Update evaluation script
    update_success = update_evaluation_script(optimal_params)
    
    # Print performance summary
    results_df = create_results_from_user_data()
    print_performance_summary(results_df)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETED")
    print(f"{'='*60}")
    
    print("\nNext steps:")
    print("1. Review the optimal hyperparameters above")
    print("2. Check optimal_hyperparameters_config.py for the configuration")
    if update_success:
        print("3. The evaluation script has been automatically updated")
    else:
        print("3. Manually update your evaluation script with the optimal parameters")
    print("4. Run the variable horizon evaluation with: python run_variable_horizon_evaluation.py")
    
    print("\nNotes:")
    print("- Parameters are based on VALIDATION performance (preferred for hyperparameter selection)")
    print("- TCN shows high RMSE - you may want to investigate this model's configuration")
    print("- Seq2Seq (without attention) shows the best validation performance")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        exit(0)
    else:
        exit(1)