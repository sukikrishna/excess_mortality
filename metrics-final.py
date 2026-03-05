import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = 'efficient_evaluation_results_more_trials'
PLOTS_DIR = 'enhanced_model_comparison_plots_fixed'
CSV_DIR = 'horizon_plotting_data_csv'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Model names and colors
MODEL_COLORS = {
    'sarima': '#2E86AB',      # Blue
    'lstm': '#A23B72',        # Purple
    'tcn': '#F18F01',         # Orange
    'seq2seq': '#C73E1D',     # Red
    'seq2seq_attn': '#2D5016', # Dark Green
    'transformer': '#8E44AD'   # Violet
}

MODEL_NAMES = {
    'sarima': 'SARIMA',
    'lstm': 'LSTM',
    'tcn': 'TCN',
    'seq2seq': 'Seq2Seq',
    'seq2seq_attn': 'Seq2Seq+Attention',
    'transformer': 'Transformer'
}

def load_efficient_results():
    """Load results from efficient evaluation script"""
    print("Loading efficient evaluation results...")
    
    # Load comprehensive results (this contains the horizon-level metrics)
    comprehensive_file = os.path.join(RESULTS_DIR, 'comprehensive_results.csv')
    if not os.path.exists(comprehensive_file):
        raise FileNotFoundError(f"Results file not found: {comprehensive_file}")
    
    comprehensive_df = pd.read_csv(comprehensive_file)
    print(f"Loaded comprehensive results: {len(comprehensive_df)} records")
    
    # Load predictions
    predictions_file = os.path.join(RESULTS_DIR, 'all_predictions.pkl')
    with open(predictions_file, 'rb') as f:
        all_predictions = pickle.load(f)
    print(f"Loaded predictions for models: {list(all_predictions.keys())}")
    
    # Load data splits
    data_splits_file = os.path.join(RESULTS_DIR, 'data_splits.pkl')
    with open(data_splits_file, 'rb') as f:
        data_splits = pickle.load(f)
    
    return comprehensive_df, all_predictions, data_splits

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = stats.norm.ppf(1 - alpha/2)  # 95% prediction interval
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

def calculate_confidence_intervals(predictions_list, alpha=0.05):
    """Calculate confidence intervals around mean predictions"""
    predictions_array = np.array(predictions_list)
    mean_pred = np.mean(predictions_array, axis=0)
    std_pred = np.std(predictions_array, axis=0, ddof=1)
    n = len(predictions_list)
    
    # t-distribution for confidence intervals
    t_score = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_score * (std_pred / np.sqrt(n))
    
    lower_bound = mean_pred - margin_of_error
    upper_bound = mean_pred + margin_of_error
    
    return mean_pred, lower_bound, upper_bound

def extract_horizon_predictions(predictions, horizon_months):
    """Extract predictions for specific horizon"""
    train_true = predictions['train_true']
    train_pred = predictions['train_pred']
    test_true = predictions['test_true'][:horizon_months]
    test_pred = predictions['test_pred'][:horizon_months]
    
    return train_true, train_pred, test_true, test_pred

def aggregate_model_predictions_for_horizon(model_predictions, horizon_months):
    """Aggregate predictions across trials for a specific horizon"""
    all_train_true = []
    all_train_pred = []
    all_test_true = []
    all_test_pred = []
    
    for trial_data in model_predictions:
        train_true, train_pred, test_true, test_pred = extract_horizon_predictions(
            trial_data, horizon_months)
        
        all_train_true.append(train_true)
        all_train_pred.append(train_pred)
        all_test_true.append(test_true)
        all_test_pred.append(test_pred)
    
    # Calculate mean and confidence intervals
    train_mean, train_ci_lower, train_ci_upper = calculate_confidence_intervals(all_train_pred)
    test_mean, test_ci_lower, test_ci_upper = calculate_confidence_intervals(all_test_pred)
    
    return {
        'train_true': all_train_true[0],  # Same across trials
        'train_mean': train_mean,
        'train_ci_lower': train_ci_lower,
        'train_ci_upper': train_ci_upper,
        'test_true': all_test_true[0][:horizon_months],  # Same across trials, truncated
        'test_mean': test_mean,
        'test_ci_lower': test_ci_lower,
        'test_ci_upper': test_ci_upper,
        'n_trials': len(all_train_pred)
    }

def create_date_axis(train_val_data, test_data, train_start_idx=0, test_months=48):
    """Create proper date axis for plotting"""
    # Training dates (after lookback if applicable)
    if train_start_idx > 0:
        train_dates = train_val_data['Month'].iloc[train_start_idx:].values
    else:
        train_dates = train_val_data['Month'].values
    
    # Test dates (truncated to specified months)
    test_dates = test_data['Month'].iloc[:test_months].values
    
    return train_dates, test_dates

def save_horizon_plotting_data_csv(sarima_data, model_data, model_name, horizon_label, 
                                   horizon_months, train_dates, test_dates):
    """Save raw plotting data for a specific horizon as CSV"""
    
    # Align SARIMA data if needed (handle lookback differences)
    if len(sarima_data['train_true']) != len(model_data['train_true']):
        lookback_diff = len(sarima_data['train_true']) - len(model_data['train_true'])
        if lookback_diff > 0:
            # SARIMA has longer training predictions, truncate to match model
            sarima_train_aligned = {
                'train_true': sarima_data['train_true'][lookback_diff:],
                'train_mean': sarima_data['train_mean'][lookback_diff:],
                'train_ci_lower': sarima_data['train_ci_lower'][lookback_diff:],
                'train_ci_upper': sarima_data['train_ci_upper'][lookback_diff:]
            }
            train_dates_aligned = train_dates
        else:
            sarima_train_aligned = {
                'train_true': sarima_data['train_true'],
                'train_mean': sarima_data['train_mean'],
                'train_ci_lower': sarima_data['train_ci_lower'],
                'train_ci_upper': sarima_data['train_ci_upper']
            }
            train_dates_aligned = train_dates
    else:
        sarima_train_aligned = {
            'train_true': sarima_data['train_true'],
            'train_mean': sarima_data['train_mean'],
            'train_ci_lower': sarima_data['train_ci_lower'],
            'train_ci_upper': sarima_data['train_ci_upper']
        }
        train_dates_aligned = train_dates
    
    # Create combined data for this horizon
    all_dates = np.concatenate([train_dates_aligned, test_dates])
    all_actual = np.concatenate([sarima_train_aligned['train_true'], model_data['test_true']])
    
    # SARIMA data
    sarima_all_mean = np.concatenate([sarima_train_aligned['train_mean'], sarima_data['test_mean']])
    sarima_all_ci_lower = np.concatenate([sarima_train_aligned['train_ci_lower'], sarima_data['test_ci_lower']])
    sarima_all_ci_upper = np.concatenate([sarima_train_aligned['train_ci_upper'], sarima_data['test_ci_upper']])
    
    # Model data
    model_all_mean = np.concatenate([model_data['train_mean'], model_data['test_mean']])
    model_all_ci_lower = np.concatenate([model_data['train_ci_lower'], model_data['test_ci_lower']])
    model_all_ci_upper = np.concatenate([model_data['train_ci_upper'], model_data['test_ci_upper']])
    
    # Calculate prediction intervals
    sarima_pi_lower, sarima_pi_upper = calculate_prediction_intervals(all_actual, sarima_all_mean)
    model_pi_lower, model_pi_upper = calculate_prediction_intervals(all_actual, model_all_mean)
    
    # Create dataset indicator
    dataset_indicator = ['Train'] * len(train_dates_aligned) + ['Test'] * len(test_dates)
    
    # Create DataFrame
    combined_df = pd.DataFrame({
        'Date': all_dates,
        'Actual': all_actual,
        'SARIMA_Mean': sarima_all_mean,
        'SARIMA_CI_Lower': sarima_all_ci_lower,
        'SARIMA_CI_Upper': sarima_all_ci_upper,
        'SARIMA_PI_Lower': sarima_pi_lower,
        'SARIMA_PI_Upper': sarima_pi_upper,
        f'{MODEL_NAMES[model_name]}_Mean': model_all_mean,
        f'{MODEL_NAMES[model_name]}_CI_Lower': model_all_ci_lower,
        f'{MODEL_NAMES[model_name]}_CI_Upper': model_all_ci_upper,
        f'{MODEL_NAMES[model_name]}_PI_Lower': model_pi_lower,
        f'{MODEL_NAMES[model_name]}_PI_Upper': model_pi_upper,
        'Dataset': dataset_indicator,
        'Horizon': horizon_label,
        'Horizon_Months': horizon_months
    })
    
    # Save CSV
    filename = f'horizon_data_{model_name}_vs_sarima_{horizon_label}.csv'
    filepath = os.path.join(CSV_DIR, filename)
    combined_df.to_csv(filepath, index=False)
    
    return combined_df

def create_enhanced_horizon_comparison_plot(sarima_data, model_data, model_name, horizon_label,
                                          horizon_months, train_dates, test_dates, save_path):
    """Create enhanced comparison plot for specific horizon with proper dates"""
    
    # Save plotting data as CSV
    combined_df = save_horizon_plotting_data_csv(sarima_data, model_data, model_name, 
                                               horizon_label, horizon_months, train_dates, test_dates)
    
    # Align SARIMA data if needed
    if len(sarima_data['train_true']) != len(model_data['train_true']):
        lookback_diff = len(sarima_data['train_true']) - len(model_data['train_true'])
        if lookback_diff > 0:
            sarima_train_aligned = {
                'train_true': sarima_data['train_true'][lookback_diff:],
                'train_mean': sarima_data['train_mean'][lookback_diff:],
                'train_ci_lower': sarima_data['train_ci_lower'][lookback_diff:],
                'train_ci_upper': sarima_data['train_ci_upper'][lookback_diff:]
            }
            train_dates_aligned = train_dates
        else:
            sarima_train_aligned = sarima_data
            train_dates_aligned = train_dates
    else:
        sarima_train_aligned = sarima_data
        train_dates_aligned = train_dates
    
    # Combine data for plotting
    all_dates = np.concatenate([train_dates_aligned, test_dates])
    all_actual = np.concatenate([sarima_train_aligned['train_true'], model_data['test_true']])
    
    sarima_all_mean = np.concatenate([sarima_train_aligned['train_mean'], sarima_data['test_mean']])
    sarima_all_ci_lower = np.concatenate([sarima_train_aligned['train_ci_lower'], sarima_data['test_ci_lower']])
    sarima_all_ci_upper = np.concatenate([sarima_train_aligned['train_ci_upper'], sarima_data['test_ci_upper']])
    
    model_all_mean = np.concatenate([model_data['train_mean'], model_data['test_mean']])
    model_all_ci_lower = np.concatenate([model_data['train_ci_lower'], model_data['test_ci_lower']])
    model_all_ci_upper = np.concatenate([model_data['train_ci_upper'], model_data['test_ci_upper']])
    
    # Calculate prediction intervals
    sarima_pi_lower, sarima_pi_upper = calculate_prediction_intervals(all_actual, sarima_all_mean)
    model_pi_lower, model_pi_upper = calculate_prediction_intervals(all_actual, model_all_mean)
    
    # Create the plot with larger size
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot actual data
    ax.plot(all_dates, all_actual, label='Actual Data', color='black', 
            linewidth=3, zorder=10, alpha=0.9)
    
    # Plot mean predictions
    ax.plot(all_dates, sarima_all_mean, label='SARIMA Mean', 
            color=MODEL_COLORS['sarima'], linewidth=2.5, alpha=0.8, zorder=8)
    ax.plot(all_dates, model_all_mean, label=f'{MODEL_NAMES[model_name]} Mean', 
            color=MODEL_COLORS[model_name], linewidth=2.5, alpha=0.8, zorder=8)
    
    # Plot confidence intervals (darker)
    ax.fill_between(all_dates, sarima_all_ci_lower, sarima_all_ci_upper, 
                    color=MODEL_COLORS['sarima'], alpha=0.4, 
                    label=f'SARIMA 95% CI (n={sarima_data["n_trials"]})')
    ax.fill_between(all_dates, model_all_ci_lower, model_all_ci_upper, 
                    color=MODEL_COLORS[model_name], alpha=0.4, 
                    label=f'{MODEL_NAMES[model_name]} 95% CI (n={model_data["n_trials"]})')
    
    # Plot prediction intervals (lighter)
    ax.fill_between(all_dates, sarima_pi_lower, sarima_pi_upper, 
                    color=MODEL_COLORS['sarima'], alpha=0.15, 
                    label='SARIMA 95% PI')
    ax.fill_between(all_dates, model_pi_lower, model_pi_upper, 
                    color=MODEL_COLORS[model_name], alpha=0.15, 
                    label=f'{MODEL_NAMES[model_name]} 95% PI')
    
    # Add vertical line at forecast start
    forecast_start = test_dates[0]
    ax.axvline(forecast_start, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label='Forecast Start', zorder=9)
    
    # Formatting
    ax.set_title(f'Mortality Forecasting: SARIMA vs {MODEL_NAMES[model_name]} - Horizon {horizon_label}', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Deaths', fontsize=14, fontweight='bold')
    
    # Format dates on x-axis
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Enhanced legend positioned outside plot area to avoid overlap
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
                      fancybox=True, shadow=True, fontsize=11, 
                      handlelength=2.5, handletextpad=0.8, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    
    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # Calculate metrics for stats box
    test_rmse_sarima = np.sqrt(np.mean((model_data['test_true'] - sarima_data['test_mean']) ** 2))
    test_rmse_model = np.sqrt(np.mean((model_data['test_true'] - model_data['test_mean']) ** 2))
    test_mae_sarima = np.mean(np.abs(model_data['test_true'] - sarima_data['test_mean']))
    test_mae_model = np.mean(np.abs(model_data['test_true'] - model_data['test_mean']))
    
    # Enhanced statistics text box
    stats_text = (f'Horizon: {horizon_months} months\n'
                 f'Test RMSE:\n'
                 f'  SARIMA: {test_rmse_sarima:.2f}\n'
                 f'  {MODEL_NAMES[model_name]}: {test_rmse_model:.2f}\n'
                 f'Test MAE:\n'
                 f'  SARIMA: {test_mae_sarima:.2f}\n'
                 f'  {MODEL_NAMES[model_name]}: {test_mae_model:.2f}\n'
                 f'Trials: {model_data["n_trials"]}')
    
    props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8, 
                edgecolor='navy', linewidth=1.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Set axis limits with padding
    y_min = min(all_actual.min(), sarima_all_mean.min(), model_all_mean.min())
    y_max = max(all_actual.max(), sarima_all_mean.max(), model_all_mean.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    
    # Save with bbox_inches='tight' to include legend
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return test_rmse_sarima, test_rmse_model

def create_horizon_overview_plot(comprehensive_df, save_path):
    """Create overview plot showing performance across all horizons"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RMSE vs Horizon
    for model in comprehensive_df['model'].unique():
        model_data = comprehensive_df[comprehensive_df['model'] == model].sort_values('horizon_months')
        ax1.plot(model_data['horizon_months'], model_data['test_rmse_mean'], 
                marker='o', linewidth=2.5, markersize=8, 
                label=MODEL_NAMES.get(model, model),
                color=MODEL_COLORS.get(model, 'gray'))
        ax1.errorbar(model_data['horizon_months'], model_data['test_rmse_mean'],
                    yerr=model_data['test_rmse_std'], alpha=0.3,
                    color=MODEL_COLORS.get(model, 'gray'), capsize=5)
    
    ax1.set_xlabel('Forecast Horizon (Months)', fontsize=12)
    ax1.set_ylabel('Test RMSE', fontsize=12)
    ax1.set_title('RMSE vs Forecast Horizon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. MAPE vs Horizon
    for model in comprehensive_df['model'].unique():
        model_data = comprehensive_df[comprehensive_df['model'] == model].sort_values('horizon_months')
        ax2.plot(model_data['horizon_months'], model_data['test_mape_mean'], 
                marker='s', linewidth=2.5, markersize=8,
                label=MODEL_NAMES.get(model, model),
                color=MODEL_COLORS.get(model, 'gray'))
        ax2.errorbar(model_data['horizon_months'], model_data['test_mape_mean'],
                    yerr=model_data['test_mape_std'], alpha=0.3,
                    color=MODEL_COLORS.get(model, 'gray'), capsize=5)
    
    ax2.set_xlabel('Forecast Horizon (Months)', fontsize=12)
    ax2.set_ylabel('Test MAPE (%)', fontsize=12)
    ax2.set_title('MAPE vs Forecast Horizon', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Model rankings heatmap
    horizons = sorted(comprehensive_df['horizon'].unique(), key=lambda x: ['2020', '2020-2021', '2020-2022', '2020-2023'].index(x))
    models = sorted(comprehensive_df['model'].unique())
    
    ranking_data = []
    for model in models:
        model_ranks = []
        for horizon in horizons:
            horizon_data = comprehensive_df[comprehensive_df['horizon'] == horizon]
            if model in horizon_data['model'].values:
                model_rank = horizon_data['test_rmse_mean'].rank()[horizon_data['model'] == model].iloc[0]
                model_ranks.append(model_rank)
            else:
                model_ranks.append(np.nan)
        ranking_data.append(model_ranks)
    
    ranking_array = np.array(ranking_data)
    im = ax3.imshow(ranking_array, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(models))
    ax3.set_xticks(range(len(horizons)))
    ax3.set_xticklabels(horizons)
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([MODEL_NAMES.get(m, m) for m in models])
    ax3.set_title('Model Rankings by Horizon\n(1=Best)', fontsize=14, fontweight='bold')
    
    # Add ranking numbers
    for i in range(len(models)):
        for j in range(len(horizons)):
            if not np.isnan(ranking_array[i, j]):
                ax3.text(j, i, f'{ranking_array[i, j]:.0f}', ha="center", va="center", 
                        color="white" if ranking_array[i, j] > len(models)/2 else "black", fontweight='bold')
    
    # 4. Performance degradation
    degradation_data = []
    for model in models:
        model_data = comprehensive_df[comprehensive_df['model'] == model].sort_values('horizon_months')
        if len(model_data) >= 2:
            baseline_rmse = model_data.iloc[0]['test_rmse_mean']  # 12 months
            final_rmse = model_data.iloc[-1]['test_rmse_mean']    # 48 months
            degradation = ((final_rmse - baseline_rmse) / baseline_rmse) * 100
            degradation_data.append(degradation)
        else:
            degradation_data.append(0)
    
    bars = ax4.bar([MODEL_NAMES.get(m, m) for m in models], degradation_data,
                   color=[MODEL_COLORS.get(m, 'gray') for m in models], alpha=0.7)
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('RMSE Degradation (%)', fontsize=12)
    ax4.set_title('Performance Degradation\n(12→48 months)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, degradation_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (max(degradation_data) * 0.01),
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Variable Horizon Forecasting Performance Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_metrics_table(comprehensive_df):
    """Create enhanced metrics table with all horizon results"""
    
    # Prepare data for table
    table_data = []
    for _, row in comprehensive_df.iterrows():
        table_data.append({
            'Model': MODEL_NAMES.get(row['model'], row['model']),
            'Horizon': row['horizon'],
            'Horizon_Months': int(row['horizon_months']),
            'Test_RMSE_Mean': f"{row['test_rmse_mean']:.3f}",
            'Test_RMSE_Std': f"{row['test_rmse_std']:.3f}",
            'Test_MAE_Mean': f"{row['test_mae_mean']:.3f}",
            'Test_MAE_Std': f"{row['test_mae_std']:.3f}",
            'Test_MAPE_Mean': f"{row['test_mape_mean']:.2f}%",
            'Test_MAPE_Std': f"{row['test_mape_std']:.2f}%",
            'Trials_Completed': int(row['trials_completed'])
        })
    
    metrics_df = pd.DataFrame(table_data)
    
    # Save to CSV
    metrics_df.to_csv(os.path.join(CSV_DIR, 'horizon_comprehensive_metrics.csv'), index=False)
    
    return metrics_df

def main():
    """Main plotting function for efficient evaluation results"""
    print("Creating enhanced plots for efficient variable horizon evaluation...")
    
    # Load results
    comprehensive_df, all_predictions, data_splits = load_efficient_results()
    
    train_val_data = data_splits['train_val_data']
    test_data = data_splits['test_data']
    
    # Create horizon overview plot
    print("Creating horizon overview plot...")
    overview_path = os.path.join(PLOTS_DIR, 'horizon_performance_overview.png')
    create_horizon_overview_plot(comprehensive_df, overview_path)
    print(f"✓ Saved: {overview_path}")
    
    # Create enhanced metrics table
    print("Creating enhanced metrics table...")
    metrics_df = create_enhanced_metrics_table(comprehensive_df)
    print(f"✓ Saved: {os.path.join(CSV_DIR, 'horizon_comprehensive_metrics.csv')}")
    
    # Create individual horizon comparison plots
    horizons = [('2020', 12), ('2020-2021', 24), ('2020-2022', 36), ('2020-2023', 48)]
    
    if 'sarima' in all_predictions and len(all_predictions['sarima']) > 0:
        print("\nCreating individual horizon comparison plots...")
        
        comparison_summary = []
        
        for horizon_label, horizon_months in horizons:
            print(f"  Processing horizon: {horizon_label} ({horizon_months} months)")
            
            # Get SARIMA data for this horizon
            sarima_data = aggregate_model_predictions_for_horizon(
                all_predictions['sarima'], horizon_months)
            
            # Determine lookback for date alignment
            if len(all_predictions['sarima']) > 0:
                sample_pred = all_predictions['sarima'][0]
                sarima_train_len = len(sample_pred['train_true'])
                train_val_len = len(train_val_data)
                sarima_lookback = train_val_len - sarima_train_len
            else:
                sarima_lookback = 0
            
            for model_name in all_predictions:
                if model_name == 'sarima' or len(all_predictions[model_name]) == 0:
                    continue
                    
                print(f"    Creating plot: SARIMA vs {model_name}")
                
                # Get model data for this horizon
                model_data = aggregate_model_predictions_for_horizon(
                    all_predictions[model_name], horizon_months)
                
                # Determine lookback for this model
                sample_pred = all_predictions[model_name][0]
                model_train_len = len(sample_pred['train_true'])
                model_lookback = train_val_len - model_train_len
                
                # Create date axes (use the larger lookback to ensure alignment)
                max_lookback = max(sarima_lookback, model_lookback)
                train_dates, test_dates = create_date_axis(
                    train_val_data, test_data, max_lookback, horizon_months)
                
                # Create plot
                save_path = os.path.join(PLOTS_DIR, 
                                       f'enhanced_sarima_vs_{model_name}_horizon_{horizon_label}.png')
                
                test_rmse_sarima, test_rmse_model = create_enhanced_horizon_comparison_plot(
                    sarima_data, model_data, model_name, horizon_label, 
                    horizon_months, train_dates, test_dates, save_path)
                
                comparison_summary.append({
                    'Horizon': horizon_label,
                    'Horizon_Months': horizon_months,
                    'Model': MODEL_NAMES[model_name],
                    'SARIMA_Test_RMSE': f"{test_rmse_sarima:.3f}",
                    'Model_Test_RMSE': f"{test_rmse_model:.3f}",
                    'RMSE_Difference': f"{test_rmse_model - test_rmse_sarima:+.3f}"
                })
                
                print(f"      ✓ Saved: {save_path}")
        
        # Save comparison summary
        if comparison_summary:
            summary_df = pd.DataFrame(comparison_summary)
            summary_df.to_csv(os.path.join(CSV_DIR, 'horizon_comparison_summary.csv'), index=False)
            print(f"✓ Saved comparison summary: {os.path.join(CSV_DIR, 'horizon_comparison_summary.csv')}")
    
    else:
        print("No SARIMA predictions available for comparison plots")
    
    # Create index of all generated files
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    png_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
    
    file_index = pd.DataFrame({
        'File_Type': ['CSV'] * len(csv_files) + ['Plot'] * len(png_files),
        'File_Name': csv_files + png_files,
        'Description': [
            'Raw plotting data for custom analysis' if 'horizon_data_' in f 
            else 'Comprehensive metrics across horizons' if 'comprehensive' in f
            else 'Comparison summary statistics' if 'comparison' in f
            else 'Enhanced visualization' for f in csv_files + png_files
        ]
    })
    
    file_index.to_csv(os.path.join(CSV_DIR, 'generated_files_index.csv'), index=False)
    
    print(f"\n{'='*60}")
    print("ENHANCED PLOTTING COMPLETED")
    print(f"{'='*60}")
    print(f"Enhanced plots saved to: {PLOTS_DIR}/")
    print(f"CSV data files saved to: {CSV_DIR}/")
    print(f"\nGenerated {len(png_files)} plots and {len(csv_files)} CSV files")
    
    print(f"\nKey improvements made:")
    print(f"✓ Fixed date axis (now shows actual month/year)")
    print(f"✓ Fixed legend overlap (positioned outside plot)")
    print(f"✓ Added confidence intervals around predictions")
    print(f"✓ Saved all plotting data as CSV for custom analysis")
    print(f"✓ Enhanced statistics and metrics")
    print(f"✓ Publication-ready formatting")

if __name__ == "__main__":
    main()