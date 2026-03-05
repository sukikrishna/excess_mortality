def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = stats.norm.ppf(1 - alpha/2)  # 95% prediction interval
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

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

# =============================================================================
# PUBLICATION SETTINGS - Easy to modify for different journals/conferences
# =============================================================================

# Font settings for easy adjustment
FONT_CONFIG = {
    'title_size': 20,           # Main plot title
    'subtitle_size': 16,        # Subplot titles  
    'axis_label_size': 16,      # X and Y axis labels
    'tick_label_size': 14,      # Axis tick numbers
    'legend_size': 14,          # Legend text
    'stats_box_size': 12,       # Statistics box text
    'annotation_size': 12       # Any annotations
}

# Plot settings
PLOT_CONFIG = {
    'figure_width': 14,         # Figure width in inches
    'figure_height': 8,         # Figure height in inches
    'dpi': 300,                 # Resolution for saving
    'line_width': 3,            # Main line thickness
    'confidence_alpha': 0.3,   # Transparency for confidence intervals
    'prediction_alpha': 0.15,   # Transparency for prediction intervals
    'grid_alpha': 0.3,          # Grid transparency
    'show_prediction_intervals': True,   # Toggle prediction intervals on/off
    'show_confidence_intervals': True,   # Toggle confidence intervals on/off
}

# Color scheme - easily customizable
MODEL_COLORS = {
    'sarima': '#1f77b4',        # Blue
    'lstm': '#ff7f0e',          # Orange  
    'tcn': '#2ca02c',           # Green
    'seq2seq': '#d62728',       # Red
    'seq2seq_attn': '#9467bd',  # Purple
    'transformer': '#8c564b'    # Brown
}

MODEL_NAMES = {
    'sarima': 'SARIMA',
    'lstm': 'LSTM', 
    'tcn': 'TCN',
    'seq2seq': 'Seq2Seq',
    'seq2seq_attn': 'Seq2Seq+Attn',
    'transformer': 'Transformer'
}

# Set global matplotlib parameters
plt.rcParams.update({
    'font.size': FONT_CONFIG['tick_label_size'],
    'axes.labelsize': FONT_CONFIG['axis_label_size'],
    'axes.titlesize': FONT_CONFIG['subtitle_size'],
    'xtick.labelsize': FONT_CONFIG['tick_label_size'],
    'ytick.labelsize': FONT_CONFIG['tick_label_size'],
    'legend.fontsize': FONT_CONFIG['legend_size'],
    'figure.titlesize': FONT_CONFIG['title_size'],
    'axes.grid': True,
    'grid.alpha': PLOT_CONFIG['grid_alpha'],
    'lines.linewidth': PLOT_CONFIG['line_width'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white'
})

# Configuration
RESULTS_DIR = 'efficient_evaluation_results'
PLOTS_DIR = 'publication_ready_plots_2'
CSV_DIR = 'publication_data_csv'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def load_efficient_results():
    """Load results from efficient evaluation script"""
    print("Loading efficient evaluation results...")
    
    comprehensive_file = os.path.join(RESULTS_DIR, 'comprehensive_results.csv')
    if not os.path.exists(comprehensive_file):
        raise FileNotFoundError(f"Results file not found: {comprehensive_file}")
    
    comprehensive_df = pd.read_csv(comprehensive_file)
    print(f"Loaded comprehensive results: {len(comprehensive_df)} records")
    
    predictions_file = os.path.join(RESULTS_DIR, 'all_predictions.pkl')
    with open(predictions_file, 'rb') as f:
        all_predictions = pickle.load(f)
    print(f"Loaded predictions for models: {list(all_predictions.keys())}")
    
    data_splits_file = os.path.join(RESULTS_DIR, 'data_splits.pkl')
    with open(data_splits_file, 'rb') as f:
        data_splits = pickle.load(f)
    
    return comprehensive_df, all_predictions, data_splits

def calculate_confidence_intervals(predictions_list, alpha=0.05):
    """Calculate confidence intervals around mean predictions"""
    predictions_array = np.array(predictions_list)
    mean_pred = np.mean(predictions_array, axis=0)
    std_pred = np.std(predictions_array, axis=0, ddof=1)
    n = len(predictions_list)
    
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
    
    train_mean, train_ci_lower, train_ci_upper = calculate_confidence_intervals(all_train_pred)
    test_mean, test_ci_lower, test_ci_upper = calculate_confidence_intervals(all_test_pred)
    
    return {
        'train_true': all_train_true[0],
        'train_mean': train_mean,
        'train_ci_lower': train_ci_lower,
        'train_ci_upper': train_ci_upper,
        'test_true': all_test_true[0][:horizon_months],
        'test_mean': test_mean,
        'test_ci_lower': test_ci_lower,
        'test_ci_upper': test_ci_upper,
        'n_trials': len(all_train_pred)
    }

def create_date_axis(train_val_data, test_data, train_start_idx=0, test_months=48):
    """Create proper date axis for plotting"""
    if train_start_idx > 0:
        train_dates = train_val_data['Month'].iloc[train_start_idx:].values
    else:
        train_dates = train_val_data['Month'].values
    
    test_dates = test_data['Month'].iloc[:test_months].values
    
    return train_dates, test_dates

def create_publication_ready_comparison_plot(sarima_data, model_data, model_name, horizon_label,
                                           horizon_months, train_dates, test_dates, save_path):
    """Create clean, publication-ready comparison plot with both confidence and prediction intervals"""
    
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
        else:
            sarima_train_aligned = sarima_data
    else:
        sarima_train_aligned = sarima_data
    
    # Combine data for plotting
    all_dates = np.concatenate([train_dates, test_dates])
    all_actual = np.concatenate([sarima_train_aligned['train_true'], model_data['test_true']])
    
    sarima_all_mean = np.concatenate([sarima_train_aligned['train_mean'], sarima_data['test_mean']])
    model_all_mean = np.concatenate([model_data['train_mean'], model_data['test_mean']])
    
    # Prepare confidence intervals data
    sarima_all_ci_lower = np.concatenate([sarima_train_aligned['train_ci_lower'], sarima_data['test_ci_lower']])
    sarima_all_ci_upper = np.concatenate([sarima_train_aligned['train_ci_upper'], sarima_data['test_ci_upper']])
    model_all_ci_lower = np.concatenate([model_data['train_ci_lower'], model_data['test_ci_lower']])
    model_all_ci_upper = np.concatenate([model_data['train_ci_upper'], model_data['test_ci_upper']])
    
    # Calculate prediction intervals
    sarima_pi_lower, sarima_pi_upper = calculate_prediction_intervals(all_actual, sarima_all_mean)
    model_pi_lower, model_pi_upper = calculate_prediction_intervals(all_actual, model_all_mean)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(PLOT_CONFIG['figure_width'], PLOT_CONFIG['figure_height']))
    
    # Plot prediction intervals first (lighter/wider bands)
    if PLOT_CONFIG['show_prediction_intervals']:
        ax.fill_between(all_dates, sarima_pi_lower, sarima_pi_upper, 
                        color=MODEL_COLORS['sarima'], alpha=PLOT_CONFIG['prediction_alpha'], 
                        label=f'SARIMA 95% PI', zorder=1)
        ax.fill_between(all_dates, model_pi_lower, model_pi_upper, 
                        color=MODEL_COLORS[model_name], alpha=PLOT_CONFIG['prediction_alpha'], 
                        label=f'{MODEL_NAMES[model_name]} 95% PI', zorder=1)
    
    # Plot confidence intervals (darker/narrower bands)
    if PLOT_CONFIG['show_confidence_intervals']:
        ax.fill_between(all_dates, sarima_all_ci_lower, sarima_all_ci_upper, 
                        color=MODEL_COLORS['sarima'], alpha=PLOT_CONFIG['confidence_alpha'], 
                        label=f'SARIMA 95% CI', zorder=2)
        ax.fill_between(all_dates, model_all_ci_lower, model_all_ci_upper, 
                        color=MODEL_COLORS[model_name], alpha=PLOT_CONFIG['confidence_alpha'], 
                        label=f'{MODEL_NAMES[model_name]} 95% CI', zorder=2)
    
    # Plot main lines
    ax.plot(all_dates, all_actual, label='Observed', color='black', 
            linewidth=PLOT_CONFIG['line_width'], zorder=10)
    ax.plot(all_dates, sarima_all_mean, label='SARIMA', 
            color=MODEL_COLORS['sarima'], linewidth=PLOT_CONFIG['line_width'], zorder=8)
    ax.plot(all_dates, model_all_mean, label=MODEL_NAMES[model_name], 
            color=MODEL_COLORS[model_name], linewidth=PLOT_CONFIG['line_width'], zorder=8)
    
    # Add forecast start line
    forecast_start = test_dates[0]
    ax.axvline(forecast_start, color='red', linestyle='--', alpha=0.8, 
               linewidth=2, label='Forecast Start', zorder=9)
    
    # Formatting
    ax.set_title(f'Mortality Forecasting: {MODEL_NAMES[model_name]} vs SARIMA\nForecast Horizon: {horizon_label}', 
                fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=FONT_CONFIG['axis_label_size'], fontweight='bold')
    ax.set_ylabel('Deaths', fontsize=FONT_CONFIG['axis_label_size'], fontweight='bold')
    
    # Clean up x-axis dates
    ax.tick_params(axis='x', rotation=45, labelsize=FONT_CONFIG['tick_label_size'])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label_size'])
    
    # Legend positioned to avoid overlap
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
              fontsize=FONT_CONFIG['legend_size'], framealpha=0.95)
    
    # Calculate and display key metrics
    projection_rmse_sarima = np.sqrt(np.mean((model_data['test_true'] - sarima_data['test_mean']) ** 2))
    projection_rmse_model = np.sqrt(np.mean((model_data['test_true'] - model_data['test_mean']) ** 2))
    
    # Stats box positioned in bottom right
    stats_text = (f'Horizon: {horizon_months} months\n'
                 f'Projection RMSE:\n'
                 f'SARIMA: {projection_rmse_sarima:.1f}\n'
                 f'{MODEL_NAMES[model_name]}: {projection_rmse_model:.1f}')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=FONT_CONFIG['stats_box_size'], verticalalignment='bottom', 
            horizontalalignment='right', bbox=props, fontweight='bold')
    
    # Clean layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return projection_rmse_sarima, projection_rmse_model

def create_performance_overview_plot(comprehensive_df, save_path):
    """Create clean overview plot showing performance across horizons"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_CONFIG['figure_width'], 
                                                   PLOT_CONFIG['figure_height']*0.6))
    
    # 1. RMSE vs Horizon
    for model in comprehensive_df['model'].unique():
        model_data = comprehensive_df[comprehensive_df['model'] == model].sort_values('horizon_months')
        ax1.plot(model_data['horizon_months'], model_data['test_rmse_mean'], 
                marker='o', linewidth=PLOT_CONFIG['line_width'], markersize=8, 
                label=MODEL_NAMES.get(model, model),
                color=MODEL_COLORS.get(model, 'gray'))
        
        # Simplified error bars
        ax1.errorbar(model_data['horizon_months'], model_data['test_rmse_mean'],
                    yerr=model_data['test_rmse_std'], alpha=0.3,
                    color=MODEL_COLORS.get(model, 'gray'), capsize=5, linewidth=2)
    
    ax1.set_xlabel('Forecast Horizon (Months)', fontsize=FONT_CONFIG['axis_label_size'])
    ax1.set_ylabel('Projection RMSE', fontsize=FONT_CONFIG['axis_label_size'])
    ax1.set_title('RMSE vs Forecast Horizon', fontsize=FONT_CONFIG['subtitle_size'], fontweight='bold')
    ax1.legend(fontsize=FONT_CONFIG['legend_size'])
    
    # 2. MAPE vs Horizon  
    for model in comprehensive_df['model'].unique():
        model_data = comprehensive_df[comprehensive_df['model'] == model].sort_values('horizon_months')
        ax2.plot(model_data['horizon_months'], model_data['test_mape_mean'], 
                marker='s', linewidth=PLOT_CONFIG['line_width'], markersize=8,
                label=MODEL_NAMES.get(model, model),
                color=MODEL_COLORS.get(model, 'gray'))
        
        ax2.errorbar(model_data['horizon_months'], model_data['test_mape_mean'],
                    yerr=model_data['test_mape_std'], alpha=0.3,
                    color=MODEL_COLORS.get(model, 'gray'), capsize=5, linewidth=2)
    
    ax2.set_xlabel('Forecast Horizon (Months)', fontsize=FONT_CONFIG['axis_label_size'])
    ax2.set_ylabel('Projection MAPE (%)', fontsize=FONT_CONFIG['axis_label_size'])
    ax2.set_title('MAPE vs Forecast Horizon', fontsize=FONT_CONFIG['subtitle_size'], fontweight='bold')
    ax2.legend(fontsize=FONT_CONFIG['legend_size'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Main plotting function with publication-ready outputs"""
    print("Creating publication-ready plots...")
    print(f"Current font settings: {FONT_CONFIG}")
    print(f"Current plot settings: {PLOT_CONFIG}")
    
    # Load results
    comprehensive_df, all_predictions, data_splits = load_efficient_results()
    
    train_val_data = data_splits['train_val_data']
    test_data = data_splits['test_data']
    
    # Create overview plot
    print("Creating performance overview plot...")
    overview_path = os.path.join(PLOTS_DIR, 'performance_overview_publication.png')
    create_performance_overview_plot(comprehensive_df, overview_path)
    print(f"✓ Saved: {overview_path}")
    
    # Create individual horizon comparison plots
    horizons = [('2020', 12), ('2020-2021', 24), ('2020-2022', 36), ('2020-2023', 48)]
    
    if 'sarima' in all_predictions and len(all_predictions['sarima']) > 0:
        print("\nCreating individual horizon comparison plots...")
        
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
                
                # Create date axes
                max_lookback = max(sarima_lookback, model_lookback)
                train_dates, test_dates = create_date_axis(
                    train_val_data, test_data, max_lookback, horizon_months)
                
                # Create plot
                save_path = os.path.join(PLOTS_DIR, 
                                       f'publication_{model_name}_vs_sarima_{horizon_label}.png')
                
                projection_rmse_sarima, projection_rmse_model = create_publication_ready_comparison_plot(
                    sarima_data, model_data, model_name, horizon_label, 
                    horizon_months, train_dates, test_dates, save_path)
                
                print(f"      ✓ Saved: {save_path}")
                print(f"      Projection RMSE - SARIMA: {projection_rmse_sarima:.1f}, {MODEL_NAMES[model_name]}: {projection_rmse_model:.1f}")
    
    print(f"\n{'='*60}")
    print("PUBLICATION-READY PLOTTING COMPLETED")
    print(f"{'='*60}")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print(f"\nKey improvements:")
    print(f"✓ Large, readable fonts (easily adjustable)")
    print(f"✓ Clean, uncluttered design")
    print(f"✓ Professional color scheme")
    print(f"✓ Proper spacing and layout")
    print(f"✓ High resolution ({PLOT_CONFIG['dpi']} DPI)")
    print(f"✓ Configurable visual elements")
    
    print(f"\nTo adjust settings:")
    print(f"- Modify FONT_CONFIG for font sizes")
    print(f"- Modify PLOT_CONFIG for visual elements") 
    print(f"- Modify MODEL_COLORS for color scheme")

if __name__ == "__main__":
    main()