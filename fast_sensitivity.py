#!/usr/bin/env python3
"""
Fast Model Sensitivity Analysis Script - LSTM_Config2 Focus

This script performs sensitivity analysis using only LSTM_Config2 as a proxy
to quickly determine optimal trials and seeds for stable model evaluation.

Key Features:
1. Fast execution with single model configuration
2. Comprehensive analysis of ranking stability
3. Determines minimum requirements for convergence
4. Easy switch back to full model analysis

Usage:
    python fast_sensitivity_analysis.py --max_trials 50 --max_seeds 50 --eval_interval 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import json
import pickle
from pathlib import Path
import time
from itertools import product
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict

# Model-specific imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class FastModelSensitivityAnalyzer:
    """
    Fast analyzer focusing on LSTM_Config2 for quick convergence assessment
    """
    
    def __init__(self, data_path='data_updated/state_month_overdose_2015_2023.xlsx', 
                 use_full_models=False):
        self.data_path = data_path
        self.use_full_models = use_full_models
        
        # Create output directories
        self.output_dir = 'fast_sensitivity_analysis_3'
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        for dir_path in [self.output_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Define model configurations
        if use_full_models:
            # Full model set for comparison
            self.model_configs = {
                'LSTM_Config1': {
                    'model_type': 'lstm',
                    'lookback': 7,
                    'batch_size': 16,
                    'epochs': 50
                },
                'LSTM_Config2': {
                    'model_type': 'lstm',
                    'lookback': 9,
                    'batch_size': 8,
                    'epochs': 30
                },
                'LSTM_Config3': {
                    'model_type': 'lstm',
                    'lookback': 12,
                    'batch_size': 32,
                    'epochs': 50
                }
            }
        else:
            # Focus on LSTM_Config2 only for fast analysis
            self.model_configs = {
                'LSTM_Config2': {
                    'model_type': 'lstm',
                    'lookback': 9,
                    'batch_size': 8,
                    'epochs': 30
                }
            }
        
        # Storage for results
        self.all_metrics = []
        self.performance_summary = {}
        self.ranking_analysis = {}
        
        analysis_type = "Full Model" if use_full_models else "Fast LSTM-focused"
        print(f"Initialized {analysis_type} Sensitivity Analyzer")
        print(f"Output directory: {self.output_dir}")
        print(f"Model configurations: {len(self.model_configs)}")
        print(f"Models: {list(self.model_configs.keys())}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the overdose data"""
        df = pd.read_excel(self.data_path)
        
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle different possible column names
        if 'Sum of Deaths' in df.columns:
            df = df.rename(columns={'Sum of Deaths': 'Deaths'})
        
        # Create proper date column
        if 'Year_Code' in df.columns and 'Month_Code' in df.columns:
            df['Month_Code'] = pd.to_numeric(df['Month_Code'], errors='coerce')
            df['Year_Code'] = pd.to_numeric(df['Year_Code'], errors='coerce')
            
            date_df = pd.DataFrame({
                'year': df['Year_Code'],
                'month': df['Month_Code'], 
                'day': 1
            })
            df['Month'] = pd.to_datetime(date_df)
        elif 'Month' in df.columns:
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        
        # Handle Deaths column
        if df['Deaths'].dtype == 'object':
            df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else float(x))
        else:
            df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['Month', 'Deaths'])
        df = df[['Month', 'Deaths']].copy()
        df = df.sort_values('Month').reset_index(drop=True)
        
        print(f"✓ Final data shape: {df.shape}")
        print(f"✓ Date range: {df['Month'].min()} to {df['Month'].max()}")
        
        return df
    
    def create_train_val_split(self, df, val_start='2019-01-01', val_end='2020-01-01'):
        """Create train/validation splits for evaluation"""
        train = df[df['Month'] < val_start].copy()
        validation = df[(df['Month'] >= val_start) & (df['Month'] < val_end)].copy()
        
        print(f"Train samples: {len(train)} ({train['Month'].min()} to {train['Month'].max()})")
        print(f"Validation samples: {len(validation)} ({validation['Month'].min()} to {validation['Month'].max()})")
        
        return train, validation
    
    def create_sequences(self, data, lookback):
        """Create sequences for time series modeling"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, config):
        """Build LSTM model matching your grid search implementation"""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(config['lookback'], 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm_model(self, train_data, val_data, config, seed):
        """Train LSTM model matching your grid search implementation"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Prepare data
        train_values = train_data['Deaths'].values.astype(np.float32)
        val_values = val_data['Deaths'].values.astype(np.float32)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_values, config['lookback'])
        X_train = X_train.reshape((X_train.shape[0], config['lookback'], 1))
        
        # Build and train model
        model = self.build_lstm_model(config)
        model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose=0)
        
        # Generate training predictions
        train_preds = []
        for i in range(config['lookback'], len(train_values)):
            input_seq = train_values[i-config['lookback']:i].reshape((1, config['lookback'], 1))
            pred = model.predict(input_seq, verbose=0)[0][0]
            train_preds.append(pred)
        
        # Generate test predictions (autoregressive)
        current_input = train_values[-config['lookback']:].reshape((1, config['lookback'], 1))
        val_preds = []
        for _ in range(len(val_values)):
            pred = model.predict(current_input, verbose=0)[0][0]
            val_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return train_values[config['lookback']:], np.array(train_preds), val_values, np.array(val_preds)
    
    def evaluate_model_single_trial(self, train_data, val_data, model_name, config, seed):
        """Evaluate a single model trial"""
        start_time = time.time()
        
        try:
            if config['model_type'] == 'lstm':
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_lstm_model(train_data, val_data, config, seed)
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
            
            # Calculate metrics on validation set (this is what we use for ranking)
            rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
            mae = mean_absolute_error(y_val_true, y_val_pred)
            mape = np.mean(np.abs((y_val_true - y_val_pred) / (y_val_true + 1e-8))) * 100
            
            # Prevent infinite or invalid values
            rmse = rmse if np.isfinite(rmse) else 1e6
            mae = mae if np.isfinite(mae) else 1e6
            mape = mape if np.isfinite(mape) else 1e6
            
            compute_time = time.time() - start_time
            
            return {
                'model_name': model_name,
                'seed': seed,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'compute_time': compute_time,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in {model_name} with seed {seed}: {e}")
            return {
                'model_name': model_name,
                'seed': seed,
                'rmse': 1e6,
                'mae': 1e6,
                'mape': 1e6,
                'compute_time': time.time() - start_time,
                'success': False
            }
    
    def run_sensitivity_analysis(self, max_trials=50, max_seeds=50, eval_interval=5):
        """Run fast sensitivity analysis"""
        print("="*80)
        print("FAST MODEL SENSITIVITY ANALYSIS - LSTM_CONFIG2 FOCUS")
        print("="*80)
        
        # Load data
        print("Loading and preprocessing data...")
        data = self.load_and_preprocess_data()
        train_data, val_data = self.create_train_val_split(data)
        
        # Define evaluation ranges
        trial_range = list(range(eval_interval, max_trials + 1, eval_interval))
        seed_range = list(range(1, max_seeds + 1, eval_interval))
        
        print(f"\nEvaluation parameters:")
        print(f"  Trial range: {min(trial_range)} to {max(trial_range)} (step: {eval_interval})")
        print(f"  Seed range: {min(seed_range)} to {max(seed_range)} (step: {eval_interval})")
        print(f"  Total combinations: {len(trial_range)} × {len(seed_range)} = {len(trial_range) * len(seed_range)}")
        print(f"  Models to evaluate: {len(self.model_configs)}")
        
        # Generate base seeds
        base_seeds = [42 + i * 1000 for i in range(max_seeds)]
        
        # Storage for all results
        all_results = []
        
        total_configs = len(self.model_configs)
        
        for config_idx, (model_name, config) in enumerate(self.model_configs.items()):
            print(f"\n{'='*60}")
            print(f"Processing model {config_idx + 1}/{total_configs}: {model_name}")
            print(f"Config: {config}")
            print(f"{'='*60}")
            
            # Run all trials for this model (max_trials × max_seeds)
            model_results = []
            
            for seed_idx in range(max_seeds):
                seed = base_seeds[seed_idx]
                
                if (seed_idx + 1) % 10 == 0:
                    print(f"  Completed {seed_idx + 1}/{max_seeds} seeds...")
                
                for trial in range(max_trials):
                    trial_seed = seed + trial
                    
                    result = self.evaluate_model_single_trial(
                        train_data, val_data, model_name, config, trial_seed
                    )
                    
                    result['seed_idx'] = seed_idx
                    result['trial'] = trial
                    model_results.append(result)
            
            all_results.extend(model_results)
            print(f"  ✓ Completed {len(model_results)} evaluations for {model_name}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        # Save raw results
        self.results_df.to_csv(os.path.join(self.results_dir, 'fast_trial_results.csv'), index=False)
        
        print(f"\n✓ All evaluations completed!")
        print(f"✓ Total evaluations: {len(all_results)}")
        print(f"✓ Success rate: {self.results_df['success'].mean():.1%}")
        
        # Analyze performance across different combinations
        self.analyze_performance_combinations(trial_range, seed_range)
        
        # Analyze convergence patterns
        self.analyze_convergence_patterns(trial_range, seed_range)
        
        return self.results_df
    
    def analyze_performance_combinations(self, trial_range, seed_range):
        """Analyze performance across different (trials, seeds) combinations"""
        print("\nAnalyzing performance combinations...")
        
        performance_data = []
        
        for n_trials in trial_range:
            for n_seeds in seed_range:
                # Filter data for this combination
                filtered_df = self.results_df[
                    (self.results_df['trial'] < n_trials) & 
                    (self.results_df['seed_idx'] < n_seeds)
                ]
                
                if len(filtered_df) == 0:
                    continue
                
                # Calculate statistics for each model
                model_stats = filtered_df.groupby('model_name').agg({
                    'rmse': ['mean', 'median', 'std', 'min', 'max', 'count'],
                    'mae': ['mean', 'median', 'std'],
                    'mape': ['mean', 'median', 'std'],
                    'compute_time': ['sum', 'mean']
                }).round(4)
                
                # Flatten column names
                model_stats.columns = [f"{col[1]}_{col[0]}" for col in model_stats.columns]
                model_stats = model_stats.reset_index()
                
                # Add combination info
                model_stats['n_trials'] = n_trials
                model_stats['n_seeds'] = n_seeds
                model_stats['total_evaluations'] = len(filtered_df)
                
                performance_data.append(model_stats)
        
        # Combine all performance data
        self.performance_df = pd.concat(performance_data, ignore_index=True)
        
        # Save performance analysis
        self.performance_df.to_csv(os.path.join(self.results_dir, 'fast_performance_analysis.csv'), index=False)
        
        print(f"✓ Performance analysis completed for {len(performance_data)} combinations")
    
    def analyze_convergence_patterns(self, trial_range, seed_range):
        """Analyze convergence patterns to determine optimal parameters"""
        print("\nAnalyzing convergence patterns...")
        
        convergence_data = []
        
        # For each model, analyze how performance stabilizes
        for model_name in self.model_configs.keys():
            model_results = self.results_df[self.results_df['model_name'] == model_name]
            
            for n_seeds in seed_range:
                # Filter by seeds
                seed_filtered = model_results[model_results['seed_idx'] < n_seeds]
                
                if len(seed_filtered) == 0:
                    continue
                
                # Calculate running statistics as we add more trials
                running_means = []
                running_stds = []
                running_cvs = []  # Coefficient of variation
                
                for n_trials in trial_range:
                    trial_filtered = seed_filtered[seed_filtered['trial'] < n_trials]
                    
                    if len(trial_filtered) == 0:
                        continue
                    
                    mean_rmse = trial_filtered['rmse'].mean()
                    std_rmse = trial_filtered['rmse'].std()
                    cv_rmse = std_rmse / mean_rmse if mean_rmse > 0 else np.inf
                    
                    running_means.append(mean_rmse)
                    running_stds.append(std_rmse)
                    running_cvs.append(cv_rmse)
                    
                    convergence_data.append({
                        'model_name': model_name,
                        'n_trials': n_trials,
                        'n_seeds': n_seeds,
                        'mean_rmse': mean_rmse,
                        'std_rmse': std_rmse,
                        'cv_rmse': cv_rmse,
                        'total_evaluations': len(trial_filtered)
                    })
        
        self.convergence_df = pd.DataFrame(convergence_data)
        
        # Save convergence analysis
        self.convergence_df.to_csv(os.path.join(self.results_dir, 'convergence_analysis.csv'), index=False)
        
        print(f"✓ Convergence analysis completed")
    
    def find_convergence_point(self, stability_threshold=0.05, min_evaluations=20):
        """Find the point where performance converges (CV < threshold)"""
        print(f"\nFinding convergence point (CV threshold: {stability_threshold})...")
        
        if not hasattr(self, 'convergence_df'):
            print("No convergence data available")
            return None
        
        convergence_points = []
        
        for model_name in self.model_configs.keys():
            model_data = self.convergence_df[
                (self.convergence_df['model_name'] == model_name) &
                (self.convergence_df['total_evaluations'] >= min_evaluations)
            ]
            
            # Find first point where CV drops below threshold
            stable_points = model_data[model_data['cv_rmse'] < stability_threshold]
            
            if len(stable_points) > 0:
                # Get the first stable point
                first_stable = stable_points.iloc[0]
                convergence_points.append({
                    'model_name': model_name,
                    'converged_trials': int(first_stable['n_trials']),
                    'converged_seeds': int(first_stable['n_seeds']),
                    'cv_at_convergence': float(first_stable['cv_rmse']),
                    'mean_rmse_at_convergence': float(first_stable['mean_rmse']),
                    'total_evaluations': int(first_stable['total_evaluations'])
                })
            else:
                # No convergence found
                best_point = model_data.loc[model_data['cv_rmse'].idxmin()]
                convergence_points.append({
                    'model_name': model_name,
                    'converged_trials': int(best_point['n_trials']),
                    'converged_seeds': int(best_point['n_seeds']),
                    'cv_at_convergence': float(best_point['cv_rmse']),
                    'mean_rmse_at_convergence': float(best_point['mean_rmse']),
                    'total_evaluations': int(best_point['total_evaluations']),
                    'converged': False
                })
        
        self.convergence_points = convergence_points
        
        # Save convergence points
        with open(os.path.join(self.results_dir, 'convergence_points.json'), 'w') as f:
            json.dump(convergence_points, f, indent=2)
        
        return convergence_points
    
    def create_fast_plots(self):
        """Create focused visualization plots"""
        print("\nCreating focused visualization plots...")
        
        # 1. Convergence heatmaps
        self.create_convergence_heatmaps()
        
        # 2. Stability analysis
        self.create_stability_plots()
        
        # 3. Recommendations plot
        self.create_recommendations_plot()
        
        print(f"✓ All plots saved to {self.plots_dir}")
    
    def create_convergence_heatmaps(self):
        """Create heatmaps showing convergence patterns"""
        
        if not hasattr(self, 'convergence_df'):
            print("No convergence data for heatmaps")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get data for primary model
        model_name = list(self.model_configs.keys())[0]
        model_data = self.convergence_df[self.convergence_df['model_name'] == model_name]
        
        # 1. Mean RMSE heatmap
        rmse_pivot = model_data.pivot(index='n_seeds', columns='n_trials', values='mean_rmse')
        sns.heatmap(rmse_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title(f'{model_name} - Mean RMSE')
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Number of Seeds')
        
        # 2. Standard deviation heatmap
        std_pivot = model_data.pivot(index='n_seeds', columns='n_trials', values='std_rmse')
        sns.heatmap(std_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'{model_name} - RMSE Standard Deviation')
        axes[0,1].set_xlabel('Number of Trials')
        axes[0,1].set_ylabel('Number of Seeds')
        
        # 3. Coefficient of variation heatmap
        cv_pivot = model_data.pivot(index='n_seeds', columns='n_trials', values='cv_rmse')
        sns.heatmap(cv_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1,0])
        axes[1,0].set_title(f'{model_name} - Coefficient of Variation')
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Number of Seeds')
        
        # Add stability threshold line
        threshold = 0.05
        axes[1,0].contour(cv_pivot.values, levels=[threshold], colors=['red'], 
                         linestyles=['--'], linewidths=2)
        
        # 4. Total evaluations heatmap
        eval_pivot = model_data.pivot(index='n_seeds', columns='n_trials', values='total_evaluations')
        sns.heatmap(eval_pivot, annot=True, fmt='d', cmap='Greens', ax=axes[1,1])
        axes[1,1].set_title(f'{model_name} - Total Evaluations')
        axes[1,1].set_xlabel('Number of Trials')
        axes[1,1].set_ylabel('Number of Seeds')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'convergence_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_stability_plots(self):
        """Create stability analysis plots"""
        
        if not hasattr(self, 'convergence_df'):
            print("No convergence data for stability plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_name = list(self.model_configs.keys())[0]
        model_data = self.convergence_df[self.convergence_df['model_name'] == model_name]
        
        # 1. CV vs Trials (different seed counts)
        unique_seeds = sorted(model_data['n_seeds'].unique())
        selected_seeds = [unique_seeds[0], unique_seeds[len(unique_seeds)//2], unique_seeds[-1]]
        
        for n_seeds in selected_seeds:
            seed_data = model_data[model_data['n_seeds'] == n_seeds].sort_values('n_trials')
            axes[0,0].plot(seed_data['n_trials'], seed_data['cv_rmse'], 
                          marker='o', label=f'{n_seeds} seeds', linewidth=2)
        
        axes[0,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Coefficient of Variation')
        axes[0,0].set_title('Stability vs Trials')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. CV vs Seeds (different trial counts)
        unique_trials = sorted(model_data['n_trials'].unique())
        selected_trials = [unique_trials[0], unique_trials[len(unique_trials)//2], unique_trials[-1]]
        
        for n_trials in selected_trials:
            trial_data = model_data[model_data['n_trials'] == n_trials].sort_values('n_seeds')
            axes[0,1].plot(trial_data['n_seeds'], trial_data['cv_rmse'], 
                          marker='s', label=f'{n_trials} trials', linewidth=2)
        
        axes[0,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
        axes[0,1].set_xlabel('Number of Seeds')
        axes[0,1].set_ylabel('Coefficient of Variation')
        axes[0,1].set_title('Stability vs Seeds')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Mean vs Std scatter
        scatter = axes[1,0].scatter(model_data['mean_rmse'], model_data['std_rmse'],
                                   c=model_data['n_trials'], s=model_data['n_seeds']*5,
                                   alpha=0.7, cmap='viridis')
        axes[1,0].set_xlabel('Mean RMSE')
        axes[1,0].set_ylabel('Standard Deviation RMSE')
        axes[1,0].set_title('Mean vs Std Relationship\n(Color=Trials, Size=Seeds)')
        
        cbar = plt.colorbar(scatter, ax=axes[1,0])
        cbar.set_label('Number of Trials')
        
        # 4. Efficiency plot (CV vs total evaluations)
        scatter = axes[1,1].scatter(model_data['total_evaluations'], model_data['cv_rmse'],
                                   c=model_data['n_trials'], s=model_data['n_seeds']*5,
                                   alpha=0.7, cmap='plasma')
        axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
        axes[1,1].set_xlabel('Total Evaluations')
        axes[1,1].set_ylabel('Coefficient of Variation')
        axes[1,1].set_title('Efficiency: Stability vs Computational Cost')
        axes[1,1].legend()
        
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label('Number of Trials')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'stability_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_recommendations_plot(self):
        """Create recommendations visualization"""
        
        if not hasattr(self, 'convergence_points') or not self.convergence_points:
            print("No convergence points for recommendations plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data from convergence points
        cp = self.convergence_points[0]  # Primary model
        
        # Create summary statistics
        model_data = self.convergence_df[self.convergence_df['model_name'] == cp['model_name']]
        
        # 1. Pareto frontier
        # Calculate efficiency (lower is better for both CV and evaluations)
        model_data['efficiency_score'] = 1 / (model_data['cv_rmse'] * model_data['total_evaluations'])
        
        # Find pareto efficient points
        pareto_points = []
        for _, point in model_data.iterrows():
            is_pareto = True
            for _, other in model_data.iterrows():
                if (other['cv_rmse'] < point['cv_rmse'] and 
                    other['total_evaluations'] <= point['total_evaluations']):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(point)
        
        pareto_df = pd.DataFrame(pareto_points)
        
        axes[0,0].scatter(model_data['total_evaluations'], model_data['cv_rmse'],
                         alpha=0.3, color='gray', label='All configurations')
        
        if len(pareto_df) > 0:
            axes[0,0].scatter(pareto_df['total_evaluations'], pareto_df['cv_rmse'],
                             color='red', s=50, label='Pareto efficient', alpha=0.8)
            
            # Highlight the recommended point
            rec_point = pareto_df[pareto_df['cv_rmse'] < 0.05]
            if len(rec_point) > 0:
                best_rec = rec_point.loc[rec_point['total_evaluations'].idxmin()]
                axes[0,0].scatter(best_rec['total_evaluations'], best_rec['cv_rmse'],
                                 color='green', s=100, marker='*', label='Recommended', alpha=1.0)
        
        axes[0,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
        axes[0,0].set_xlabel('Total Evaluations')
        axes[0,0].set_ylabel('Coefficient of Variation')
        axes[0,0].set_title('Pareto Frontier Analysis')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Recommendation regions
        trial_range = sorted(model_data['n_trials'].unique())
        seed_range = sorted(model_data['n_seeds'].unique())
        
        # Create decision matrix
        decision_matrix = np.zeros((len(seed_range), len(trial_range)))
        
        for i, n_seeds in enumerate(seed_range):
            for j, n_trials in enumerate(trial_range):
                point_data = model_data[
                    (model_data['n_seeds'] == n_seeds) & 
                    (model_data['n_trials'] == n_trials)
                ]
                if len(point_data) > 0:
                    cv = point_data['cv_rmse'].iloc[0]
                    if cv < 0.02:
                        decision_matrix[i, j] = 3  # Excellent
                    elif cv < 0.05:
                        decision_matrix[i, j] = 2  # Good
                    elif cv < 0.1:
                        decision_matrix[i, j] = 1  # Acceptable
                    else:
                        decision_matrix[i, j] = 0  # Poor
        
        im = axes[0,1].imshow(decision_matrix, cmap='RdYlGn', aspect='auto')
        axes[0,1].set_xticks(range(len(trial_range)))
        axes[0,1].set_xticklabels(trial_range)
        axes[0,1].set_yticks(range(len(seed_range)))
        axes[0,1].set_yticklabels(seed_range)
        axes[0,1].set_xlabel('Number of Trials')
        axes[0,1].set_ylabel('Number of Seeds')
        axes[0,1].set_title('Recommendation Regions')
        
        cbar = plt.colorbar(im, ax=axes[0,1])
        cbar.set_label('Quality (0=Poor, 3=Excellent)')
        
        # 3. Cost-benefit analysis
        costs = model_data['total_evaluations']
        benefits = 1 / model_data['cv_rmse']  # Higher is better
        
        axes[1,0].scatter(costs, benefits, c=model_data['n_trials'], 
                         s=model_data['n_seeds']*5, alpha=0.7, cmap='viridis')
        axes[1,0].set_xlabel('Computational Cost (Total Evaluations)')
        axes[1,0].set_ylabel('Benefit (1/CV)')
        axes[1,0].set_title('Cost-Benefit Analysis')
        
        cbar = plt.colorbar(axes[1,0].scatter(costs, benefits, c=model_data['n_trials'], 
                                             s=model_data['n_seeds']*5, alpha=0.7, cmap='viridis'), 
                           ax=axes[1,0])
        cbar.set_label('Number of Trials')
        
        # 4. Summary table
        axes[1,1].axis('off')
        
        # Create recommendation summary
        summary_data = []
        
        # Find different quality tiers
        excellent = model_data[model_data['cv_rmse'] < 0.02]
        good = model_data[(model_data['cv_rmse'] >= 0.02) & (model_data['cv_rmse'] < 0.05)]
        acceptable = model_data[(model_data['cv_rmse'] >= 0.05) & (model_data['cv_rmse'] < 0.1)]
        
        if len(excellent) > 0:
            best = excellent.loc[excellent['total_evaluations'].idxmin()]
            summary_data.append(['Excellent (CV<0.02)', f"{int(best['n_trials'])}", 
                               f"{int(best['n_seeds'])}", f"{best['cv_rmse']:.3f}", 
                               f"{int(best['total_evaluations'])}"])
        
        if len(good) > 0:
            best = good.loc[good['total_evaluations'].idxmin()]
            summary_data.append(['Good (CV<0.05)', f"{int(best['n_trials'])}", 
                               f"{int(best['n_seeds'])}", f"{best['cv_rmse']:.3f}", 
                               f"{int(best['total_evaluations'])}"])
        
        if len(acceptable) > 0:
            best = acceptable.loc[acceptable['total_evaluations'].idxmin()]
            summary_data.append(['Acceptable (CV<0.1)', f"{int(best['n_trials'])}", 
                               f"{int(best['n_seeds'])}", f"{best['cv_rmse']:.3f}", 
                               f"{int(best['total_evaluations'])}"])
        
        if summary_data:
            table = axes[1,1].table(cellText=summary_data,
                                   colLabels=['Quality Tier', 'Trials', 'Seeds', 'CV', 'Evaluations'],
                                   cellLoc='center',
                                   loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        axes[1,1].set_title('Recommendations by Quality Tier')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'recommendations_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_fast_report(self):
        """Generate focused analysis report"""
        
        print("\nGenerating focused analysis report...")
        
        # Find convergence points
        convergence_points = self.find_convergence_point()
        
        # Calculate summary statistics
        models = list(self.model_configs.keys())
        total_evaluations = len(self.results_df)
        success_rate = self.results_df['success'].mean()
        
        # Get performance statistics
        if hasattr(self, 'convergence_df') and len(self.convergence_df) > 0:
            best_cv = self.convergence_df['cv_rmse'].min()
            best_config = self.convergence_df.loc[self.convergence_df['cv_rmse'].idxmin()]
        else:
            best_cv = None
            best_config = None
        
        # Create report
        report = {
            'analysis_summary': {
                'analysis_type': 'Fast LSTM-focused sensitivity analysis',
                'models_evaluated': models,
                'total_evaluations': int(total_evaluations),
                'success_rate': float(success_rate),
                'analysis_date': datetime.now().isoformat()
            },
            'convergence_analysis': convergence_points[0] if convergence_points else None,
            'best_configuration': {
                'trials': int(best_config['n_trials']) if best_config is not None else None,
                'seeds': int(best_config['n_seeds']) if best_config is not None else None,
                'cv_rmse': float(best_cv) if best_cv is not None else None,
                'mean_rmse': float(best_config['mean_rmse']) if best_config is not None else None,
                'total_evaluations': int(best_config['total_evaluations']) if best_config is not None else None
            },
            'computational_analysis': {
                'total_compute_time': float(self.results_df['compute_time'].sum()),
                'avg_time_per_evaluation': float(self.results_df['compute_time'].mean()),
                'evaluations_per_second': float(total_evaluations / self.results_df['compute_time'].sum()) if self.results_df['compute_time'].sum() > 0 else 0
            }
        }
        
        # Save detailed report
        with open(os.path.join(self.results_dir, 'fast_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_text = f"""
FAST MODEL SENSITIVITY ANALYSIS - FOCUSED REPORT
================================================

ANALYSIS OVERVIEW:
- Analysis Type: LSTM_Config2 Focused Analysis
- Total Evaluations: {total_evaluations:,}
- Success Rate: {success_rate:.1%}
- Focus Model: {models[0] if models else 'N/A'}

CONVERGENCE ANALYSIS:
"""
        
        if convergence_points and convergence_points[0]:
            cp = convergence_points[0]
            converged = cp.get('converged', True)
            
            if converged:
                summary_text += f"""
✓ CONVERGENCE ACHIEVED:
  - Recommended Trials: {cp['converged_trials']}
  - Recommended Seeds: {cp['converged_seeds']}
  - Coefficient of Variation: {cp['cv_at_convergence']:.4f}
  - Mean RMSE at Convergence: {cp['mean_rmse_at_convergence']:.4f}
  - Total Evaluations Required: {cp['total_evaluations']}
"""
            else:
                summary_text += f"""
⚠ PARTIAL CONVERGENCE:
  - Best Available Trials: {cp['converged_trials']}
  - Best Available Seeds: {cp['converged_seeds']}
  - Best CV Achieved: {cp['cv_at_convergence']:.4f}
  - Mean RMSE: {cp['mean_rmse_at_convergence']:.4f}
  - Consider increasing max parameters for full convergence
"""
        else:
            summary_text += """
⚠ NO CONVERGENCE DATA AVAILABLE
  - Analysis may have encountered issues
  - Check raw results for details
"""
        
        if best_config is not None:
            summary_text += f"""
BEST CONFIGURATION FOUND:
- Optimal Trials: {int(best_config['n_trials'])}
- Optimal Seeds: {int(best_config['n_seeds'])}
- Coefficient of Variation: {best_cv:.4f}
- Mean RMSE: {best_config['mean_rmse']:.4f}
- Total Evaluations: {int(best_config['total_evaluations'])}
"""
        
        summary_text += f"""
COMPUTATIONAL EFFICIENCY:
- Total Compute Time: {report['computational_analysis']['total_compute_time']:.1f} seconds
- Average Time per Evaluation: {report['computational_analysis']['avg_time_per_evaluation']:.2f} seconds
- Throughput: {report['computational_analysis']['evaluations_per_second']:.1f} evaluations/second

RECOMMENDATIONS FOR FULL MODEL EVALUATION:
"""
        
        if convergence_points and convergence_points[0]:
            cp = convergence_points[0]
            summary_text += f"""
1. PROXY-BASED RECOMMENDATION:
   - Use {cp['converged_trials']} trials and {cp['converged_seeds']} seeds for all models
   - Expected stability: CV ≈ {cp['cv_at_convergence']:.4f}
   - Computational cost per model: ~{cp['total_evaluations']} evaluations

2. CONFIDENCE LEVEL: {'High' if cp.get('converged', True) else 'Medium'}
   - LSTM_Config2 showed {'stable' if cp.get('converged', True) else 'improving'} performance
   - Other models likely to follow similar convergence patterns
   - Consider validation with 1-2 additional models if critical

3. IMPLEMENTATION STRATEGY:
   - Start with recommended parameters for full hyperparameter search
   - Monitor early results for consistency with proxy analysis
   - Adjust if other models show significantly different behavior
"""
        else:
            summary_text += f"""
1. FALLBACK RECOMMENDATION:
   - Use maximum tested parameters as conservative estimate
   - Monitor convergence during full model evaluation
   - Consider increasing evaluation scope if instability observed

2. VALIDATION NEEDED:
   - Proxy analysis inconclusive
   - Test 2-3 additional models before full implementation
   - Use adaptive approach during hyperparameter search
"""
        
        summary_text += f"""
OUTPUT FILES:
- Raw Results: {self.results_dir}/fast_trial_results.csv
- Performance Analysis: {self.results_dir}/fast_performance_analysis.csv
- Convergence Analysis: {self.results_dir}/convergence_analysis.csv
- Convergence Points: {self.results_dir}/convergence_points.json
- Analysis Report: {self.results_dir}/fast_analysis_report.json
- Visualization Plots: {self.plots_dir}/

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save summary
        with open(os.path.join(self.output_dir, 'FAST_ANALYSIS_SUMMARY.txt'), 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\nFast analysis report saved to {self.output_dir}")
        
        return report


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Fast Model Sensitivity Analysis - LSTM Focus')
    
    parser.add_argument('--max_trials', type=int, default=50,
                       help='Maximum number of trials to test (default: 50)')
    
    parser.add_argument('--max_seeds', type=int, default=50,
                       help='Maximum number of seeds to test (default: 50)')
    
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval for trials and seeds (default: 5)')
    
    parser.add_argument('--data_path', type=str, 
                       default='data_updated/state_month_overdose_2015_2023.xlsx',
                       help='Path to data file')
    
    parser.add_argument('--stability_threshold', type=float, default=0.05,
                       help='CV stability threshold (default: 0.05)')
    
    parser.add_argument('--use_full_models', action='store_true',
                       help='Use multiple LSTM configs instead of just LSTM_Config2')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_trials < args.eval_interval:
        print(f"Error: max_trials ({args.max_trials}) must be >= eval_interval ({args.eval_interval})")
        return
    
    if args.max_seeds < args.eval_interval:
        print(f"Error: max_seeds ({args.max_seeds}) must be >= eval_interval ({args.eval_interval})")
        return
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    print("="*80)
    print("FAST MODEL SENSITIVITY ANALYSIS - LSTM FOCUS")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Max Trials: {args.max_trials}")
    print(f"Max Seeds: {args.max_seeds}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Stability Threshold: {args.stability_threshold}")
    print(f"Use Full Models: {args.use_full_models}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = FastModelSensitivityAnalyzer(args.data_path, args.use_full_models)
    
    try:
        # Run fast sensitivity analysis
        print("Starting fast sensitivity analysis...")
        results_df = analyzer.run_sensitivity_analysis(
            max_trials=args.max_trials,
            max_seeds=args.max_seeds,
            eval_interval=args.eval_interval
        )
        
        # Find convergence points
        convergence_points = analyzer.find_convergence_point(
            stability_threshold=args.stability_threshold
        )
        
        # Create visualizations
        print("\nGenerating focused visualizations...")
        analyzer.create_fast_plots()
        
        # Generate final report
        print("\nGenerating analysis report...")
        final_report = analyzer.generate_fast_report()
        
        print("\n" + "="*80)
        print("FAST ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {analyzer.output_dir}")
        
        # Display key recommendations
        if convergence_points and convergence_points[0]:
            cp = convergence_points[0]
            print(f"\n🎯 KEY RECOMMENDATION:")
            print(f"   Use {cp['converged_trials']} trials and {cp['converged_seeds']} seeds")
            print(f"   Coefficient of Variation: {cp['cv_at_convergence']:.4f}")
            print(f"   Total evaluations per model: {cp['total_evaluations']}")
            print(f"   Confidence: {'High' if cp.get('converged', True) else 'Medium'}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        print(f"Partial results may be available in: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be available in: {analyzer.output_dir}")
        return


if __name__ == "__main__":
    main()