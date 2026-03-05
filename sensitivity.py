#!/usr/bin/env python3
"""
Enhanced Model Sensitivity Analysis Script

This script performs comprehensive sensitivity analysis to determine the optimal
combination of trials and random seeds for stable model evaluation and ranking.

Key Features:
1. Evaluates model performance across different (trials, seeds) combinations
2. Analyzes ranking stability using correlation metrics
3. Creates comprehensive visualizations including heatmaps and convergence plots
4. Determines minimum requirements for stable model rankings
5. Optimizes for computational efficiency

Usage:
    python enhanced_sensitivity_analysis.py --max_trials 100 --max_seeds 100 --eval_interval 5
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

try:
    from tcn import TCN
except ImportError:
    print("Warning: TCN not available. Install with: pip install keras-tcn")
    TCN = None

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class EnhancedModelSensitivityAnalyzer:
    """
    Enhanced analyzer for model sensitivity to trials and seeds with comprehensive evaluation
    """
    
    def __init__(self, data_path='data_updated/state_month_overdose_2015_2023.xlsx'):
        self.data_path = data_path
        
        # Create output directories
        self.output_dir = 'enhanced_sensitivity_analysis'
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        for dir_path in [self.output_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Define model configurations matching your grid search script
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
                'epochs': 100
            },
            'LSTM_Config3': {
                'model_type': 'lstm',
                'lookback': 12,
                'batch_size': 32,
                'epochs': 50
            },
            'TCN_Config1': {
                'model_type': 'tcn',
                'lookback': 5,
                'batch_size': 16,
                'epochs': 50
            },
            'TCN_Config2': {
                'model_type': 'tcn',
                'lookback': 9,
                'batch_size': 8,
                'epochs': 100
            },
            'SARIMA_Config1': {
                'model_type': 'sarima',
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 12)
            },
            'SARIMA_Config2': {
                'model_type': 'sarima',
                'order': (2, 1, 1),
                'seasonal_order': (2, 2, 2, 12)
            },
            'Seq2Seq_Config1': {
                'model_type': 'seq2seq',
                'lookback': 7,
                'batch_size': 16,
                'epochs': 50,
                'encoder_units': 64,
                'decoder_units': 64,
                'use_attention': False
            },
            'Seq2SeqAttn_Config1': {
                'model_type': 'seq2seq_attn',
                'lookback': 9,
                'batch_size': 8,
                'epochs': 75,
                'encoder_units': 128,
                'decoder_units': 128,
                'use_attention': True
            },
            'Transformer_Config1': {
                'model_type': 'transformer',
                'lookback': 7,
                'batch_size': 16,
                'epochs': 50,
                'd_model': 64,
                'n_heads': 2
            }
        }
        
        # Storage for results
        self.all_metrics = []
        self.performance_summary = {}
        self.ranking_analysis = {}
        
        print(f"Initialized Enhanced Model Sensitivity Analyzer")
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
    
    def build_tcn_model(self, config):
        """Build TCN model matching your grid search implementation"""
        if TCN is None:
            raise ValueError("TCN not available")
        
        model = Sequential([
            TCN(input_shape=(config['lookback'], 1), dilations=[1, 2, 4, 8], 
                nb_filters=64, kernel_size=3, dropout_rate=0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_seq2seq_model(self, config):
        """Build seq2seq model matching your grid search implementation"""
        lookback = config['lookback']
        encoder_units = config['encoder_units']
        decoder_units = config['decoder_units']
        use_attention = config['use_attention']
        
        encoder_inputs = Input(shape=(lookback, 1), name='encoder_input')
        
        if use_attention:
            encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
            encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
            
            if encoder_units != decoder_units:
                encoder_outputs_proj = Dense(decoder_units, name='encoder_proj')(encoder_outputs)
                encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
            else:
                encoder_outputs_proj = encoder_outputs
                
            decoder_inputs = Input(shape=(1, 1), name='decoder_input')
            decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
            decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
            
            # Use TensorFlow's Attention layer
            from tensorflow.keras.layers import Attention
            attention_layer = Attention(name='attention')
            context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
            decoder_combined = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
            decoder_hidden = Dense(decoder_units, activation='relu', name='decoder_hidden')(decoder_combined)
            decoder_outputs = Dense(1, name='output_dense')(decoder_hidden)
        else:
            encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
            _, encoder_state = encoder_gru(encoder_inputs)
            
            if encoder_units != decoder_units:
                encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
                
            decoder_inputs = Input(shape=(1, 1), name='decoder_input')
            decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
            decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
            decoder_outputs = Dense(1, name='decoder_dense')(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
        return model
    
    class PositionalEncoding(tf.keras.layers.Layer):
        """Positional encoding for transformer"""
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = np.zeros((max_len, d_model))
            for pos in range(max_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                    if i+1 < d_model:
                        pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
            self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

        def call(self, x):
            return x + self.pe[:, :tf.shape(x)[1], :]
    
    def build_transformer_model(self, config):
        """Build transformer model matching your grid search implementation"""
        lookback = config['lookback']
        d_model = config['d_model']
        n_heads = config['n_heads']
        
        inputs = Input(shape=(lookback, 1))
        x = Dense(d_model)(inputs)
        x = self.PositionalEncoding(d_model)(x)
        
        attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
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
    
    def train_tcn_model(self, train_data, val_data, config, seed):
        """Train TCN model matching your grid search implementation"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Prepare data
        train_values = train_data['Deaths'].values.astype(np.float32)
        val_values = val_data['Deaths'].values.astype(np.float32)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_values, config['lookback'])
        X_train = X_train.reshape((X_train.shape[0], config['lookback'], 1))
        
        # Build and train model
        model = self.build_tcn_model(config)
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
    
    def train_sarima_model(self, train_data, val_data, config, seed):
        """Train SARIMA model matching your grid search implementation"""
        np.random.seed(seed)
        
        train_series = train_data['Deaths'].astype(float)
        val_series = val_data['Deaths'].astype(float)
        
        try:
            model = SARIMAX(train_series,
                           order=config['order'],
                           seasonal_order=config['seasonal_order'],
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            results = model.fit(disp=False, maxiter=100)
            
            # Get fitted values and forecast
            fitted = results.fittedvalues
            forecast = results.predict(start=len(train_series), end=len(train_series) + len(val_series) - 1)
            
            return train_series.values, fitted.values, val_series.values, forecast.values
            
        except Exception as e:
            print(f"SARIMA training failed with seed {seed}: {e}")
            # Fallback to simple forecast
            train_mean = train_series.mean()
            fitted = np.full_like(train_series, train_mean)
            forecast = np.full_like(val_series, train_mean)
            return train_series.values, fitted, val_series.values, forecast
    
    def train_seq2seq_model(self, train_data, val_data, config, seed):
        """Train seq2seq model matching your grid search implementation"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Prepare data
        train_values = train_data['Deaths'].values.astype(np.float32)
        val_values = val_data['Deaths'].values.astype(np.float32)
        
        # Scaling
        full_series = np.concatenate([train_values, val_values])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_scaled = scaled_full[:len(train_values)]
        val_scaled = scaled_full[len(train_values):]
        
        # Prepare training data
        X_train, y_train = self.create_sequences(train_scaled, config['lookback'])
        X_train = X_train.reshape((X_train.shape[0], config['lookback'], 1))
        decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
        y_train = y_train.reshape((-1, 1, 1))
        
        # Build and train model
        model = self.build_seq2seq_model(config)
        
        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        model.fit([X_train, decoder_input_train], y_train, epochs=config['epochs'], 
                 batch_size=config['batch_size'], verbose=0, callbacks=[early_stopping], validation_split=0.1)
        
        # Generate training predictions
        train_preds_scaled = []
        for i in range(config['lookback'], len(train_values)):
            encoder_input = train_scaled[i-config['lookback']:i].reshape((1, config['lookback'], 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            train_preds_scaled.append(pred_scaled)
        
        # Generate test predictions (autoregressive)
        val_preds_scaled = []
        current_sequence = train_scaled[-config['lookback']:].copy()
        
        for _ in range(len(val_values)):
            encoder_input = current_sequence.reshape((1, config['lookback'], 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            val_preds_scaled.append(pred_scaled)
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        val_preds_original = scaler.inverse_transform(np.array(val_preds_scaled).reshape(-1, 1)).flatten()
        
        return train_values[config['lookback']:], train_preds_original, val_values, val_preds_original
    
    def train_transformer_model(self, train_data, val_data, config, seed):
        """Train transformer model matching your grid search implementation"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Prepare data
        train_values = train_data['Deaths'].values.astype(np.float32)
        val_values = val_data['Deaths'].values.astype(np.float32)
        
        # Scaling
        full_series = np.concatenate([train_values, val_values])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_scaled = scaled_full[:len(train_values)]
        val_scaled = scaled_full[len(train_values):]
        
        # Prepare data
        X_train, y_train = self.create_sequences(train_scaled, config['lookback'])
        X_train = X_train.reshape((X_train.shape[0], config['lookback'], 1))
        y_train = y_train.reshape((-1, 1))
        
        # Build and train model
        model = self.build_transformer_model(config)
        model.fit(X_train, y_train, batch_size=config['batch_size'], epochs=config['epochs'], verbose=0)
        
        # Generate training predictions
        train_preds_scaled = []
        for i in range(config['lookback'], len(train_values)):
            input_seq = train_scaled[i-config['lookback']:i].reshape((1, config['lookback'], 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            train_preds_scaled.append(pred_scaled)
        
        # Generate test predictions (autoregressive)
        current_seq = train_scaled[-config['lookback']:].copy()
        val_preds_scaled = []
        for _ in range(len(val_values)):
            input_seq = current_seq.reshape((1, config['lookback'], 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            val_preds_scaled.append(pred_scaled)
            current_seq = np.append(current_seq[1:], pred_scaled)
        
        # Inverse transform predictions
        train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        val_preds_original = scaler.inverse_transform(np.array(val_preds_scaled).reshape(-1, 1)).flatten()
        
        return train_values[config['lookback']:], train_preds_original, val_values, val_preds_original
    
    def evaluate_model_single_trial(self, train_data, val_data, model_name, config, seed):
        """Evaluate a single model trial"""
        start_time = time.time()
        
        try:
            if config['model_type'] == 'lstm':
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_lstm_model(train_data, val_data, config, seed)
            elif config['model_type'] == 'tcn':
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_tcn_model(train_data, val_data, config, seed)
            elif config['model_type'] == 'sarima':
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_sarima_model(train_data, val_data, config, seed)
            elif config['model_type'] in ['seq2seq', 'seq2seq_attn']:
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_seq2seq_model(train_data, val_data, config, seed)
            elif config['model_type'] == 'transformer':
                y_train_true, y_train_pred, y_val_true, y_val_pred = self.train_transformer_model(train_data, val_data, config, seed)
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
    
    def run_sensitivity_analysis(self, max_trials=100, max_seeds=100, eval_interval=5):
        """Run comprehensive sensitivity analysis"""
        print("="*80)
        print("ENHANCED MODEL SENSITIVITY ANALYSIS")
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
                
                if (seed_idx + 1) % 20 == 0:
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
        self.results_df.to_csv(os.path.join(self.results_dir, 'all_trial_results.csv'), index=False)
        
        print(f"\n✓ All evaluations completed!")
        print(f"✓ Total evaluations: {len(all_results)}")
        print(f"✓ Success rate: {self.results_df['success'].mean():.1%}")
        
        # Analyze performance across different combinations
        self.analyze_performance_combinations(trial_range, seed_range)
        
        # Analyze ranking stability
        self.analyze_ranking_stability(trial_range, seed_range)
        
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
                    'rmse': ['mean', 'median', 'std', 'min', 'max'],
                    'mae': ['mean', 'median', 'std', 'min', 'max'],
                    'mape': ['mean', 'median', 'std', 'min', 'max'],
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
        self.performance_df.to_csv(os.path.join(self.results_dir, 'performance_analysis.csv'), index=False)
        
        print(f"✓ Performance analysis completed for {len(performance_data)} combinations")
    
    def analyze_ranking_stability(self, trial_range, seed_range):
        """Analyze ranking stability across combinations"""
        print("\nAnalyzing ranking stability...")
        
        ranking_data = []
        
        for n_trials in trial_range:
            for n_seeds in seed_range:
                # Filter data
                filtered_df = self.results_df[
                    (self.results_df['trial'] < n_trials) & 
                    (self.results_df['seed_idx'] < n_seeds)
                ]
                
                if len(filtered_df) == 0:
                    continue
                
                # Calculate model rankings based on mean RMSE
                model_performance = filtered_df.groupby('model_name')['rmse'].mean()
                rankings = model_performance.rank(ascending=True)
                
                ranking_data.append({
                    'n_trials': n_trials,
                    'n_seeds': n_seeds,
                    'rankings': rankings.to_dict(),
                    'total_compute_time': filtered_df['compute_time'].sum(),
                    'mean_rmse': model_performance.to_dict()
                })
        
        # Calculate ranking correlations with reference (max trials, max seeds)
        if ranking_data:
            reference_rankings = ranking_data[-1]['rankings']  # Last entry should be max combination
            
            for entry in ranking_data:
                current_rankings = entry['rankings']
                
                # Calculate correlation with reference
                common_models = set(reference_rankings.keys()) & set(current_rankings.keys())
                if len(common_models) >= 2:
                    ref_ranks = [reference_rankings[model] for model in common_models]
                    cur_ranks = [current_rankings[model] for model in common_models]
                    
                    spearman_corr, _ = spearmanr(ref_ranks, cur_ranks)
                    kendall_corr, _ = kendalltau(ref_ranks, cur_ranks)
                    
                    entry['spearman_correlation'] = spearman_corr
                    entry['kendall_correlation'] = kendall_corr
                else:
                    entry['spearman_correlation'] = np.nan
                    entry['kendall_correlation'] = np.nan
        
        self.ranking_analysis = ranking_data
        
        # Save ranking analysis
        with open(os.path.join(self.results_dir, 'ranking_analysis.json'), 'w') as f:
            # Convert to serializable format
            serializable_data = []
            for entry in ranking_data:
                serializable_entry = {
                    'n_trials': int(entry['n_trials']),
                    'n_seeds': int(entry['n_seeds']),
                    'rankings': {k: float(v) for k, v in entry['rankings'].items()},
                    'total_compute_time': float(entry['total_compute_time']),
                    'mean_rmse': {k: float(v) for k, v in entry['mean_rmse'].items()},
                    'spearman_correlation': float(entry.get('spearman_correlation', 0)),
                    'kendall_correlation': float(entry.get('kendall_correlation', 0))
                }
                serializable_data.append(serializable_entry)
            
            json.dump(serializable_data, f, indent=2)
        
        print(f"✓ Ranking stability analysis completed for {len(ranking_data)} combinations")
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots"""
        print("\nCreating comprehensive plots...")
        
        # 1. Performance heatmaps
        self.create_performance_heatmaps()
        
        # 2. Convergence plots
        self.create_convergence_plots()
        
        # 3. Ranking stability plots
        self.create_ranking_stability_plots()
        
        # 4. Efficiency analysis plots
        self.create_efficiency_plots()
        
        # 5. Model comparison plots
        self.create_model_comparison_plots()
        
        print(f"✓ All plots saved to {self.plots_dir}")
    
    def create_performance_heatmaps(self):
        """Create heatmaps for different performance metrics"""
        
        # Get unique combinations
        trial_values = sorted(self.performance_df['n_trials'].unique())
        seed_values = sorted(self.performance_df['n_seeds'].unique())
        models = sorted(self.performance_df['model_name'].unique())
        
        # Create subplot for each model
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # RMSE heatmaps
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model in enumerate(models):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create pivot table for heatmap
            model_data = self.performance_df[self.performance_df['model_name'] == model]
            pivot_data = model_data.pivot(index='n_seeds', columns='n_trials', values='mean_rmse')
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{model} - Mean RMSE')
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Number of Seeds')
        
        # Hide unused subplots
        if n_models < n_rows * n_cols:
            for idx in range(n_models, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rmse_heatmaps_by_model.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Overall heatmap showing best model performance
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean RMSE across all models
        overall_rmse = self.performance_df.groupby(['n_trials', 'n_seeds'])['mean_rmse'].mean().reset_index()
        rmse_pivot = overall_rmse.pivot(index='n_seeds', columns='n_trials', values='mean_rmse')
        
        sns.heatmap(rmse_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Overall Mean RMSE')
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Number of Seeds')
        
        # Standard deviation of RMSE
        overall_std = self.performance_df.groupby(['n_trials', 'n_seeds'])['std_rmse'].mean().reset_index()
        std_pivot = overall_std.pivot(index='n_seeds', columns='n_trials', values='std_rmse')
        
        sns.heatmap(std_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Mean RMSE Standard Deviation')
        axes[0,1].set_xlabel('Number of Trials')
        axes[0,1].set_ylabel('Number of Seeds')
        
        # Compute time
        compute_time = self.performance_df.groupby(['n_trials', 'n_seeds'])['sum_compute_time'].mean().reset_index()
        time_pivot = compute_time.pivot(index='n_seeds', columns='n_trials', values='sum_compute_time')
        
        sns.heatmap(time_pivot, annot=True, fmt='.1f', cmap='Greens', ax=axes[1,0])
        axes[1,0].set_title('Total Compute Time (seconds)')
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Number of Seeds')
        
        # Ranking stability (Spearman correlation)
        if hasattr(self, 'ranking_analysis') and self.ranking_analysis:
            ranking_df = pd.DataFrame(self.ranking_analysis)
            corr_pivot = ranking_df.pivot(index='n_seeds', columns='n_trials', values='spearman_correlation')
            
            sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1,1])
            axes[1,1].set_title('Ranking Stability (Spearman Correlation)')
            axes[1,1].set_xlabel('Number of Trials')
            axes[1,1].set_ylabel('Number of Seeds')
        else:
            axes[1,1].text(0.5, 0.5, 'Ranking analysis\nnot available', 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title('Ranking Stability')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'comprehensive_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_convergence_plots(self):
        """Create convergence analysis plots"""
        
        # Get unique values
        trial_values = sorted(self.performance_df['n_trials'].unique())
        seed_values = sorted(self.performance_df['n_seeds'].unique())
        models = sorted(self.performance_df['model_name'].unique())
        
        # Set up color palette for models
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        color_map = dict(zip(models, colors))
        
        # Create convergence plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RMSE vs Trials (fixed seeds)
        selected_seeds = [seed_values[0], seed_values[len(seed_values)//2], seed_values[-1]]
        
        for seed_val in selected_seeds:
            for model in models:
                model_data = self.performance_df[
                    (self.performance_df['model_name'] == model) & 
                    (self.performance_df['n_seeds'] == seed_val)
                ].sort_values('n_trials')
                
                if len(model_data) > 0:
                    axes[0,0].plot(model_data['n_trials'], model_data['mean_rmse'], 
                                  color=color_map[model], marker='o', 
                                  label=f'{model} ({seed_val} seeds)' if seed_val == selected_seeds[0] else "",
                                  alpha=0.7, linewidth=2)
        
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Mean RMSE')
        axes[0,0].set_title('RMSE Convergence vs Trials\n(Different seed counts)')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. RMSE vs Seeds (fixed trials)
        selected_trials = [trial_values[0], trial_values[len(trial_values)//2], trial_values[-1]]
        
        for trial_val in selected_trials:
            for model in models:
                model_data = self.performance_df[
                    (self.performance_df['model_name'] == model) & 
                    (self.performance_df['n_trials'] == trial_val)
                ].sort_values('n_seeds')
                
                if len(model_data) > 0:
                    axes[0,1].plot(model_data['n_seeds'], model_data['mean_rmse'], 
                                  color=color_map[model], marker='s', 
                                  label=f'{model} ({trial_val} trials)' if trial_val == selected_trials[0] else "",
                                  alpha=0.7, linewidth=2)
        
        axes[0,1].set_xlabel('Number of Seeds')
        axes[0,1].set_ylabel('Mean RMSE')
        axes[0,1].set_title('RMSE Convergence vs Seeds\n(Different trial counts)')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Standard deviation convergence
        for seed_val in selected_seeds:
            for model in models:
                model_data = self.performance_df[
                    (self.performance_df['model_name'] == model) & 
                    (self.performance_df['n_seeds'] == seed_val)
                ].sort_values('n_trials')
                
                if len(model_data) > 0:
                    axes[1,0].plot(model_data['n_trials'], model_data['std_rmse'], 
                                  color=color_map[model], marker='o', 
                                  label=f'{model} ({seed_val} seeds)' if seed_val == selected_seeds[0] else "",
                                  alpha=0.7, linewidth=2)
        
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('RMSE Standard Deviation')
        axes[1,0].set_title('RMSE Variability vs Trials')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Coefficient of variation (std/mean)
        for seed_val in selected_seeds:
            for model in models:
                model_data = self.performance_df[
                    (self.performance_df['model_name'] == model) & 
                    (self.performance_df['n_seeds'] == seed_val)
                ].sort_values('n_trials')
                
                if len(model_data) > 0:
                    cv = model_data['std_rmse'] / model_data['mean_rmse']
                    axes[1,1].plot(model_data['n_trials'], cv, 
                                  color=color_map[model], marker='o', 
                                  label=f'{model} ({seed_val} seeds)' if seed_val == selected_seeds[0] else "",
                                  alpha=0.7, linewidth=2)
        
        axes[1,1].set_xlabel('Number of Trials')
        axes[1,1].set_ylabel('Coefficient of Variation (Std/Mean)')
        axes[1,1].set_title('Relative Variability vs Trials')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'convergence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ranking_stability_plots(self):
        """Create ranking stability analysis plots"""
        
        if not hasattr(self, 'ranking_analysis') or not self.ranking_analysis:
            print("No ranking analysis data available")
            return
        
        ranking_df = pd.DataFrame(self.ranking_analysis)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Ranking stability heatmap (already created in heatmaps, but more detailed)
        corr_pivot = ranking_df.pivot(index='n_seeds', columns='n_trials', values='spearman_correlation')
        
        sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,0], cbar_kws={'label': 'Spearman Correlation'})
        axes[0,0].set_title('Ranking Stability (Spearman Correlation)')
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Number of Seeds')
        
        # Add stability threshold line
        threshold = 0.9
        axes[0,0].contour(corr_pivot.values, levels=[threshold], colors=['red'], 
                         linestyles=['--'], linewidths=2)
        
        # 2. Kendall correlation
        kendall_pivot = ranking_df.pivot(index='n_seeds', columns='n_trials', values='kendall_correlation')
        
        sns.heatmap(kendall_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,1], cbar_kws={'label': 'Kendall Correlation'})
        axes[0,1].set_title('Ranking Stability (Kendall Correlation)')
        axes[0,1].set_xlabel('Number of Trials')
        axes[0,1].set_ylabel('Number of Seeds')
        
        # 3. Convergence of ranking stability
        trial_values = sorted(ranking_df['n_trials'].unique())
        seed_values = sorted(ranking_df['n_seeds'].unique())
        
        # For different seed counts, show correlation vs trials
        selected_seeds = [seed_values[0], seed_values[len(seed_values)//2], seed_values[-1]]
        
        for seed_val in selected_seeds:
            seed_data = ranking_df[ranking_df['n_seeds'] == seed_val].sort_values('n_trials')
            axes[1,0].plot(seed_data['n_trials'], seed_data['spearman_correlation'], 
                          marker='o', label=f'{seed_val} seeds', linewidth=2)
        
        axes[1,0].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                         label=f'Stability threshold ({threshold})')
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Spearman Correlation')
        axes[1,0].set_title('Ranking Stability vs Trials')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Efficiency plot (correlation vs compute time)
        scatter = axes[1,1].scatter(ranking_df['total_compute_time'], 
                                   ranking_df['spearman_correlation'],
                                   c=ranking_df['n_trials'], 
                                   s=ranking_df['n_seeds']*5,
                                   alpha=0.7, cmap='viridis')
        
        axes[1,1].axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
        axes[1,1].set_xlabel('Total Compute Time (seconds)')
        axes[1,1].set_ylabel('Spearman Correlation')
        axes[1,1].set_title('Efficiency: Correlation vs Compute Time\n(Color=Trials, Size=Seeds)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label('Number of Trials')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'ranking_stability_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_efficiency_plots(self):
        """Create efficiency analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Compute time vs performance trade-off
        trial_values = sorted(self.performance_df['n_trials'].unique())
        seed_values = sorted(self.performance_df['n_seeds'].unique())
        
        # Aggregate performance across models
        agg_performance = self.performance_df.groupby(['n_trials', 'n_seeds']).agg({
            'mean_rmse': 'mean',
            'std_rmse': 'mean',
            'sum_compute_time': 'sum'
        }).reset_index()
        
        scatter = axes[0,0].scatter(agg_performance['sum_compute_time'], 
                                   agg_performance['mean_rmse'],
                                   c=agg_performance['n_trials'], 
                                   s=agg_performance['n_seeds']*3,
                                   alpha=0.7, cmap='plasma')
        
        axes[0,0].set_xlabel('Total Compute Time (seconds)')
        axes[0,0].set_ylabel('Mean RMSE')
        axes[0,0].set_title('Performance vs Compute Time\n(Color=Trials, Size=Seeds)')
        
        cbar = plt.colorbar(scatter, ax=axes[0,0])
        cbar.set_label('Number of Trials')
        
        # 2. Pareto frontier analysis
        # Calculate efficiency score (inverse RMSE per compute time)
        agg_performance['efficiency'] = 1 / (agg_performance['mean_rmse'] * agg_performance['sum_compute_time'])
        
        # Sort by efficiency and highlight top configurations
        top_configs = agg_performance.nlargest(10, 'efficiency')
        
        axes[0,1].scatter(agg_performance['sum_compute_time'], agg_performance['mean_rmse'],
                         alpha=0.3, color='gray', label='All configurations')
        axes[0,1].scatter(top_configs['sum_compute_time'], top_configs['mean_rmse'],
                         color='red', s=50, label='Top 10 efficient', alpha=0.8)
        
        axes[0,1].set_xlabel('Total Compute Time (seconds)')
        axes[0,1].set_ylabel('Mean RMSE')
        axes[0,1].set_title('Efficiency Analysis (Pareto Frontier)')
        axes[0,1].legend()
        
        # 3. Stability vs efficiency
        if hasattr(self, 'ranking_analysis') and self.ranking_analysis:
            ranking_df = pd.DataFrame(self.ranking_analysis)
            
            # Merge with performance data
            merged_data = pd.merge(agg_performance, ranking_df, on=['n_trials', 'n_seeds'])
            
            scatter = axes[1,0].scatter(merged_data['efficiency'], 
                                       merged_data['spearman_correlation'],
                                       c=merged_data['n_trials'], 
                                       s=merged_data['n_seeds']*3,
                                       alpha=0.7, cmap='viridis')
            
            axes[1,0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
            axes[1,0].set_xlabel('Efficiency Score')
            axes[1,0].set_ylabel('Ranking Stability (Spearman)')
            axes[1,0].set_title('Stability vs Efficiency')
            axes[1,0].legend()
            
            cbar = plt.colorbar(scatter, ax=axes[1,0])
            cbar.set_label('Number of Trials')
        
        # 4. Diminishing returns analysis
        # Show how additional trials/seeds provide diminishing returns
        selected_seeds = [seed_values[0], seed_values[len(seed_values)//2], seed_values[-1]]
        
        for seed_val in selected_seeds:
            seed_data = agg_performance[agg_performance['n_seeds'] == seed_val].sort_values('n_trials')
            if len(seed_data) > 1:
                # Calculate marginal improvement
                rmse_improvement = seed_data['mean_rmse'].iloc[0] - seed_data['mean_rmse']
                time_cost = seed_data['sum_compute_time']
                marginal_efficiency = rmse_improvement / time_cost
                
                axes[1,1].plot(seed_data['n_trials'], marginal_efficiency, 
                              marker='o', label=f'{seed_val} seeds', linewidth=2)
        
        axes[1,1].set_xlabel('Number of Trials')
        axes[1,1].set_ylabel('Marginal Efficiency (RMSE improvement / time)')
        axes[1,1].set_title('Diminishing Returns Analysis')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'efficiency_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison_plots(self):
        """Create detailed model comparison plots"""
        
        models = sorted(self.performance_df['model_name'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        color_map = dict(zip(models, colors))
        
        # Create comprehensive model comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Overall performance comparison (boxplot)
        # Get data for maximum trials and seeds combination
        max_trials = self.performance_df['n_trials'].max()
        max_seeds = self.performance_df['n_seeds'].max()
        
        best_config_data = self.performance_df[
            (self.performance_df['n_trials'] == max_trials) & 
            (self.performance_df['n_seeds'] == max_seeds)
        ]
        
        if len(best_config_data) > 0:
            # Create boxplot data from individual trials
            boxplot_data = []
            model_names = []
            
            for model in models:
                model_trials = self.results_df[
                    (self.results_df['model_name'] == model) & 
                    (self.results_df['trial'] < max_trials) & 
                    (self.results_df['seed_idx'] < max_seeds)
                ]['rmse'].values
                
                if len(model_trials) > 0:
                    boxplot_data.append(model_trials)
                    model_names.append(model)
            
            if boxplot_data:
                bp = axes[0,0].boxplot(boxplot_data, labels=model_names, patch_artist=True)
                for patch, color in zip(bp['boxes'], [color_map[m] for m in model_names]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[0,0].set_ylabel('RMSE')
                axes[0,0].set_title(f'Model Performance Distribution\n({max_trials} trials, {max_seeds} seeds)')
                axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Performance vs complexity (compute time)
        mean_performance = best_config_data.groupby('model_name').agg({
            'mean_rmse': 'mean',
            'mean_compute_time': 'mean',
            'std_rmse': 'mean'
        }).reset_index()
        
        for _, row in mean_performance.iterrows():
            axes[0,1].scatter(row['mean_compute_time'], row['mean_rmse'], 
                             color=color_map[row['model_name']], s=100, 
                             label=row['model_name'], alpha=0.8)
            axes[0,1].errorbar(row['mean_compute_time'], row['mean_rmse'], 
                              yerr=row['std_rmse'], color=color_map[row['model_name']], 
                              alpha=0.5, capsize=5)
        
        axes[0,1].set_xlabel('Mean Compute Time per Trial (seconds)')
        axes[0,1].set_ylabel('Mean RMSE')
        axes[0,1].set_title('Performance vs Computational Cost')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Stability comparison (how performance varies with different configurations)
        for model in models:
            model_data = self.performance_df[self.performance_df['model_name'] == model]
            stability_metric = model_data['std_rmse'] / model_data['mean_rmse']  # Coefficient of variation
            
            axes[0,2].scatter(model_data['n_trials'], stability_metric, 
                             color=color_map[model], alpha=0.6, label=model)
        
        axes[0,2].set_xlabel('Number of Trials')
        axes[0,2].set_ylabel('Coefficient of Variation (Std/Mean)')
        axes[0,2].set_title('Model Stability Across Configurations')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Convergence comparison
        # Show how quickly each model's performance stabilizes
        selected_seeds = self.performance_df['n_seeds'].max()
        
        for model in models:
            model_data = self.performance_df[
                (self.performance_df['model_name'] == model) & 
                (self.performance_df['n_seeds'] == selected_seeds)
            ].sort_values('n_trials')
            
            if len(model_data) > 0:
                axes[1,0].plot(model_data['n_trials'], model_data['mean_rmse'], 
                              color=color_map[model], marker='o', label=model, linewidth=2)
        
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Mean RMSE')
        axes[1,0].set_title(f'Performance Convergence ({selected_seeds} seeds)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Ranking evolution
        if hasattr(self, 'ranking_analysis') and self.ranking_analysis:
            ranking_df = pd.DataFrame(self.ranking_analysis)
            
            # Show how rankings change with different configurations
            for model in models:
                model_rankings = []
                trial_vals = []
                
                for _, row in ranking_df.iterrows():
                    if model in row['rankings']:
                        model_rankings.append(row['rankings'][model])
                        trial_vals.append(row['n_trials'])
                
                if model_rankings:
                    axes[1,1].plot(trial_vals, model_rankings, 
                                  color=color_map[model], marker='s', label=model, linewidth=2)
            
            axes[1,1].set_xlabel('Number of Trials')
            axes[1,1].set_ylabel('Ranking (lower is better)')
            axes[1,1].set_title('Model Ranking Evolution')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].invert_yaxis()  # Lower rankings at top
        
        # 6. Summary statistics table
        axes[1,2].axis('off')
        
        # Create summary table
        summary_stats = best_config_data.set_index('model_name')[
            ['mean_rmse', 'median_rmse', 'std_rmse', 'min_rmse', 'max_rmse', 'mean_compute_time']
        ].round(3)
        
        # Add ranking
        if hasattr(self, 'ranking_analysis') and self.ranking_analysis:
            final_rankings = self.ranking_analysis[-1]['rankings']
            summary_stats['ranking'] = [final_rankings.get(model, np.nan) for model in summary_stats.index]
        
        # Create table
        table_data = []
        for model in summary_stats.index:
            row = summary_stats.loc[model]
            table_data.append([
                model,
                f"{row['mean_rmse']:.3f}",
                f"{row['std_rmse']:.3f}",
                f"{row['mean_compute_time']:.2f}",
                f"{row.get('ranking', 'N/A')}"
            ])
        
        table = axes[1,2].table(cellText=table_data,
                               colLabels=['Model', 'Mean RMSE', 'Std RMSE', 'Avg Time', 'Ranking'],
                               cellLoc='center',
                               loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,2].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'comprehensive_model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def find_optimal_configuration(self):
        """Find optimal (trials, seeds) configuration"""
        
        if not hasattr(self, 'ranking_analysis') or not self.ranking_analysis:
            print("No ranking analysis available for optimization")
            return None
        
        ranking_df = pd.DataFrame(self.ranking_analysis)
        
        # Define stability threshold
        stability_threshold = 0.9
        
        # Find configurations meeting stability threshold
        stable_configs = ranking_df[ranking_df['spearman_correlation'] >= stability_threshold]
        
        if len(stable_configs) == 0:
            print(f"No configurations meet stability threshold of {stability_threshold}")
            print(f"Best correlation achieved: {ranking_df['spearman_correlation'].max():.3f}")
            
            # Return best available configuration
            best_config = ranking_df.loc[ranking_df['spearman_correlation'].idxmax()]
            return {
                'n_trials': int(best_config['n_trials']),
                'n_seeds': int(best_config['n_seeds']),
                'spearman_correlation': float(best_config['spearman_correlation']),
                'total_compute_time': float(best_config['total_compute_time']),
                'meets_threshold': False
            }
        
        # Find most efficient stable configuration (minimum compute time)
        optimal_config = stable_configs.loc[stable_configs['total_compute_time'].idxmin()]
        
        return {
            'n_trials': int(optimal_config['n_trials']),
            'n_seeds': int(optimal_config['n_seeds']),
            'spearman_correlation': float(optimal_config['spearman_correlation']),
            'total_compute_time': float(optimal_config['total_compute_time']),
            'meets_threshold': True,
            'stable_configs_count': len(stable_configs)
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        print("\nGenerating comprehensive report...")
        
        # Find optimal configuration
        optimal_config = self.find_optimal_configuration()
        
        # Calculate summary statistics
        models = self.performance_df['model_name'].unique()
        trial_range = sorted(self.performance_df['n_trials'].unique())
        seed_range = sorted(self.performance_df['n_seeds'].unique())
        
        # Get best performance for each model
        max_config = self.performance_df[
            (self.performance_df['n_trials'] == max(trial_range)) & 
            (self.performance_df['n_seeds'] == max(seed_range))
        ]
        
        best_model = max_config.loc[max_config['mean_rmse'].idxmin(), 'model_name']
        worst_model = max_config.loc[max_config['mean_rmse'].idxmax(), 'model_name']
        
        # Create report
        report = {
            'analysis_summary': {
                'total_models': len(models),
                'models_evaluated': list(models),
                'trial_range': f"{min(trial_range)} to {max(trial_range)}",
                'seed_range': f"{min(seed_range)} to {max(seed_range)}",
                'total_evaluations': len(self.results_df),
                'success_rate': float(self.results_df['success'].mean()),
                'analysis_date': datetime.now().isoformat()
            },
            'optimal_configuration': optimal_config,
            'model_performance': {
                'best_model': best_model,
                'worst_model': worst_model,
                'performance_gap': float(max_config.loc[max_config['model_name'] == worst_model, 'mean_rmse'].iloc[0] - 
                                       max_config.loc[max_config['model_name'] == best_model, 'mean_rmse'].iloc[0])
            },
            'computational_analysis': {
                'total_compute_time': float(self.results_df['compute_time'].sum()),
                'avg_time_per_evaluation': float(self.results_df['compute_time'].mean()),
                'fastest_model': max_config.loc[max_config['mean_compute_time'].idxmin(), 'model_name'],
                'slowest_model': max_config.loc[max_config['mean_compute_time'].idxmax(), 'model_name']
            }
        }
        
        # Save detailed report
        with open(os.path.join(self.results_dir, 'comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_text = f"""
ENHANCED MODEL SENSITIVITY ANALYSIS - COMPREHENSIVE REPORT
=========================================================

ANALYSIS OVERVIEW:
- Models evaluated: {len(models)}
- Total evaluations: {len(self.results_df):,}
- Success rate: {self.results_df['success'].mean():.1%}
- Trial range: {min(trial_range)} to {max(trial_range)}
- Seed range: {min(seed_range)} to {max(seed_range)}

OPTIMAL CONFIGURATION:
"""
        
        if optimal_config:
            if optimal_config['meets_threshold']:
                summary_text += f"""
✓ STABLE CONFIGURATION FOUND:
  - Recommended trials: {optimal_config['n_trials']}
  - Recommended seeds: {optimal_config['n_seeds']}
  - Ranking stability: {optimal_config['spearman_correlation']:.3f}
  - Total compute time: {optimal_config['total_compute_time']:.1f} seconds
  - Stable configurations found: {optimal_config['stable_configs_count']}
"""
            else:
                summary_text += f"""
⚠ NO FULLY STABLE CONFIGURATION FOUND:
  - Best available trials: {optimal_config['n_trials']}
  - Best available seeds: {optimal_config['n_seeds']}
  - Best correlation achieved: {optimal_config['spearman_correlation']:.3f}
  - Consider increasing max_trials or max_seeds for better stability
"""
        
        summary_text += f"""
MODEL PERFORMANCE SUMMARY:
- Best performing model: {report['model_performance']['best_model']}
- Worst performing model: {report['model_performance']['worst_model']}
- Performance gap (RMSE): {report['model_performance']['performance_gap']:.3f}

COMPUTATIONAL EFFICIENCY:
- Total compute time: {report['computational_analysis']['total_compute_time']:.1f} seconds
- Average time per evaluation: {report['computational_analysis']['avg_time_per_evaluation']:.2f} seconds
- Fastest model: {report['computational_analysis']['fastest_model']}
- Slowest model: {report['computational_analysis']['slowest_model']}

RECOMMENDATIONS:
"""
        
        if optimal_config and optimal_config['meets_threshold']:
            summary_text += f"""
1. Use {optimal_config['n_trials']} trials and {optimal_config['n_seeds']} seeds for hyperparameter search
2. Model rankings are stable at this configuration
3. Proceed with confidence to full hyperparameter optimization
4. Expected compute time per model configuration: ~{optimal_config['total_compute_time']/len(models):.1f} seconds
"""
        else:
            summary_text += f"""
1. Current analysis suggests model rankings may not be fully stable
2. Consider increasing evaluation scope (more trials/seeds)
3. Use the best available configuration with caution
4. Monitor ranking consistency in actual hyperparameter search
5. Consider running additional validation experiments
"""
        
        summary_text += f"""
KEY INSIGHTS:
1. Model performance varies significantly across configurations
2. {'Stable rankings require substantial computational investment' if optimal_config and optimal_config['meets_threshold'] else 'Ranking stability is challenging to achieve with current parameters'}
3. Some models show more consistent performance than others
4. Trade-off exists between computational cost and ranking reliability

OUTPUT FILES:
- Raw results: {self.results_dir}/all_trial_results.csv
- Performance analysis: {self.results_dir}/performance_analysis.csv
- Ranking analysis: {self.results_dir}/ranking_analysis.json
- Comprehensive report: {self.results_dir}/comprehensive_report.json
- All plots: {self.plots_dir}/

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save summary
        with open(os.path.join(self.output_dir, 'COMPREHENSIVE_SUMMARY.txt'), 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\nComprehensive report saved to {self.output_dir}")
        
        return report


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced Model Sensitivity Analysis')
    
    parser.add_argument('--max_trials', type=int, default=100,
                       help='Maximum number of trials to test (default: 100)')
    
    parser.add_argument('--max_seeds', type=int, default=100,
                       help='Maximum number of seeds to test (default: 100)')
    
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval for trials and seeds (default: 5)')
    
    parser.add_argument('--data_path', type=str, 
                       default='data_updated/state_month_overdose_2015_2023.xlsx',
                       help='Path to data file')
    
    parser.add_argument('--val_start', type=str, default='2019-01-01',
                       help='Validation period start date (default: 2019-01-01)')
    
    parser.add_argument('--val_end', type=str, default='2020-01-01',
                       help='Validation period end date (default: 2020-01-01)')
    
    parser.add_argument('--stability_threshold', type=float, default=0.9,
                       help='Ranking stability threshold (default: 0.9)')
    
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
    print("ENHANCED MODEL SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Validation Period: {args.val_start} to {args.val_end}")
    print(f"Max Trials: {args.max_trials}")
    print(f"Max Seeds: {args.max_seeds}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Stability Threshold: {args.stability_threshold}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = EnhancedModelSensitivityAnalyzer(args.data_path)
    
    try:
        # Run comprehensive sensitivity analysis
        print("Starting comprehensive sensitivity analysis...")
        results_df = analyzer.run_sensitivity_analysis(
            max_trials=args.max_trials,
            max_seeds=args.max_seeds,
            eval_interval=args.eval_interval
        )
        
        # Create all visualizations
        print("\nGenerating comprehensive visualizations...")
        analyzer.create_comprehensive_plots()
        
        # Generate final report
        print("\nGenerating comprehensive report...")
        final_report = analyzer.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {analyzer.output_dir}")
        print("\nGenerated files:")
        print(f"  ├── results/")
        print(f"  │   ├── all_trial_results.csv")
        print(f"  │   ├── performance_analysis.csv")
        print(f"  │   ├── ranking_analysis.json")
        print(f"  │   └── comprehensive_report.json")
        print(f"  ├── plots/")
        print(f"  │   ├── rmse_heatmaps_by_model.png")
        print(f"  │   ├── comprehensive_heatmaps.png")
        print(f"  │   ├── convergence_analysis.png")
        print(f"  │   ├── ranking_stability_analysis.png")
        print(f"  │   ├── efficiency_analysis.png")
        print(f"  │   └── comprehensive_model_comparison.png")
        print(f"  └── COMPREHENSIVE_SUMMARY.txt")
        
        # Display key recommendations
        if final_report and final_report.get('optimal_configuration'):
            opt_config = final_report['optimal_configuration']
            print(f"\n🎯 KEY RECOMMENDATION:")
            if opt_config['meets_threshold']:
                print(f"   Use {opt_config['n_trials']} trials and {opt_config['n_seeds']} seeds")
                print(f"   Ranking stability: {opt_config['spearman_correlation']:.3f}")
                print(f"   Estimated time per model: {opt_config['total_compute_time']/len(analyzer.model_configs):.1f}s")
            else:
                print(f"   Best available: {opt_config['n_trials']} trials, {opt_config['n_seeds']} seeds")
                print(f"   Correlation: {opt_config['spearman_correlation']:.3f} (below threshold)")
                print(f"   Consider increasing max_trials or max_seeds")
        
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