#!/usr/bin/env python3
"""
Efficient Variable Horizon Forecasting Evaluation

Trains each model ONCE on train+val data (2015-2019), then extracts results
for different horizons from the same 48-month prediction sequence.

Much more efficient approach:
- Train once per model on 2015-2019 data
- Predict full 48-month sequence (2020-2023)
- Extract horizon results: 12, 24, 36, 48 months from same predictions
- Multiple trials for robust statistics
- Comprehensive metrics and visualizations

Usage:
    python efficient_variable_horizon_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Deep learning imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, GRU, Input, Add, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Statistical modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX

# TCN import
try:
    from tcn import TCN
except ImportError:
    print("Warning: TCN not available. Install with: pip install keras-tcn")
    TCN = None

import math

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ==================== CONFIGURATION ====================

# Optimal hyperparameters from grid search results
OPTIMAL_PARAMS = {
    'lstm': {'lookback': 5, 'batch_size': 8, 'epochs': 50},
    'seq2seq': {'lookback': 7, 'batch_size': 16, 'epochs': 100, 'encoder_units': 64, 'decoder_units': 64},
    'seq2seq_attn': {'lookback': 5, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 64},
    'tcn': {'lookback': 7, 'batch_size': 8, 'epochs': 100},
    'transformer': {'lookback': 7, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2},
    'sarima': {'order': (1, 0, 0), 'seasonal_order': (1, 1, 1, 12)},
}

# OPTIMAL_PARAMS = {
#     'lstm': {'lookback': 24, 'batch_size': 16, 'epochs': 150},
#     'seq2seq': {'lookback': 3, 'batch_size': 32, 'epochs': 150, 'encoder_units': 64, 'decoder_units': 64},
#     'seq2seq_attn': {'lookback': 3, 'batch_size': 32, 'epochs': 150, 'encoder_units': 64, 'decoder_units': 64},
#     'tcn': {'lookback': 8, 'batch_size': 8, 'epochs': 200},
#     'transformer': {'lookback': 6, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 4},
#     'sarima': {'order': (1, 1, 0), 'seasonal_order': (2, 1, 1, 12)},
# }

# Evaluation configuration
HORIZON_MONTHS = [12, 24, 36, 48]  # Corresponding to 2020, 2020-2021, 2020-2022, 2020-2023
HORIZON_LABELS = ['2020', '2020-2021', '2020-2022', '2020-2023']
SEEDS = [42, 123, 456, 789, 1000]
TRIALS_PER_SEED = 10  # Total trials = 5 seeds × 20 = 100 per model

# Paths
DATA_PATH = 'data/state_month_overdose_2015_2023.xlsx'
RESULTS_DIR = 'efficient_evaluation_results_more_trials'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model colors for plots
MODEL_COLORS = {
    'sarima': '#2E86AB',
    'lstm': '#A23B72',
    'tcn': '#F18F01',
    'seq2seq': '#C73E1D',
    'seq2seq_attn': '#2D5016',
    'transformer': '#8E44AD'
}

MODEL_NAMES = {
    'sarima': 'SARIMA',
    'lstm': 'LSTM',
    'tcn': 'TCN',
    'seq2seq': 'Seq2Seq',
    'seq2seq_attn': 'Seq2Seq+Attention',
    'transformer': 'Transformer'
}

# ==================== DATA LOADING ====================

def load_and_preprocess_data():
    """Load and preprocess the overdose mortality data"""
    print("Loading and preprocessing data...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_excel(DATA_PATH)
    print(f"Raw data shape: {df.shape}")
    
    # Handle column names
    if 'Sum of Deaths' in df.columns:
        df = df.rename(columns={'Sum of Deaths': 'Deaths'})
    
    # Create date column
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
    else:
        raise ValueError("No suitable date column found")
    
    # Handle Deaths column
    if 'Deaths' not in df.columns:
        raise ValueError("Deaths column not found")
    
    if df['Deaths'].dtype == 'object':
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else float(x))
    else:
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
    
    # Clean and sort
    df = df.dropna(subset=['Month', 'Deaths'])
    df = df[['Month', 'Deaths']].copy()
    df = df.sort_values('Month').reset_index(drop=True)
    
    print(f"Processed data shape: {df.shape}")
    print(f"Date range: {df['Month'].min()} to {df['Month'].max()}")
    
    return df

def create_data_splits(df):
    """Create single train+val and test split"""
    train_val_end = pd.to_datetime('2020-01-01')
    
    train_val = df[df['Month'] < train_val_end]
    test = df[df['Month'] >= train_val_end]
    
    print(f"Train+Val: {len(train_val)} samples ({train_val['Month'].min()} to {train_val['Month'].max()})")
    print(f"Test: {len(test)} samples ({test['Month'].min()} to {test['Month'].max()})")
    
    return train_val, test

# ==================== UTILITY FUNCTIONS ====================

def create_dataset(series, look_back):
    """Create dataset for supervised learning"""
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    mse = mean_squared_error(y_true, y_pred)
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse
    }

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = 1.96  # 95% confidence interval
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

def extract_horizon_results(train_true, train_pred, test_true, test_pred, horizon_months):
    """Extract results for specific horizon from full prediction sequence"""
    # Training results (always full sequence available after lookback)
    horizon_train_true = train_true
    horizon_train_pred = train_pred
    
    # Test results (extract first N months)
    horizon_test_true = test_true[:horizon_months]
    horizon_test_pred = test_pred[:horizon_months]
    
    return horizon_train_true, horizon_train_pred, horizon_test_true, horizon_test_pred

# ==================== MODEL IMPLEMENTATIONS ====================

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

def train_and_predict_sarima(train_val_data, test_data, order, seasonal_order, seed):
    """Train SARIMA once and predict full test sequence"""
    np.random.seed(seed)
    
    train_val_series = train_val_data['Deaths'].astype(float)
    test_series = test_data['Deaths'].astype(float)
    
    try:
        model = SARIMAX(train_val_series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=100)
        
        # Training predictions
        train_predictions = results.fittedvalues.values
        
        # Test predictions for full sequence
        test_predictions = results.predict(start=len(train_val_series), 
                                          end=len(train_val_series) + len(test_series) - 1).values
        
        return train_val_series.values, train_predictions, test_series.values, test_predictions
        
    except Exception as e:
        print(f"SARIMA failed: {e}")
        # Fallback to mean prediction
        train_mean = train_val_series.mean()
        fitted = np.full_like(train_val_series, train_mean)
        forecast = np.full_like(test_series, train_mean)
        return train_val_series.values, fitted, test_series.values, forecast

def train_and_predict_lstm(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Train LSTM once and predict full test sequence"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Prepare training data
    X_train, y_train = create_dataset(train_val_data, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Build and train model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Generate training predictions
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    # Generate test predictions for full sequence (autoregressive)
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        # Update input sequence for next prediction
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return train_val_data[lookback:], np.array(train_preds), test_data, np.array(test_preds)

def train_and_predict_tcn(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Train TCN once and predict full test sequence"""
    if TCN is None:
        raise ImportError("TCN not available")
        
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Prepare training data
    X_train, y_train = create_dataset(train_val_data, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Build and train model
    model = Sequential([
        TCN(input_shape=(lookback, 1), dilations=[1, 2, 4, 8], 
            nb_filters=64, kernel_size=3, dropout_rate=0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Generate predictions (same logic as LSTM)
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return train_val_data[lookback:], np.array(train_preds), test_data, np.array(test_preds)

def build_seq2seq_model(lookback, encoder_units=64, decoder_units=64, use_attention=False):
    """Build seq2seq model"""
    encoder_inputs = Input(shape=(lookback, 1))
    
    if use_attention:
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units)(encoder_outputs)
            encoder_state = Dense(decoder_units)(encoder_state)
        else:
            encoder_outputs_proj = encoder_outputs
            
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention()
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu')(decoder_combined)
        decoder_outputs = Dense(1)(decoder_hidden)
    else:
        encoder_gru = GRU(encoder_units, return_state=True)
        _, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units)(encoder_state)
            
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(decoder_units, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        decoder_outputs = Dense(1)(decoder_outputs)
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

def train_and_predict_seq2seq(train_val_data, test_data, lookback, batch_size, epochs, seed, 
                              encoder_units, decoder_units, use_attention=False):
    """Train Seq2Seq once and predict full test sequence"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Scaling
    full_series = np.concatenate([train_val_data, test_data])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare training data
    X_train, y_train = create_dataset(train_val_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
    y_train = y_train.reshape((-1, 1, 1))
    
    # Build and train model
    model = build_seq2seq_model(lookback, encoder_units, decoder_units, use_attention)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size, 
              verbose=0, callbacks=[early_stopping])
    
    # Generate training predictions
    train_preds_scaled = []
    for i in range(lookback, len(train_val_data)):
        encoder_input = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        train_preds_scaled.append(pred_scaled)
    
    # Generate test predictions for full sequence
    test_preds_scaled = []
    current_sequence = train_val_scaled[-lookback:].copy()
    
    for _ in range(len(test_data)):
        encoder_input = current_sequence.reshape((1, lookback, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        test_preds_scaled.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Inverse transform
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
    return train_val_data[lookback:], train_preds_original, test_data, test_preds_original

def train_and_predict_transformer(train_val_data, test_data, lookback, batch_size, epochs, seed, d_model, n_heads):
    """Train Transformer once and predict full test sequence"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Scaling
    full_series = np.concatenate([train_val_data, test_data])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare training data
    X_train, y_train = create_dataset(train_val_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Build transformer model
    inputs = Input(shape=(lookback, 1))
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Generate training predictions
    train_preds_scaled = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        train_preds_scaled.append(pred_scaled)
    
    # Generate test predictions for full sequence
    current_seq = train_val_scaled[-lookback:].copy()
    test_preds_scaled = []
    for _ in range(len(test_data)):
        input_seq = current_seq.reshape((1, lookback, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        test_preds_scaled.append(pred_scaled)
        current_seq = np.append(current_seq[1:], pred_scaled)
    
    # Inverse transform
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
    return train_val_data[lookback:], train_preds_original, test_data, test_preds_original

# ==================== MAIN EVALUATION ====================

def train_model_once(model_name, train_val_data, test_data, params, seed):
    """Train model once and get full predictions"""
    
    if model_name == 'sarima':
        return train_and_predict_sarima(train_val_data, test_data, 
                                       params['order'], params['seasonal_order'], seed)
    
    elif model_name == 'lstm':
        return train_and_predict_lstm(train_val_data['Deaths'].values, test_data['Deaths'].values,
                                     params['lookback'], params['batch_size'], params['epochs'], seed)
    
    elif model_name == 'tcn':
        if TCN is None:
            print(f"Skipping TCN - not available")
            return None
        return train_and_predict_tcn(train_val_data['Deaths'].values, test_data['Deaths'].values,
                                    params['lookback'], params['batch_size'], params['epochs'], seed)
    
    elif model_name == 'seq2seq':
        return train_and_predict_seq2seq(train_val_data['Deaths'].values, test_data['Deaths'].values,
                                        params['lookback'], params['batch_size'], params['epochs'], seed,
                                        params['encoder_units'], params['decoder_units'], use_attention=False)
    
    elif model_name == 'seq2seq_attn':
        return train_and_predict_seq2seq(train_val_data['Deaths'].values, test_data['Deaths'].values,
                                        params['lookback'], params['batch_size'], params['epochs'], seed,
                                        params['encoder_units'], params['decoder_units'], use_attention=True)
    
    elif model_name == 'transformer':
        return train_and_predict_transformer(train_val_data['Deaths'].values, test_data['Deaths'].values,
                                           params['lookback'], params['batch_size'], params['epochs'], seed,
                                           params['d_model'], params['n_heads'])
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def evaluate_all_models():
    """Main evaluation function - train once per model, extract multiple horizons"""
    print("="*80)
    print("EFFICIENT VARIABLE HORIZON EVALUATION")
    print("="*80)
    
    # Load data
    data = load_and_preprocess_data()
    train_val_data, test_data = create_data_splits(data)
    
    # Check test data length
    if len(test_data) < max(HORIZON_MONTHS):
        print(f"Warning: Test data has only {len(test_data)} months, but max horizon is {max(HORIZON_MONTHS)} months")
    
    # Store all results
    all_results = {}
    all_predictions = {}
    
    # Evaluate each model
    for model_name in OPTIMAL_PARAMS.keys():
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*50}")
        
        params = OPTIMAL_PARAMS[model_name]
        model_results = {}
        model_predictions = {}
        
        # Multiple trials
        trial_results = []
        trial_predictions = []
        
        total_trials = len(SEEDS) * TRIALS_PER_SEED
        trial_count = 0
        
        for seed in SEEDS:
            for trial in range(TRIALS_PER_SEED):
                trial_seed = seed + trial * 1000
                trial_count += 1
                
                if trial_count % 20 == 0:
                    print(f"  Trial {trial_count}/{total_trials}")
                
                try:
                    # Train once and get full predictions
                    result = train_model_once(model_name, train_val_data, test_data, params, trial_seed)
                    
                    if result is None:
                        continue
                    
                    train_true, train_pred, test_true, test_pred = result
                    
                    # Extract results for each horizon from the same predictions
                    trial_horizon_results = {}
                    
                    for horizon_idx, horizon_months in enumerate(HORIZON_MONTHS):
                        horizon_label = HORIZON_LABELS[horizon_idx]
                        
                        # Extract horizon-specific results
                        h_train_true, h_train_pred, h_test_true, h_test_pred = extract_horizon_results(
                            train_true, train_pred, test_true, test_pred, horizon_months)
                        
                        # Calculate metrics for this horizon
                        train_metrics = evaluate_metrics(h_train_true, h_train_pred)
                        test_metrics = evaluate_metrics(h_test_true, h_test_pred)
                        
                        # Store results for this horizon
                        horizon_result = {
                            'horizon': horizon_label,
                            'horizon_months': horizon_months,
                            'model': model_name,
                            'seed': seed,
                            'trial': trial,
                            'trial_seed': trial_seed,
                            'train_rmse': train_metrics['RMSE'],
                            'train_mae': train_metrics['MAE'],
                            'train_mape': train_metrics['MAPE'],
                            'test_rmse': test_metrics['RMSE'],
                            'test_mae': test_metrics['MAE'],
                            'test_mape': test_metrics['MAPE'],
                        }
                        
                        trial_horizon_results[horizon_label] = horizon_result
                    
                    # Store full predictions (for first few trials only to save space)
                    if trial_count <= 10:
                        # Calculate prediction intervals for full sequence
                        train_lower, train_upper = calculate_prediction_intervals(train_true, train_pred)
                        test_lower, test_upper = calculate_prediction_intervals(test_true, test_pred)
                        
                        prediction_result = {
                            'trial_seed': trial_seed,
                            'train_true': train_true,
                            'train_pred': train_pred,
                            'train_lower': train_lower,
                            'train_upper': train_upper,
                            'test_true': test_true,
                            'test_pred': test_pred,
                            'test_lower': test_lower,
                            'test_upper': test_upper,
                            'train_val_dates': train_val_data['Month'].values,
                            'test_dates': test_data['Month'].values
                        }
                        trial_predictions.append(prediction_result)
                    
                    # Add horizon results to trial results
                    trial_results.extend(list(trial_horizon_results.values()))
                    
                except Exception as e:
                    print(f"    Error in trial {trial_count}: {e}")
                    continue
        
        # Process results for this model
        if trial_results:
            results_df = pd.DataFrame(trial_results)
            
            # Calculate summary statistics for each horizon
            for horizon_label in HORIZON_LABELS:
                horizon_data = results_df[results_df['horizon'] == horizon_label]
                
                if len(horizon_data) > 0:
                    summary = {
                        'horizon': horizon_label,
                        'horizon_months': horizon_data['horizon_months'].iloc[0],
                        'model': model_name,
                        'trials_completed': len(horizon_data),
                        'test_rmse_mean': horizon_data['test_rmse'].mean(),
                        'test_rmse_median': horizon_data['test_rmse'].median(),
                        'test_rmse_std': horizon_data['test_rmse'].std(),
                        'test_mae_mean': horizon_data['test_mae'].mean(),
                        'test_mae_median': horizon_data['test_mae'].median(),
                        'test_mae_std': horizon_data['test_mae'].std(),
                        'test_mape_mean': horizon_data['test_mape'].mean(),
                        'test_mape_median': horizon_data['test_mape'].median(),
                        'test_mape_std': horizon_data['test_mape'].std(),
                    }
                    
                    model_results[horizon_label] = summary
            
            # Store predictions
            model_predictions = trial_predictions
            
            print(f"  Completed {len(trial_results)} horizon evaluations from {trial_count} model training runs")
            
            # Print summary for each horizon
            for horizon_label in HORIZON_LABELS:
                if horizon_label in model_results:
                    summary = model_results[horizon_label]
                    print(f"    {horizon_label}: RMSE {summary['test_rmse_mean']:.3f} ± {summary['test_rmse_std']:.3f}")
        
        all_results[model_name] = model_results
        all_predictions[model_name] = model_predictions
    
    return all_results, all_predictions, data, train_val_data, test_data

# ==================== VISUALIZATION ====================

def create_comprehensive_plots(all_results, all_predictions, data, train_val_data, test_data):
    """Create comprehensive comparison plots"""
    print(f"\n{'='*50}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*50}")
    
    # Create comprehensive results DataFrame
    results_list = []
    for model_name in all_results:
        for horizon_label in all_results[model_name]:
            result = all_results[model_name][horizon_label].copy()
            results_list.append(result)
    
    if not results_list:
        print("No results to plot")
        return
    
    comprehensive_df = pd.DataFrame(results_list)
    
    # 1. Main results plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE vs Horizon
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
    
    # MAPE vs Horizon
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
    
    # Model rankings heatmap
    ranking_data = []
    horizons = sorted(comprehensive_df['horizon'].unique(), key=lambda x: HORIZON_LABELS.index(x))
    models = sorted(comprehensive_df['model'].unique())
    
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
    
    # Performance degradation
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
    
    plt.suptitle('Efficient Variable Horizon Evaluation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual SARIMA comparison plots
    if 'sarima' in all_predictions:
        create_sarima_comparison_plots(all_predictions, train_val_data, test_data)
    
    print("✓ Created comprehensive results plot")

def create_sarima_comparison_plots(all_predictions, train_val_data, test_data):
    """Create individual model vs SARIMA comparison plots for each horizon"""
    
    if 'sarima' not in all_predictions or len(all_predictions['sarima']) == 0:
        print("No SARIMA predictions available for comparison plots")
        return
    
    sarima_preds = all_predictions['sarima'][0]  # First trial
    
    for model_name in all_predictions:
        if model_name == 'sarima' or len(all_predictions[model_name]) == 0:
            continue
        
        model_preds = all_predictions[model_name][0]  # First trial
        
        # Create comparison plots for each horizon
        for horizon_idx, horizon_months in enumerate(HORIZON_MONTHS):
            horizon_label = HORIZON_LABELS[horizon_idx]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Get full prediction data
            sarima_train_true = sarima_preds['train_true']
            sarima_train_pred = sarima_preds['train_pred']
            sarima_test_true = sarima_preds['test_true']
            sarima_test_pred = sarima_preds['test_pred']
            
            model_train_true = model_preds['train_true']
            model_train_pred = model_preds['train_pred']
            model_test_true = model_preds['test_true']
            model_test_pred = model_preds['test_pred']
            
            # Extract horizon-specific results
            s_train_true, s_train_pred, s_test_true, s_test_pred = extract_horizon_results(
                sarima_train_true, sarima_train_pred, sarima_test_true, sarima_test_pred, horizon_months)
            
            m_train_true, m_train_pred, m_test_true, m_test_pred = extract_horizon_results(
                model_train_true, model_train_pred, model_test_true, model_test_pred, horizon_months)
            
            # Align training predictions if needed
            if len(s_train_true) != len(m_train_true):
                lookback_diff = len(s_train_true) - len(m_train_true)
                if lookback_diff > 0:
                    s_train_true = s_train_true[lookback_diff:]
                    s_train_pred = s_train_pred[lookback_diff:]
            
            # Combine train and test
            all_true = np.concatenate([s_train_true, s_test_true])
            all_pred_sarima = np.concatenate([s_train_pred, s_test_pred])
            all_pred_model = np.concatenate([m_train_pred, m_test_pred])
            
            # Create time index
            x_range = range(len(all_true))
            
            # Plot
            ax.plot(x_range, all_true, label='Actual', color='black', linewidth=2.5, alpha=0.8)
            ax.plot(x_range, all_pred_sarima, label='SARIMA', 
                   color=MODEL_COLORS['sarima'], linewidth=2, alpha=0.7)
            ax.plot(x_range, all_pred_model, label=MODEL_NAMES.get(model_name, model_name), 
                   color=MODEL_COLORS.get(model_name, 'gray'), linewidth=2, alpha=0.7)
            
            # Add vertical line at forecast start
            forecast_start = len(s_train_true)
            ax.axvline(forecast_start, color='red', linestyle='--', alpha=0.7, linewidth=2,
                      label='Forecast Start')
            
            # Calculate metrics for this horizon
            test_rmse_sarima = np.sqrt(np.mean((s_test_true - s_test_pred) ** 2))
            test_rmse_model = np.sqrt(np.mean((m_test_true - m_test_pred) ** 2))
            test_mae_sarima = np.mean(np.abs(s_test_true - s_test_pred))
            test_mae_model = np.mean(np.abs(m_test_true - m_test_pred))
            
            ax.set_title(f'SARIMA vs {MODEL_NAMES.get(model_name, model_name)} - Horizon {horizon_label}',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Deaths', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = (f'Test RMSE:\n'
                           f'SARIMA: {test_rmse_sarima:.2f}\n'
                           f'{MODEL_NAMES.get(model_name, model_name)}: {test_rmse_model:.2f}\n\n'
                           f'Test MAE:\n'
                           f'SARIMA: {test_mae_sarima:.2f}\n'
                           f'{MODEL_NAMES.get(model_name, model_name)}: {test_mae_model:.2f}')
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'sarima_vs_{model_name}_horizon_{horizon_label}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✓ Created SARIMA comparison plots for all horizons")

# ==================== SAVE RESULTS ====================

def save_results(all_results, all_predictions, data, train_val_data, test_data):
    """Save all results and raw data"""
    print(f"\n{'='*50}")
    print("SAVING RESULTS")
    print(f"{'='*50}")
    
    # Create comprehensive results DataFrame
    results_list = []
    for model_name in all_results:
        for horizon_label in all_results[model_name]:
            result = all_results[model_name][horizon_label].copy()
            results_list.append(result)
    
    if results_list:
        comprehensive_df = pd.DataFrame(results_list)
        comprehensive_df = comprehensive_df.round(4)
        comprehensive_df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_results.csv'), index=False)
        print("✓ Saved comprehensive_results.csv")
        
        # Create detailed metrics table
        metrics_cols = ['horizon', 'model', 'horizon_months','test_rmse_mean', 'test_rmse_median', 'test_rmse_std',
                       'test_mae_mean', 'test_mae_median', 'test_mae_std',
                       'test_mape_mean', 'test_mape_median', 'test_mape_std', 'trials_completed']
        detailed_metrics = comprehensive_df[metrics_cols].copy()
        detailed_metrics['model_name'] = detailed_metrics['model'].map(MODEL_NAMES)
        detailed_metrics.to_csv(os.path.join(RESULTS_DIR, 'detailed_metrics.csv'), index=False)
        print("✓ Saved detailed_metrics.csv")
        
        # Create best models summary
        summary_data = []
        for horizon in comprehensive_df['horizon'].unique():
            horizon_data = comprehensive_df[comprehensive_df['horizon'] == horizon]
            
            best_rmse = horizon_data.loc[horizon_data['test_rmse_mean'].idxmin()]
            best_mape = horizon_data.loc[horizon_data['test_mape_mean'].idxmin()]
            
            summary_data.append({
                'Horizon': horizon,
                'Horizon_Months': int(best_rmse['horizon_months']),
                'Best_RMSE_Model': MODEL_NAMES.get(best_rmse['model'], best_rmse['model']),
                'Best_RMSE_Value': f"{best_rmse['test_rmse_mean']:.2f} ± {best_rmse['test_rmse_std']:.2f}",
                'Best_MAPE_Model': MODEL_NAMES.get(best_mape['model'], best_mape['model']),
                'Best_MAPE_Value': f"{best_mape['test_mape_mean']:.2f}% ± {best_mape['test_mape_std']:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(RESULTS_DIR, 'best_models_summary.csv'), index=False)
        print("✓ Saved best_models_summary.csv")
    
    # Save predictions
    with open(os.path.join(RESULTS_DIR, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    print("✓ Saved all_predictions.pkl")
    
    # Save data splits
    data_splits = {
        'original_data': data,
        'train_val_data': train_val_data,
        'test_data': test_data
    }
    with open(os.path.join(RESULTS_DIR, 'data_splits.pkl'), 'wb') as f:
        pickle.dump(data_splits, f)
    print("✓ Saved data_splits.pkl")
    
    # Save configuration
    config = {
        'optimal_params': OPTIMAL_PARAMS,
        'horizon_months': HORIZON_MONTHS,
        'horizon_labels': HORIZON_LABELS,
        'seeds': SEEDS,
        'trials_per_seed': TRIALS_PER_SEED,
        'evaluation_date': datetime.now().isoformat(),
        'approach': 'efficient_single_training'
    }
    
    with open(os.path.join(RESULTS_DIR, 'evaluation_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Saved evaluation_config.json")
    
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    
    # Print summary
    if results_list:
        print(f"\nSUMMARY - BEST MODELS BY HORIZON:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Horizon']}: {row['Best_RMSE_Model']} (RMSE: {row['Best_RMSE_Value']})")

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("Starting efficient variable horizon evaluation...")
    print("="*80)
    print("EFFICIENT APPROACH:")
    print("- Train each model ONCE on 2015-2019 data")
    print("- Predict full 48-month sequence (2020-2023)")
    print("- Extract results for 12, 24, 36, 48 month horizons")
    print("="*80)
    
    print(f"Models to evaluate: {list(OPTIMAL_PARAMS.keys())}")
    print(f"Horizons: {HORIZON_LABELS} ({HORIZON_MONTHS} months)")
    print(f"Total trials per model: {len(SEEDS) * TRIALS_PER_SEED}")
    
    start_time = datetime.now()
    
    try:
        # Run evaluation
        all_results, all_predictions, data, train_val_data, test_data = evaluate_all_models()
        
        # Create visualizations
        create_comprehensive_plots(all_results, all_predictions, data, train_val_data, test_data)
        
        # Save results
        save_results(all_results, all_predictions, data, train_val_data, test_data)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print("EFFICIENT EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Duration: {duration}")
        print(f"Results saved to: {RESULTS_DIR}/")
        print(f"Plots saved to: {PLOTS_DIR}/")
        
        print(f"\nGenerated files:")
        print(f"├── comprehensive_results.csv          # Complete results table")
        print(f"├── detailed_metrics.csv              # Detailed metrics breakdown")
        print(f"├── best_models_summary.csv           # Best model per horizon")
        print(f"├── all_predictions.pkl               # Raw predictions")
        print(f"├── data_splits.pkl                   # Data splits used")
        print(f"├── evaluation_config.json            # Configuration")
        print(f"└── plots/")
        print(f"    ├── comprehensive_results.png     # 4-panel summary plot")
        print(f"    └── sarima_vs_[model]_horizon_[horizon].png  # Individual comparisons")
        
        print(f"\nEfficiency gains:")
        print(f"- Traditional approach: {len(OPTIMAL_PARAMS) * len(HORIZON_MONTHS) * len(SEEDS) * TRIALS_PER_SEED} training runs")
        print(f"- Efficient approach: {len(OPTIMAL_PARAMS) * len(SEEDS) * TRIALS_PER_SEED} training runs")
        print(f"- Reduction: {75}% fewer training runs!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)