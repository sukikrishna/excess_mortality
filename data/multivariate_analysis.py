import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, GRU, Input, Add, Dropout, 
                                      Flatten, Concatenate, LayerNormalization, 
                                      MultiHeadAttention, Attention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
import math
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
OPTIMAL_PARAMS = {
    'varmax': {'order': (2, 1), 'trend': 'c'},
    'lstm': {'lookback': 5, 'batch_size': 8, 'epochs': 50, 'units': 64},
    'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50, 'filters': 64},
    'seq2seq': {'lookback': 7, 'batch_size': 16, 'epochs': 100, 'encoder_units': 64, 'decoder_units': 64},
    'seq2seq_attn': {'lookback': 5, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 64},
    'transformer': {'lookback': 7, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
}

SEEDS = [12, 23, 34, 45, 56, 67, 78, 89, 90, 42]
TRIALS_PER_SEED = 50

PROCESSED_DATA_DIR = 'processed_data'
RESULTS_DIR = 'hundo_multivariate_evaluation_results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------ HELPER FUNCTIONS ------------------ #

def load_processed_data(aggregation_type='sex'):
    """
    Load preprocessed data.
    
    Parameters:
    - aggregation_type: 'sex' or 'state'
    
    Returns:
    - train, val, test DataFrames
    """
    prefix = f'{aggregation_type}_'
    
    train = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, f'{prefix}train.csv'),
        index_col=0,
        parse_dates=True
    )
    val = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, f'{prefix}val.csv'),
        index_col=0,
        parse_dates=True
    )
    test = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, f'{prefix}test.csv'),
        index_col=0,
        parse_dates=True
    )
    
    return train, val, test


def create_dataset_multivariate(data, look_back):
    """Create dataset for supervised learning with multivariate data"""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)


def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for multivariate data"""
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # Avoid division by zero in MAPE
    mask = y_true_flat != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = np.nan
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse
    }


def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals for multivariate predictions"""
    residuals = actual - predictions
    std_residual = np.std(residuals, axis=0)
    z_score = stats.norm.ppf(1 - alpha/2)
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound


def calculate_pi_coverage(actual, lower_bound, upper_bound):
    """Calculate prediction interval coverage"""
    coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound))
    return coverage * 100


# ------------------ MODEL IMPLEMENTATIONS ------------------ #

def run_varmax_full_predictions(train_val_df, test_df, order=(2, 1), trend='c', seed=42):
    """Run VARMAX model for multivariate time series"""
    np.random.seed(seed)
    
    train_val_data = train_val_df.values
    test_data = test_df.values
    
    n_series = train_val_data.shape[1]
    
    try:
        # Fit VARMAX model
        model = VARMAX(train_val_data, order=order, trend=trend, 
                      enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=200)
        
        # Training predictions
        train_predictions = results.fittedvalues
        
        # Test predictions
        test_predictions = results.forecast(steps=len(test_data))
        
        return train_val_data, train_predictions, test_data, test_predictions
        
    except Exception as e:
        print(f"    VARMAX failed: {str(e)}")
        # Return dummy predictions on failure
        train_predictions = np.zeros_like(train_val_data)
        test_predictions = np.zeros_like(test_data)
        return train_val_data, train_predictions, test_data, test_predictions


def run_lstm_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, units, seed):
    """Run LSTM model for multivariate time series"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    n_features = train_val_data.shape[1]
    
    # Prepare training data
    X_train, y_train = create_dataset_multivariate(train_val_data, lookback)
    
    # Build model
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, 
             input_shape=(lookback, n_features)),
        LSTM(units, activation='relu'),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Training predictions
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, n_features))
        pred = model.predict(input_seq, verbose=0)[0]
        train_preds.append(pred)
    
    # Test predictions (autoregressive)
    current_input = train_val_data[-lookback:].copy()
    test_preds = []
    for _ in range(len(test_data)):
        input_seq = current_input.reshape((1, lookback, n_features))
        pred = model.predict(input_seq, verbose=0)[0]
        test_preds.append(pred)
        current_input = np.vstack([current_input[1:], pred])
    
    return (train_val_data[lookback:], np.array(train_preds),
            test_data, np.array(test_preds))


def run_tcn_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, filters, seed):
    """Run TCN model for multivariate time series"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    n_features = train_val_data.shape[1]
    
    # Prepare training data
    X_train, y_train = create_dataset_multivariate(train_val_data, lookback)
    
    # Build model
    model = Sequential([
        TCN(input_shape=(lookback, n_features), 
            dilations=[1, 2, 4, 8],
            nb_filters=filters,
            kernel_size=3,
            dropout_rate=0.1),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Training predictions
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, n_features))
        pred = model.predict(input_seq, verbose=0)[0]
        train_preds.append(pred)
    
    # Test predictions (autoregressive)
    current_input = train_val_data[-lookback:].copy()
    test_preds = []
    for _ in range(len(test_data)):
        input_seq = current_input.reshape((1, lookback, n_features))
        pred = model.predict(input_seq, verbose=0)[0]
        test_preds.append(pred)
        current_input = np.vstack([current_input[1:], pred])
    
    return (train_val_data[lookback:], np.array(train_preds),
            test_data, np.array(test_preds))


def build_seq2seq_model_multivariate(lookback, n_features, encoder_units=128, 
                                     decoder_units=128, use_attention=True):
    """Build seq2seq model for multivariate time series"""
    encoder_inputs = Input(shape=(lookback, n_features), name='encoder_input')
    
    if use_attention:
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units, name='encoder_proj')(encoder_outputs)
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
        else:
            encoder_outputs_proj = encoder_outputs
        
        decoder_inputs = Input(shape=(1, n_features), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention(name='attention')
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu', name='decoder_hidden')(decoder_combined)
        decoder_outputs = Dense(n_features, name='output_dense')(decoder_hidden)
    else:
        encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
        _, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
        
        decoder_inputs = Input(shape=(1, n_features), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        decoder_outputs = Dense(n_features, name='decoder_dense')(decoder_outputs)
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def run_seq2seq_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, seed,
                                 encoder_units=128, decoder_units=128, use_attention=True):
    """Run seq2seq model for multivariate time series"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    n_features = train_val_data.shape[1]
    
    # Scaling
    full_series = np.vstack([train_val_data, test_data])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series)
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare training data
    X_train, y_train = create_dataset_multivariate(train_val_scaled, lookback)
    decoder_input_train = np.zeros((X_train.shape[0], 1, n_features))
    y_train = y_train.reshape((-1, 1, n_features))
    
    # Build and train model
    model = build_seq2seq_model_multivariate(lookback, n_features, encoder_units, 
                                            decoder_units, use_attention)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=[early_stopping], validation_split=0.1)
    
    # Training predictions
    train_preds_scaled = []
    for i in range(lookback, len(train_val_data)):
        encoder_input = train_val_scaled[i-lookback:i].reshape((1, lookback, n_features))
        decoder_input = np.zeros((1, 1, n_features))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, :]
        train_preds_scaled.append(pred_scaled)
    
    # Test predictions (autoregressive)
    test_preds_scaled = []
    current_sequence = train_val_scaled[-lookback:].copy()
    
    for _ in range(len(test_data)):
        encoder_input = current_sequence.reshape((1, lookback, n_features))
        decoder_input = np.zeros((1, 1, n_features))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, :]
        test_preds_scaled.append(pred_scaled)
        current_sequence = np.vstack([current_sequence[1:], pred_scaled])
    
    # Inverse transform
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled))
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled))
    
    return (train_val_data[lookback:], train_preds_original,
            test_data, test_preds_original)


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


def run_transformer_full_predictions(train_val_data, test_data, lookback, batch_size, 
                                    epochs, seed, d_model=64, n_heads=2):
    """Run transformer model for multivariate time series"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    n_features = train_val_data.shape[1]
    
    # Scaling
    full_series = np.vstack([train_val_data, test_data])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series)
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare data
    X_train, y_train = create_dataset_multivariate(train_val_scaled, lookback)
    
    # Build transformer model
    inputs = Input(shape=(lookback, n_features))
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(n_features)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Training predictions
    train_preds_scaled = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_scaled[i-lookback:i].reshape((1, lookback, n_features))
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        train_preds_scaled.append(pred_scaled)
    
    # Test predictions (autoregressive)
    current_seq = train_val_scaled[-lookback:].copy()
    test_preds_scaled = []
    for _ in range(len(test_data)):
        input_seq = current_seq.reshape((1, lookback, n_features))
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        test_preds_scaled.append(pred_scaled)
        current_seq = np.vstack([current_seq[1:], pred_scaled])
    
    # Inverse transform
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled))
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled))
    
    return (train_val_data[lookback:], train_preds_original,
            test_data, test_preds_original)


# ------------------ MAIN EVALUATION LOOP ------------------ #

def main():
    """Main evaluation function for multivariate time series"""
    
    # Choose aggregation type
    import sys
    if len(sys.argv) > 1:
        aggregation_type = sys.argv[1]
    else:
        aggregation_type = 'state'  # Default to sex aggregation
    
    if aggregation_type not in ['sex', 'state', 'age', 'age_sex']:
        print("Invalid aggregation type. Choose 'sex' or 'state'")
        return
    
    print("="*60)
    print(f"MULTIVARIATE TIME SERIES ANALYSIS - {aggregation_type.upper()} AGGREGATION")
    print("="*60)
    
    # Load data
    print("\nLoading processed data...")
    train_data, validation_data, test_data = load_processed_data(aggregation_type)
    
    # Combine train and validation
    train_val_data = pd.concat([train_data, validation_data])
    
    print(f"Train+Val shape: {train_val_data.shape}")
    print(f"Test shape: {test_data.shape}")
    print(f"Number of series: {train_val_data.shape[1]}")
    
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, aggregation_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    all_results = {}
    all_predictions = {}
    
    # Models to evaluate
    models_to_evaluate = ['varmax', 'lstm', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer']
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print('='*60)
        
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        params = OPTIMAL_PARAMS[model_name]
        
        seed_results = {}
        seed_predictions = {}
        
        for seed in SEEDS:
            print(f"\n  Processing seed {seed}...")
            
            seed_dir = os.path.join(model_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            
            trial_results = []
            trial_predictions = []
            
            for trial in range(TRIALS_PER_SEED):
                trial_seed = seed + trial * 1000
                print(f"    Trial {trial + 1}/{TRIALS_PER_SEED}", end='')
                
                try:
                    if model_name == 'varmax':
                        train_true, train_pred, test_true, test_pred = run_varmax_full_predictions(
                            train_val_data, test_data,
                            order=params['order'],
                            trend=params['trend'],
                            seed=trial_seed
                        )
                    
                    elif model_name == 'lstm':
                        train_true, train_pred, test_true, test_pred = run_lstm_full_predictions(
                            train_val_data.values, test_data.values,
                            params['lookback'], params['batch_size'], params['epochs'],
                            params['units'], trial_seed
                        )
                    
                    elif model_name == 'tcn':
                        train_true, train_pred, test_true, test_pred = run_tcn_full_predictions(
                            train_val_data.values, test_data.values,
                            params['lookback'], params['batch_size'], params['epochs'],
                            params['filters'], trial_seed
                        )
                    
                    elif model_name == 'seq2seq':
                        train_true, train_pred, test_true, test_pred = run_seq2seq_full_predictions(
                            train_val_data.values, test_data.values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['encoder_units'], params['decoder_units'], use_attention=False
                        )
                    
                    elif model_name == 'seq2seq_attn':
                        train_true, train_pred, test_true, test_pred = run_seq2seq_full_predictions(
                            train_val_data.values, test_data.values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['encoder_units'], params['decoder_units'], use_attention=True
                        )
                    
                    elif model_name == 'transformer':
                        train_true, train_pred, test_true, test_pred = run_transformer_full_predictions(
                            train_val_data.values, test_data.values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['d_model'], params['n_heads']
                        )
                    
                    # Calculate metrics
                    train_metrics = evaluate_metrics(train_true, train_pred)
                    test_metrics = evaluate_metrics(test_true, test_pred)
                    
                    # Calculate prediction intervals
                    train_lower, train_upper = calculate_prediction_intervals(train_true, train_pred)
                    test_lower, test_upper = calculate_prediction_intervals(test_true, test_pred)
                    
                    # Calculate PI coverage
                    train_coverage = calculate_pi_coverage(train_true, train_lower, train_upper)
                    test_coverage = calculate_pi_coverage(test_true, test_lower, test_upper)
                    
                    # Store results
                    combined_metrics = {
                        'Trial': trial + 1,
                        'Seed': seed,
                        'Trial_Seed': trial_seed,
                        'Train_RMSE': train_metrics['RMSE'],
                        'Train_MAE': train_metrics['MAE'],
                        'Train_MAPE': train_metrics['MAPE'],
                        'Train_MSE': train_metrics['MSE'],
                        'Train_PI_Coverage': train_coverage,
                        'Test_RMSE': test_metrics['RMSE'],
                        'Test_MAE': test_metrics['MAE'],
                        'Test_MAPE': test_metrics['MAPE'],
                        'Test_MSE': test_metrics['MSE'],
                        'Test_PI_Coverage': test_coverage
                    }
                    trial_results.append(combined_metrics)
                    
                    # Store predictions
                    prediction_data = {
                        'train_true': train_true,
                        'train_pred': train_pred,
                        'train_lower': train_lower,
                        'train_upper': train_upper,
                        'test_true': test_true,
                        'test_pred': test_pred,
                        'test_lower': test_lower,
                        'test_upper': test_upper
                    }
                    trial_predictions.append(prediction_data)
                    
                    # Save individual trial predictions (save first series only for size)
                    trial_df = pd.DataFrame({
                        f'Train_True_Series{i}': train_true[:, i] if i < train_true.shape[1] else []
                        for i in range(min(5, train_true.shape[1]))
                    })
                    for i in range(min(5, train_pred.shape[1])):
                        trial_df[f'Train_Pred_Series{i}'] = train_pred[:, i]
                    trial_df.to_csv(os.path.join(seed_dir, f'trial_{trial + 1}_train_predictions.csv'), index=False)
                    
                    test_df = pd.DataFrame({
                        f'Test_True_Series{i}': test_true[:, i] if i < test_true.shape[1] else []
                        for i in range(min(5, test_true.shape[1]))
                    })
                    for i in range(min(5, test_pred.shape[1])):
                        test_df[f'Test_Pred_Series{i}'] = test_pred[:, i]
                    test_df.to_csv(os.path.join(seed_dir, f'trial_{trial + 1}_test_predictions.csv'), index=False)
                    
                    print(f" ✓ RMSE: {test_metrics['RMSE']:.2f}")
                    
                except Exception as e:
                    print(f" ✗ Error: {str(e)[:50]}")
                    continue
            
            # Process results for this seed
            if trial_results:
                trial_results_df = pd.DataFrame(trial_results)
                trial_results_df.to_csv(os.path.join(seed_dir, 'all_trials_metrics.csv'), index=False)
                
                # Calculate summary statistics
                summary_stats = trial_results_df.select_dtypes(include=[np.number]).agg(['mean', 'std', 'min', 'max'])
                summary_stats.to_csv(os.path.join(seed_dir, 'seed_summary_statistics.csv'))
                
                # Store seed results
                seed_results[seed] = {
                    'train_mean_rmse': trial_results_df['Train_RMSE'].mean(),
                    'train_std_rmse': trial_results_df['Train_RMSE'].std(),
                    'test_mean_rmse': trial_results_df['Test_RMSE'].mean(),
                    'test_std_rmse': trial_results_df['Test_RMSE'].std(),
                    'train_mean_mae': trial_results_df['Train_MAE'].mean(),
                    'train_std_mae': trial_results_df['Train_MAE'].std(),
                    'test_mean_mae': trial_results_df['Test_MAE'].mean(),
                    'test_std_mae': trial_results_df['Test_MAE'].std(),
                    'train_mean_mape': trial_results_df['Train_MAPE'].mean(),
                    'train_std_mape': trial_results_df['Train_MAPE'].std(),
                    'test_mean_mape': trial_results_df['Test_MAPE'].mean(),
                    'test_std_mape': trial_results_df['Test_MAPE'].std(),
                    'train_mean_pi_cov': trial_results_df['Train_PI_Coverage'].mean(),
                    'train_std_pi_cov': trial_results_df['Train_PI_Coverage'].std(),
                    'test_mean_pi_cov': trial_results_df['Test_PI_Coverage'].mean(),
                    'test_std_pi_cov': trial_results_df['Test_PI_Coverage'].std(),
                    'trials_completed': len(trial_results_df)
                }
                
                seed_predictions[seed] = trial_predictions
                
                print(f"    Seed {seed} completed: Test RMSE = {trial_results_df['Test_RMSE'].mean():.4f} ± {trial_results_df['Test_RMSE'].std():.4f}")
        
        # Save seed comparison
        if seed_results:
            seed_comparison_df = pd.DataFrame(seed_results).T
            seed_comparison_df = seed_comparison_df.round(4)
            seed_comparison_df.to_csv(os.path.join(model_dir, 'seed_comparison.csv'))
            
            # Calculate overall model statistics
            test_rmse_means = [seed_results[s]['test_mean_rmse'] for s in seed_results]
            test_mape_means = [seed_results[s]['test_mean_mape'] for s in seed_results]
            test_mae_means = [seed_results[s]['test_mean_mae'] for s in seed_results]
            test_pic_means = [seed_results[s]['test_mean_pi_cov'] for s in seed_results]
            
            all_results[model_name] = {
                'test_mean_rmse': np.mean(test_rmse_means),
                'test_std_rmse': np.std(test_rmse_means),
                'test_mean_mae': np.mean(test_mae_means),
                'test_std_mae': np.std(test_mae_means),
                'test_mean_mape': np.mean(test_mape_means),
                'test_std_mape': np.std(test_mape_means),
                'test_mean_pi_cov': np.mean(test_pic_means),
                'test_std_pi_cov': np.std(test_pic_means),
                'seeds_completed': len(seed_results)
            }
            
            all_predictions[model_name] = seed_predictions
            
            print(f"\n  {model_name.upper()} Overall:")
            print(f"    Test RMSE: {np.mean(test_rmse_means):.2f} ± {np.std(test_rmse_means):.2f}")
            print(f"    Test MAE: {np.mean(test_mae_means):.2f} ± {np.std(test_mae_means):.2f}")
            print(f"    Test MAPE: {np.mean(test_mape_means):.2f}% ± {np.std(test_mape_means):.2f}%")
            print(f"    Test PI Coverage: {np.mean(test_pic_means):.1f}% ± {np.std(test_pic_means):.1f}%")
    
    # Create final comparison table
    if all_results:
        final_comparison_df = pd.DataFrame(all_results).T
        final_comparison_df = final_comparison_df.round(4)
        final_comparison_df.to_csv(os.path.join(results_dir, 'final_model_comparison.csv'))
        
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        print(final_comparison_df.to_string())
    
    # Save aggregated predictions
    import pickle
    with open(os.path.join(results_dir, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    
    # Save data info
    data_info = {
        'train_data': train_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'train_val_data': train_val_data,
        'aggregation_type': aggregation_type
    }
    with open(os.path.join(results_dir, 'data_info.pkl'), 'wb') as f:
        pickle.dump(data_info, f)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()