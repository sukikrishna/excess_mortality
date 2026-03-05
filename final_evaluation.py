import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import RepeatVector, Concatenate
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN  # pip install keras-tcn
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
# Optimal hyperparameters from grid search
OPTIMAL_PARAMS = {
    # 'sarima': {'order': (1,0,0), 'seasonal_order': (1,1,1,12)},
    'sarima': {'order': (2,1,1), 'seasonal_order': (1,1,1,12)},
    'lstm': {'lookback': 5, 'batch_size': 8, 'epochs': 50},
    'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50},
    'seq2seq': {'lookback': 7, 'batch_size': 16, 'epochs': 100, 'encoder_units': 64, 'decoder_units': 64},
    'seq2seq_attn': {'lookback': 5, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 64},
    'transformer': {'lookback': 7, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
}

# OPTIMAL_PARAMS = {
#     'sarima': {'order': (2,1,1), 'seasonal_order': (1,1,1,12)},
#     'lstm': {'lookback': 9, 'batch_size': 8, 'epochs': 100},
#     'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50},
#     'seq2seq': {'lookback': 7, 'batch_size': 16, 'epochs': 100, 'encoder_units': 64, 'decoder_units': 64},
#     'seq2seq_attn': {'lookback': 11, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 128},
#     'transformer': {'lookback': 5, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
# }

# Multiple seeds for robust evaluation
SEEDS = [42, 123, 456, 789, 1000]
# SEEDS = [42]
TRIALS_PER_SEED = 10  # Number of trials per model per seed

# DATA_PATH = 'data/state_month_overdose.xlsx'
DATA_PATH = 'data/national_month_overdose_2010_2023.xlsx'
RESULTS_DIR = 'national_final_evaluation_results'

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------ HELPER FUNCTIONS ------------------ #
# def load_and_preprocess_data():
#     """Load and preprocess the overdose data"""
#     df = pd.read_excel(DATA_PATH)
#     df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
#     df['Month'] = pd.to_datetime(df['Month'])
#     df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
#     return df

def load_and_preprocess_data():
    """Load and preprocess the overdose data from Excel with flexible column handling."""
    # Read, keep codes as strings for safe concatenation later
    df = pd.read_excel(DATA_PATH, dtype={"Month_Code": str, "Year_Code": str})

    # Normalize headers (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # --- Choose the date source WITHOUT creating duplicate column names ---
    date_series = None
    if "Row Labels" in df.columns:
        # Full dates like 1/1/2015 live here — use this and ignore the textual Month column
        date_series = pd.to_datetime(df["Row Labels"], errors="coerce", infer_datetime_format=True)
    elif {"Year_Code", "Month_Code"}.issubset(df.columns):
        # Build YYYY-MM-01 from codes
        df["Month_Code"] = df["Month_Code"].astype(str).str.zfill(2)
        df["Year_Code"] = df["Year_Code"].astype(str)
        date_series = pd.to_datetime(df["Year_Code"] + "-" + df["Month_Code"] + "-01", errors="coerce")
    elif "Month" in df.columns:
        # As a last resort, try to parse Month directly (handles real dates, not bare 'Jan' without a year)
        date_series = pd.to_datetime(df["Month"], errors="coerce", infer_datetime_format=True)
    else:
        raise KeyError("No usable date columns found (expected 'Row Labels' or Year/Month codes).")

    # Create a clean Month column from the chosen source
    df["Month"] = date_series

    # --- Standardize deaths ---
    if "Deaths" not in df.columns:
        if "Sum of Deaths" in df.columns:
            df = df.rename(columns={"Sum of Deaths": "Deaths"})
        else:
            raise KeyError("Neither 'Deaths' nor 'Sum of Deaths' column found.")

    df["Deaths"] = (
        df["Deaths"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"Suppressed": "0", "NaN": "0", "nan": "0", "": "0"})
        .astype(float)
    )

    # Drop rows where date couldn’t be parsed
    df = df.dropna(subset=["Month"])

    # Ensure unique column names going forward (keep first if duplicates exist)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Aggregate to monthly total, sort, reset index
    df = df.groupby("Month", as_index=False)["Deaths"].sum().sort_values("Month").reset_index(drop=True)
    return df


def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01'):
    """Create train/validation/test splits"""
    train = df[df['Month'] < train_end]
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[df['Month'] >= val_end]
    return train, validation, test

def create_dataset(series, look_back):
    """Create dataset for supervised learning"""
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse
    }

# def calculate_prediction_intervals(actual, predictions, alpha=0.05):
#     """Calculate prediction intervals for a set of predictions"""
#     residuals = actual - predictions
#     std_residual = np.std(residuals)
#     # Z-score for 95% confidence interval
#     z_score = 1.96
#     margin_of_error = z_score * std_residual
#     lower_bound = predictions - margin_of_error
#     upper_bound = predictions + margin_of_error
#     return lower_bound, upper_bound

def calculate_pi_coverage(actual, lower_bound, upper_bound):
    """Calculate prediction interval coverage"""
    coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound))
    return coverage * 100

def calculate_pi_overlap(lower1, upper1, lower2, upper2):
    """Calculate prediction interval overlap between two models"""
    overlap_lower = np.maximum(lower1, lower2)
    overlap_upper = np.minimum(upper1, upper2)
    overlap_width = np.maximum(0, overlap_upper - overlap_lower)
    width1 = upper1 - lower1
    width2 = upper2 - lower2
    avg_width = (width1 + width2) / 2
    overlap_percentage = np.mean(overlap_width / avg_width) * 100
    return overlap_percentage

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals for a set of predictions"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = stats.norm.ppf(1 - alpha/2)   # 95% PI (matches plotting code)
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound


# ------------------ MODEL IMPLEMENTATIONS ------------------ #

def run_sarima_full_predictions(train_val_df, test_df, order=(1, 0, 0), seasonal_order=(2, 2, 2, 12), seed=42):
    """Run SARIMA model and return both training and test predictions"""
    np.random.seed(seed)
    
    # Convert to series
    train_val_series = train_val_df['Deaths'].reset_index(drop=True).astype(float)
    test_series = test_df['Deaths'].reset_index(drop=True).astype(float)
    
    # Fit model
    model = SARIMAX(train_val_series, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Get training predictions (fitted values)
    train_predictions = results.fittedvalues.values
    
    # Get test predictions (forecast)
    test_predictions = results.predict(start=len(train_val_series), 
                                      end=len(train_val_series) + len(test_series) - 1).values
    
    return train_val_series.values, train_predictions, test_series.values, test_predictions

def run_lstm_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Run LSTM model and return both training and test predictions"""
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
    
    # Generate training predictions (autoregressive for consistency)
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    # Generate test predictions (autoregressive)
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return (train_val_data[lookback:], np.array(train_preds), 
            test_data, np.array(test_preds))

def run_tcn_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Run TCN model and return both training and test predictions"""
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
    
    # Generate training predictions
    train_preds = []
    for i in range(lookback, len(train_val_data)):
        input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    # Generate test predictions (autoregressive)
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return (train_val_data[lookback:], np.array(train_preds), 
            test_data, np.array(test_preds))

def build_seq2seq_model(lookback, encoder_units=128, decoder_units=128, use_attention=True):
    """Build seq2seq model with optional attention"""
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
        
        attention_layer = Attention(name='attention')
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
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
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

def run_seq2seq_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, seed, 
                                encoder_units=128, decoder_units=128, use_attention=True):
    """Run seq2seq model and return both training and test predictions"""
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
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size, 
              verbose=0, callbacks=[early_stopping], validation_split=0.1)
    
    # Generate training predictions
    train_preds_scaled = []
    for i in range(lookback, len(train_val_data)):
        encoder_input = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        train_preds_scaled.append(pred_scaled)
    
    # Generate test predictions (autoregressive)
    test_preds_scaled = []
    current_sequence = train_val_scaled[-lookback:].copy()
    
    for _ in range(len(test_data)):
        encoder_input = current_sequence.reshape((1, lookback, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        test_preds_scaled.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Inverse transform predictions
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
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

def run_transformer_full_predictions(train_val_data, test_data, lookback, batch_size, epochs, seed, d_model=64, n_heads=2):
    """Run transformer model and return both training and test predictions"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Scaling
    full_series = np.concatenate([train_val_data, test_data])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare data
    X_train, y_train = create_dataset(train_val_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    y_train = y_train.reshape((-1, 1))
    
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
    
    # Generate test predictions (autoregressive)
    current_seq = train_val_scaled[-lookback:].copy()
    test_preds_scaled = []
    for _ in range(len(test_data)):
        input_seq = current_seq.reshape((1, lookback, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        test_preds_scaled.append(pred_scaled)
        current_seq = np.append(current_seq[1:], pred_scaled)
    
    # Inverse transform predictions
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
    return (train_val_data[lookback:], train_preds_original, 
            test_data, test_preds_original)

# ------------------ MAIN EVALUATION LOOP ------------------ #

def main():
    """Main evaluation function"""
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    train_data, validation_data, test_data = create_train_val_test_split(data)
    
    # Combine train and validation for final training
    train_val_data = pd.concat([train_data, validation_data], ignore_index=True)
    
    print(f"Train+Val data shape: {train_val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Results storage
    all_results = {}
    all_predictions = {}
    
    # Models to evaluate
    models_to_evaluate = ['sarima', 'lstm', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer']
    
    for model_name in models_to_evaluate:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        
        model_dir = os.path.join(RESULTS_DIR, model_name)
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
                print(f"    Trial {trial + 1}/{TRIALS_PER_SEED}")
                
                try:
                    if model_name == 'sarima':
                        train_true, train_pred, test_true, test_pred = run_sarima_full_predictions(
                            train_val_data, test_data, 
                            order=params['order'], 
                            seasonal_order=params['seasonal_order'], 
                            seed=trial_seed)
                    
                    elif model_name == 'lstm':
                        train_true, train_pred, test_true, test_pred = run_lstm_full_predictions(
                            train_val_data['Deaths'].values, test_data['Deaths'].values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed)
                    
                    elif model_name == 'tcn':
                        train_true, train_pred, test_true, test_pred = run_tcn_full_predictions(
                            train_val_data['Deaths'].values, test_data['Deaths'].values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed)
                        
                    elif model_name == 'seq2seq':
                        train_true, train_pred, test_true, test_pred = run_seq2seq_full_predictions(
                            train_val_data['Deaths'].values, test_data['Deaths'].values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['encoder_units'], params['decoder_units'], use_attention=False)
                    
                    elif model_name == 'seq2seq_attn':
                        train_true, train_pred, test_true, test_pred = run_seq2seq_full_predictions(
                            train_val_data['Deaths'].values, test_data['Deaths'].values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['encoder_units'], params['decoder_units'], use_attention=True)
                    
                    elif model_name == 'transformer':
                        train_true, train_pred, test_true, test_pred = run_transformer_full_predictions(
                            train_val_data['Deaths'].values, test_data['Deaths'].values,
                            params['lookback'], params['batch_size'], params['epochs'], trial_seed,
                            params['d_model'], params['n_heads'])
                    
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
                    
                    # Save individual trial predictions
                    trial_df = pd.DataFrame({
                        'Train_True': pd.Series(train_true),
                        'Train_Pred': pd.Series(train_pred),
                        'Train_Lower': pd.Series(train_lower),
                        'Train_Upper': pd.Series(train_upper)
                    })
                    trial_df.to_csv(os.path.join(seed_dir, f'trial_{trial + 1}_train_predictions.csv'), index=False)
                    
                    test_df = pd.DataFrame({
                        'Test_True': pd.Series(test_true),
                        'Test_Pred': pd.Series(test_pred),
                        'Test_Lower': pd.Series(test_lower),
                        'Test_Upper': pd.Series(test_upper)
                    })
                    test_df.to_csv(os.path.join(seed_dir, f'trial_{trial + 1}_test_predictions.csv'), index=False)
                    
                except Exception as e:
                    print(f"      Error in trial {trial + 1}: {str(e)}")
                    continue
            
            # Process results for this seed
            if trial_results:
                trial_results_df = pd.DataFrame(trial_results)
                trial_results_df.to_csv(os.path.join(seed_dir, 'all_trials_metrics.csv'), index=False)
                
                # Calculate summary statistics
                summary_stats = trial_results_df.select_dtypes(include=[np.number]).agg(['mean', 'std', 'min', 'max'])
                summary_stats.to_csv(os.path.join(seed_dir, 'seed_summary_statistics.csv'))
                
                # Store seed results
                # seed_results[seed] = {
                #     'train_mean_rmse': trial_results_df['Train_RMSE'].mean(),
                #     'train_std_rmse': trial_results_df['Train_RMSE'].std(),
                #     'test_mean_rmse': trial_results_df['Test_RMSE'].mean(),
                #     'test_std_rmse': trial_results_df['Test_RMSE'].std(),
                #     'train_mean_mape': trial_results_df['Train_MAPE'].mean(),
                #     'train_std_mape': trial_results_df['Train_MAPE'].std(),
                #     'test_mean_mape': trial_results_df['Test_MAPE'].mean(),
                #     'test_std_mape': trial_results_df['Test_MAPE'].std(),
                #     'trials_completed': len(trial_results_df)
                # }
                
                seed_results[seed] = {
                    # RMSE
                    'train_mean_rmse': trial_results_df['Train_RMSE'].mean(),
                    'train_std_rmse':  trial_results_df['Train_RMSE'].std(),
                    'test_mean_rmse':  trial_results_df['Test_RMSE'].mean(),
                    'test_std_rmse':   trial_results_df['Test_RMSE'].std(),

                    # MAE  (NEW)
                    'train_mean_mae':  trial_results_df['Train_MAE'].mean(),
                    'train_std_mae':   trial_results_df['Train_MAE'].std(),
                    'test_mean_mae':   trial_results_df['Test_MAE'].mean(),
                    'test_std_mae':    trial_results_df['Test_MAE'].std(),

                    # MAPE
                    'train_mean_mape': trial_results_df['Train_MAPE'].mean(),
                    'train_std_mape':  trial_results_df['Train_MAPE'].std(),
                    'test_mean_mape':  trial_results_df['Test_MAPE'].mean(),
                    'test_std_mape':   trial_results_df['Test_MAPE'].std(),

                    # PI Coverage %  (NEW)
                    'train_mean_pi_cov': trial_results_df['Train_PI_Coverage'].mean(),
                    'train_std_pi_cov':  trial_results_df['Train_PI_Coverage'].std(),
                    'test_mean_pi_cov':  trial_results_df['Test_PI_Coverage'].mean(),
                    'test_std_pi_cov':   trial_results_df['Test_PI_Coverage'].std(),

                    'trials_completed': len(trial_results_df)
                }

                # Store average predictions for this seed
                seed_predictions[seed] = trial_predictions
                
                print(f"    Seed {seed} completed {len(trial_results_df)} trials")
                print(f"    Test RMSE: {trial_results_df['Test_RMSE'].mean():.4f} ± {trial_results_df['Test_RMSE'].std():.4f}")
        
        # Save seed comparison
        if seed_results:
            seed_comparison_df = pd.DataFrame(seed_results).T
            seed_comparison_df = seed_comparison_df.round(4)
            seed_comparison_df.to_csv(os.path.join(model_dir, 'seed_comparison.csv'))
            
            # Calculate overall model statistics
#             test_rmse_means = [seed_results[seed]['test_mean_rmse'] for seed in seed_results]
#             test_mape_means = [seed_results[seed]['test_mean_mape'] for seed in seed_results]
            
#             all_results[model_name] = {
#                 'test_mean_rmse': np.mean(test_rmse_means),
#                 'test_std_rmse': np.std(test_rmse_means),
#                 'test_mean_mape': np.mean(test_mape_means),
#                 'test_std_mape': np.std(test_mape_means),
#                 'seeds_completed': len(seed_results)
#             }
            
            
        
            # collect lists from seeds for each metric
            test_rmse_means = [seed_results[s]['test_mean_rmse'] for s in seed_results]
            test_mape_means = [seed_results[s]['test_mean_mape'] for s in seed_results]
            test_mae_means  = [seed_results[s]['test_mean_mae']  for s in seed_results]   # NEW
            test_pic_means  = [seed_results[s]['test_mean_pi_cov'] for s in seed_results] # NEW

            all_results[model_name] = {
                # RMSE
                'test_mean_rmse': np.mean(test_rmse_means),
                'test_std_rmse':  np.std(test_rmse_means),

                # MAE (NEW)
                'test_mean_mae':  np.mean(test_mae_means),
                'test_std_mae':   np.std(test_mae_means),

                # MAPE
                'test_mean_mape': np.mean(test_mape_means),
                'test_std_mape':  np.std(test_mape_means),

                # PI Coverage % (NEW)
                'test_mean_pi_cov': np.mean(test_pic_means),
                'test_std_pi_cov':  np.std(test_pic_means),

                'seeds_completed': len(seed_results)
            }


            all_predictions[model_name] = seed_predictions
            
            print(f"  Model {model_name} overall Test RMSE: {np.mean(test_rmse_means):.4f} ± {np.std(test_rmse_means):.4f}")
            # print(f"  Model {model_name} overall:"
            #   f" Test RMSE {np.mean(test_rmse_means):.2f}±{np.std(test_rmse_means):.2f},"
            #   f" MAE {np.mean(test_mae_means):.2f}±{np.std(test_mae_means):.2f},"
            #   f" MAPE {np.mean(test_mape_means):.2f}±{np.std(test_mape_means):.2f},"
            #   f" PI Cov {np.mean(test_pic_means):.1f}%±{np.std(test_pic_means):.1f}%")

    
    # Create final comparison table
    if all_results:
        final_comparison_df = pd.DataFrame(all_results).T
        final_comparison_df = final_comparison_df.round(4)
        final_comparison_df.to_csv(os.path.join(RESULTS_DIR, 'final_model_comparison.csv'))
        
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        print(final_comparison_df.to_string())
        
    # Save aggregated predictions for plotting
    import pickle
    with open(os.path.join(RESULTS_DIR, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(all_predictions, f)
    
    # Save data splits for plotting
    data_info = {
        'full_data': data,
        'train_data': train_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'train_val_data': train_val_data
    }
    with open(os.path.join(RESULTS_DIR, 'data_splits.pkl'), 'wb') as f:
        pickle.dump(data_info, f)
        
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("All predictions and data splits saved for plotting scripts.")

if __name__ == "__main__":
    main()