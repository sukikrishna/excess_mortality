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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math

# ------------------ CONFIGURATION ------------------ #
MODEL_TYPE = 'transformer'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'transformer', 'tcn_updated', 'tcn_fixed', 'seq2seq_attn'
TRIAL_MODE = 'fixed_seed'  # Options: 'fixed_seed', 'multi_seed'
SEEDS = [42] if TRIAL_MODE == 'fixed_seed' else [123, 456, 11, 245, 56712, 23467, 98, 38, 1506, 42]
TRIALS_PER_CONFIG = 30

LOOKBACKS = [3, 5, 7, 9, 11, 12]
BATCH_SIZES = [8, 16, 32]
EPOCHS_LIST = [50, 100]

DATA_PATH = 'data/state_month_overdose.xlsx'
RESULTS_DIR = 'results'

# ------------------ HELPER FUNCTIONS ------------------ #
def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH)
    df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
    return df

def create_dataset(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'PI Width': 0,
        'CI Coverage': 0,
        'PI Overlap': 0
    }


def create_train_val_test_split_lstm(df: pd.DataFrame, 
                                  train_end: str = '2019-01-01',
                                  val_end: str = '2020-01-01', 
                                  test_end: str = '2020-12-01'):
    """Create proper train/validation/test splits"""
    train = df[df['Month'] < train_end]#['Deaths'].values
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]#['Deaths'].values
    test = df[(df['Month'] >= val_end)]# & (df['Month'] <= test_end)]#['Deaths'].values
    
    return train, validation, test




def run_lstm(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(np.concatenate([train[-look_back:], test]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))
    model = Sequential([LSTM(50, activation='relu', input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = []
    current_input = X_test[0].reshape((1, look_back, 1))
    for _ in range(len(y_test)):
        pred = model.predict(current_input)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    return y_test, np.array(preds)


def run_sarima(train_df, test_df):
    np.random.seed(seed)
    train_series = train_df['Deaths'].astype(float)
    test_series = test_df['Deaths'].astype(float)

    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # In-sample fit (for training)
    fitted = results.fittedvalues

    # Forecast (for test)
    forecast = results.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
    
    return train_series.values, fitted.values, test_series.values, forecast.values


def run_tcn(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = create_dataset(train, look_back)
    extended_test = np.concatenate([train[-look_back:], test])
    y_test = extended_test[look_back:]

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = Sequential([TCN(input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # convert to NumPy array before reshaping
    current_input = np.array(train[-look_back:]).reshape((1, look_back, 1))
    preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    return y_test, np.array(preds)

    
# def run_seq2seq(train, test, look_back, batch_size, epochs, seed):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

#     # Prepare training data
#     X_train, y_train = create_dataset(train, look_back)
#     X_train = X_train.reshape((X_train.shape[0], look_back, 1))
#     y_train = y_train.reshape((-1, 1))

#     # Encoder
#     encoder_inputs = Input(shape=(look_back, 1))
#     encoder_outputs, state_h = GRU(64, return_state=True)(encoder_inputs)

#     # Decoder
#     decoder_input = Input(shape=(1, 1))  # feed previous value
#     decoder_gru = GRU(64, return_sequences=False, return_state=False)
#     decoder_dense = Dense(1)

#     decoder_output = decoder_gru(decoder_input, initial_state=state_h)
#     decoder_output = decoder_dense(decoder_output)

#     model = Model(inputs=[encoder_inputs, decoder_input], outputs=decoder_output)
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     # Training decoder input is last time step
#     decoder_inputs_train = np.array([X_train[i, -1, 0] for i in range(X_train.shape[0])])
#     decoder_inputs_train = decoder_inputs_train.reshape((-1, 1, 1))

#     model.fit([X_train, decoder_inputs_train], y_train, epochs=epochs, batch_size=batch_size, verbose=0)

#     # ----------------- Inference: rolling prediction -----------------
#     input_seq = np.array(train[-look_back:]).reshape((1, look_back, 1))
#     preds = []

#     for _ in range(len(test)):
#         decoder_input = np.array([[[input_seq[0, -1, 0]]]])  # feed last value
#         pred = model.predict([input_seq, decoder_input], verbose=0)[0][0]
#         preds.append(pred)
#         # roll input forward with prediction
#         input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

#     return test, np.array(preds)



def build_seq2seq_model(look_back, encoder_units=64, decoder_units=64, use_attention=True):
    encoder_inputs = Input(shape=(look_back, 1), name='encoder_input')

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

def run_seq2seq(train, test, look_back, batch_size, epochs, seed,
                encoder_units=64, decoder_units=64, use_attention=True):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ----------- Scaling -----------
    full_series = np.concatenate([train, test])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]

    # ----------- Dataset -----------
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    decoder_input_train = np.zeros((X_train.shape[0], 1, 1))  # start token
    y_train = y_train.reshape((-1, 1, 1))

    # ----------- Model -----------
    model = build_seq2seq_model(look_back, encoder_units, decoder_units, use_attention)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])

    model.fit(
        [X_train, decoder_input_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        ],
        validation_split=0.1
    )

    # ----------- Autoregressive Inference -----------
    preds_scaled = []
    current_sequence = train_scaled[-look_back:].copy()

    for _ in range(len(test)):
        encoder_input = current_sequence.reshape((1, look_back, 1))
        decoder_input = np.zeros((1, 1, 1))  # start token

        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        preds_scaled.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)

    preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    y_test = test
    return y_test, preds_original




    

class PositionalEncoding(tf.keras.layers.Layer):
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


def run_transformer(train, test, look_back, batch_size, epochs, seed, d_model=64, n_heads=2):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ----------- Scaling -----------
    full_series = np.concatenate([train, test])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]

    # ----------- Prepare data -----------
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    y_train = y_train.reshape((-1, 1))

    # ----------- Build model -----------
    inputs = Input(shape=(look_back, 1))
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # ----------- Autoregressive Forecasting -----------
    current_seq = train_scaled[-look_back:].copy()
    preds_scaled = []
    for _ in range(len(test)):
        input_seq = current_seq.reshape((1, look_back, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(pred_scaled)
        current_seq = np.append(current_seq[1:], pred_scaled)

    preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    y_test = test
    return y_test, preds_original


data = load_and_preprocess_data()
train_data, validation_data, test_data = create_train_val_test_split_lstm(data)

# train_data = data[data['Month'] < '2019-01-01']
# validation_data = data[(data['Month'] >= '2019-01-01') & (data['Month'] <= '2019-12-01')]

for seed in SEEDS:
    for look_back in LOOKBACKS:
        for bs in BATCH_SIZES:
            for ep in EPOCHS_LIST:
                config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}'
                print(config_name)
                base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                    'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                    config_name)
                os.makedirs(base_dir, exist_ok=True)
                metrics_all_test = []
                metrics_all_train = []
                for trial in range(TRIALS_PER_CONFIG):
                    if MODEL_TYPE == 'lstm':
                        y_train_true, y_train_pred = run_lstm(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_lstm(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                    elif MODEL_TYPE == 'sarima':
                        # y_train_true, y_train_pred = run_sarima(train_data, train_data.iloc[look_back:])
                        # y_test_true, y_test_pred = run_sarima(train_data, validation_data)
                        y_train_true, y_train_pred, y_test_true, y_test_pred = run_sarima(train_data, validation_data)

                    elif MODEL_TYPE == 'tcn':
                        y_train_true, y_train_pred = run_tcn(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_tcn(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                        
                    # elif MODEL_TYPE == 'tcn_updated':
                    #     y_train_true, y_train_pred = run_tcn_updated(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                    #     y_test_true, y_test_pred = run_tcn_updated(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)

                    # elif MODEL_TYPE == 'tcn_fixed':
                    #     y_train_true, y_train_pred = run_tcn_fixed(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                    #     y_test_true, y_test_pred = run_tcn_fixed(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                        
                    elif MODEL_TYPE == 'seq2seq_attn':
                        y_train_true, y_train_pred = run_seq2seq(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed, use_attention=True)
                        y_test_true, y_test_pred = run_seq2seq(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed, use_attention=True)

                    elif MODEL_TYPE == 'transformer':
                        y_train_true, y_train_pred = run_transformer(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_transformer(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)

                    else:
                        raise ValueError("Unknown model type")

                    train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                    test_df = pd.DataFrame({'True': y_test_true, 'Pred': y_test_pred})

                    train_df.to_csv(
                        os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                    test_df.to_csv(
                        os.path.join(base_dir, f'trial_{trial}_test.csv'), index=False)
                    metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                    metrics_test = evaluate_metrics(y_test_true, y_test_pred)
                    metrics_all_train.append(metrics_train)
                    metrics_all_test.append(metrics_test)
                pd.DataFrame(metrics_all_train).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_train.csv'))
                pd.DataFrame(metrics_all_test).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_test.csv'))
