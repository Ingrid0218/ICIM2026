import os
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from math import sqrt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings

# -----------------------------------------------------------
# 環境設定
# -----------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 設定檔名變數 (SimpleRNN)
FILE_DATA = r"C:\Users\hscy\Desktop\AI\final\prices-split-adjusted.csv"
FILE_OPTUNA_HISTORY = "RNN_Optuna_All_Trials_History.csv"
FILE_METRICS = "RNN_Optuna_Best_Model_Metrics.csv"
FILE_PREDS = "RNN_Optuna_Best_Model_Preds.csv"

# -----------------------------------------------------------
# 1. 資料處理函數
# -----------------------------------------------------------
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# -----------------------------------------------------------
# 2. 載入與準備資料
# -----------------------------------------------------------
print("正在載入資料...")
df = pd.read_csv(FILE_DATA)
df_goog = df[df.symbol == 'GOOG'].copy()
dataset = df_goog[['open', 'high', 'low', 'close', 'volume']]

values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n_ts = 3          
n_features = 5    

reframed = series_to_supervised(scaled, n_ts, 1)
values = reframed.values

n_test = 100
n_train = len(values) - n_test
train = values[:n_train, :]
test = values[n_train:, :]

n_obs = n_ts * n_features
train_x_full, train_y_full = train[:, :n_obs], train[:, -2]
test_x, test_y = test[:, :n_obs], test[:, -2]

train_x_full = train_x_full.reshape((train_x_full.shape[0], n_ts, n_features))
test_x = test_x.reshape((test_x.shape[0], n_ts, n_features))

val_split_idx = int(len(train_x_full) * 0.9)
X_train_opt = train_x_full[:val_split_idx]
y_train_opt = train_y_full[:val_split_idx]
X_val_opt = train_x_full[val_split_idx:]
y_val_opt = train_y_full[val_split_idx:]

# -----------------------------------------------------------
# 3. 定義 Optuna Objective Function (SimpleRNN)
# -----------------------------------------------------------
def objective(trial):
    tf.keras.backend.clear_session()
    
    # --- 定義搜尋空間 ---
    # RNN 容易梯度爆炸，建議 Units 不用太大，最大設到 128 即可
    n_rnn_units = trial.suggest_int('rnn_units', 32, 128, step=16)
    n_dense_units = trial.suggest_int('dense_units', 16, 128, step=16)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    activation_func = trial.suggest_categorical('activation', ['relu', 'swish', 'tanh'])
    # --- 建立 SimpleRNN 模型 ---
    input_layer = Input(shape=(n_ts, n_features))
    
    # 使用 SimpleRNN 層
    x = SimpleRNN(n_rnn_units, return_sequences=False)(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_dense_units, activation=activation_func)(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_opt, y_train_opt,
        validation_data=(X_val_opt, y_val_opt),
        epochs=30,
        batch_size=batch_size,
        verbose=0,
        callbacks=[earlystop]
    )
    
    val_loss = min(history.history['val_loss'])
    val_mae = min(history.history['val_mae'])
    
    trial.set_user_attr("val_mae", val_mae)
    
    return val_loss

# -----------------------------------------------------------
# 4. 執行 Optuna 優化
# -----------------------------------------------------------
print("\n[Step 1] 開始執行 SimpleRNN Optuna 超參數優化...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) 

print("\n[Step 2] 優化完成，正在匯出詳細實驗記錄...")

df_trials = study.trials_dataframe()

rename_map = {
    'number': 'Trial_ID',
    'value': 'Val_MSE_Loss',
    'params_rnn_units': 'RNN_Units',     # 改名
    'params_dense_units': 'Dense_Units',
    'params_activation': 'Activation',  # 新增
    'params_dropout': 'Dropout',
    'params_lr': 'Learning_Rate',
    'params_batch_size': 'Batch_Size',
    'user_attrs_val_mae': 'Val_MAE',
    'state': 'Status'
}

cols_to_keep = [c for c in rename_map.keys() if c in df_trials.columns]
df_trials_clean = df_trials[cols_to_keep].rename(columns=rename_map)
df_trials_clean = df_trials_clean.sort_values(by='Val_MSE_Loss')

df_trials_clean.to_csv(FILE_OPTUNA_HISTORY, index=False)
print(f"詳細實驗記錄已儲存至: {FILE_OPTUNA_HISTORY}")
print("前 5 名最佳參數組合：")
print(df_trials_clean.head())

# -----------------------------------------------------------
# 5. 使用最佳參數進行最終驗證 (5次循環) [RNN 修正版]
# -----------------------------------------------------------
best_params = study.best_params
print(f"\n[Step 3] 使用最佳參數進行最終測試: {best_params}")

metrics_records = []
col_idx = 3

for run in range(1, 6):
    print(f"  > 重複測試 Run {run}/5 ...")
    tf.keras.backend.clear_session()

    # 重建最佳 RNN 模型
    input_layer = Input(shape=(n_ts, n_features))
    
    x = SimpleRNN(best_params['rnn_units'], return_sequences=False)(input_layer)
    x = Dropout(best_params['dropout'])(x)
    
    # [修正] 使用最佳激活函數
    x = Dense(best_params['dense_units'], activation=best_params['activation'])(x)
    
    x = Dropout(best_params['dropout'])(x)
    output = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mse', metrics=['mae'])
    
    earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        train_x_full, train_y_full,
        validation_data=(test_x, test_y),
        epochs=50,
        batch_size=best_params['batch_size'],
        verbose=0, 
        callbacks=[earlystop]
    )
    
    yhat = model.predict(test_x).flatten()
    
    inv_pred = []
    inv_true = []
    arr_template = np.zeros((n_features,))
    
    for i in range(len(yhat)):
        arr_pred = arr_template.copy()
        arr_true = arr_template.copy()
        arr_pred[col_idx] = yhat[i]
        arr_true[col_idx] = test_y[i]
        inv_pred.append(scaler.inverse_transform(arr_pred.reshape(1, -1))[0, col_idx])
        inv_true.append(scaler.inverse_transform(arr_true.reshape(1, -1))[0, col_idx])
        
    inv_pred = np.array(inv_pred).reshape(-1, 1)
    inv_true = np.array(inv_true).reshape(-1, 1)
    
    rmse = sqrt(mean_squared_error(inv_true, inv_pred))
    mae = mean_absolute_error(inv_true, inv_pred)
    r2 = r2_score(inv_true, inv_pred)
    mape = mean_absolute_percentage_error(inv_true, inv_pred)
    
    # [修正] 加入 activation
    metrics_records.append([
        run, 
        rmse, mae, r2, mape,
        best_params['rnn_units'],
        best_params['dense_units'],
        best_params['activation'], # 新增
        best_params['dropout'],
        best_params['lr'],
        best_params['batch_size']
    ])
    
    if run == 5:
        final_pred = inv_pred
        final_true = inv_true

# -----------------------------------------------------------
# 6. 輸出最終指標與預測值 [RNN 修正版]
# -----------------------------------------------------------
columns = [
    "Run", "RMSE", "MAE", "R2", "MAPE", 
    "RNN_Units", "Dense_Units", "Activation", "Dropout", "Learning_Rate", "Batch_Size"
]

metric_df = pd.DataFrame(metrics_records, columns=columns)

# [修正] 平均值計算邏輯 (避開文字欄位)
numeric_cols = metric_df.select_dtypes(include=[np.number]).columns
mean_vals = metric_df[numeric_cols].mean()

mean_row = pd.Series(index=columns)
mean_row["Run"] = "Average"
for col in numeric_cols:
    if col != "Run":
        mean_row[col] = mean_vals[col]
        
# 補回文字欄位
mean_row["Activation"] = metric_df.iloc[0]["Activation"]

metric_df.loc[len(metric_df)] = mean_row

metric_df.to_csv(FILE_METRICS, index=False)
print(f"\n最佳模型 5 次測試平均結果 (含參數) 已儲存至: {FILE_METRICS}")
print(metric_df)

results = pd.DataFrame({
    "Actual_Close": final_true.flatten(),
    "Pred_Close": final_pred.flatten()
})
results.to_csv(FILE_PREDS, index=False)
print(f"預測股價數據已儲存至: {FILE_PREDS}")