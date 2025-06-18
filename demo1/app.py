import matplotlib
matplotlib.use('Agg')
import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from utils_lstm import load_and_clean_data, create_lstm_sequences

# --- 1. App 初始化與設定 ---
app = Flask(__name__)
app.secret_key = 'a-very-secret-and-robust-key'

# 定義目錄路徑
DATA_DIR = 'data'
TEST_DIR = 'test'
MODEL_DIR = 'model'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_gold_predictor.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
SEQUENCE_LENGTH = 60

# --- 2. 基礎路由 ---
@app.route('/')
def index():
    return redirect(url_for('predict_page'))

# --- 3. 模型訓練功能 ---
@app.route('/train', methods=['GET', 'POST'])
def train_page():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            flash('錯誤：未選擇任何檔案。', 'danger')
            return redirect(request.url)
        
        # 清空舊數據並儲存新檔案
        for f in os.listdir(DATA_DIR):
            try:
                os.remove(os.path.join(DATA_DIR, f))
            except Exception as e:
                print(f"無法刪除檔案 {f}: {e}")

        for file in files:
            if file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                file.save(os.path.join(DATA_DIR, filename))
        
        flash(f'成功上傳 {len(files)} 個檔案。開始訓練...', 'info')

        try:
            train_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
            price_data = load_and_clean_data(train_files)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(price_data['Price'].values.reshape(-1, 1))
            X_train, y_train = create_lstm_sequences(scaled_prices, SEQUENCE_LENGTH)

            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            # 增加 verbose=1 可以在伺服器日誌中看到訓練進度
            model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1) 
            
            model.save(MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            
            flash('模型訓練成功，並已儲存！現在您可以進行預測和測試了。', 'success')
        except Exception as e:
            flash(f'模型訓練失敗：{e}', 'danger')

        return redirect(url_for('train_page'))

    return render_template('train.html')

# --- 4. 模型測試功能 ---
@app.route('/test', methods=['GET', 'POST'])
def test_page():
    if request.method == 'POST':
        # 每次請求時都重新載入，確保使用最新模型和數據
        try:
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            train_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
            df_hist = load_and_clean_data(train_files)
            if df_hist is None or df_hist.empty:
                raise FileNotFoundError("找不到或無法載入 'data' 資料夾中的歷史數據。")
        except FileNotFoundError as e:
            flash(f'模型或數據尚未準備好。請先訓練模型。錯誤: {e}', 'danger')
            return redirect(url_for('test_page'))
            
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('錯誤：未選擇測試檔案。', 'danger')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            # ... (後續的測試邏輯與之前相同，這裡省略以保持簡潔) ...
            # 確保這部分的 try-except 塊也存在
            try:
                # ... 完整的測試和繪圖邏輯 ...
                 filename = secure_filename(file.filename)
                 test_filepath = os.path.join(TEST_DIR, filename)
                 file.save(test_filepath)
                 df_new_test = load_and_clean_data([test_filepath])
                 test_input_data = pd.concat([df_hist.tail(SEQUENCE_LENGTH), df_new_test], ignore_index=True)
                 scaled_test_prices = scaler.transform(test_input_data['Price'].values.reshape(-1, 1))
                 X_test, y_test_scaled = create_lstm_sequences(scaled_test_prices, SEQUENCE_LENGTH)
                 if len(X_test) == 0:
                     flash(f"測試檔案數據不足 ({len(df_new_test)}行)，無法生成測試序列。", 'danger')
                     return redirect(url_for('test_page'))
                 y_pred_scaled = model.predict(X_test)
                 y_pred = scaler.inverse_transform(y_pred_scaled)
                 y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))
                 # ... 指標計算和圖表生成 ...
                 mae = mean_absolute_error(y_test, y_pred)
                 mse = mean_squared_error(y_test, y_pred)
                 rmse = np.sqrt(mse)
                 r2 = r2_score(y_test, y_pred)
                 residuals = y_test.flatten() - y_pred.flatten()
                 plots = {}
                 fig1 = plt.figure(figsize=(10, 5)); plt.plot(df_new_test['Date'], y_test, 'b-o', label='真實價格'); plt.plot(df_new_test['Date'], y_pred, 'r--x', label='預測價格'); plt.legend(); plt.grid(True); buf1 = BytesIO(); fig1.savefig(buf1, format="png"); plots['plot1'] = base64.b64encode(buf1.getbuffer()).decode("ascii"); plt.close(fig1)
                 fig2 = plt.figure(figsize=(10, 5)); plt.scatter(y_pred, residuals, alpha=0.6); plt.axhline(y=0, color='r', linestyle='--'); plt.xlabel("預測價格"); plt.ylabel("殘差"); plt.grid(True); buf2 = BytesIO(); fig2.savefig(buf2, format="png"); plots['plot2'] = base64.b64encode(buf2.getbuffer()).decode("ascii"); plt.close(fig2)
                 result_data = {'mae': f'{mae:.2f}', 'mse': f'{mse:.2f}', 'rmse': f'{rmse:.2f}', 'r2': f'{r2:.4f}', 'plots': plots}
                 return render_template('test.html', result=result_data)
            except Exception as e:
                flash(f'模型測試失敗：{e}', 'danger')

    return render_template('test.html')

# --- 5. 預測明日價格功能 ---
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    # 每次都嘗試獲取最新日期，以便日期選擇器能正常工作
    latest_date = ""
    try:
        train_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if train_files:
            df_for_date = load_and_clean_data(train_files)
            if not df_for_date.empty:
                latest_date = df_for_date['Date'].max().strftime('%Y-%m-%d')
    except Exception:
        pass # 靜默處理，如果沒有數據，日期選擇器為空即可

    if request.method == 'POST':
        target_date_str = request.form.get('date')
        if not target_date_str:
            flash('請選擇一個基準日期。', 'warning')
            return redirect(url_for('predict_page'))

        try:
            # 在需要時才載入模型和數據
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            full_df = load_and_clean_data(train_files).set_index('Date')
            
            target_date = pd.to_datetime(target_date_str)
            if target_date not in full_df.index:
                raise KeyError

            last_60_days = full_df.loc[:target_date].tail(SEQUENCE_LENGTH)

            if len(last_60_days) < SEQUENCE_LENGTH:
                flash(f"歷史數據不足（需要 {SEQUENCE_LENGTH} 天）。", "warning")
                return render_template('predict.html', latest_date=latest_date)

            # ... (後續預測邏輯不變) ...
            today_price = full_df.loc[target_date]['Price']
            last_60_days_scaled = scaler.transform(last_60_days['Price'].values.reshape(-1, 1))
            input_feature = np.reshape(last_60_days_scaled, (1, SEQUENCE_LENGTH, 1))
            predicted_price_scaled = model.predict(input_feature)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
            result = {'target_date': target_date.strftime('%Y-%m-%d'), 'today_price': f"${today_price:,.2f}", 'next_day': (target_date + timedelta(days=1)).strftime('%Y-%m-%d'), 'predicted_price': f"${predicted_price:,.2f}"}
            return render_template('predict.html', result=result, latest_date=latest_date)

        except FileNotFoundError:
            flash('模型檔案或訓練數據不存在，請先前往「訓練模型」頁面。', 'danger')
        except KeyError:
            flash(f"資料庫中找不到日期 '{target_date_str}' 的價格。", 'danger')
        except Exception as e:
            flash(f'預測時發生錯誤: {e}', 'danger')
            
        return redirect(url_for('predict_page'))

    # GET 請求
    return render_template('predict.html', latest_date=latest_date)

# --- 6. 啟動伺服器 ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)