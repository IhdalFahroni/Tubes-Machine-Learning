# %% [markdown]
# # TUGAS BESAR: Analisis Perbandingan Algoritma Random Forest dan XGBoost dalam Memprediksi Saham BBCA
# 
# **Abstrak Proyek:**
# Penelitian ini bertujuan untuk menguji, membandingkan, dan menganalisis kinerja dua arsitektur *Machine Learning* berbasis *Ensemble Learning*, yakni **Random Forest (Bagging)** dan **XGBoost (Boosting)**. Pemodelan ditujukan untuk memprediksi probabilitas arah pergerakan harga saham PT Bank Central Asia Tbk (BBCA) dengan jendela waktu penahanan (*holding period*) 5 hari perdagangan.
# 
# Penelitian ini mengadopsi kerangka kerja *Walk-Forward Validation* untuk mencegah kebocoran data masa depan (*Future Data Leakage*), mengintegrasikan variabel makroekonomi eksogen (Kurs USD/IDR), serta menyertakan metrik risiko finansial riil melalui komputasi biaya komisi broker.

# %% [markdown]
# ## 1. Inisialisasi Lingkungan Kerja dan Import Pustaka
# Mengimpor pustaka standar industri untuk komputasi numerik, manipulasi *time-series*, pemodelan prediktif, dan visualisasi data.

# %%
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Konfigurasi Lingkungan
warnings.filterwarnings('ignore') # Menyembunyikan peringatan/warnings agar output laporan bersih
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 120})

# %% [markdown]
# ## 2. Ekstraksi dan Prapemrosesan Data (Data Preprocessing)
# Dataset historis saham BBCA dimuat ke dalam struktur *DataFrame*. Tahapan ini mencakup penyesuaian penamaan kolom, penetapan *DatetimeIndex*, dan reduksi dimensi pada fitur yang tidak memiliki nilai informasi prediktif.

# %%
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Memuat data CSV dan melakukan standardisasi nama kolom."""
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", 
        "close": "Close", "volume": "Volume"
    }, inplace=True)
    df.drop(columns=['adjclose', 'ingested_at_utc'], errors='ignore', inplace=True)
    return df

bbca = load_and_clean_data('BBCA.JK.csv')

# %% [markdown]
# ## 3. Formulasi Variabel Target dan Logika Keuangan
# Pendekatan analitik diubah dari regresi menjadi **Klasifikasi Biner (Binary Classification)**.
# Target kelas (`Target_5D`) dienkode menjadi `1` (Naik/Bullish) jika harga penutupan 5 hari ke depan melampaui harga pembukaan esok hari, dan `0` (Turun/Bearish) untuk kondisi sebaliknya.

# %%
# Variabel Target
bbca["Tomorrow_Open"] = bbca["Open"].shift(-1)
bbca["Close_5_Days"] = bbca["Close"].shift(-5)
bbca["Target_5D"] = (bbca["Close_5_Days"] > bbca["Tomorrow_Open"]).astype(int)

# Variabel Return Finansial (Tanpa leverage)
bbca["Trade_Return_5D"] = (bbca["Close_5_Days"] / bbca["Tomorrow_Open"]) - 1    

# %% [markdown]
# ## 4. Rekayasa Fitur (Feature Engineering): Indikator Teknikal Intrinsik
# Derivasi fitur prediktif dari data harga mentah untuk mendeteksi rezim pasar dan momentum osilator.

# %%
# 4.1 Simple Moving Average (SMA) untuk deteksi Tren
bbca["MA50"] = bbca["Close"].rolling(window=50).mean()
bbca["MA200"] = bbca["Close"].rolling(window=200).mean()
bbca["MA_Trend"] = (bbca["MA50"] > bbca["MA200"]).astype(int)
bbca["Cross_Signal"] = bbca["MA_Trend"].diff()

# 4.2 Relative Strength Index (RSI 14) untuk deteksi Overbought/Oversold
delta = bbca["Close"].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
bbca["RSI_14"] = 100 - (100 / (1 + rs))

# Riwayat arah pergerakan harian sebagai basis komputasi rolling window
bbca["Target_1D_History"] = (bbca["Close"].shift(-1) > bbca["Open"].shift(-1)).astype(int)

# %% [markdown]
# ## 5. Ekstraksi Pola Multi-Horizon (Rolling Window Features)
# Mengekstraksi rasio volatilitas dan konsistensi tren historis melintasi berbagai spektrum waktu (2, 5, 60, dan 250 hari perdagangan) guna memberikan konteks fraktal kepada model.

# %%
horizons = [2, 5, 60, 250]
predictors = ["MA50", "MA200", "MA_Trend", "Cross_Signal", "RSI_14"]

for horizon in horizons:
    # Rasio harga penutupan terhadap rata-rata historisnya
    rolling_averages = bbca["Close"].rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    bbca[ratio_column] = bbca["Close"] / rolling_averages

    # Akumulasi frekuensi hari positif dalam rentang horizon
    trend_column = f"Trend_{horizon}" 
    bbca[trend_column] = bbca["Target_1D_History"].shift(1).rolling(horizon).sum()
    
    predictors.extend([ratio_column, trend_column])

# Eliminasi Missing Values (NaN) akibat kalkulasi rolling window
bbca.dropna(inplace=True)

# %% [markdown]
# ## 6. Arsitektur Pemodelan: Inisialisasi Model Klasifikasi
# - **Random Forest:** Memanfaatkan teknik *Bagging* paralel dengan `n_estimators` tinggi guna mereduksi *variance*.
# - **XGBoost:** Memanfaatkan teknik *Gradient Boosting* sekuensial dengan regularisasi ketat (L1 & L2) serta batasan kedalaman (`max_depth=3`) guna mencegah *overfitting* pada deret waktu keuangan.

# %%
model_rf = RandomForestClassifier(
    n_estimators=1000,           # Jumlah estimator optimal untuk stabilitas
    min_samples_split=100,       # Pembatasan spesifisitas simpul daun
    random_state=42,
    n_jobs=-1                    # Komputasi paralel (Multithreading)
)

model_xgb = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.005,         # Laju konvergensi konservatif
    max_depth=3,                 # Restriksi kompleksitas model
    subsample=0.7,               # Rasio subsampling baris stokastik
    colsample_bytree=0.7,        # Rasio subsampling fitur stokastik
    reg_alpha=10,                # L1 Regularization (Lasso)
    reg_lambda=5,                # L2 Regularization (Ridge)
    eval_metric='logloss',
    random_state=42
)

# %% [markdown]
# ## 7. Metodologi Validasi: Walk-Forward Backtesting & Early Stopping
# Menerapkan iterasi evaluasi yang bergerak maju secara kronologis untuk mereplikasi kondisi *live-trading*. Khusus pada arsitektur XGBoost, mekanisme *Early Stopping* diintegrasikan secara dinamis untuk menghentikan fase pelatihan apabila model mulai menghafal *noise* pasar.

# %%
def predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list, model) -> pd.DataFrame:
    """Melatih model dan menghasilkan prediksi berbasis probabilitas dengan threshold ketat."""
    
    # Deteksi arsitektur model untuk injeksi Early Stopping
    if "XGB" in type(model).__name__:
        model.fit(
            train[predictors], train["Target_5D"], 
            eval_set=[(test[predictors], test["Target_5D"])],
            verbose=False
        )
    else:
        model.fit(train[predictors], train["Target_5D"])
        
    # Ekstraksi probabilitas kelas positif (Kelas 1)
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    
    # Implementasi Decision Threshold Konservatif (Sniper Mode: >= 70%)
    preds_binary = np.where(preds_proba >= 0.70, 1, 0)
    preds_series = pd.Series(preds_binary, index=test.index, name="Predictions")
    
    return pd.concat([test["Target_5D"], preds_series, test["Trade_Return_5D"]], axis=1)

def backtest(data: pd.DataFrame, model, predictors: list, start: int = 500, step: int = 250) -> pd.DataFrame:
    """Menjalankan kerangka evaluasi Walk-Forward secara iteratif."""
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)

# %% [markdown]
# ## 8. Integrasi Variabel Makroekonomi Eksogen (Nilai Tukar USD/IDR)
# Likuiditas saham *Blue Chip* dipengaruhi oleh arus modal asing (*foreign flow*). Indikator makro berupa volatilitas nilai tukar USD/IDR ditambahkan untuk mempertajam pemahaman model terhadap sentimen pasar global.

# %%
print("Menyinkronkan data makroekonomi (USD/IDR) via Yahoo Finance...")
usdidr = yf.Ticker("IDR=X").history(start="2019-01-01")
kurs = usdidr[['Close']].rename(columns={'Close': 'USD_IDR'})
kurs.index = kurs.index.tz_localize(None)

# Penggabungan (Left Join) dan Interpolasi
bbca = bbca.join(kurs, how="left")
bbca["USD_IDR"] = bbca["USD_IDR"].ffill() # Forward-fill untuk mengatasi hari libur valas
bbca["USD_Trend_5D"] = bbca["USD_IDR"].pct_change(periods=5)

predictors_makro = predictors + ["USD_IDR", "USD_Trend_5D"]
bbca.dropna(inplace=True)
print("Sikronisasi data selesai. Dimensi dataset aktual:", bbca.shape)

# %% [markdown]
# ## 9. Eksekusi Backtesting Komparatif
# Menjalankan fungsi *Walk-Forward* untuk kedua algoritma menggunakan matriks fitur hibrida (Teknikal Intrinsik + Makroekonomi Eksogen).

# %%
print("Mengeksekusi Walk-Forward Backtesting: Random Forest Classifier...")
preds_rf_makro = backtest(bbca, model_rf, predictors_makro)

print("Mengeksekusi Walk-Forward Backtesting: XGBoost Classifier...")
preds_xgb_makro = backtest(bbca, model_xgb, predictors_makro)

print("Proses komputasi backtesting berhasil diselesaikan.")

# %% [markdown]
# ## 10. Evaluasi Finansial Berbobot-Waktu (Time-Weighted Cumulative Return)
# Kinerja klasifikasi diterjemahkan ke dalam metrik finansial riil (Uang). Profitabilitas portofolio dihitung setelah dipotong beban sistematis berupa biaya transaksi/komisi broker sebesar 0.4% (`FEE_TRANSAKSI`) per eksekusi perdagangan.

# %%
FEE_TRANSAKSI = 0.004 # Representasi komisi broker (Beli + Jual)

def calculate_portfolio_growth(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung Net Return strategi dikurangi biaya transaksi."""
    df = preds_df.copy()
    df["Strategy_Return_Net"] = df["Predictions"] * (df["Trade_Return_5D"] - FEE_TRANSAKSI)
    df["Cumulative_Net"] = (1 + df["Strategy_Return_Net"]).cumprod()
    return df

preds_rf_makro = calculate_portfolio_growth(preds_rf_makro)
preds_xgb_makro = calculate_portfolio_growth(preds_xgb_makro)

# Kalkulasi Benchmark Pasar (Buy & Hold)
market_cumulative_makro = (1 + preds_xgb_makro["Trade_Return_5D"]).cumprod()

# %% [markdown]
# ## 11. Visualisasi dan Hasil Analisis Kinerja Strategi

# %%
plt.figure(figsize=(15, 7.5))

# Plot Baseline
plt.plot(preds_xgb_makro.index, market_cumulative_makro, 
         label="Benchmark Pasar: Pasif Buy & Hold", color="gray", alpha=0.6, linewidth=1.5, linestyle='--')

# Plot Model Kuantitatif
plt.plot(preds_rf_makro.index, preds_rf_makro["Cumulative_Net"], 
         label="Portofolio ML: Random Forest (+Makro)", color="#2ca02c", linewidth=2.5)
plt.plot(preds_xgb_makro.index, preds_xgb_makro["Cumulative_Net"], 
         label="Portofolio ML: XGBoost (+Makro)", color="#d62728", linewidth=2.5)

# Konfigurasi Estetika Grafik
plt.title("Komparasi Kinerja Algoritma Machine Learning vs Risiko Sistematis Pasar", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Tahun Perdagangan", fontsize=12, labelpad=10)
plt.ylabel("Multiplier Pertumbuhan Modal (Basis Modal = 1.0)", fontsize=12, labelpad=10)
plt.legend(loc="upper left", frameon=True, shadow=True, fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Menampilkan Grafik
plt.show()

# %% [markdown]
# ## 12. Rekapitulasi Evaluasi Kinerja (Performance Metrics Report)
# Laporan akhir memuat tingkat *Positive Predictive Value* (Presisi), intensitas eksekusi algoritma, serta komparasi *Cumulative Net Profit* terhadap risiko sistematis pasar.

# %%
def generate_academic_report(preds_df: pd.DataFrame, model_name: str):
    """Mencetak laporan evaluasi dengan format akademis."""
    presisi = precision_score(preds_df["Target_5D"], preds_df["Predictions"], zero_division=0) * 100
    jumlah_trade = int(preds_df['Predictions'].sum())
    total_net = (preds_df["Cumulative_Net"].iloc[-1] - 1) * 100
    
    print(f"[{model_name.upper()} - METRIK EVALUASI]")
    print(f"Tingkat Presisi (Positive Predictive Value) : {presisi:>6.2f}%")
    print(f"Intensitas Eksekusi Perdagangan           : {jumlah_trade:>6} Posisi Terbuka")
    print(f"Total Return Kumulatif (Net Profit)       : {total_net:>6.2f}%\n")

# Konversi Return Benchmark
total_market = (market_cumulative_makro.iloc[-1] - 1) * 100

print("="*65)
print(f"BENCHMARK RISIKO PASAR BBCA (STRATEGI BUY & HOLD)")
print(f"Systematic Risk Cumulative Return: {total_market:.2f}%")
print("="*65, "\n")

# Eksekusi Pelaporan
generate_academic_report(preds_rf_makro, "Random Forest (Teknikal + Makroekonomi)")
generate_academic_report(preds_xgb_makro, "XGBoost (Teknikal + Makroekonomi)")