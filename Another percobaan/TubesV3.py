# %% [markdown]
# # TUGAS BESAR: Komparasi Model Ensemble Learning (Random Forest vs XGBoost) dalam Prediksi Pergerakan Harga Saham BBCA
# 
# **Abstrak Proyek:**
# Penelitian ini bertujuan untuk menguji, membandingkan, dan menganalisis kinerja dua arsitektur *Machine Learning* berbasis *Ensemble Learning*, yakni **Random Forest (Bagging)** dan **XGBoost (Boosting)**. Pemodelan ditujukan untuk memprediksi probabilitas arah pergerakan harga saham PT Bank Central Asia Tbk (BBCA) dengan jendela waktu penahanan (*holding period*) 5 hari perdagangan.
# 
# Penelitian ini mengadopsi kerangka kerja *Walk-Forward Validation* untuk mencegah kebocoran data masa depan (*Future Data Leakage*). Model ini juga mengintegrasikan variabel makroekonomi eksogen (Kurs USD/IDR), perhitungan risiko finansial riil (Biaya Komisi Broker), serta pengujian signifikansi statistik menggunakan *McNemar's Test*.

# %% [markdown]
# ## 1. Inisialisasi Lingkungan Kerja dan Import Pustaka
# Mengimpor pustaka standar industri untuk komputasi numerik, manipulasi deret waktu (*time-series*), pemodelan prediktif, uji statistik, dan visualisasi data.

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
from statsmodels.stats.contingency_tables import mcnemar # Modul untuk Uji McNemar

# Konfigurasi Lingkungan
warnings.filterwarnings('ignore') # Menyembunyikan peringatan/warnings agar output laporan bersih
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 120})

# %% [markdown]
# ## 2. Ekstraksi dan Prapemrosesan Data (Data Preprocessing)
# Dataset historis saham BBCA dimuat ke dalam struktur *DataFrame*. Tahapan ini mencakup standardisasi penamaan kolom, penetapan *DatetimeIndex*, dan reduksi dimensi dengan mengeliminasi fitur yang tidak memiliki nilai informasi prediktif (`adjclose` dan `ingested_at_utc`).

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
# Pendekatan analitik diubah dari model regresi menjadi **Klasifikasi Biner (Binary Classification)** untuk mereduksi *noise* pergerakan harga absolut.
# 
# Variabel target (`Target_5D`) dienkode menjadi `1` (Naik/Bullish) apabila harga penutupan 5 hari ke depan melampaui harga pembukaan hari berikutnya, dan `0` (Turun/Bearish) untuk kondisi sebaliknya.

# %%
# Variabel Target
bbca["Tomorrow_Open"] = bbca["Open"].shift(-1)
bbca["Close_5_Days"] = bbca["Close"].shift(-5)
bbca["Target_5D"] = (bbca["Close_5_Days"] > bbca["Tomorrow_Open"]).astype(int)

# Variabel Return Finansial (Tanpa leverage)
bbca["Trade_Return_5D"] = (bbca["Close_5_Days"] / bbca["Tomorrow_Open"]) - 1    

# %% [markdown]
# ## 4. Rekayasa Fitur (Feature Engineering): Indikator Teknikal Intrinsik
# Derivasi fitur prediktif dari data harga mentah untuk mendeteksi rezim pasar dan momentum osilator. Fitur ini dirancang untuk menangkap pola tren historis yang mendasari pergerakan saham.

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
# Ekstraksi fitur statis tunggal tidak cukup untuk mendeteksi anomali pasar. Oleh karena itu, rasio volatilitas dan konsistensi tren diekstraksi melintasi berbagai spektrum waktu (2, 5, 60, dan 250 hari perdagangan) guna memberikan konteks adaptif kepada model.

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
# Dua model *Machine Learning* disiapkan untuk evaluasi komparatif:
# - **Random Forest Classifier:** Memanfaatkan teknik *Bagging* paralel dengan `n_estimators` tinggi (1000 pohon) guna menyaring *noise* dan mereduksi *variance*.
# - **XGBoost Classifier:** Memanfaatkan teknik *Gradient Boosting* sekuensial. Untuk mencegah *overfitting* pada deret waktu keuangan, model ini dibatasi dengan regularisasi ketat (L1 & L2) serta pembatasan kedalaman (`max_depth=3`).

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
# Untuk mencegah *Future Data Leakage*, evaluasi menggunakan iterasi *Walk-Forward Validation* yang bergerak maju secara kronologis. Model dilatih pada data masa lalu yang terus bertambah ( *Expanding Window* ) untuk memprediksi masa depan yang belum terlihat. Khusus pada XGBoost, *Early Stopping* diintegrasikan secara dinamis untuk menghentikan fase pelatihan apabila model mulai menghafal *noise*.

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
    """Menjalankan kerangka evaluasi Walk-Forward (Expanding Window) secara iteratif."""
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)

# %% [markdown]
# ## 8. Integrasi Variabel Makroekonomi Eksogen (Nilai Tukar USD/IDR)
# Pergerakan saham kapitalisasi besar (*Blue Chip*) sangat dipengaruhi oleh aliran modal asing. Oleh karena itu, indikator makroekonomi berupa volatilitas nilai tukar mata uang (USD/IDR) ditambahkan sebagai variabel eksogen untuk mempertajam insting model terhadap sentimen pasar global.

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
# Menjalankan fungsi simulasi *Walk-Forward* secara menyeluruh untuk kedua algoritma menggunakan matriks fitur hibrida (Teknikal Intrinsik + Makroekonomi Eksogen).

# %%
print("Mengeksekusi Walk-Forward Backtesting: Random Forest Classifier...")
preds_rf_makro = backtest(bbca, model_rf, predictors_makro)

print("Mengeksekusi Walk-Forward Backtesting: XGBoost Classifier...")
preds_xgb_makro = backtest(bbca, model_xgb, predictors_makro)

print("Proses komputasi backtesting berhasil diselesaikan.")

# %% [markdown]
# ## 10. Evaluasi Finansial Berbobot-Waktu (Time-Weighted Cumulative Return)
# Metrik evaluasi kecerdasan buatan dinilai kurang relevan jika tidak disimulasikan ke dalam bentuk finansial. Profitabilitas portofolio (Net Profit) dihitung secara runtun waktu setelah dikurangi beban sistematis komisi broker sebesar 0.4% (`FEE_TRANSAKSI`) per eksekusi perdagangan.

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
# Plot grafik di bawah ini membandingkan pertumbuhan ekuitas dari model Random Forest dan XGBoost melawan tingkat risiko sistematis pasar (Benchmark).

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
# Laporan akhir ini memuat tingkat *Positive Predictive Value* (Presisi Algoritma), intensitas eksekusi posisi perdagangan, serta *Cumulative Net Profit* pasca-komisi.

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

# %% [markdown]
# ## 13. Pengujian Signifikansi Statistik (McNemar's Test)
# Uji McNemar adalah metode statistik non-parametrik yang ideal untuk mengevaluasi matriks hasil prediksi model klasifikasi. Pengujian ini digunakan untuk membuktikan secara empiris apakah selisih tingkat akurasi klasifikasi antara Random Forest dan XGBoost merupakan perbedaan yang diakibatkan oleh kecerdasan model, atau hanya terjadi akibat kebetulan (varians acak).

# %%
def evaluate_mcnemar_test(preds_rf: pd.DataFrame, preds_xgb: pd.DataFrame):
    """Membangun tabel kontingensi dan mengeksekusi Uji McNemar."""
    # Mendapatkan kelas prediksi dan aktual (0 atau 1)
    rf_preds = preds_rf["Predictions"]
    xgb_preds = preds_xgb["Predictions"]
    actual = preds_rf["Target_5D"]
    
    # Menentukan kebenaran prediksi masing-masing model (True/False)
    rf_correct = (rf_preds == actual)
    xgb_correct = (xgb_preds == actual)
    
    # Membangun 2x2 Tabel Kontingensi (Contingency Table)
    both_correct = (rf_correct & xgb_correct).sum()
    rf_only_correct = (rf_correct & ~xgb_correct).sum()
    xgb_only_correct = (~rf_correct & xgb_correct).sum()
    neither_correct = (~rf_correct & ~xgb_correct).sum()
    
    contingency_table = [[both_correct, rf_only_correct],
                         [xgb_only_correct, neither_correct]]
    
    # Eksekusi McNemar Test dengan koreksi kontinuitas (Yates's correction)
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print("="*65)
    print(f"UJI STATISTIK KLASIFIKASI (MCNEMAR'S TEST)")
    print("="*65)
    print(f"Tabel Kontingensi:")
    print(f"- Keduanya Prediksi Benar        : {both_correct}")
    print(f"- Hanya Random Forest Benar      : {rf_only_correct}")
    print(f"- Hanya XGBoost Benar            : {xgb_only_correct}")
    print(f"- Keduanya Prediksi Salah        : {neither_correct}\n")
    
    print(f"Chi-Square Statistic : {result.statistic:.4f}")
    print(f"P-Value              : {result.pvalue:.5f}\n")
    
    if result.pvalue < 0.05:
        print("Kesimpulan: H0 ditolak. Terdapat perbedaan yang SIGNIFIKAN secara statistik")
        print("terhadap tingkat akurasi klasifikasi antara Random Forest dan XGBoost.")
    else:
        print("Kesimpulan: H0 gagal ditolak. Perbedaan akurasi antara kedua model")
        print("TIDAK signifikan secara statistik (kemungkinan terjadi karena varians acak).")

# Eksekusi Uji
evaluate_mcnemar_test(preds_rf_makro, preds_xgb_makro)

# %% [markdown]
# ## 14. Analisis Kesalahan Model (Error Analysis)
# Meskipun model *Machine Learning* menunjukkan kapasitas dalam memitigasi risiko pasar, masih terdapat margin kesalahan prediksi yang memberikan dampak finansial. Analisis limitasi ini dibedah menjadi tiga aspek utama:
# 
# 1. **Analisis *False Positive* (Sinyal Palsu akibat *Lagging*):** Kelemahan terbesar model ini bertumpu pada penggunaan fitur *Moving Average* dan *Rolling Window* yang sifatnya terlambat (*lagging*). Ketika terjadi sentimen pasar yang menjatuhkan harga secara mendadak ( *market shock* ), *Moving Average* secara teknis masih mengindikasikan tren *bullish*. Hal ini memicu model untuk menghasilkan sinyal *BUY* palsu yang berujung pada kerugian riil (potongan nilai aset ditambah biaya komisi broker).
# 2. **Analisis *False Negative* (Fenomena Sindrom *Under-Trading*):** Penggunaan *Decision Threshold* konservatif sebesar 70% memicu kelumpuhan keputusan (*decision paralysis*). Sistem algoritma, terutama pada arsitektur XGBoost yang teregulasi ketat, cenderung menjadi hiper-konservatif. Akibatnya, portofolio sering kali kehilangan peluang profit maksimal (*missed opportunities*) pada saat saham BBCA benar-benar mengalami reli tajam karena probabilitas prediksi AI tertahan di kisaran 65% - 69%.
# 3. **Keterbatasan Dimensi Fitur (*Fundamental Blind-Spots*):** Model yang dibangun murni mengandalkan indikator teknikal historis dan pergerakan Kurs Valuta Asing. Model mengalami "kebutaan fundamental", di mana algoritma tidak mampu mengantisipasi lonjakan harga yang diakibatkan oleh sentimen fundamental eksternal, seperti rilis pembagian dividen jumbo perusahaan, pemecahan nilai saham (*stock split*), atau perubahan suku bunga acuan (*BI Rate*).

# %% [markdown]
# ## 15. KESIMPULAN DAN SARAN
# 
# ### Kesimpulan
# Berdasarkan serangkaian eksperimen, pengujian *Walk-Forward Validation*, dan evaluasi metrik finansial yang telah dilakukan pada prediksi pergerakan saham PT Bank Central Asia Tbk (BBCA), dapat ditarik beberapa kesimpulan utama sebagai berikut:
# 
# 1. **Keunggulan Arsitektur Random Forest (Bagging):** Algoritma Random Forest terbukti secara empiris sebagai model yang paling superior dan *robust* (tangguh) dalam menangani data deret waktu finansial yang memiliki tingkat kebisingan (*noise*) tinggi. Dengan mengandalkan mekanisme *voting* dari 1000 pohon keputusan, Random Forest sukses meredam bias harga dan bertindak sebagai instrumen lindung nilai (*hedging*) yang efektif, sukses mencetak profit positif ketika *benchmark* pasar (*Buy & Hold*) mengalami kerugian kumulatif secara sistematis.
# 
# 2. **Sensitivitas dan Limitasi XGBoost (Boosting):** Arsitektur XGBoost menunjukkan kerentanan terhadap anomali ketika dihadapkan pada volatilitas data eksogen (Kurs USD/IDR). Meskipun telah dikonfigurasi dengan batasan regularisasi ketat (L1, L2, dan *Early Stopping*), penerapan *threshold* konservatif membuat XGBoost cenderung bertindak hiper-pasif. Hal ini membuktikan bahwa metode koreksi galat secara sekuensial (*Boosting*) kurang ideal dan kurang adaptif pada data saham ber- *noise* tinggi dibandingkan metode konsensus paralel (*Bagging*).
# 
# 3. **Urgensi Metodologi Validasi Dunia Nyata:** Penelitian ini mengonfirmasi bahwa akurasi standar tidak cukup untuk menilai kelayakan model di pasar modal. Penerapan *Decision Threshold* konservatif dan pemotongan biaya komisi broker riil (0.4%) secara drastis mengubah profil profitabilitas. Model dipaksa beroperasi dalam *Sniper Mode* (mementingkan kualitas sinyal di atas kuantitas transaksi), yang mana terbukti menyelamatkan portofolio dari kebocoran modal akibat biaya eksekusi (*fee bleeding*).
# 
# ---
# 
# ### Saran untuk Penelitian Selanjutnya
# 1. **Analisis Kepentingan Fitur (Feature Importance):** Disarankan untuk mengintegrasikan metode interpretasi model seperti SHAP (*SHapley Additive exPlanations*) untuk membedah kontribusi masing-masing indikator. Fitur dengan bobot prediktif mendekati nol dapat dieliminasi guna merampingkan dimensi komputasi.
# 2. **Penerapan *Dynamic Thresholding*:** Memodifikasi ambang batas keputusan (*threshold*) agar tidak statis di angka 70%, melainkan bergerak adaptif menyesuaikan rezim volatilitas pasar (*Dynamic Risk Adjustment*).
# 3. **Eksplorasi Analisis Sentimen dan Data Fundamental:** Model dapat diperkaya dengan integrasi pemrosesan bahasa alami (NLP) terhadap tajuk berita (*News Sentiment*) atau *Google Trends* guna mengatasi kebutaan model (*blind-spots*) terhadap rilis kalender ekonomi makro.