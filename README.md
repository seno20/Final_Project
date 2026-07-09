# 🌧️ Rainfall Forecasting & Water Deficit Analysis

### Prediksi Curah Hujan Bulanan dengan SARIMA & XGBoost

---

## 📌 Tentang Proyek Ini

Notebook ini adalah pipeline **end-to-end** untuk memprediksi curah hujan bulanan serta menghitung kondisi neraca air (*Soil Water Reserve* & *Water Deficit*) yang berguna untuk mendukung keputusan di bidang pertanian, irigasi, dan mitigasi kekeringan.

Proyek ini membandingkan pendekatan **statistik klasik (SARIMA)** dengan pendekatan **machine learning modern (XGBoost)** untuk menemukan model peramalan terbaik, lalu menggunakannya untuk memproyeksikan curah hujan hingga akhir tahun **2026**.

---

## 🧭 Alur Kerja (Workflow)

### 1️⃣ Import Library & Load Dataset
- Membaca data curah hujan harian dari `rainfall.csv`
- Parsing tanggal & konversi kolom numerik (format desimal koma → titik)

### 2️⃣ Exploratory Data Analysis (EDA)
- Pembersihan kolom yang tidak relevan
- Penanganan *missing value* (imputasi rata-rata)
- Visualisasi tren curah hujan harian
- Deteksi *Dry Spell* / musim kering (hari kering berturut-turut)
- Agregasi curah hujan bulanan (total & rata-rata)

### 3️⃣ Analisis Neraca Air (SWR & Water Deficit)
- Perhitungan **Soil Water Reserve (SWR)** bulanan
- Perhitungan **Water Deficit (WD)** berdasarkan evapotranspirasi (ETP)
- Klasifikasi status air: **Deficit** vs **Surplus**

### 4️⃣ Train-Test Split
- Pembagian data time series secara kronologis (80% train / 20% test)

### 5️⃣ Pemodelan Statistik — SARIMA
- Uji stasioneritas (*Augmented Dickey-Fuller Test*)
- Analisis ACF & PACF untuk menentukan parameter model
- Perbandingan beberapa kombinasi parameter SARIMA
- Validasi dengan **AUTO_ARIMA** (pmdarima)

### 6️⃣ Pemodelan Machine Learning — XGBoost
- Feature engineering: lag features (1, 2, 3, 6, 12 bulan), rolling mean, bulan, dan kuartal
- Training model **XGBoost Regressor**
- Feature importance analysis
- Hyperparameter tuning dengan **GridSearchCV** + `TimeSeriesSplit`

### 7️⃣ Perbandingan Performa Model

| Model                  | RMSE    | MAE     | MAPE (%) |
|-------------------------|--------:|--------:|---------:|
| 🏆 **XGBoost (Tuned)**   | 1457.68 | 1228.11 | 43.57    |
| XGBoost (Untuned)       | 1741.55 | 1497.89 | 46.94    |
| SARIMA (Best Manual)    | 1874.87 | 1300.35 | 40.80    |
| AUTO_ARIMA              | 2323.77 | 1977.95 | 51.48    |

> ✅ **Model terbaik berdasarkan RMSE:** XGBoost (Tuned)

### 8️⃣ Prediksi 2026
- Forecast curah hujan bulanan hingga Desember 2026 menggunakan model XGBoost terbaik (peramalan rekursif dengan lag & rolling features)
- Perhitungan proyeksi SWR & Water Deficit untuk sisa tahun 2026

### 9️⃣ Deployment
- Prototipe aplikasi web sederhana dengan **Streamlit** untuk menyajikan hasil prediksi model XGBoost secara interaktif

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---|---|
| `pandas`, `numpy` | Manipulasi & analisis data |
| `matplotlib`, `seaborn` | Visualisasi data |
| `statsmodels` (SARIMAX, ADF) | Pemodelan time series statistik |
| `pmdarima` (auto_arima) | Pencarian parameter SARIMA otomatis |
| `scikit-learn` | Evaluasi model & GridSearchCV |
| `xgboost` | Pemodelan machine learning |
| `streamlit` | Deployment aplikasi prediksi |

---

## 📂 Struktur Data yang Dibutuhkan

```
rainfall.csv
├── Tanggal          : format dd/mm/yyyy
├── AF01, AF02, ...  : kolom curah hujan per periode/pos pengukuran
└── (kolom numerik lain dengan format desimal koma)
```

---

## 🚀 Cara Menjalankan

1. Pastikan `rainfall.csv` berada di direktori yang sama dengan notebook
2. Install dependency yang diperlukan:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels
   pip install scikit-learn pmdarima xgboost streamlit
   ```
3. Jalankan notebook secara berurutan dari sel pertama hingga terakhir
4. (Opsional) Jalankan aplikasi Streamlit untuk mode prediksi interaktif:
   ```bash
   streamlit run app.py
   ```

---

## 💡 Catatan

- Nilai `ETP_MONTHLY` (120 mm) dan `SWR_MAX_CAPACITY` (200 mm) merupakan asumsi parameter neraca air yang dapat disesuaikan dengan kondisi wilayah studi.
- Hasil evaluasi menunjukkan model berbasis machine learning (XGBoost) memberikan RMSE & MAE lebih baik dibanding pendekatan SARIMA klasik, meski SARIMA sedikit unggul pada metrik MAPE.

---

<p align="center">🌦️ <i>Selamat bereksplorasi dengan data hujan!</i> 🌦️</p>
