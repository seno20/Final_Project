import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="XGBoost Rainfall Forecast",
    page_icon="🌧️",
    layout="wide"
)

# Konstanta Kalkulasi Air
ETP_MONTHLY = 120
SWR_MAX = 200

def preprocess_data(df):
    """Fungsi untuk membersihkan data sesuai notebook asli"""
    try:
        # Ubah Tanggal
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Tanggal'])
        
        # Ubah semua kolom numerik (hapus koma ganti titik)
        for col in df.columns.drop('Tanggal'):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )
            
        # Isi missing values dengan mean
        for col in df.columns:
            if df[col].isnull().any() and df[col].dtype == 'float64':
                df[col] = df[col].fillna(df[col].mean())
                
        # Hitung Total Curah Hujan
        rainfall_cols = [col for col in df.columns if str(col).startswith('AF')]
        if not rainfall_cols:
            st.error("Kolom yang diawali 'AF' tidak ditemukan untuk menghitung curah hujan.")
            return None
            
        df['Total_Curah_Hujan'] = df[rainfall_cols].sum(axis=1)
        
        # Agregasi Bulanan
        df_monthly = df.set_index('Tanggal')['Total_Curah_Hujan'].resample('ME').sum()
        
        return pd.DataFrame({'Monthly_Rainfall': df_monthly})
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

def create_features(df_monthly):
    """Membuat fitur lag dan rolling untuk ML"""
    df_ml = df_monthly.copy()
    df_ml['lag_1'] = df_ml['Monthly_Rainfall'].shift(1)
    df_ml['lag_2'] = df_ml['Monthly_Rainfall'].shift(2)
    df_ml['lag_3'] = df_ml['Monthly_Rainfall'].shift(3)
    df_ml['lag_6'] = df_ml['Monthly_Rainfall'].shift(6)
    df_ml['lag_12'] = df_ml['Monthly_Rainfall'].shift(12)
    df_ml['month'] = df_ml.index.month
    df_ml['quarter'] = df_ml.index.quarter
    df_ml['rolling_3'] = df_ml['Monthly_Rainfall'].rolling(3).mean()
    df_ml['rolling_6'] = df_ml['Monthly_Rainfall'].rolling(6).mean()
    
    return df_ml.dropna()

def iterative_forecast(model, combined_df, feature_cols, steps=12):
    """Fungsi untuk memprediksi N bulan ke depan secara iteratif"""
    last_known_date = combined_df.index.max()
    future_dates = pd.date_range(start=last_known_date + pd.DateOffset(months=1), periods=steps, freq='ME')
    
    forecast_values = []
    
    for current_date in future_dates:
        temp_features = pd.DataFrame(index=[current_date])
        temp_features['month'] = current_date.month
        temp_features['quarter'] = current_date.quarter
        
        # Ekstrak Lags
        for lag in [1, 2, 3, 6, 12]:
            lag_date = current_date - pd.DateOffset(months=lag)
            if lag_date in combined_df.index:
                temp_features[f'lag_{lag}'] = combined_df.loc[lag_date, 'Monthly_Rainfall']
            else:
                temp_features[f'lag_{lag}'] = combined_df['Monthly_Rainfall'].mean() # fallback
                
        # Ekstrak Rolling
        temp_features['rolling_3'] = combined_df['Monthly_Rainfall'].iloc[-3:].mean()
        temp_features['rolling_6'] = combined_df['Monthly_Rainfall'].iloc[-6:].mean()
        
        # Urutkan fitur sesuai training
        X_predict = temp_features[feature_cols]
        
        # Prediksi
        pred = model.predict(X_predict)[0]
        forecast_values.append(pred)
        
        # Tambahkan ke combined_df agar bisa dipakai untuk lag bulan berikutnya
        new_row = pd.DataFrame({'Monthly_Rainfall': pred}, index=[current_date])
        combined_df = pd.concat([combined_df, new_row])
        
    return pd.DataFrame({'Forecasted_Rainfall_mm': forecast_values}, index=future_dates)

def calculate_water_status(forecast_df, initial_swr=200):
    """Menghitung SWR, WD, dan Status Air"""
    swr_vals = []
    wd_vals = []
    status_vals = []
    
    curr_swr = initial_swr
    
    for _, row in forecast_df.iterrows():
        ch = row['Forecasted_Rainfall_mm']
        
        # Hitung SWR
        pot_swr = curr_swr + ch - ETP_MONTHLY
        curr_swr = max(0, min(SWR_MAX, pot_swr))
        swr_vals.append(curr_swr)
        
        # Hitung WD
        wd = max(0, ETP_MONTHLY - (ch + curr_swr))
        wd_vals.append(wd)
        
        # Status
        if wd > 0:
            status_vals.append('Deficit')
        else:
            status_vals.append('Surplus')
            
    forecast_df['SWR_forecast'] = swr_vals
    forecast_df['WD_forecast'] = wd_vals
    forecast_df['Water_Status'] = status_vals
    
    return forecast_df


# ==========================================
# UI APLIKASI
# ==========================================
st.title("🌧️ Prediksi Curah Hujan Bulanan & Status Air")
st.markdown("""
Aplikasi ini memprediksi **Total Curah Hujan Bulanan** menggunakan model **XGBoost Regressor** dan secara otomatis menghitung *Soil Water Reserve (SWR)* serta *Water Deficit (WD)* untuk menentukan status air di masa depan.
""")

st.sidebar.header("📁 Upload Data Historis")
uploaded_file = st.sidebar.file_uploader("Upload file rainfall.csv", type=["csv"])

if uploaded_file is not None:
    # Membaca data
    raw_df = pd.read_csv(uploaded_file, sep=';')
    
    with st.spinner("Memproses data dan mengekstrak fitur..."):
        df_monthly = preprocess_data(raw_df)
        
    if df_monthly is not None:
        df_ml = create_features(df_monthly)
        
        st.subheader("📊 Pratinjau Data Historis (Bulan Terakhir)")
        st.dataframe(df_ml.tail(3), use_container_width=True)
        
        # Memisahkan X dan Y
        X = df_ml.drop(columns=['Monthly_Rainfall'])
        y = df_ml['Monthly_Rainfall']
        feature_cols = X.columns.tolist()
        
        # Training Model
        with st.spinner("Melatih model XGBoost..."):
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.01,
                objective='reg:squarederror',
                random_state=42
            )
            model.fit(X, y)
            st.toast("Model berhasil dilatih!", icon="✅")
            
        # Pilihan Prediksi
        st.divider()
        st.subheader("🔮 Pengaturan Peramalan (Forecasting)")
        forecast_steps = st.slider("Berapa bulan ke depan yang ingin diprediksi?", min_value=1, max_value=24, value=7)
        
        if st.button("Jalankan Prediksi", type="primary"):
            with st.spinner("Memprediksi curah hujan dan menghitung status air..."):
                # Melakukan prediksi iteratif
                forecast_df = iterative_forecast(model, df_monthly.copy(), feature_cols, steps=forecast_steps)
                
                # Menghitung status air
                final_df = calculate_water_status(forecast_df, initial_swr=SWR_MAX)
                
                # Menampilkan Hasil
                st.subheader("📈 Hasil Prediksi")
                
                # Metrik Singkat
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata Prediksi CH", f"{final_df['Forecasted_Rainfall_mm'].mean():.2f} mm")
                col2.metric("Total Bulan Defisit", len(final_df[final_df['Water_Status'] == 'Deficit']))
                col3.metric("Total Bulan Surplus", len(final_df[final_df['Water_Status'] == 'Surplus']))
                
                # Plotly Native Streamlit
                st.write("**Grafik Prediksi Curah Hujan Bulanan**")
                st.line_chart(final_df['Forecasted_Rainfall_mm'])
                
                # Tabel Hasil
                st.write("**Detail Data Prediksi & Kalkulasi Air**")
                # Format index agar hanya menampilkan Bulan-Tahun
                final_df.index = final_df.index.strftime('%Y-%m')
                st.dataframe(
                    final_df.style.highlight_max(axis=0, subset=['Forecasted_Rainfall_mm'])
                                  .map(lambda x: 'background-color: #ffcccc' if x == 'Deficit' else 'background-color: #e6ffe6', subset=['Water_Status']),
                    use_container_width=True
                )
else:
    st.info("👋 Silakan unggah file CSV di sidebar sebelah kiri untuk memulai prediksi.")
    st.markdown("""
    **Format data yang diharapkan:**
    - Pemisah kolom menggunakan titik koma (`;`).
    - Memiliki kolom `Tanggal` dengan format `DD/MM/YYYY`.
    - Memiliki kolom-kolom stasiun pengamat yang diawali huruf `AF` (misal: `AF01`, `AF02`, dst).
    """)