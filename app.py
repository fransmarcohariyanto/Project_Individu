# app.py - KODE FINAL DAN CLEAN (Solusi Anti-ValueError ColumnTransformer)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Load Model dan Fitur ---
# Menggunakan @st.cache_resource untuk menghindari loading berulang
@st.cache_resource
def load_resources():
    try:
        # Load FULL PIPELINE
        model = joblib.load('best_dt_model.joblib')
        
        # Load daftar 8 kolom input mentah (dari file input_columns.joblib)
        # File ini dibuat di model_creation.py (harus di-push!)
        input_cols = joblib.load('input_columns.joblib') 
        
        return model, input_cols
    except FileNotFoundError:
        st.error("❌ ERROR GEDE: File model (.joblib) atau input_columns (.joblib) tidak ditemukan di server. Cek kembali Git push.")
        st.stop()
    except Exception as e:
        # Menangkap error loading model (biasanya mismatch versi scikit-learn)
        st.error(f"⚠️ GAGAL LOAD MODEL! Cek versi scikit-learn di requirements.txt. Detail: {e}")
        st.stop()

# Panggil fungsi caching
model, input_cols = load_resources()


# --- 2. Fungsi Prediksi (Menerima data mentah untuk Pipeline) ---
# input_cols digunakan untuk memastikan urutan kolom DataFrame input
def predict_diabetes(input_data, model, input_cols):
    
    # 1. BUAT DATAFRAME HANYA DARI KOLOM MENTAH (8 KOLOM ASLI)
    # Semua data di-wrapping dalam list untuk membuat satu row
    data_dict = {
        # URUTAN KEY DI SINI TIDAK TERLALU PENTING, asalkan NAMA KEY SAMA
        'gender': [input_data['gender']],
        'age': [input_data['age']],
        'hypertension': [input_data['hypertension']],
        'heart_disease': [input_data['heart_disease']],
        'smoking_history': [input_data['smoking_history']],
        'bmi': [input_data['bmi']],
        'HbA1c_level': [input_data['HbA1c_level']],
        'blood_glucose_level': [input_data['blood_glucose_level']],
    }
    
    # KUNCI UTAMA: Membuat DataFrame dengan urutan kolom yang BENAR
    # Urutan diambil dari input_cols (yang dibuat saat training)
    input_df = pd.DataFrame(data_dict, columns=input_cols) 
    
    # 2. MODEL.PREDICT: Pipeline menjalankan ColumnTransformer dengan input mentah
    prediction = model.predict(input_df) 
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- 3. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("👨‍🔬 Aplikasi Prediksi Diabetes (Decision Tree)")
st.markdown("---")

st.sidebar.header("Input Data Pasien")

with st.sidebar.form("input_form"):
    # Input Kategori
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
    smoking_history = st.selectbox("Riwayat Merokok", [
        'Tidak Pernah', 'Saat Ini', 'Dahulu (Former)',
        'Pernah', 'Tidak Rutin', 'Tidak Diketahui'
    ])
    hypertension = st.selectbox("Riwayat Hipertensi", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')

    # Input Numerik
    st.markdown("---")
    st.markdown("**Data Biometrik & Laboratorium**")
    
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30, step=1, format="%d")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.01, format="%.2f")
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.01, format="%.2f")
    blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=140, step=1, format="%d")
    
    st.markdown("---")
    submitted = st.form_submit_button("Prediksi Sekarang!")

# Pemanggilan fungsi (sekitar Line 124 lo)
if submitted:
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': blood_glucose
    }
    
    # Memanggil dengan variabel input_cols yang sudah diperbaiki
    result, proba = predict_diabetes(input_data, model, input_cols) 
    
    st.subheader("Hasil Prediksi")
    
    if result == 1:
        st.error(f"⚠️ Pasien DIPREDIKSI **MENGIDAP DIABETES**")
    else:
        st.success(f"✅ Pasien DIPREDIKSI **TIDAK MENGIDAP DIABETES**")
        
    st.markdown(f"**Tingkat Keyakinan Model:**")
    col1, col2 = st.columns(2)
    
    col1.metric("Probabilitas Tidak Diabetes", f"{proba[0]*100:.2f}%")
    col2.metric("Probabilitas Diabetes", f"{proba[1]*100:.2f}%")

    st.caption("Disclaimer: Hasil ini hanya prediksi Machine Learning, bukan diagnosis medis.")
