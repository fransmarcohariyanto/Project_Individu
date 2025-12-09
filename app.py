# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# DEBUGGING - UNTUK MENGHILANGKAN KERAGUAN FILE
st.subheader("--- DEBUGGING FILES CHECK ---")
try:
    if 'best_dt_model.joblib' in os.listdir('.'):
        st.success("✅ File model TERLIHAT di server.")
    else:
        st.error("❌ File model TIDAK TERLIHAT di server.")
    st.markdown("---")
except:
    pass
# END DEBUGGING

# --- 1. Load Model dan Fitur (Pake Caching Streamlit) ---
@st.cache_resource
def load_resources():
    try:
        # Pengecekan file dijamin BISA karena kita sudah konfirmasi file ada
        model = joblib.load('best_dt_model.joblib')
        feature_cols = joblib.load('model_features.joblib')
        return model, feature_cols
    except FileNotFoundError:
        st.error("❌ ERROR: File model tidak ditemukan di server. Cek kembali Git push.")
        st.stop()
    except Exception as e:
        # CATCHING ERROR LOADING MODEL (Misal versi scikit-learn beda)
        st.error(f"⚠️ GAGAL LOAD MODEL! Cek requirements.txt. Detail: {e}")
        st.stop()

# Panggil fungsi caching
model, feature_cols = load_resources()


# --- 2. Fungsi Prediksi (Logic sudah diperkuat untuk mencegah ValueError) ---
def predict_diabetes(input_data, model, feature_cols):
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # **PASTIKAN TIPE DATA KONSISTEN**
    input_df['age'] = int(input_data['age'])
    input_df['hypertension'] = int(input_data['hypertension'])
    input_df['heart_disease'] = int(input_data['heart_disease'])
    input_df['bmi'] = float(input_data['bmi'])
    input_df['HbA1c_level'] = float(input_data['HbA1c_level'])
    input_df['blood_glucose_level'] = int(input_data['blood_glucose_level'])
    
    # Mapping One-Hot Encoding
    
    # Gender
    gender_map = {'Female': 'gender_Female', 'Male': 'gender_Male', 'Other': 'gender_Other'}
    col_gender = gender_map.get(input_data['gender'])
    if col_gender in input_df.columns:
        input_df[col_gender] = 1

    # Smoking History
    # NOTE: Nama kolom harus sesuai dengan hasil get_feature_names_out!
    smoking_map = {
        'Tidak Pernah': 'smoking_history_never',
        'Saat Ini': 'smoking_history_current',
        'Dahulu (Former)': 'smoking_history_former',
        'Pernah': 'smoking_history_ever',
        'Tidak Rutin': 'smoking_history_not current',
        'Tidak Diketahui': 'smoking_history_No Info' 
    }
    col_name = smoking_map.get(input_data['smoking_history'])
    if col_name in input_df.columns:
        input_df[col_name] = 1
    
    # Prediksi menggunakan Pipeline
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
    
    # Panggil fungsi prediksi
    result, proba = predict_diabetes(input_data, model, feature_cols)
    
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
