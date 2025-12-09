# [CELL 4 - REVISI TERAKHIR] Otomasi Pembuatan app.py (Input Integer Age)
print("\n--- [START] CELL 4: Creating app.py (Final Fix Input Age) ---")
import os
# Path ke direktori script Streamlit lo
BASE_DIR = os.path.dirname(os.path.abspath("C:\Users\ACER\Downloads\Project_datnal_keseluruhan\FransMarcoHaryanto_Individu_Datnal_IS388_C_00000108659")) 
# Gabungin dengan nama file model
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'model.h5')

# Contoh Load
# model = load_model(MODEL_PATH)
app_py_content_final_fix = """
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model dan Fitur ---
try:
    model = joblib.load('best_dt_model.joblib')
    feature_cols = joblib.load('model_features.joblib')
except FileNotFoundError:
    st.error("Error: File model atau fitur tidak ditemukan. Pastikan sudah menjalankan semua cell di Jupyter Notebook.")
    st.stop()

# --- 2. Fungsi Prediksi (Logic tetap sama dan sudah benar) ---
def predict_diabetes(input_data):
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Mengisi kolom numerik
    # Pastikan age yang di-input ke model adalah integer
    input_df['age'] = int(input_data['age']) 
    input_df['hypertension'] = input_data['hypertension']
    input_df['heart_disease'] = input_data['heart_disease']
    input_df['bmi'] = input_data['bmi']
    input_df['HbA1c_level'] = input_data['HbA1c_level']
    input_df['blood_glucose_level'] = input_data['blood_glucose_level']
    
    # Mengisi kolom One-Hot Encoding
    if input_data['gender'] == 'Female':
        input_df['gender_Female'] = 1
    elif input_data['gender'] == 'Male':
        input_df['gender_Male'] = 1
    elif input_data['gender'] == 'Other':
        input_df['gender_Other'] = 1
        
    smoking_map = {
        'Tidak Pernah': 'smoking_history_never',
        'Saat Ini': 'smoking_history_current',
        'Dahulu (Former)': 'smoking_history_former',
        'Pernah': 'smoking_history_ever',
        'Tidak Rutin': 'smoking_history_not current',
        'Tidak Diketahui': 'smoking_history_No Info'
    }
    col_name = smoking_map.get(input_data['smoking_history'])
    if col_name:
        input_df[col_name] = 1

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- 3. Tampilan Streamlit (FINAL FIX INPUT AGE) ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

st.title("👨‍🔬 Aplikasi Prediksi Diabetes (Decision Tree)")
st.markdown("---")

st.sidebar.header("Input Data Pasien")

with st.sidebar.form("input_form"):
    # Input Kategori (TETAP)
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
    
    # UMUR (SUDAH DIPAKSA INTEGER: format="%d")
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30, step=1, format="%d") 
    
    # BMI (Desimal 2 Angka)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.01, format="%.2f")
    
    # HbA1c Level (Desimal 2 Angka)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.01, format="%.2f")
    
    # Blood Glucose Level (INTEGER)
    blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=140, step=1, format="%d")
    
    # Tombol Submit
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
    
    # Lakukan Prediksi
    result, proba = predict_diabetes(input_data)
    
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
"""

# Tulis konten app.py yang sudah direvisi ke file
try:
    with open('app.py', 'w') as f:
        f.write(app_py_content_final_fix)
    print("🚀 File 'app.py' sudah DI-FIX! Input Usia sekarang **PASTI** bilangan bulat.")
except Exception as e:
    print(f"❌ Gagal membuat file app.py: {e}")

