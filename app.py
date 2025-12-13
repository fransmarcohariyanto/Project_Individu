import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model dan Fitur ---
@st.cache_resource
def load_resources():
    try:
        # PANGGIL NAMA FILE RANDOM FOREST YANG BARU DAN FIX!
        model = joblib.load('random_forest_fix.joblib')
        # feature_cols (list kolom OHE)
        feature_cols = joblib.load('model_features.joblib') 
        return model, feature_cols
    except FileNotFoundError:
        st.error("❌ ERROR GEDE: Pastikan 4 file (app.py, random_forest_fix.joblib, model_features.joblib, requirements.txt) sudah di-push ke GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ GAGAL LOAD MODEL! Pastikan library di requirements.txt sudah benar. Detail: {e}")
        st.stop()

model, feature_cols = load_resources()


# --- 2. Fungsi Prediksi (MENGGUNAKAN REINDEX MUTLAK) ---
def predict_diabetes(input_data, model, feature_cols):
    
    # Kumpulkan semua data input (numerical dan OHE) ke dictionary
    data_dict = {
        # Input Numerik
        'age': [int(input_data['age'])],
        'hypertension': [int(input_data['hypertension'])],
        'heart_disease': [int(input_data['heart_disease'])],
        'bmi': [float(input_data['bmi'])],
        'HbA1c_level': [float(input_data['HbA1c_level'])],
        'blood_glucose_level': [int(input_data['blood_glucose_level'])]
    }

    # Tambahkan kolom OHE
    
    # Gender
    gender_map = {'Female': 'gender_Female', 'Male': 'gender_Male', 'Other': 'gender_Other'}
    col_gender = gender_map.get(input_data['gender'])
    if col_gender:
        data_dict[col_gender] = [1] 

    # Smoking History
    smoking_map = {
        'Tidak Pernah': 'smoking_history_never', 'Saat Ini': 'smoking_history_current', 
        'Dahulu (Former)': 'smoking_history_former', 'Pernah': 'smoking_history_ever', 
        'Tidak Rutin': 'smoking_history_not current', 'Tidak Diketahui': 'smoking_history_No Info'
    }
    col_smoking = smoking_map.get(input_data['smoking_history'])
    if col_smoking:
        data_dict[col_smoking] = [1]
        
    # Buat DataFrame dari input yang ada
    temp_df = pd.DataFrame.from_dict(data_dict)

    # REINDEX MUTLAK: KUNCI FIX-NYA. Memastikan urutan kolom input SAMA PERSIS dengan urutan kolom training
    input_df = temp_df.reindex(columns=feature_cols, fill_value=0)
    
    # PREDIKSI
    prediction = model.predict(input_df) 
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- 3. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("👨‍🔬 MACHINE PREDIKSI DIABETES RANDOM FOREST X DECISION TREE")
st.markdown("---")

st.sidebar.header("Input Data Pasien")

with st.sidebar.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
    smoking_history = st.selectbox("Riwayat Merokok", [
        'Tidak Pernah', 'Saat Ini', 'Dahulu (Former)',
        'Pernah', 'Tidak Rutin', 'Tidak Diketahui'
    ])
    hypertension = st.selectbox("Riwayat Hipertensi", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')

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
        'gender': gender, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 
        'smoking_history': smoking_history, 'bmi': bmi, 'HbA1c_level': hba1c, 'blood_glucose_level': blood_glucose
    }
    
    try:
        result, proba = predict_diabetes(input_data, model, feature_cols) 
    except ValueError as ve:
        st.error(f"❌ ERROR PREDIKSI: Ada masalah fatal pada kolom input. Pastikan file 'model_features.joblib' sesuai dengan model.")
        st.stop()
    
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

