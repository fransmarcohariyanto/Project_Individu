import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Streamlit & Load Model/Fitur ---
# TEMA PROFESIONAL: Tetap wide, tapi judul lebih serius
st.set_page_config(
    page_title="Sistem Prediksi Risiko Diabetes (Model Random Forest)", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🔬" 
) 

# Hapus CSS yang kurang perlu, biarkan Streamlit default bekerja.
st.markdown(
    """
    <style>
    /* Styling minimalis untuk tampilan yang lebih bersih */
    .stApp {
        background-color: #f0f2f6; /* Tetap pakai warna terang default untuk kesan formal */
    }
    .stButton>button {
        border-radius: 8px; /* Lebih kecil, lebih formal */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

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


# --- 2. Fungsi Prediksi (Gak Berubah) ---
def predict_diabetes(input_data, model, feature_cols):
    
    # Kumpulkan semua data input (numerical dan OHE) ke dictionary
    data_dict = {
        'age': [int(input_data['age'])],
        'hypertension': [int(input_data['hypertension'])],
        'heart_disease': [int(input_data['heart_disease'])],
        'bmi': [float(input_data['bmi'])],
        'HbA1c_level': [float(input_data['HbA1c_level'])],
        'blood_glucose_level': [int(input_data['blood_glucose_level'])]
    }

    # Tambahkan kolom OHE
    gender_map = {'Female': 'gender_Female', 'Male': 'gender_Male', 'Other': 'gender_Other'}
    col_gender = gender_map.get(input_data['gender'])
    if col_gender:
        data_dict[col_gender] = [1] 

    smoking_map = {
        'Tidak Pernah': 'smoking_history_never', 'Saat Ini': 'smoking_history_current', 
        'Dahulu (Former)': 'smoking_history_former', 'Pernah': 'smoking_history_ever', 
        'Tidak Rutin': 'smoking_history_not current', 'Tidak Diketahui': 'smoking_history_No Info'
    }
    col_smoking = smoking_map.get(input_data['smoking_history'])
    if col_smoking:
        data_dict[col_smoking] = [1]
        
    temp_df = pd.DataFrame.from_dict(data_dict)
    input_df = temp_df.reindex(columns=feature_cols, fill_value=0)
    
    prediction = model.predict(input_df) 
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- 3. Tampilan Streamlit (Revisi Total Profesional) ---

st.title("Sistem Prediksi Risiko Diabetes (Random Forest Classifier)")
st.markdown("Aplikasi ini menggunakan Model Machine Learning Random Forest untuk mengestimasi risiko diabetes berdasarkan parameter pasien.")
st.markdown("---")

# Gunakan tabs untuk memisahkan Input dan Hasil
tab1, tab2 = st.tabs(["📝 Input Data Pasien", "📊 Hasil Prediksi & Analisis"])

with tab1:
    
    st.subheader("Parameter Input Pasien")
    
    # Gunakan st.form untuk mengelompokkan input
    with st.form("input_form_profesional"):
        
        # Section 1: Data Dasar dan Riwayat
        st.markdown("#### 1. Data Demografi & Riwayat Medis")
        col_A1, col_A2, col_A3 = st.columns(3)
        
        with col_A1:
            gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
            hypertension = st.selectbox("Riwayat Hipertensi", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')
        
        with col_A2:
            age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30, step=1, format="%d")
            heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')
            
        with col_A3:
            smoking_history = st.selectbox("Riwayat Merokok", [
                'Tidak Pernah', 'Saat Ini', 'Dahulu (Former)',
                'Pernah', 'Tidak Rutin', 'Tidak Diketahui'
            ], help="Riwayat kebiasaan merokok pasien.")

        st.markdown("---")
        
        # Section 2: Data Biometrik dan Laboratorium
        st.markdown("#### 2. Data Biometrik dan Laboratorium")
        col_B1, col_B2, col_B3 = st.columns(3)
        
        with col_B1:
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.01, format="%.2f")
        with col_B2:
            hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.01, format="%.2f")
        with col_B3:
            blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=140, step=1, format="%d")
        
        st.markdown("---")
        submitted = st.form_submit_button("Lakukan Prediksi")


if submitted:
    input_data = {
        'gender': gender, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 
        'smoking_history': smoking_history, 'bmi': bmi, 'HbA1c_level': hba1c, 'blood_glucose_level': blood_glucose
    }
    
    try:
        result, proba = predict_diabetes(input_data, model, feature_cols) 
    except ValueError as ve:
        st.error(f"❌ ERROR PREDIKSI: Terjadi kesalahan pada proses input data. Mohon cek file model_features.joblib.")
        st.stop()
    
    # Pindah hasil ke tab2
    with tab2:
        st.subheader("Ringkasan Hasil Estimasi Risiko")
        st.markdown("---")
        
        # Hasil Utama (Gunakan st.container dan st.info/error yang lebih formal)
        if result == 1:
            with st.container(border=True):
                st.error("### KESIMPULAN: PASIEN DIPREDIKSI MENGIDAP DIABETES")
                st.markdown("**Rekomendasi:** Berdasarkan model, risiko diabetes tinggi. Disarankan konsultasi medis lebih lanjut.")
        else:
            with st.container(border=True):
                st.success("### KESIMPULAN: PASIEN DIPREDIKSI TIDAK MENGIDAP DIABETES")
                st.markdown("**Rekomendasi:** Risiko diabetes rendah. Tetap pertahankan pola hidup sehat.")
            
        st.markdown("---")
        st.subheader("Detail Probabilitas Model")
        
        col_prob_A, col_prob_B = st.columns(2)
        
        # Pake metric yang bersih
        col_prob_A.metric("Probabilitas (Kelas 0 - Tidak Diabetes)", 
                         f"{proba[0]*100:.2f}%", help="Tingkat keyakinan model untuk Kelas 0.")
        
        col_prob_B.metric("Probabilitas (Kelas 1 - Diabetes)", 
                         f"{proba[1]*100:.2f}%", help="Tingkat keyakinan model untuk Kelas 1.")
        
        # Tampilkan data input yang digunakan
        st.markdown("---")
        st.markdown("#### Parameter Input yang Digunakan:")
        input_display = pd.DataFrame([input_data]).T.rename(columns={0: 'Nilai Input'})
        st.dataframe(input_display, use_container_width=True)
        
        st.caption("Disclaimer: Hasil prediksi ini adalah estimasi statistik Machine Learning dan bukan merupakan diagnosis klinis. Pengambilan keputusan medis harus didasarkan pada pemeriksaan dokter profesional.")
