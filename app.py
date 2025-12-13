import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Streamlit & Load Model/Fitur ---
# SET PAGE CONFIG: Pake tema yang lebih dark/wide, dan kasih emoji kece!
st.set_page_config(
    page_title="Prediksi Diabetes Gen Z", 
    layout="wide", 
    initial_sidebar_state="expanded",
    # Kasih ikon yang relevan
    page_icon="🤖" 
) 

# Inject custom CSS (opsional, biar makin beda)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Warna latar belakang lebih terang */
    }
    .css-1d391kg, .stButton>button {
        border-radius: 12px; /* Border radius yang lebih modern */
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #FF4B4B; /* Warna buat hasil prediksi */
    }
    .succes-font {
        font-size:30px !important;
        font-weight: bold;
        color: #09E58A; /* Warna buat hasil prediksi sukses */
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


# --- 2. Fungsi Prediksi (Gak Berubah, Logic-nya udah OK) ---
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


# --- 3. Tampilan Streamlit (Ini yang diganti total!) ---

st.title("🔬 Diabetes Predictor: Edisi Gen Z Cepat Tepat! 💨")
st.caption("🚀 Powered by Random Forest Magic & Data Science. Cek kadar diabetes kamu sekarang!")
st.markdown("---")


# SIDEBAR: Pake emoji dan judul yang lebih menarik
st.sidebar.title("⚙️ Input Data Pasien Kece")
st.sidebar.caption("Isi data ini seakurat mungkin ya, *bestie*!")

with st.sidebar.form("input_form"):
    
    st.markdown("### 🧑‍⚕️ Data Umum")
    # Pake kolom di sidebar biar lebih ringkas
    col_gender, col_age = st.columns(2)
    with col_gender:
        gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'], help="Penting buat model!")
    with col_age:
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30, step=1, format="%d", help="Usia biologis kamu.")

    st.markdown("---")
    st.markdown("### 🏥 Riwayat Kesehatan")
    
    # Pake kolom lagi biar rapi
    col_hyper, col_heart = st.columns(2)
    with col_hyper:
        hypertension = st.selectbox("Hipertensi (Tekanan Darah Tinggi)", [0, 1], 
                                     format_func=lambda x: '✅ Ya' if x==1 else '❌ Tidak')
    with col_heart:
        heart_disease = st.selectbox("Penyakit Jantung", [0, 1], 
                                      format_func=lambda x: '✅ Ya' if x==1 else '❌ Tidak')

    smoking_history = st.selectbox("Riwayat Merokok", [
        'Tidak Pernah', 'Saat Ini', 'Dahulu (Former)',
        'Pernah', 'Tidak Rutin', 'Tidak Diketahui'
    ], help="Pilih yang paling menggambarkan kebiasaan kamu.")

    st.markdown("---")
    st.markdown("### 🧪 Data Lab & Biometrik")
    
    # Biometrik & Lab di 4 kolom
    col_bmi, col_hba1c, col_glucose, _ = st.columns(4)
    with col_bmi:
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.01, format="%.2f", help="Body Mass Index.")
    with col_hba1c:
        hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.01, format="%.2f", help="Rata-rata gula darah 3 bulan.")
    with col_glucose:
        blood_glucose = st.number_input("Gula Darah (mg/dL)", min_value=80, max_value=300, value=140, step=1, format="%d", help="Kadar gula darah saat ini.")
    
    st.markdown("---")
    # Tombol submit yang eye-catching
    submitted = st.form_submit_button("🔥 Gas, Prediksi Sekarang! 🔥")

# MAIN CONTENT
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
    
    st.subheader("🎉 Result Check: Hasil Prediksi Model")
    st.markdown("---")
    
    # Tampilan Hasil Utama (Paling penting, pakai custom CSS biar gede)
    if result == 1:
        st.markdown(f"<p class='big-font'>🛑 Pasien DIPREDIKSI **MENGIDAP DIABETES**</p>", unsafe_allow_html=True)
        st.warning("🚨 **Waduh!** Model bilang risikonya tinggi. Yuk, segera cek ke dokter betulan!")
    else:
        st.markdown(f"<p class='succes-font'>🥳 Pasien DIPREDIKSI **TIDAK MENGIDAP DIABETES**</p>", unsafe_allow_html=True)
        st.info("👍 **Chill!** Prediksi model aman. Tetap jaga pola hidup sehat ya!")
        
    st.markdown("---")
    st.markdown("### 📈 Detail Keyakinan Model (Probabilitas)")
    
    # Tampilkan probabilitas pake 'st.expander' biar gak menuh-menuhin layar
    with st.expander("Klik buat lihat angka probabilitas *valid*:", expanded=True):
        col_proba1, col_proba2, col_grafik = st.columns([1, 1, 2])
        
        # Pake metric yang warna-warni
        col_proba1.metric("Prob. **TIDAK** Diabetes", 
                         f"{proba[0]*100:.2f}%", 
                         delta_color="off", help="Tingkat keyakinan model untuk kelas 0.")
        
        col_proba2.metric("Prob. **DIABETES**", 
                         f"{proba[1]*100:.2f}%", 
                         delta_color="off", help="Tingkat keyakinan model untuk kelas 1.")
        
        # Pake chart sederhana buat visualisasi probabilitas
        data_proba = pd.DataFrame({
            'Kategori': ['TIDAK Diabetes (0)', 'DIABETES (1)'],
            'Probabilitas': [proba[0], proba[1]]
        })
        col_grafik.bar_chart(data_proba.set_index('Kategori'))

    st.markdown("---")
    st.markdown("💡 *Reminder: Hasil ini murni prediksi Machine Learning. Tetap konsultasi ke profesional kesehatan untuk diagnosis valid.*")
