# --- [START] KODE FULL NOTEBOOK UNTUK DEPLOYMENT ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. SIMULASI DATA LOADING
# *Anggap lo punya data 'diabetes_prediction_dataset.csv' di folder yang sama*
print("1. Loading Data...")
try:
    # Ganti dengan path data lo yang sebenarnya
    data = pd.read_csv('diabetes_prediction_dataset.csv')
except FileNotFoundError:
    print("❌ ERROR: File 'diabetes_prediction_dataset.csv' tidak ditemukan. Pastikan data ada di folder yang sama.")
    exit()

# 2. PREPROCESSING DAN FEATURE ENGINEERING
print("2. Preprocessing Data...")

# Handle 'Other' di gender (ganti dengan 'Other' yang konsisten)
data['gender'] = data['gender'].replace('Other', 'Other') 

# Definisikan Target dan Fitur
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Definisikan kolom untuk transformasi
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

# Buat Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# 3. TRAINING MODEL (Decision Tree)
print("3. Training Model...")
# Pisahkan data (Walaupun untuk deployment tidak selalu perlu, ini untuk memastikan model dilatih)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat Pipeline
dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', dt_model)])

full_pipeline.fit(X_train, y_train)
print("   Model Decision Tree berhasil dilatih.")


# 4. PENYIMPANAN MODEL DAN FITUR (THE CRUCIAL STEP!)
print("4. Menyimpan Model dan Fitur...")

# Simpan Model Pipeline yang sudah di-train
joblib.dump(full_pipeline, 'best_dt_model.joblib')

# Ambil nama fitur akhir (Setelah Preprocessing)
# Ini penting buat memastikan app.py menggunakan urutan dan nama kolom yang SAMA
# Kita ambil nama kolom dari OneHotEncoder dan gabungkan dengan numerik/binary
ohe_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
final_features = numerical_features + ohe_cols + binary_features

joblib.dump(final_features, 'model_features.joblib')

print("   ✅ File 'best_dt_model.joblib' dan 'model_features.joblib' berhasil dibuat!")


# 5. PEMBUATAN FILE app.py (DEPLOYMENT SCRIPT)
print("5. Membuat file app.py...")

app_py_content = f"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model dan Fitur (Pake Caching Streamlit biar gak di-load terus!) ---
@st.cache_resource
def load_resources():
    try:
        # PATH-nya sekarang dijamin BISA karena kita udah bikin file ini di root folder
        model = joblib.load('best_dt_model.joblib')
        feature_cols = joblib.load('model_features.joblib')
        return model, feature_cols
    except FileNotFoundError:
        # Jika masih error, 99% masalahnya Git LFS/Deployment/Lupa Push
        st.error("❌ Error: File model atau fitur (.joblib) tidak ditemukan di server deployment.")
        st.info("Pastikan file 'best_dt_model.joblib' dan 'model_features.joblib' sudah di-commit dan di-push ke GitHub!")
        st.stop()

model, feature_cols = load_resources()


# --- 2. Fungsi Prediksi ---
def predict_diabetes(input_data):
    # Membuat DataFrame dummy dengan semua kolom fitur yang dibutuhkan model
    # Urutan kolom dijamin benar karena menggunakan 'feature_cols' yang sudah disimpan
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Mengisi kolom input user
    input_df['age'] = int(input_data['age'])
    input_df['hypertension'] = input_data['hypertension']
    input_df['heart_disease'] = input_data['heart_disease']
    input_df['bmi'] = input_data['bmi']
    input_df['HbA1c_level'] = input_data['HbA1c_level']
    input_df['blood_glucose_level'] = input_data['blood_glucose_level']
    
    # Mapping dan mengisi kolom One-Hot Encoding
    # Logika untuk gender
    gender_map = {{'Female': 'gender_Female', 'Male': 'gender_Male', 'Other': 'gender_Other'}}
    col_gender = gender_map.get(input_data['gender'])
    if col_gender in input_df.columns:
        input_df[col_gender] = 1

    # Logika untuk smoking_history
    # NOTE: Nama kolom harus SESUAI dengan hasil get_feature_names_out dari OneHotEncoder!
    smoking_map = {{
        'Tidak Pernah': 'smoking_history_never',
        'Saat Ini': 'smoking_history_current',
        'Dahulu (Former)': 'smoking_history_former',
        'Pernah': 'smoking_history_ever',
        'Tidak Rutin': 'smoking_history_not current',
        'Tidak Diketahui': 'smoking_history_No Info' # Harusnya 'smoking_history_No Info'
    }}
    col_name = smoking_map.get(input_data['smoking_history'])
    if col_name in input_df.columns:
        input_df[col_name] = 1
    
    # IMPORTANT: Preprocessor di pipeline akan mengurus Scaling dan OHE pada input_df
    # Tidak perlu scaling manual di sini karena 'full_pipeline' sudah mencakup 'preprocessor' dan 'classifier'
    
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
    
    # UMUR (Integer)
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
    
    # Probabilitas di indeks 0 adalah kelas 0 (Tidak Diabetes), indeks 1 adalah kelas 1 (Diabetes)
    col1.metric("Probabilitas Tidak Diabetes", f"{{proba[0]*100:.2f}}%")
    col2.metric("Probabilitas Diabetes", f"{{proba[1]*100:.2f}}%")

    st.caption("Disclaimer: Hasil ini hanya prediksi Machine Learning, bukan diagnosis medis.")
"""

# Tulis konten app.py ke file
try:
    with open('app.py', 'w') as f:
        f.write(app_py_content)
    print("----------------------------------------------------------------------")
    print("✅ SCRIPT SELESAI! Tiga file penting sudah siap:")
    print("   1. app.py (Streamlit Code)")
    print("   2. best_dt_model.joblib (Model Pipeline)")
    print("   3. model_features.joblib (Urutan Fitur)")
    print("----------------------------------------------------------------------")
    print(">>> ACTION WAJIB: Lakukan 'git add .', 'git commit', dan 'git push' KETIGA file di atas ke GitHub!")
except Exception as e:
    print(f"❌ Gagal membuat file app.py: {e}")

# --- [END] KODE FULL NOTEBOOK ---
