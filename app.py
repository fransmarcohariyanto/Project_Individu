# Import semua library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import joblib
import os
import sys # Tambahkan ini kalau lo mau menjalankan Streamlit dari notebook (opsional)

# ==============================================================================
# [CELL 1] Data Preprocessing dan Splitting
# ==============================================================================
print("--- [START] CELL 1: Preprocessing Data ---")

# 1. Load Data
df = pd.read_csv('diabetes_prediction_dataset.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 3. One-Hot Encoding
# drop_first=False penting buat memastikan semua kolom kategori ada
X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=False)

# 4. Ambil dan Simpan Nama-nama Kolom (WAJIB buat Deployment)
# feature_cols ini akan menentukan urutan dan nama kolom input di app.py
feature_cols = X.columns.tolist()

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Preprocessing selesai. Shape X_train:", X_train.shape)

# ==============================================================================
# [CELL 2] Model Training dengan RandomizedSearchCV (Decision Tree)
# ==============================================================================
print("\n--- [START] CELL 2: Training Model Decision Tree ---")

# Parameter distribution (sesuai permintaan lo)
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': [None, 'sqrt', 'log2'],
}

dt_model = DecisionTreeClassifier(random_state=42)

# RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    estimator=dt_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit/Training Model
randomized_search.fit(X_train, y_train)

best_model = randomized_search.best_estimator_

print("Best Parameters:", randomized_search.best_params_)
print("Best Cross-Validation Accuracy:", randomized_search.best_score_)


# ==============================================================================
# [CELL 3] Simpan Model dan Feature Columns
# ==============================================================================
print("\n--- [START] CELL 3: Saving Model Files ---")

# Simpan model terbaik
model_filename = 'best_dt_model.joblib'
joblib.dump(best_model, model_filename)

# Simpan feature columns
columns_filename = 'model_features.joblib'
joblib.dump(feature_cols, columns_filename)

print(f"Model tersimpan: {model_filename}")
print(f"Fitur tersimpan: {columns_filename}")


# ==============================================================================
# [CELL 4] Otomasi Pembuatan app.py (Kode Streamlit LENGKAP)
# ==============================================================================
print("\n--- [START] CELL 4: Creating app.py ---")

app_py_content = """
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

# --- 2. Fungsi Prediksi ---
def predict_diabetes(input_data):
    # Buat DataFrame kosong dengan semua 15 kolom fitur yang dibutuhkan
    # Ini menjamin urutan kolom sama seperti saat training
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Isi kolom yang bersifat numerik
    input_df['age'] = input_data['age']
    input_df['hypertension'] = input_data['hypertension']
    input_df['heart_disease'] = input_data['heart_disease']
    input_df['bmi'] = input_data['bmi']
    input_df['HbA1c_level'] = input_data['HbA1c_level']
    input_df['blood_glucose_level'] = input_data['blood_glucose_level']
    
    # Isi kolom One-Hot Encoding (sesuai pilihan user)
    # Gender
    if input_data['gender'] == 'Female':
        input_df['gender_Female'] = 1
    elif input_data['gender'] == 'Male':
        input_df['gender_Male'] = 1
    elif input_data['gender'] == 'Other':
        input_df['gender_Other'] = 1
        
    # Smoking History Mapping
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

    # Prediksi
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
    age = st.slider("Usia", min_value=1.0, max_value=100.0, value=30.0, step=0.1)
    bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.1)
    blood_glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=140, step=1)
    
    # Tombol Submit
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

# Tulis konten app.py ke file
try:
    with open('app.py', 'w') as f:
        f.write(app_py_content)
    print("🚀 File 'app.py' sudah terbuat secara otomatis (LENGKAP).")
except Exception as e:
    print(f"❌ Gagal membuat file app.py: {e}")


# ==============================================================================
# [CELL 5] Otomasi Pembuatan requirements.txt (Hanya yang Perlu)
# ==============================================================================
print("\n--- [START] CELL 5: Creating requirements.txt ---")

requirements_content = """
streamlit
pandas
scikit-learn
joblib
numpy
"""

# Tulis konten requirements.txt ke file
try:
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("📋 File 'requirements.txt' sudah terbuat (Hanya untuk DT/Streamlit).")
except Exception as e:
    print(f"❌ Gagal membuat file requirements.txt: {e}")

print("\n--- SEMUA FILE DEPLOYMENT SUDAH SIAP DI FOLDER LOKAL LO! ---")
