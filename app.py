import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import joblib

# ==============================================================================
# A. PREPROCESSING DAN TRAINING (Decision Tree Randomized Search)
# ==============================================================================
print("--- [START] SINKRONISASI FILE DEPLOYMENT ---")

# 1. Load Data
df = pd.read_csv('diabetes_prediction_dataset.csv')
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 2. One-Hot Encoding (Manual)
X_processed = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=False)

# 3. Simpan Daftar Kolom OHE (INI KUNCI URUTAN)
feature_cols = X_processed.columns.tolist()

# 4. Training (Diulang untuk menjamin best_dt_model.joblib terbaru)
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, None], 'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20), 'max_features': [None, 'sqrt', 'log2'],
}
dt_model = DecisionTreeClassifier(random_state=42)
randomized_search = RandomizedSearchCV(dt_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
randomized_search.fit(X_processed, y) # Fit pada seluruh X_processed
best_model = randomized_search.best_estimator_

# 5. Simpan Model dan Daftar Kolom
joblib.dump(best_model, 'best_dt_model.joblib')
joblib.dump(feature_cols, 'model_features.joblib')
print("✅ 2 file joblib tersimpan dan sinkron.")

# ==============================================================================
# B. BUAT app.py (FIXED DENGAN .loc DAN HARDCASTING)
# ==============================================================================
app_py_content = """
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model dan Fitur ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('best_dt_model.joblib')
        feature_cols = joblib.load('model_features.joblib') 
        return model, feature_cols
    except FileNotFoundError:
        st.error("❌ ERROR GEDE: Pastikan 4 file deployment sudah di-push.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ GAGAL LOAD MODEL! Cek requirements.txt. Detail: {e}")
        st.stop()

model, feature_cols = load_resources()


# --- 2. Fungsi Prediksi (ANTI-GAGAL KOLOM) ---
def predict_diabetes(input_data, model, feature_cols):
    
    # 1. BUAT DATAFRAME KOSONG DENGAN SEMUA 15 KOLOM OHE (Default: 0)
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. ISI KOLOM NUMERIK (DENGAN HARDCASTING DAN .loc)
    # Gunakan .loc[0, 'kolom'] untuk assignment yang aman ke DataFrame
    input_df.loc[0, 'age'] = int(input_data['age']) 
    input_df.loc[0, 'hypertension'] = int(input_data['hypertension']) 
    input_df.loc[0, 'heart_disease'] = int(input_data['heart_disease']) 
    input_df.loc[0, 'bmi'] = float(input_data['bmi'])
    input_df.loc[0, 'HbA1c_level'] = float(input_data['HbA1c_level'])
    input_df.loc[0, 'blood_glucose_level'] = int(input_data['blood_glucose_level'])
    
    # 3. ISI KOLOM OHE (Manual Mapping dan .loc)
    
    # Gender
    gender_map = {'Female': 'gender_Female', 'Male': 'gender_Male', 'Other': 'gender_Other'}
    col_gender = gender_map.get(input_data['gender'])
    if col_gender in feature_cols:
        input_df.loc[0, col_gender] = 1 

    # Smoking History
    smoking_map = {
        'Tidak Pernah': 'smoking_history_never', 'Saat Ini': 'smoking_history_current', 
        'Dahulu (Former)': 'smoking_history_former', 'Pernah': 'smoking_history_ever', 
        'Tidak Rutin': 'smoking_history_not current', 'Tidak Diketahui': 'smoking_history_No Info'
    }
    col_smoking = smoking_map.get(input_data['smoking_history'])
    if col_smoking in feature_cols:
        input_df.loc[0, col_smoking] = 1 

    # 4. PREDIKSI (Urutan Kolom Dijamin Aman)
    prediction = model.predict(input_df) 
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- 3. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("👨‍🔬 Aplikasi Prediksi Diabetes (Decision Tree Randomized Search)")
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
"""
with open('app.py', 'w') as f:
    f.write(app_py_content)
print("✅ app.py tersimpan.")


# ==============================================================================
# C. BUAT requirements.txt (Final)
# ==============================================================================
requirements_content = """
streamlit
pandas
scikit-learn
joblib
numpy
"""
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)
print("✅ requirements.txt tersimpan.")
print("\n--- SEMUA FILE SUDAH SINKRON DAN SIAP PUSH ---")
