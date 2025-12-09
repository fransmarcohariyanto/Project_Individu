# model_creation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("--- START: Model Creation Script ---")

# 1. LOAD DATA
try:
    data = pd.read_csv('diabetes_prediction_dataset.csv')
except FileNotFoundError:
    print("❌ ERROR: File data 'diabetes_prediction_dataset.csv' tidak ditemukan. Harap letakkan di folder yang sama.")
    exit()

# Handle 'Other' di gender (ganti dengan 'Other' yang konsisten)
data['gender'] = data['gender'].replace('Other', 'Other') 

X = data.drop('diabetes', axis=1)
y = data['diabetes']

# 2. PREPROCESSING PIPELINE
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# 3. TRAINING MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', dt_model)])

full_pipeline.fit(X_train, y_train)
print("✅ Model Decision Tree berhasil dilatih.")

# 4. SIMPAN MODEL DAN FITUR (CRUCIAL STEP!)
# Simpan Model Pipeline
joblib.dump(full_pipeline, 'best_dt_model.joblib')

# Ambil nama fitur akhir (PENTING untuk app.py)
ohe_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
final_features = numerical_features + ohe_cols + binary_features
joblib.dump(final_features, 'model_features.joblib')

print("🚀 Dua file .joblib siap: 'best_dt_model.joblib' & 'model_features.joblib'")
print("--- END: Model Creation Script ---")
