# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# Preprocessing and pipeline utilities
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier

# Utility functions
from collections import Counter 

# Import joblib để lưu model
import joblib

df=pd.read_csv(r'C:\Users\ta an\Desktop\Risk_warning\Credit Risk Benchmark Dataset (1).csv')

df.duplicated().sum()
#duplicados = df[df.duplicated()]
#print(duplicados)
df.drop_duplicates(inplace=True)

# Relación ingreso-deuda
df['income_to_debt'] = df['monthly_inc'] / (df['debt_ratio'] + 1e-5)  # evitamos división por cero

# Total de morosidades (past)
df['total_late'] = df['late_30_59'] + df['late_60_89'] + df['late_90']

# Edad por crédito activo (¿tiene experiencia crediticia?)
df['age_per_credit'] = df['age'] / (df['open_credit'] + 1)

# Ratio real estate vs total créditos (¿cuántos créditos son hipotecarios?)
df['real_estate_ratio'] = df['real_estate'] / (df['open_credit'] + 1)

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")

results_list = []

features = df.drop(columns='dlq_2yrs')
target = df['dlq_2yrs']

features_temp, features_test, target_temp, target_test = train_test_split(
    features, target, test_size=0.20, stratify=target, random_state=42
)

features_train, features_val, target_train, target_val = train_test_split(
    features_temp, target_temp, test_size=0.20, stratify=target_temp, random_state=42
)

print(f"Train: {features_train.shape}, Val: {features_val.shape}, Test: {features_test.shape}")

# 1. Crear pipeline
logreg_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# 2. Entrenar el modelo
logreg_pipeline.fit(features_train, target_train)

# 3. Evaluar en el conjunto de validación
val_preds = logreg_pipeline.predict(features_val)
val_proba = logreg_pipeline.predict_proba(features_val)[:, 1]

test_preds = logreg_pipeline.predict(features_test)
test_proba = logreg_pipeline.predict_proba(features_test)[:, 1]

# 4. Resultados
print("📊 Classification Report (Validation):")
print(classification_report(target_val, val_preds))

print("\n🔍 Confusion Matrix (Validation):")
print(confusion_matrix(target_val, val_preds))

print("📊 Classification Report (Test):")
print(classification_report(target_test, test_preds))

print("\n🔍 Confusion Matrix (Test):")
print(confusion_matrix(target_test, test_preds))

print("\n🎯 ROC-AUC Score (Validation):", roc_auc_score(target_val, val_proba))

# LƯU MODEL
joblib.dump(logreg_pipeline, 'credit_risk_model.pkl')
print("✅ Đã lưu model vào file credit_risk_model.pkl")