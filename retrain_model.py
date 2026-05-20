"""
retrain_model.py — إعادة تدريب النموذج بحجم أصغر
==================================================
يقوم بإعادة تدريب نموذج Random Forest مع:
- عدد أشجار أقل (200 بدلاً من 800+)
- max_depth محدود لتقليل حجم الملف
- حفظ مع compress لتصغير ملف pkl
- استخدام preprocessing.py المشترك
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# استيراد دوال المعالجة المشتركة
from preprocessing import (
    extract_date_features,
    clean_and_extract_features,
    date_transformer,
)

# ──────────────────────────────────────────────
# تحميل البيانات
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data_Train.xlsx")
print("📂 جاري تحميل البيانات...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")
print(f"   أبعاد البيانات: {df.shape}")

# ──────────────────────────────────────────────
# ملء القيم الفارغة
# ──────────────────────────────────────────────
df["Total_Stops"] = df["Total_Stops"].fillna(df["Total_Stops"].mode()[0])
df["Route"] = df["Route"].fillna(df["Route"].mode()[0])

# ──────────────────────────────────────────────
# تنظيف واستخراج الميزات
# ──────────────────────────────────────────────
print("🔧 جاري تنظيف البيانات...")
df = clean_and_extract_features(df)

# ──────────────────────────────────────────────
# إعداد X و y
# ──────────────────────────────────────────────
X = df.drop(["Price", "Route", "Dep_Time", "Arrival_Time", "Duration"], axis=1)
y = df["Price"]

# تحويل لوغاريتمي
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"🟢 بيانات التدريب: {X_train.shape}")
print(f"🔵 بيانات الاختبار: {X_test.shape}")

# ──────────────────────────────────────────────
# بناء Pipeline
# ──────────────────────────────────────────────
categorical_cols = ["Airline", "Source", "Destination", "Additional_Info"]
numeric_cols = [col for col in X.columns if col not in categorical_cols + ["Date_of_Journey"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)

pipeline = Pipeline(steps=[
    ("date_features", date_transformer),
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42)),
])

# ──────────────────────────────────────────────
# Hyperparameter Tuning (مُحسَّن للحجم)
# ──────────────────────────────────────────────
rf_params = {
    "model__n_estimators": [150, 200, 300],       # أقل من قبل
    "model__max_depth": [10, 15],
    "model__min_samples_split": [5, 10, 15],
    "model__min_samples_leaf": [4, 6, 8],
    "model__max_features": ["sqrt", "log2"],
    "model__bootstrap": [True],
}

print("\n🔍 جاري البحث عن أفضل المعلمات (RandomizedSearchCV)...")
rf_search = RandomizedSearchCV(
    pipeline, rf_params, n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=5, random_state=42, n_jobs=-1, verbose=1,
)

rf_search.fit(X_train, y_train)
best_pipeline = rf_search.best_estimator_
print(f"\n✅ أفضل المعلمات: {rf_search.best_params_}")

# ──────────────────────────────────────────────
# التقييم
# ──────────────────────────────────────────────
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n📊 {dataset_name}:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")
    print(f"   MAPE = {mape:.2f}%")
    return mae, rmse, r2, mape

y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")

# ──────────────────────────────────────────────
# حفظ النموذج مع ضغط
# ──────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
print(f"\n💾 جاري حفظ النموذج مع ضغط...")
joblib.dump(best_pipeline, OUTPUT_PATH, compress=3)

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"✅ تم حفظ النموذج: {OUTPUT_PATH}")
print(f"📦 حجم الملف: {file_size_mb:.1f} MB")
