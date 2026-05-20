"""
preprocessing.py — وحدة المعالجة المشتركة بين التدريب والتطبيق
================================================================
تحتوي على جميع دوال تحويل وتنظيف البيانات المستخدمة في Pipeline التدريب
وتطبيق Streamlit، لضمان التطابق الكامل بين مرحلتي التدريب والاستنتاج.
"""

import pandas as pd
from sklearn.preprocessing import FunctionTransformer


# ──────────────────────────────────────────────
# 📅 استخراج ميزات التاريخ
# ──────────────────────────────────────────────
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    تحويل عمود Date_of_Journey إلى ثلاثة أعمدة رقمية:
    Journey_Day, Journey_Month, Journey_Year
    ثم حذف العمود الأصلي.
    """
    df = df.copy()
    if "Date_of_Journey" in df.columns:
        df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
        df["Journey_Day"] = df["Date_of_Journey"].dt.day
        df["Journey_Month"] = df["Date_of_Journey"].dt.month
        df["Journey_Year"] = df["Date_of_Journey"].dt.year
        df.drop("Date_of_Journey", axis=1, inplace=True)
    return df


# كائن FunctionTransformer جاهز للاستخدام في Pipeline
date_transformer = FunctionTransformer(extract_date_features)


# ──────────────────────────────────────────────
# 🧹 تحويل Duration إلى دقائق
# ──────────────────────────────────────────────
def convert_duration(duration: str) -> int:
    """
    تحويل نص المدة مثل '2h 30m' أو '5h' أو '45m' إلى عدد صحيح من الدقائق.
    """
    duration = str(duration).strip()
    total = 0
    if "h" in duration and "m" in duration:
        parts = duration.split("h")
        total = int(parts[0].strip()) * 60 + int(parts[1].replace("m", "").strip())
    elif "h" in duration:
        total = int(duration.replace("h", "").strip()) * 60
    elif "m" in duration:
        total = int(duration.replace("m", "").strip())
    return total


# ──────────────────────────────────────────────
# 🔧 تنظيف واستخراج الميزات الشامل
# ──────────────────────────────────────────────
STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


def clean_and_extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    تطبيق جميع تحويلات الميزات على DataFrame:
    - استخراج ساعة ودقيقة الإقلاع
    - تحويل Duration إلى دقائق
    - تحويل Total_Stops إلى رقم
    """
    df = df.copy()

    # ✈️ وقت الإقلاع
    if "Dep_Time" in df.columns:
        dep_parsed = pd.to_datetime(df["Dep_Time"], format="%H:%M")
        df["Dep_Hour"] = dep_parsed.dt.hour
        df["Dep_Minute"] = dep_parsed.dt.minute

    # ⏱️ المدة بالدقائق
    if "Duration" in df.columns:
        df["Duration_Minutes"] = df["Duration"].apply(convert_duration)

    # 🔁 عدد التوقفات
    if "Total_Stops" in df.columns and df["Total_Stops"].dtype == object:
        df["Total_Stops"] = df["Total_Stops"].map(STOPS_MAP)

    return df
