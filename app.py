"""
app.py — تطبيق Streamlit للتنبؤ بأسعار تذاكر الطيران ✈️
========================================================
يستخدم النموذج المدرب (best_model.pkl) للتنبؤ بسعر التذكرة
بناءً على بيانات الرحلة المدخلة من المستخدم.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from io import BytesIO

# استيراد دوال المعالجة المشتركة
from preprocessing import extract_date_features, clean_and_extract_features, STOPS_MAP

# ===============================
# إعداد الصفحة
# ===============================
st.set_page_config(page_title="تطبيق تسعير الطيران", page_icon="✈️", layout="wide")
st.title("✈️ تطبيق تسعير تذاكر الطيران")
st.markdown("أدخل بيانات الرحلة لتوقع سعر التذكرة بناءً على الموديل المدرب 🎯")

# ===============================
# تحميل الموديل والبيانات
# ===============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data_Train.xlsx")


@st.cache_resource
def load_model():
    """تحميل النموذج مع التخزين المؤقت لتسريع إعادة التشغيل."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """تحميل بيانات التدريب للحصول على القيم الفريدة للقوائم المنسدلة."""
    return pd.read_excel(DATA_PATH, engine="openpyxl")


try:
    model = load_model()
    st.sidebar.success("✅ تم تحميل الموديل بنجاح")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل الموديل: {e}")
    model = None

try:
    data = load_data()
    st.sidebar.success("✅ تم تحميل البيانات بنجاح")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل البيانات: {e}")
    data = None

# ===============================
# تهيئة تخزين التوقعات في session_state
# ===============================
if "predictions" not in st.session_state:
    st.session_state["predictions"] = pd.DataFrame(columns=[
        "Airline", "Source", "Destination", "Total_Stops",
        "Duration_Minutes", "Days_Left", "Additional_Info", "Predicted_Price"
    ])

# ===============================
# واجهة إدخال البيانات
# ===============================
if data is not None and model is not None:
    st.header("🧾 إدخال بيانات الرحلة")

    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("شركة الطيران", sorted(data["Airline"].dropna().unique()))
        source = st.selectbox("مكان الإقلاع", sorted(data["Source"].dropna().unique()))
        destination = st.selectbox("الوجهة", sorted(data["Destination"].dropna().unique()))
        additional_info = st.selectbox(
            "معلومات إضافية",
            sorted(data["Additional_Info"].dropna().unique()),
            index=sorted(data["Additional_Info"].dropna().unique()).index("No info")
            if "No info" in data["Additional_Info"].dropna().unique()
            else 0,
        )
    with col2:
        total_stops = st.selectbox(
            "عدد التوقفات",
            ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"],
        )
        duration_hours = st.number_input("⏱️ مدة الرحلة — ساعات", min_value=0, max_value=40, value=5)
        duration_minutes = st.number_input("⏱️ مدة الرحلة — دقائق", min_value=0, max_value=59, value=0)
        days_left = st.slider("📆 عدد الأيام قبل الرحلة", 0, 60, 10)

    date_of_journey = st.date_input("📅 تاريخ الرحلة")
    dep_time = st.time_input("⏰ وقت الإقلاع")

    # بناء Duration بالتنسيق الأصلي مثل "2h 30m"
    duration_str = f"{duration_hours}h"
    if duration_minutes > 0:
        duration_str += f" {duration_minutes}m"

    # بناء الداتا فريم بالأعمدة الأصلية
    input_df = pd.DataFrame({
        "Airline": [airline],
        "Source": [source],
        "Destination": [destination],
        "Total_Stops": [total_stops],
        "Duration": [duration_str],
        "Days_Left": [days_left],
        "Date_of_Journey": [pd.to_datetime(date_of_journey)],
        "Dep_Time": [dep_time.strftime("%H:%M")],
        "Additional_Info": [additional_info],
    })

    # تطبيق المعالجة المشتركة
    input_df = extract_date_features(input_df)
    input_df = clean_and_extract_features(input_df)

    # حذف الأعمدة الخام التي تم تحويلها
    cols_to_drop = [c for c in ["Dep_Time", "Duration"] if c in input_df.columns]
    input_df.drop(columns=cols_to_drop, inplace=True)

    st.write("### 👇 البيانات المدخلة بعد المعالجة")
    st.dataframe(input_df)

    # ===============================
    # التنبؤ
    # ===============================
    if st.button("🔮 توقع السعر"):
        try:
            prediction_log = model.predict(input_df)
            prediction = np.expm1(prediction_log)
            price = float(prediction[0])

            st.success(f"💰 السعر المتوقع للتذكرة هو: {price:,.2f} جنيه")

            # حفظ النتيجة في الداشبورد
            new_row = {
                "Airline": airline,
                "Source": source,
                "Destination": destination,
                "Total_Stops": total_stops,
                "Duration_Minutes": duration_hours * 60 + duration_minutes,
                "Days_Left": days_left,
                "Additional_Info": additional_info,
                "Predicted_Price": price,
            }
            st.session_state["predictions"] = pd.concat(
                [st.session_state["predictions"], pd.DataFrame([new_row])],
                ignore_index=True,
            )

        except Exception as e:
            st.error(f"⚠️ حدث خطأ أثناء التنبؤ: {e}")

    # ===============================
    # 📊 عرض الداشبورد التفاعلية
    # ===============================
    if not st.session_state["predictions"].empty:
        st.markdown("---")
        st.subheader("📊 لوحة التوقعات السابقة")

        df_pred = st.session_state["predictions"]

        col1, col2, col3 = st.columns(3)
        col1.metric("عدد التوقعات", len(df_pred))
        col2.metric("أرخص سعر", f"{df_pred['Predicted_Price'].min():,.2f} جنيه")
        col3.metric("أعلى سعر", f"{df_pred['Predicted_Price'].max():,.2f} جنيه")

        st.dataframe(df_pred)

        st.bar_chart(df_pred.groupby("Airline")["Predicted_Price"].mean())

        # ===============================
        # 📤 تحميل النتائج و 🗑️ إعادة التعيين
        # ===============================
        st.markdown("### ⚙️ أدوات إضافية")

        colA, colB = st.columns(2)

        # 🔹 زر التحميل كملف Excel
        with colA:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_pred.to_excel(writer, index=False, sheet_name="Predictions")
            st.download_button(
                label="📤 تحميل النتائج كملف Excel",
                data=buffer.getvalue(),
                file_name="Predicted_Flights.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # 🔹 زر إعادة التعيين
        with colB:
            if st.button("🗑️ إعادة تعيين التوقعات"):
                st.session_state["predictions"] = pd.DataFrame(columns=df_pred.columns)
                st.success("✅ تم مسح جميع التوقعات.")

else:
    st.error("⚠️ لم يتم تحميل البيانات أو الموديل بشكل صحيح. تأكد من وجود الملفات في نفس المجلد.")

st.markdown("---")
st.caption("🚀 تم إنشاء هذا التطبيق باستخدام Streamlit و Scikit-learn")
