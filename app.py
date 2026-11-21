import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import sklearn.compose._column_transformer as ct
from sklearn.preprocessing import FunctionTransformer
from io import BytesIO
import requests
from streamlit_lottie import st_lottie
import plotly.express as px

# ===============================
# إعداد الصفحة
# ===============================
st.set_page_config(page_title="توقع أسعار الطيران", layout="wide", initial_sidebar_state="collapsed")

# ===============================
# إعدادات التصميم (CSS)
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
}

/* خلفية احترافية رمادية/زرقاء هادئة بدلاً من الأبيض الفاقع */
.stApp {
    background: #f4f6f9;
}

/* تصميم الكروت الإحصائية */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e1e5eb;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2c3e50;
}
[data-testid="stMetricLabel"] {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7f8c8d;
}

/* تحسين الأزرار */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    background: #2c3e50;
    color: white !important;
    font-weight: 600;
    font-size: 1.15rem;
    border: none;
    box-shadow: 0 2px 4px rgba(44, 62, 80, 0.2);
    transition: background 0.2s;
}
.stButton>button:hover {
    background: #1a252f;
}

.gradient-text {
    color: #2c3e50;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# تحميل رسوم Lottie
# ===============================
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_flight = load_lottieurl("https://lottie.host/9e47f2ef-cbef-4fa1-8ca6-0b81831c18bf/G8E1Uj3tX7.json")
lottie_dashboard = load_lottieurl("https://lottie.host/bba4c4be-e2d1-42ae-932d-32cd835c1650/474e6O8rW7.json")

# ===============================
# تعريف الدوال والكلاسات الخاصة بالنموذج
# ===============================
def extract_date_features(df):
    df = df.copy()
    if "Date_of_Journey" in df.columns:
        df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
        df["Journey_Day"] = df["Date_of_Journey"].dt.day
        df["Journey_Month"] = df["Date_of_Journey"].dt.month
        df["Journey_Year"] = df["Date_of_Journey"].dt.year
        df.drop("Date_of_Journey", axis=1, inplace=True)
    return df

class _RemainderColsList(list):
    pass

setattr(ct, "_RemainderColsList", _RemainderColsList)

# ===============================
# تحميل الموديل والبيانات
# ===============================
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.getcwd(), "best_model.pkl")
    try:
        globals()["extract_date_features"] = extract_date_features
        globals()["_RemainderColsList"] = _RemainderColsList
        globals()["FunctionTransformer"] = FunctionTransformer
        model = joblib.load(MODEL_PATH)
        return model, True, None
    except Exception as e:
        return None, False, str(e)

@st.cache_data
def load_data():
    DATA_PATH = os.path.join(os.getcwd(), "Data_Train.xlsx")
    try:
        data = pd.read_excel(DATA_PATH, engine="openpyxl")
        return data, True, None
    except Exception as e:
        return None, False, str(e)

if "predictions" not in st.session_state:
    st.session_state["predictions"] = pd.DataFrame(columns=[
        "Airline", "Source", "Destination", "Total_Stops",
        "Duration_Minutes", "Predicted_Price"
    ])

model, is_model_loaded, model_err = load_model()
data, is_data_loaded, data_err = load_data()

# ===============================
# الهيكل الرئيسي للتطبيق
# ===============================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["حاسبة الأسعار", "لوحة المتابعة", "معلومات النظام"])

with tab1:
    h_col1, h_col2 = st.columns([2.5, 1])
    with h_col1:
        st.markdown("<h1 class='gradient-text' style='font-size: 3rem; margin-bottom: 0px;'>نظام توقع أسعار تذاكر الطيران</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size: 1.15rem; color: #475569; margin-top: 15px; line-height: 1.8; font-weight: 600;'>
        يعتمد هذا النظام على نماذج تحليل البيانات لتقدير أسعار رحلات الطيران. <br>
        يرجى إدخال تفاصيل الرحلة المخطط لها أدناه للحصول على التكلفة التقديرية.
        </p>
        """, unsafe_allow_html=True)
    with h_col2:
        if lottie_flight:
            st_lottie(lottie_flight, height=160, key="flight_anim")

    st.markdown("<br>", unsafe_allow_html=True)

    if not is_model_loaded:
        st.error(f"خطأ في تحميل النموذج: {model_err}")
    elif not is_data_loaded:
        st.error(f"خطأ في تحميل ملف البيانات: {data_err}")
    else:
        st.markdown("""
        <div style='background: #ffffff; border-radius: 10px; padding: 25px; border: 1px solid #e1e5eb; margin-bottom: 30px;'>
            <h3 style='color: #2c3e50; font-weight: 700; margin-bottom: 25px; font-size: 1.5rem;'>مسار الرحلة وتفاصيلها</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            airline = st.selectbox("شركة الطيران", sorted(data["Airline"].dropna().unique()))
            source = st.selectbox("مدينة الإقلاع", sorted(data["Source"].dropna().unique()))
            destination = st.selectbox("وجهة الوصول", sorted(data["Destination"].dropna().unique()))
        
        with col2:
            total_stops = st.selectbox("عدد التوقفات", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
            duration = st.slider("مدة الرحلة المتوقعة (ساعات)", 1, 40, 5)
            
        with col3:
            date_of_journey = st.date_input("تاريخ الإقلاع")
            dep_time = st.time_input("وقت الإقلاع التقريبي")

        st.markdown("</div>", unsafe_allow_html=True)
        
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1.5, 1])
        with predict_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.button("احسب السعر المتوقع")

        if predict_button:
            with st.spinner('جاري حساب السعر التقديري بناءً على المعطيات...'):
                input_df = pd.DataFrame({
                    "Airline": [airline],
                    "Source": [source],
                    "Destination": [destination],
                    "Total_Stops": [total_stops],
                    "Duration": [f"{duration}h"],
                    "Date_of_Journey": [pd.to_datetime(date_of_journey)],
                    "Dep_Time": [dep_time.strftime("%H:%M")],
                    "Additional_Info": ["No info"]
                })

                input_df = extract_date_features(input_df)
                input_df["Dep_Hour"] = pd.to_datetime(input_df["Dep_Time"]).dt.hour
                input_df["Dep_Minute"] = pd.to_datetime(input_df["Dep_Time"]).dt.minute
                input_df.drop("Dep_Time", axis=1, inplace=True)
                stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
                input_df["Total_Stops"] = input_df["Total_Stops"].map(stops_map)
                input_df["Duration_Minutes"] = input_df["Duration"].str.replace("h", "").astype(float) * 60
                input_df.drop("Duration", axis=1, inplace=True)

                try:
                    prediction_log = model.predict(input_df)
                    prediction = np.expm1(prediction_log)
                    price = float(prediction[0])
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.success(f"التكلفة التقديرية للتذكرة: {price:,.2f} جنيه / روبية")

                    new_row = {
                        "Airline": airline,
                        "Source": source,
                        "Destination": destination,
                        "Total_Stops": total_stops,
                        "Duration_Minutes": duration * 60,
                        "Predicted_Price": price
                    }
                    st.session_state["predictions"] = pd.concat(
                        [st.session_state["predictions"], pd.DataFrame([new_row])],
                        ignore_index=True
                    )

                except Exception as e:
                    st.error(f"حدث خطأ أثناء المعالجة: {e}")

with tab2:
    st.markdown("<h2 class='gradient-text' style='font-size: 2.2rem; margin-bottom: 20px;'>لوحة المتابعة والتحليلات</h2>", unsafe_allow_html=True)
    
    if st.session_state["predictions"].empty:
        st.info("لم يتم تسجيل أي بيانات بعد. استخدم حاسبة الأسعار لتوليد سجل التوقعات.")
        if lottie_dashboard:
            st_lottie(lottie_dashboard, height=350, key="empty_dashboard")
    else:
        df_pred = st.session_state["predictions"]

        col1, col2, col3 = st.columns(3)
        col1.metric("إجمالي العمليات", len(df_pred), "عملية حديثة")
        col2.metric("أرخص سعر متوقع", f"{df_pred['Predicted_Price'].min():,.0f}")
        col3.metric("أعلى سعر متوقع", f"{df_pred['Predicted_Price'].max():,.0f}")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #2c3e50; font-weight: 700; font-size: 1.5rem;'>سجل الرحلات المفصل</h3>", unsafe_allow_html=True)
        
        st.dataframe(
            df_pred,
            column_config={
                "Airline": st.column_config.TextColumn("الشركة", width="medium"),
                "Source": "الإقلاع",
                "Destination": "الوجهة",
                "Total_Stops": st.column_config.TextColumn("التوقفات"),
                "Duration_Minutes": st.column_config.NumberColumn("المدة", format="%d min"),
                "Predicted_Price": st.column_config.ProgressColumn(
                    "السعر المتوقع",
                    format="%f",
                    min_value=0, 
                    max_value=float(df_pred["Predicted_Price"].max() * 1.2),
                ),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #2c3e50; font-weight: 700; font-size: 1.5rem;'>الرسوم البيانية</h3>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            fig_bar = px.bar(
                df_pred.groupby("Airline", as_index=False)["Predicted_Price"].mean().sort_values("Predicted_Price", ascending=False),
                x="Airline", y="Predicted_Price",
                title="متوسط السعر حسب شركة الطيران",
                labels={"Predicted_Price": "متوسط السعر", "Airline": "الشركة"},
                color="Airline",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font=dict(family="Cairo", size=13))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            fig_pie = px.pie(
                df_pred, names="Airline",
                title="توزيع عمليات البحث حسب الشركة",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Cairo", size=13))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            df_pred, x="Duration_Minutes", y="Predicted_Price",
            color="Airline", size="Predicted_Price",
            hover_data=["Source", "Destination", "Total_Stops"],
            title="علاقة مدة الرحلة بالسعر التقديري",
            labels={"Predicted_Price": "السعر المرجح", "Duration_Minutes": "المدة (دقائق)"},
            size_max=25,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_scatter.update_layout(plot_bgcolor="rgba(255, 255, 255, 0.4)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest", font=dict(family="Cairo", size=13))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("<hr style='border: 1px solid #e1e5eb; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #475569; font-weight: 700; font-size: 1.3rem;'>أدوات إدارة البيانات</h3>", unsafe_allow_html=True)

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_pred.to_excel(writer, index=False, sheet_name="Data")
            st.download_button(
                label="تصدير السجل (Excel)",
                data=buffer.getvalue(),
                file_name="Flight_Analysis_Export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with colB:
            if st.button("مسح السجل الحالي", use_container_width=True):
                st.session_state["predictions"] = pd.DataFrame(columns=df_pred.columns)
                st.rerun()

with tab3:
    st.markdown("<h2 class='gradient-text' style='font-size: 2.2rem; margin-bottom: 20px;'>معلومات النظام</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: #ffffff; padding: 30px; border-radius: 10px; border: 1px solid #e1e5eb; margin-top: 20px;'>
        <p style='font-size: 1.15rem; color: #334155; line-height: 1.8;'>
        تم بناء هذا التطبيق لتقديم واجهة مبسطة لنموذج توقع أسعار تذاكر الطيران الخاص بالمشروع.
        الهدف هو مساعدة المستخدمين على معرفة النطاق السعري التقديري للرحلات بناءً على التواريخ والوجهات وشركات الطيران المدخلة.
        </p>
        <br>
        <h4 style='color: #2c3e50; font-weight: 700; font-size: 1.3rem; margin-bottom: 15px;'>المواصفات التقنية:</h4>
        <ul style='font-size: 1.05rem; color: #475569; line-height: 2; list-style-type: disc; padding-right: 20px;'>
            <li>محرك عرض البيانات: مكتبة Streamlit لتكوين واجهة المستخدم.</li>
            <li>الرسوم البيانية: تم استخدام Plotly للحصول على رسومات إحصائية تفاعلية.</li>
            <li>النموذج التحليلي: Scikit-Learn (خوارزمية Regression).</li>
            <li>معالجة البيانات: مكتبة Pandas لتنظيم وحفظ سجل التوقعات.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
