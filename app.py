# app/streamlit_app.py  (Dashboard - main page)
import streamlit as st
import pandas as pd, plotly.express as px, joblib

st.set_page_config(page_title="ChurnSense",
                   page_icon="🏦", layout="wide")

@st.cache_resource   # Load once, cache in memory
def load_model():
    return joblib.load('models/churn_model_v1.pkl')

@st.cache_data       # Cache data processing
def load_data():
    return pd.read_csv('data/WA_Fn-UseC_Telco.csv')

model = load_model()
df = load_data()

st.title("🏦 ChurnSense — Customer Analytics")
st.markdown("*AI-Powered Churn Prediction Dashboard*")

# KPI Cards Row
col1,col2,col3,col4 = st.columns(4)
churn_rate = (df['Churn']=='Yes').mean()*100
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churn Rate", f"{churn_rate:.1f}%",
            delta="-1.2%", delta_color="inverse")
col3.metric("Avg Monthly Charges",
            "$64.76")
col4.metric("Model AUC Score", "0.87")

# app/pages/01_predict.py  (Prediction page)
import streamlit as st, pandas as pd, joblib

model = joblib.load('models/churn_model_v1.pkl')
st.title("🔮 Predict Customer Churn")

with st.form("predict_form"):
    c1,c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)",0,72,12)
        monthly = st.number_input(
            "Monthly Charges ($)", 18.0, 120.0, 65.0)
        contract = st.selectbox("Contract Type",
            ["Month-to-month","One year","Two year"])
    with c2:
        internet = st.selectbox("Internet Service",
            ["DSL","Fiber optic","No"])
        tech_support = st.selectbox("Tech Support",
            ["Yes","No","No internet service"])
        payment = st.selectbox("Payment Method",
            ["Electronic check","Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])
    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    customer = {'tenure':tenure,
                'MonthlyCharges':monthly}
    prob = model.predict_proba(
        pd.DataFrame([customer]))[0][1]
    st.metric("Churn Probability",f"{prob:.1%}")
    if prob>0.5: st.error("⚠️ HIGH churn risk!")
    else: st.success("✅ Low churn risk")
