import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")

st.write("""
This project predicts customer churn using
Logistic Regression and Random Forest.
""")

st.header("Project Overview")

st.write("""
Dataset: IBM Telco Customer Churn Dataset
""")

st.header("EDA Visualizations")

st.image("images/churn_distribution.png")
st.image("images/contract_churn.png")
st.image("images/feature_importance.png")