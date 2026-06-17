import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Customer Retention Analytics System",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Model
# ==========================================

model = joblib.load("customer_churn_pipeline.pkl")

# ==========================================
# Session State
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# Title
# ==========================================

st.title("📊 Customer Retention Analytics & Churn Prediction System")

st.markdown("""
This application combines:

- 📈 Customer Churn Analytics
- 🤖 Machine Learning Prediction
- 🎯 Customer Retention Insights
""")

# ==========================================
# KPI Cards
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Customers", "7043")

with col2:
    st.metric("📉 Churn Rate", "26.5%")

with col3:
    st.metric("✅ Retention Rate", "73.5%")

with col4:
    st.metric("🤖 Model Accuracy", "79.5%")

# ==========================================
# Tabs
# ==========================================

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Prediction"])

# ==========================================
# Dashboard
# ==========================================

with tab1:

    st.header("📊 Customer Churn Dashboard")

    try:
        st.image("images/confusion_matrix.png")
    except:
        pass

    try:
        st.image("images/feature_importance.png")
    except:
        pass

    try:
        st.image("images/contract_churn.png")
    except:
        pass

    try:
        st.image("images/gender_churn.png")
    except:
        pass

    try:
        st.image("images/internet_churn.png")
    except:
        pass

    try:
        st.image("images/tenure_distribution.png")
    except:
        pass

    st.subheader("🥧 Customer Retention Overview")

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        [73.5,26.5],
        labels=["Retained","Churned"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.subheader("💡 Business Insights")

    st.success("""
• Customers with Month-to-Month contracts churn the most.

• Longer tenure customers are more loyal.

• Online Security and Tech Support significantly reduce churn.

• High Monthly Charges increase churn probability.
""")

# ==========================================
# Prediction
# ==========================================

with tab2:

    st.header("🤖 Customer Churn Prediction")

    col1,col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0,1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes","No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes","No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes","No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes","No"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes","No"]
        )

    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes","No"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes","No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes","No"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes","No"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes","No"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes","No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            value=1000.0
        )

    if st.button("🚀 Predict Churn"):

        input_data = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior_citizen],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone_service],
            "MultipleLines":[multiple_lines],
            "InternetService":[internet_service],
            "OnlineSecurity":[online_security],
            "OnlineBackup":[online_backup],
            "DeviceProtection":[device_protection],
            "TechSupport":[tech_support],
            "StreamingTV":[streaming_tv],
            "StreamingMovies":[streaming_movies],
            "Contract":[contract],
            "PaperlessBilling":[paperless_billing],
            "PaymentMethod":[payment_method],
            "MonthlyCharges":[monthly_charges],
            "TotalCharges":[total_charges]
        })
                # =====================================
        # Make Prediction
        # =====================================

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]
        churn_probability = probability[1] * 100

        # =====================================
        # Save Prediction History
        # =====================================

        st.session_state.history.append({
            "Prediction": "Churn" if prediction == 1 else "Stay",
            "Probability (%)": f"{churn_probability:.2f}",
            "Tenure": tenure,
            "Monthly Charges": monthly_charges
        })

        # =====================================
        # Risk Level
        # =====================================

        if churn_probability >= 70:
            risk = "🔴 High Risk"

        elif churn_probability >= 40:
            risk = "🟡 Medium Risk"

        else:
            risk = "🟢 Low Risk"

        # =====================================
        # Display Probability
        # =====================================

        st.subheader("📈 Churn Probability")

        st.progress(int(churn_probability))

        st.metric(
            "Customer Risk Score",
            f"{churn_probability:.2f}%"
        )

        # =====================================
        # Prediction Result
        # =====================================

        if prediction == 1:

            st.error("🚨 Customer Will Churn")

        else:

            st.success("✅ Customer Will Stay")

        st.write(f"### Risk Level: {risk}")

        # =====================================
        # Retention Recommendation
        # =====================================

        st.subheader("💡 Recommended Retention Actions")

        if prediction == 1:

            if churn_probability >= 70:

                st.warning("""
### Immediate Action Required

• Offer a loyalty discount

• Contact the customer personally

• Upgrade their current plan

• Assign dedicated customer support
""")

            elif churn_probability >= 40:

                st.info("""
### Moderate Risk

• Send promotional offers

• Recommend annual contract

• Provide additional benefits
""")

            else:

                st.success("""
### Low Risk

Continue engaging with the customer through
regular offers and communication.
""")

        else:

            st.success("""
🎉 Excellent!

This customer is likely to stay.

Continue providing excellent service.
""")
            # =====================================
# Prediction History
# =====================================

st.markdown("---")

col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("📜 Prediction History")

with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        width="stretch"
    )

else:

    st.info("No predictions made yet.")

# =====================================
# Footer
# =====================================

st.markdown("---")

st.markdown("""
# 👨‍💻 Developed By Shweta Singh

### Customer Retention Analytics & Churn Prediction System

Built Using

- Python
- Streamlit
- Scikit-Learn
- Random Forest
- Pandas
- Matplotlib

### Features

✅ Customer Churn Prediction

✅ Churn Probability Score

✅ Risk Classification

✅ Retention Recommendations

✅ Prediction History

✅ Interactive Dashboard
""")