import pandas as pd
import streamlit as st


from cloud_inference import SageMakerCreditScoreInference

inference = SageMakerCreditScoreInference()

st.set_page_config(page_title="Credit Score Prediction", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
h1 {
    font-size: 34px !important;
}
h2, h3 {
    margin-top: 1rem;
}
.stButton > button {
    height: 3rem;
    font-weight: 600;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


st.title("Credit Score Prediction")
st.write("Input customer financial and credit information to predict the credit score category.")

st.divider()


st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    month = st.selectbox(
        "Month",
        ["January", "February", "March", "April", "May", "June", "July", "August"]
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        step=1
    )

with col2:
    occupation = st.selectbox(
        "Occupation",
        [
            "Accountant", "Architect", "Developer", "Doctor", "Engineer",
            "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
            "Media_Manager", "Musician", "Scientist", "Teacher", "Writer", "Unknown"
        ]
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col3:
    monthly_salary = st.number_input(
        "Monthly Inhand Salary",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    monthly_balance = st.number_input(
        "Monthly Balance",
        min_value=0.0,
        value=400.0,
        step=50.0
    )

    credit_history = st.number_input(
        "Credit History Age (Months)",
        min_value=0.0,
        value=200.0,
        step=1.0
    )


st.divider()
st.subheader("Credit Account Information")

col1, col2, col3 = st.columns(3)

with col1:
    num_bank = st.number_input(
        "Number of Bank Accounts",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )

    num_card = st.number_input(
        "Number of Credit Cards",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )

with col2:
    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

    num_loan = st.number_input(
        "Number of Loan",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

with col3:
    credit_inquiry = st.number_input(
        "Number of Credit Inquiries",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    credit_mix = st.selectbox(
        "Credit Mix",
        ["Good", "Standard", "Bad", "Unknown"]
    )

    loan_type = st.selectbox(
        "Type of Loan",
        [
            "Personal Loan",
            "Credit-Builder Loan",
            "Debt Consolidation Loan",
            "Auto Loan",
            "Student Loan",
            "Mortgage Loan",
            "Payday Loan",
            "Home Equity Loan",
            "Unknown"
        ]
    )


st.divider()
st.subheader("Payment Information")

col1, col2, col3 = st.columns(3)

with col1:
    delay_due = st.number_input(
        "Delay from Due Date",
        min_value=-5,
        value=10,
        step=1
    )

    delayed_payment = st.number_input(
        "Number of Delayed Payment",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

    min_payment = st.selectbox(
        "Payment of Minimum Amount",
        ["Yes", "No", "Unknown"]
    )

with col2:
    changed_limit = st.number_input(
        "Changed Credit Limit",
        value=9.0,
        step=1.0
    )

    outstanding_debt = st.number_input(
        "Outstanding Debt",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    utilization = st.number_input(
        "Credit Utilization Ratio",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

with col3:
    total_emi = st.number_input(
        "Total EMI per Month",
        min_value=0.0,
        value=100.0,
        step=50.0
    )

    amount_invested = st.number_input(
        "Amount Invested Monthly",
        min_value=0.0,
        value=100.0,
        step=50.0
    )

    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "High_spent_Large_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Small_value_payments",
            "Unknown"
        ]
    )


st.divider()

predict = st.button("Predict Credit Score", type="primary")

if predict:
    input_data = {
        "Month": month,
        "Age": float(age),
        "Occupation": occupation,
        "Annual_Income": float(annual_income),
        "Monthly_Inhand_Salary": float(monthly_salary),
        "Num_Bank_Accounts": float(num_bank),
        "Num_Credit_Card": float(num_card),
        "Interest_Rate": float(interest_rate),
        "Num_of_Loan": float(num_loan),
        "Type_of_Loan": loan_type,
        "Delay_from_due_date": int(delay_due),
        "Num_of_Delayed_Payment": float(delayed_payment),
        "Changed_Credit_Limit": float(changed_limit),
        "Num_Credit_Inquiries": float(credit_inquiry),
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": float(outstanding_debt),
        "Credit_Utilization_Ratio": float(utilization),
        "Payment_of_Min_Amount": min_payment,
        "Total_EMI_per_month": float(total_emi),
        "Amount_invested_monthly": float(amount_invested),
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": float(monthly_balance),
        "Credit_History_Age_Months": float(credit_history)
    }

    prediction, prob_df = inference.predict(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted Credit Score: {prediction}")

    st.subheader("Prediction Probability")
    st.dataframe(prob_df, use_container_width=True)