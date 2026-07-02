import streamlit as st
import pandas as pd
from cloud_inference import SageMakerCreditScoreInference


st.set_page_config(page_title="Credit Score Prediction", layout="wide")

inference = SageMakerCreditScoreInference()

st.title("Credit Score Prediction")
st.write("This Streamlit application runs on EC2 and invokes a deployed SageMaker endpoint for prediction.")

test_cases = {
    "Good": {
        "Month": "January",
        "Age": 42.0,
        "Occupation": "Teacher",
        "Annual_Income": 19214.965,
        "Monthly_Inhand_Salary": 1730.247083,
        "Num_Bank_Accounts": 0.0,
        "Num_Credit_Card": 4.0,
        "Interest_Rate": 11.0,
        "Num_of_Loan": 0.0,
        "Type_of_Loan": "Unknown",
        "Delay_from_due_date": 10,
        "Num_of_Delayed_Payment": 10.0,
        "Changed_Credit_Limit": 4.18,
        "Num_Credit_Inquiries": 0.0,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 498.81,
        "Credit_Utilization_Ratio": 37.600265,
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 0.0,
        "Amount_invested_monthly": 217.780472,
        "Payment_Behaviour": "Unknown",
        "Monthly_Balance": 245.244236,
        "Credit_History_Age_Months": 345.0
    },
    "Standard": {
        "Month": "March",
        "Age": 35.0,
        "Occupation": "Engineer",
        "Annual_Income": 45000.0,
        "Monthly_Inhand_Salary": 3500.0,
        "Num_Bank_Accounts": 5.0,
        "Num_Credit_Card": 5.0,
        "Interest_Rate": 14.0,
        "Num_of_Loan": 3.0,
        "Type_of_Loan": "Personal Loan",
        "Delay_from_due_date": 18,
        "Num_of_Delayed_Payment": 8.0,
        "Changed_Credit_Limit": 10.0,
        "Num_Credit_Inquiries": 4.0,
        "Credit_Mix": "Standard",
        "Outstanding_Debt": 1200.0,
        "Credit_Utilization_Ratio": 35.0,
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 120.0,
        "Amount_invested_monthly": 100.0,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 350.0,
        "Credit_History_Age_Months": 180.0
    },
    "Poor": {
        "Month": "July",
        "Age": 25.0,
        "Occupation": "Unknown",
        "Annual_Income": 15000.0,
        "Monthly_Inhand_Salary": 1200.0,
        "Num_Bank_Accounts": 9.0,
        "Num_Credit_Card": 9.0,
        "Interest_Rate": 32.0,
        "Num_of_Loan": 8.0,
        "Type_of_Loan": "Payday Loan",
        "Delay_from_due_date": 55,
        "Num_of_Delayed_Payment": 22.0,
        "Changed_Credit_Limit": 25.0,
        "Num_Credit_Inquiries": 12.0,
        "Credit_Mix": "Bad",
        "Outstanding_Debt": 4500.0,
        "Credit_Utilization_Ratio": 75.0,
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 350.0,
        "Amount_invested_monthly": 10.0,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 80.0,
        "Credit_History_Age_Months": 50.0
    }
}

selected_case = st.selectbox("Select test case", ["Good", "Standard", "Poor"])

input_data = test_cases[selected_case]

st.subheader("Input Data")
st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

if st.button("Predict Credit Score", type="primary"):
    try:
        prediction, probability_df = inference.predict(input_data)

        st.subheader("Prediction Result")
        st.success(f"Predicted Credit Score: {prediction}")

        st.subheader("Prediction Probability")
        st.dataframe(probability_df, use_container_width=True)

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
