import streamlit as st
import pandas as pd
import numpy as np
import joblib


model=joblib.load('pipeline.pkl')

st.title("Fraud Detection Prediction APP")

st.markdown("Please enter the transaction details and click on the Predict button to see if the transaction is fraudulent or not.")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "TRANSFER", "DEPOSITE"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=0.01)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=0.01)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)


if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("The transaction is predicted to be Fraudulent.")
    else:
        st.success("The transaction is predicted to be Non-Fraudulent.")

    st.write(f"Prediction Probability: Non-Fraudulent: {prediction_proba[0][0]:.4f}, Fraudulent: {prediction_proba[0][1]:.4f}")