import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer 360 Dashboard",  
    page_icon="%%",
    layout="wide"
)

rf_model = pickle.load(open("segmentation_model.pkl", "rb"))
intent_model = pickle.load(open("intent_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
le_type = pickle.load(open("le_type.pkl", "rb"))
le_channel = pickle.load(open("le_channel.pkl", "rb"))
le_occupation = pickle.load(open("le_occupation.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv(r"C:\Users\USER\Documents\bank_transaction_data.csv")

st.sidebar.title("Customer 360")
page = st.sidebar.radio("Navigate",[
    "Overview",
    "Predict Segment",
    "Predict Intent"
])

if page == "Overview":
    st.title("Customer 360 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df["AccountID"].unique()))
    col2.metric("Total Transactions", len(df))
    col3.metric("Average Transactions", f"${df['TransactionAmount'].mean():.2f}")

    st.subheader("Transaction Channel Distribution")
    fig, ax = plt.subplots()
    df["Channel"].value_counts().plot(kind="bar", ax=ax, color=["steelblue","orange", "blue"])
    st.pyplot(fig)


elif page == "Predict Segment":
    st.title("Predict Customer Segment")

    recency = st.slider("Recency (days since last transaction)", 1, 365, 30)
    frequency = st.slider("Transaction Frequency", 1, 50, 10)
    monetary = st.number_input("Total Transaction Amount", 100, 100000, 5000)

    if st.button("Predict Segment"):
        input_data = scaler.transform([[recency, frequency, monetary]])
        pred = rf_model.predict(input_data)[0]
        segment = le.inverse_transform([pred])[0]

        colors = {
            "Champion": " ",
            "Loyal Customer": " ",
            "At Risk": " ",
            "Potential Loyalist": " "
        }
        st.success(f"{colors.get(segment, ' ')} **Segment: {segment}**")

elif page == " Predict Intent":
    st.title("Predict Customer Intent")

    col1, col2 = st.columns(2)
    with col1:
        trans_type = st.selectbox("Transaction Type", ["Debit", "Credit"])
        channel = st.selectbox("Channel", ["ATM", "Online", "Branch"])
        occupation = st.selectbox("Occupation", df["CustomerOccupation"].unique())

    with col2:
        amount = st.number_input("Transaction Amount", 10, 50000, 500)
        balance = st.number_input("Account Balance", 100, 100000, 5000)
        age = st.slider("Customer Age", 18, 80, 35)
        login_attempts = st.slider("Login Attempts", 1, 10, 1)
        duration = st.slider("Transaction Duration(mins)", 1, 60, 5)

    
    if st.button("Predict Intent"):
        type_enc = le_type.transform([trans_type])[0]
        channel_enc = le_channel.transform([channel])[0]
        occ_enc = le_occupation.transform([occupation])[0]

        features = [[amount, balance, type_enc,
                     channel_enc, occ_enc, age,
                     login_attempts, duration]]
        
        intent = intent_model.predict(features)[0]
        st.success(f"** Predicted Intent: {intent}**")
    