
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="NexGuard Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# 1. Generate sample transactions
# -----------------------------
np.random.seed(42)

normal = pd.DataFrame({
    "amount": np.random.uniform(10, 5000, 200),
    "transactions_per_hour": np.random.randint(1, 4, 200),
    "distance_from_usual_km": np.random.uniform(0, 30, 200)
})

unusual = pd.DataFrame({
    "amount": np.random.uniform(10000, 50000, 10),
    "transactions_per_hour": np.random.randint(8, 15, 10),
    "distance_from_usual_km": np.random.uniform(100, 1000, 10)
})

transactions = pd.concat(
    [normal, unusual],
    ignore_index=True
)

# -----------------------------
# 2. Train AI model
# -----------------------------
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(transactions)

transactions["AI_Result"] = model.predict(transactions)

transactions["Status"] = transactions["AI_Result"].map({
    1: "Normal",
    -1: "Suspicious"
})

# -----------------------------
# 3. Dashboard heading
# -----------------------------
st.title("🛡️ NEX GUARD")
st.subheader("AI-Powered Banking Fraud Detection System")

st.write(
    "Monitor financial transactions and identify "
    "unusual activity using machine learning."
)

# -----------------------------
# 4. Dashboard metrics
# -----------------------------
total = len(transactions)
suspicious_count = (transactions["Status"] == "Suspicious").sum()
normal_count = (transactions["Status"] == "Normal").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total)
col2.metric("Suspicious Transactions", suspicious_count)
col3.metric("Normal Transactions", normal_count)

# -----------------------------
# 5. Transaction analysis graph
# -----------------------------
st.divider()
st.subheader("Transaction Analysis")

counts = transactions["Status"].value_counts()

fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)

ax.set_xlabel("Transaction Status")
ax.set_ylabel("Number of Transactions")
ax.set_title("Normal vs Suspicious Transactions")

st.pyplot(fig)
plt.close(fig)

# -----------------------------
# 6. INTERACTIVE TRANSACTION TEST
# -----------------------------
st.divider()
st.header("🧪 Test a Transaction")

st.write(
    "Enter transaction details below to see how "
    "the AI model classifies a new transaction."
)

with st.form("transaction_test"):

    amount = st.number_input(
        "Transaction Amount (KSh)",
        min_value=1.0,
        value=2000.0,
        step=100.0
    )

    frequency = st.number_input(
        "Transactions per Hour",
        min_value=1,
        value=2,
        step=1
    )

    distance = st.number_input(
        "Distance from Usual Location (km)",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    test_button = st.form_submit_button(
        "Analyse Transaction"
    )

if test_button:

    new_transaction = pd.DataFrame({
        "amount": [amount],
        "transactions_per_hour": [frequency],
        "distance_from_usual_km": [distance]
    })

    prediction = model.predict(new_transaction)[0]

    if prediction == -1:
        st.error(
            "⚠️ SUSPICIOUS TRANSACTION: "
            "The AI has identified unusual activity."
        )
        st.write(
            "This transaction should be reviewed "
            "before any decision is made."
        )

    else:
        st.success(
            "✅ NORMAL TRANSACTION: "
            "The AI has not identified unusual activity."
        )

    st.write("Transaction details:")
    st.dataframe(new_transaction, use_container_width=True)

# -----------------------------
# 7. Suspicious transactions
# -----------------------------
st.divider()
st.subheader("Suspicious Transactions")

suspicious = transactions[
    transactions["Status"] == "Suspicious"
]

st.dataframe(
    suspicious,
    use_container_width=True
)

# -----------------------------
# 8. Download results
# -----------------------------
csv = transactions.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Transaction Results",
    data=csv,
    file_name="NexGuard_Results.csv",
    mime="text/csv"
)

# -----------------------------
# 9. Disclaimer
# -----------------------------
st.divider()

st.caption(
    "NexGuard is an educational prototype using "
    "simulated transaction data. An anomaly does not "
    "necessarily mean fraud. Real transactions require "
    "further investigation."
)
