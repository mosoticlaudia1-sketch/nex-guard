
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --------------------------------
# NEX GUARD - AI FRAUD DETECTION
# --------------------------------

st.set_page_config(
    page_title="NEX GUARD",
    page_icon="🛡️",
    layout="wide"
)

# Generate sample transaction data
@st.cache_data
def create_data():
    np.random.seed(42)

    normal = pd.DataFrame({
        "amount": np.random.normal(1500, 500, 200).clip(100),
        "transactions_per_hour": np.random.poisson(1, 200),
        "distance_from_usual_km": np.random.normal(5, 3, 200).clip(0)
    })

    unusual = pd.DataFrame({
        "amount": np.random.uniform(20000, 100000, 10),
        "transactions_per_hour": np.random.randint(8, 20, 10),
        "distance_from_usual_km": np.random.uniform(100, 1000, 10)
    })

    data = pd.concat(
        [normal, unusual],
        ignore_index=True
    )

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    model.fit(data)

    data["AI_Result"] = model.predict(data)

    data["Status"] = data["AI_Result"].map({
        1: "Normal",
        -1: "Suspicious"
    })

    data.insert(
        0,
        "Transaction_ID",
        [f"TXN-{i:03d}" for i in range(1, len(data) + 1)]
    )

    data["AI_Score"] = model.decision_function(
        data[["amount", "transactions_per_hour",
              "distance_from_usual_km"]]
    )

    return data

transactions = create_data()

# Dashboard heading
st.title("🛡️ NEX GUARD")
st.subheader("AI-Powered Banking Fraud Detection System")

st.write(
    "Monitor financial transactions and identify "
    "unusual activity using machine learning."
)

st.divider()

# Calculate statistics
total = len(transactions)
suspicious_count = (transactions["Status"] == "Suspicious").sum()
normal_count = (transactions["Status"] == "Normal").sum()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total)
col2.metric("Suspicious Transactions", suspicious_count)
col3.metric("Normal Transactions", normal_count)

st.divider()

# Transaction chart
st.subheader("Transaction Analysis")

chart_data = transactions["Status"].value_counts()

st.bar_chart(chart_data)

# Suspicious transactions
st.subheader("⚠️ Suspicious Transaction Alerts")

suspicious = transactions[
    transactions["Status"] == "Suspicious"
]

st.dataframe(
    suspicious,
    use_container_width=True,
    hide_index=True
)

# Search and filter
st.subheader("All Transaction Records")

status_filter = st.selectbox(
    "Filter transactions",
    ["All", "Normal", "Suspicious"]
)

if status_filter == "All":
    filtered = transactions
else:
    filtered = transactions[
        transactions["Status"] == status_filter
    ]

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

# Download report
csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Transaction Report",
    data=csv,
    file_name="NexGuard_Fraud_Report.csv",
    mime="text/csv"
)

st.divider()

st.info(
    "Note: This is an educational prototype using "
    "simulated transactions. Suspicious flags indicate "
    "unusual patterns and do not confirm actual fraud."
)

st.caption("NEX GUARD | Group 7 | IST 1010F")
