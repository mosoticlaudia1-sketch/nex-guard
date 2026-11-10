
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# --------------------------------
# NEX GUARD - AI FRAUD DETECTION
# Interactive educational prototype
# --------------------------------

st.set_page_config(
    page_title="NEX GUARD",
    page_icon="🛡️",
    layout="wide"
)

FEATURES = ["amount", "transactions_per_hour", "distance_from_usual_km"]


@st.cache_data
def create_sample_data():
    """Create simulated transactions for the dashboard and model demo."""
    rng = np.random.default_rng(42)

    normal = pd.DataFrame({
        "amount": rng.normal(1500, 500, 200).clip(100),
        "transactions_per_hour": rng.poisson(1, 200),
        "distance_from_usual_km": rng.normal(5, 3, 200).clip(0)
    })

    unusual = pd.DataFrame({
        "amount": rng.uniform(20000, 100000, 10),
        "transactions_per_hour": rng.integers(8, 20, 10),
        "distance_from_usual_km": rng.uniform(100, 1000, 10)
    })

    data = pd.concat([normal, unusual], ignore_index=True)
    data.insert(0, "Transaction_ID", [f"TXN-{i:03d}" for i in range(1, len(data) + 1)])
    return data


@st.cache_resource
def train_model():
    """Train the demo anomaly detector on the simulated transaction set."""
    training_data = create_sample_data()
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(training_data[FEATURES])
    return model


transactions = create_sample_data().copy()
model = train_model()

# Score the sample transactions for the dashboard.
transactions["AI_Result"] = model.predict(transactions[FEATURES])
transactions["Status"] = transactions["AI_Result"].map({1: "Normal", -1: "Suspicious"})
transactions["AI_Score"] = model.decision_function(transactions[FEATURES])

# Store manually tested transactions for this browser session.
if "tested_transactions" not in st.session_state:
    st.session_state.tested_transactions = []

st.title("🛡️ NEX GUARD")
st.subheader("AI-Powered Banking Fraud Detection System")
st.write(
    "Monitor financial transactions and identify unusual activity using "
    "machine learning. You can also enter a transaction below and test it."
)
st.caption("Prototype note: this version uses simulated training data and is for demonstration, not real banking decisions.")
st.divider()

# Dashboard metrics
sample_total = len(transactions)
sample_suspicious = int((transactions["Status"] == "Suspicious").sum())
sample_normal = int((transactions["Status"] == "Normal").sum())

col1, col2, col3 = st.columns(3)
col1.metric("Sample Transactions", sample_total)
col2.metric("Sample Flagged", sample_suspicious)
col3.metric("Sample Normal", sample_normal)

st.divider()

# Interactive transaction test
st.header("🧪 Test a Transaction")
st.write("Enter the transaction details, then select **Check Transaction**.")

with st.form("transaction_test_form"):
    amount = st.number_input(
        "Transaction amount (KSh)",
        min_value=1.0,
        max_value=10000000.0,
        value=1500.0,
        step=100.0
    )
    frequency = st.number_input(
        "Number of transactions in the last hour",
        min_value=0,
        max_value=500,
        value=1,
        step=1
    )
    distance = st.number_input(
        "Distance from the customer's usual location (km)",
        min_value=0.0,
        max_value=20000.0,
        value=5.0,
        step=1.0
    )
    submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)

if submitted:
    new_transaction = pd.DataFrame([{
        "amount": amount,
        "transactions_per_hour": frequency,
        "distance_from_usual_km": distance
    }])
    prediction = int(model.predict(new_transaction[FEATURES])[0])
    score = float(model.decision_function(new_transaction[FEATURES])[0])
    status = "Suspicious" if prediction == -1 else "Normal"
    transaction_id = f"TEST-{len(st.session_state.tested_transactions) + 1:03d}"

    result = {
        "Transaction_ID": transaction_id,
        "amount": float(amount),
        "transactions_per_hour": int(frequency),
        "distance_from_usual_km": float(distance),
        "Status": status,
        "AI_Score": score
    }
    st.session_state.tested_transactions.insert(0, result)

    if status == "Suspicious":
        st.error("⚠️ NexGuard has flagged this transaction as suspicious.")
    else:
        st.success("✅ NexGuard has classified this transaction as normal.")

    st.metric("AI anomaly score", f"{score:.4f}")
    st.caption(
        "The anomaly score is a model score, not a fraud probability. "
        "Lower scores indicate more unusual patterns for this model."
    )
    st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)

# Previously tested transactions
st.subheader("Your Test History")
if st.session_state.tested_transactions:
    tested_df = pd.DataFrame(st.session_state.tested_transactions)
    st.dataframe(tested_df, use_container_width=True, hide_index=True)
    test_csv = tested_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Test History (CSV)",
        data=test_csv,
        file_name="NexGuard_Test_History.csv",
        mime="text/csv"
    )
    if st.button("Clear Test History"):
        st.session_state.tested_transactions = []
        st.rerun()
else:
    st.info("No transactions tested yet. Enter details above to begin.")

st.divider()

# Sample transaction analysis
st.header("📊 Sample Transaction Analysis")
chart_data = transactions["Status"].value_counts()
st.bar_chart(chart_data)

st.subheader("⚠️ Sample Suspicious Transaction Alerts")
suspicious = transactions[transactions["Status"] == "Suspicious"]
st.dataframe(suspicious, use_container_width=True, hide_index=True)

st.subheader("All Sample Transaction Records")
status_filter = st.selectbox("Filter sample transactions", ["All", "Normal", "Suspicious"])
if status_filter == "All":
    filtered = transactions
else:
    filtered = transactions[transactions["Status"] == status_filter]

st.dataframe(filtered, use_container_width=True, hide_index=True)

sample_csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Sample Transaction Report",
    data=sample_csv,
    file_name="NexGuard_Sample_Fraud_Report.csv",
    mime="text/csv"
)

st.divider()
st.info(
    "Important: NexGuard is an educational prototype using simulated transactions. "
    "A suspicious flag means the transaction has an unusual pattern; it does not prove fraud. "
    "Do not use this demo to approve, reject, or block real financial transactions."
)
st.caption("NEX GUARD | Group 7 | IST 1010F")
0
