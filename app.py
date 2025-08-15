import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Page Setup ---
st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")

# --- Custom Dark Theme ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #3399ff !important;
        color: black !important;
    }
    .stDataFrame, .stTable {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    .stCodeBlock {
        background-color: #1f1f1f !important;
        color: #eee !important;
    }
    .stExpander {
        background-color: #141414 !important;
        color: white !important;
    }
    .stMetric label, .stMetric div {
        color: white !important;
    }
    hr {
        border: 1px solid #444;
    }
    table {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("📊 Bank Marketing Prediction Dashboard")

# --- Load CSV ---
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv", sep=';')

df = load_data()
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --- Preprocess ---
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# --- Display Accuracy ---
st.markdown(f"### ✅ Accuracy: <span style='color:#00ffcc'>{acc * 100:.2f}%</span>", unsafe_allow_html=True)

# --- Display Classification Report ---
with st.expander("📋 Classification Report"):
    st.code(report, language="text")

# --- Feature Importances ---
importances = model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# --- Plot ---
fig = px.bar(feature_df.head(10),
             x='Importance', y='Feature',
             orientation='h',
             title="🌟 Top 10 Feature Importances",
             color='Importance',
             color_continuous_scale='plasma',
             template='plotly_dark')

st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🔁 Model Used: <strong style='color:#00ffaa'>Decision Tree Classifier (max_depth=5)</strong>", unsafe_allow_html=True)
