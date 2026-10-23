from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "AI-Data.csv"
FEATURES = [
    "raisedhands",
    "VisITedResources",
    "Discussion",
    "StudentAbsenceDays",
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(data):
    model_data = data[FEATURES + ["Class"]].copy()

    model_data["StudentAbsenceDays"] = model_data[
        "StudentAbsenceDays"
    ].map({"Under-7": 1, "Above-7": 0})

    X = model_data[FEATURES]
    y = model_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


data = load_data()
model, accuracy = train_model(data)

st.title("Student Performance Prediction")
st.write(
    "Explore student-performance factors and predict performance "
    "using a Random Forest classifier."
)

col1, col2, col3 = st.columns(3)
col1.metric("Students", len(data))
col2.metric("Features", len(data.columns) - 1)
col3.metric("Model Accuracy", f"{accuracy:.1%}")

overview_tab, prediction_tab = st.tabs(
    ["Data Analysis", "Performance Prediction"]
)

with overview_tab:
    left, right = st.columns(2)

    with left:
        st.subheader("Performance Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(
            data=data,
            x="Class",
            order=["L", "M", "H"],
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Performance by Absence Level")
        fig, ax = plt.subplots()
        sns.countplot(
            data=data,
            x="StudentAbsenceDays",
            hue="Class",
            hue_order=["L", "M", "H"],
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data.corr(numeric_only=True),
        cmap="coolwarm",
        annot=True,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View dataset"):
        st.dataframe(data, use_container_width=True)

with prediction_tab:
    st.subheader("Enter Student Information")

    raised_hands = st.slider("Raised Hands", 0, 100, 30)
    resources = st.slider("Visited Resources", 0, 100, 40)
    discussions = st.slider("Discussion Participation", 0, 100, 30)
    absence = st.selectbox(
        "Student Absence Days",
        ["Under-7", "Above-7"],
    )

    if st.button("Predict Performance", type="primary"):
        input_data = pd.DataFrame(
            {
                "raisedhands": [raised_hands],
                "VisITedResources": [resources],
                "Discussion": [discussions],
                "StudentAbsenceDays": [
                    1 if absence == "Under-7" else 0
                ],
            }
        )

        prediction = model.predict(input_data)[0]

        descriptions = {
            "H": "High Performance",
            "M": "Medium Performance",
            "L": "Low Performance",
        }

        st.success(
            f"Predicted Class: {prediction} — "
            f"{descriptions[prediction]}"
        )