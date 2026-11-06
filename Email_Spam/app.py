from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Email Spam Detection")
st.write("A machine learning application using Logistic Regression.")
st.caption("Predict whether an email is spam or not spam using email features.")

# ------------------------------------------------------------
# 2. PROJECT CONFIGURATION
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "email_spam.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 3. LOAD AND CLEAN DATA
# ------------------------------------------------------------
@st.cache_data
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    data = data.dropna()
    return data

if not DATASET_PATH.exists():
    st.error(
        "Dataset not found. Put **email_spam.csv** in the same folder as app.py."
    )
    st.code("Email_Spam/\\n  app.py\\n  email_spam.csv")
    st.stop()

df = load_dataset(str(DATASET_PATH))

required_columns = [
    "Message_ID",
    "Email_Length",
    "Word_Count",
    "Link_Count",
    "Has_Attachment",
    "Spam",
]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"These required columns are missing from the CSV: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The dataset has no usable rows after removing missing and duplicate records.")
    st.stop()

# The supplied dataset uses 0 = Not Spam and 1 = Spam.
if not set(df["Spam"].unique()).issubset({0, 1}):
    st.error("The Spam column must contain only 0 (Not Spam) and 1 (Spam).")
    st.stop()

# ------------------------------------------------------------
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------
@st.cache_resource
def train_model(data):
    feature_columns = [
        "Email_Length",
        "Word_Count",
        "Link_Count",
        "Has_Attachment",
    ]

    X = data[feature_columns]
    y = data["Spam"].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("The dataset needs at least two examples in each target class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_probability),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        output_dict=True,
        zero_division=0,
    )

    # Save the trained model beside this app.py file.
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return model, X_test, y_test, y_pred, y_probability, scores, cm, report

try:
    (
        model,
        X_test,
        y_test,
        y_pred,
        y_probability,
        scores,
        cm,
        report,
    ) = train_model(df)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Project Information")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write(f"**Dataset:** {DATASET_PATH.name}")
st.sidebar.write(f"**Usable records:** {len(df)}")
st.sidebar.write("**Target:** Spam")
st.sidebar.write("**0:** Not Spam")
st.sidebar.write("**1:** Spam")

# ------------------------------------------------------------
# 6. DATASET OVERVIEW
# ------------------------------------------------------------
st.header("1. Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Emails", f"{len(df):,}")
c2.metric("Spam Emails", f"{int((df['Spam'] == 1).sum()):,}")
c3.metric("Not Spam Emails", f"{int((df['Spam'] == 0).sum()):,}")

with st.expander("View dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("View data quality information"):
    quality = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Data Type": df.dtypes.astype(str),
    })
    st.dataframe(quality, use_container_width=True)

# ------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
st.header("2. Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Spam vs Not Spam")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Spam", ax=ax)
    ax.set_title("Spam vs Not Spam Emails")
    ax.set_xlabel("Email Type (0 = Not Spam, 1 = Spam)")
    ax.set_ylabel("Number of Emails")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Email Length Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Email_Length", hue="Spam", kde=True, bins=15, ax=ax)
    ax.set_title("Email Length by Email Type")
    ax.set_xlabel("Email Length")
    ax.set_ylabel("Number of Emails")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.subheader("Links in Emails")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Link_Count", hue="Spam", ax=ax)
    ax.set_title("Links in Spam and Not Spam Emails")
    ax.set_xlabel("Number of Links")
    ax.set_ylabel("Number of Emails")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Feature Correlation")
    numeric_data = df.drop(columns=["Message_ID"], errors="ignore")
    correlation = numeric_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------
# 8. MODEL PERFORMANCE
# ------------------------------------------------------------
st.header("3. Model Performance")

metric_columns = st.columns(5)
for column, (name, value) in zip(metric_columns, scores.items()):
    column.metric(name, f"{value:.2%}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Spam", "Spam"],
        yticklabels=["Not Spam", "Spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Performance Metrics")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [scores[name] for name in metric_names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metric_names, metric_values)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Logistic Regression Performance")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
        )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("View classification report"):
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

# ------------------------------------------------------------
# 9. FEATURE COEFFICIENTS
# ------------------------------------------------------------
st.header("4. Feature Coefficients")

feature_names = [
    "Email_Length",
    "Word_Count",
    "Link_Count",
    "Has_Attachment",
]
feature_coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0],
})
feature_coefficients["Absolute Importance"] = feature_coefficients["Coefficient"].abs()
feature_coefficients = feature_coefficients.sort_values(
    by="Absolute Importance",
    ascending=False,
)

st.dataframe(feature_coefficients.round(4), use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    data=feature_coefficients,
    x="Coefficient",
    y="Feature",
    ax=ax,
)
ax.set_title("Logistic Regression Feature Coefficients")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------
# 10. ROC CURVE
# ------------------------------------------------------------
st.header("5. ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_probability)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, label=f"ROC AUC = {scores['ROC AUC']:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------
# 11. INTERACTIVE EMAIL PREDICTION
# ------------------------------------------------------------
st.header("6. Try Your Own Email")
st.write("Enter the email details below and click **Predict Email**.")

with st.form("email_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        email_length = st.number_input(
            "Email Length",
            min_value=0,
            max_value=100000,
            value=80,
            step=1,
        )
        word_count = st.number_input(
            "Word Count",
            min_value=0,
            max_value=10000,
            value=12,
            step=1,
        )

    with col2:
        link_count = st.number_input(
            "Link Count",
            min_value=0,
            max_value=1000,
            value=2,
            step=1,
        )
        has_attachment = st.selectbox(
            "Has Attachment?",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

    submitted = st.form_submit_button("Predict Email", type="primary")

if submitted:
    sample_email = pd.DataFrame({
        "Email_Length": [email_length],
        "Word_Count": [word_count],
        "Link_Count": [link_count],
        "Has_Attachment": [has_attachment],
    })

    prediction = int(model.predict(sample_email)[0])
    probabilities = model.predict_proba(sample_email)[0]

    not_spam_probability = probabilities[0]
    spam_probability = probabilities[1]

    if prediction == 1:
        st.error("🚨 Prediction: SPAM EMAIL")
    else:
        st.success("✅ Prediction: NOT SPAM EMAIL")

    p1, p2 = st.columns(2)
    p1.metric("Not Spam Probability", f"{not_spam_probability:.2%}")
    p2.metric("Spam Probability", f"{spam_probability:.2%}")

# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------
st.divider()
st.subheader("Project Summary")
st.write(
    "This application uses Logistic Regression to classify emails as spam or "
    "not spam. The model is evaluated using accuracy, precision, recall, F1 "
    "score, a confusion matrix, and ROC-AUC."
)
st.caption(f"Trained model saved as: {MODEL_PATH.name}")
