"""Streamlit dashboard for credit card fraud detection.

Uses the modular ``app.ml`` package for model loading, prediction, and
metrics. All displayed metrics come from ``artifacts/metrics.json`` —
no hardcoded values.
"""

import warnings
from datetime import datetime

import joblib  # noqa: F401  (kept for backward compat if user loads old pkl)
import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import DATA_PATH, SAMPLE_DATA_PATH
from app.ml.predict import (
    ModelBundle,
    get_feature_importances,
    get_model_info,
    load_bundle,
    predict_transaction,
)
from app.ml.schema import FEATURE_COLUMNS

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .legit-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> ModelBundle | None:
    """Load the trained model bundle from artifacts/."""
    try:
        return load_bundle()
    except FileNotFoundError as exc:
        st.error(f"Model artifacts not found: {exc}")
        return None


@st.cache_data
def load_sample_data() -> tuple[pd.DataFrame | None, str | None]:
    """Load the dataset, falling back to the demonstration sample.

    Tries ``creditcard.csv`` first, then ``creditcard_sample.csv``.
    Returns ``(df, source)`` where source is ``"full"``, ``"sample"``,
    or ``None`` when neither file is available (e.g. minimal deployments).
    """
    for path, source in ((DATA_PATH, "full"), (SAMPLE_DATA_PATH, "sample")):
        try:
            return pd.read_csv(path), source
        except FileNotFoundError:
            continue
    return None, None


def get_ai_explanation(result: dict, features: dict, feature_names: list[str],
                       importances: dict[str, float] | None) -> str:
    """Generate an explanation for the prediction using real feature importances."""
    fraud_prob = result["fraud_probability"] * 100

    top_features = []
    if importances:
        for name in feature_names:
            if name in importances:
                contribution = importances[name] * abs(features.get(name, 0.0))
                top_features.append((name, features.get(name, 0.0), contribution))
        top_features.sort(key=lambda x: x[2], reverse=True)
        top_3 = top_features[:3]
    else:
        top_3 = [(name, features.get(name, 0.0), 0.0) for name in feature_names[:3]]

    if result["predicted_class"] == 1:
        explanation = f"🚨 **FRAUD DETECTED** (Confidence: {fraud_prob:.1f}%)\n\n"
        explanation += "**Key Risk Indicators:**\n"
        for feature, value, _ in top_3:
            explanation += f"• **{feature}**: {value:.3f}\n"
        explanation += (
            f"\n**Recommendation**: Block this transaction immediately and flag "
            f"the account for review. (Threshold: {result['threshold']:.2f})"
        )
    else:
        explanation = f"✅ **LEGITIMATE TRANSACTION** (Confidence: {100 - fraud_prob:.1f}%)\n\n"
        explanation += "**Transaction appears normal based on:**\n"
        for feature, value, _ in top_3:
            explanation += f"• **{feature}**: {value:.3f}\n"
        explanation += (
            f"\n**Recommendation**: Process transaction normally. "
            f"(Threshold: {result['threshold']:.2f})"
        )
    return explanation


def get_ai_insights(df: pd.DataFrame) -> str:
    """Generate insights about the dataset."""
    try:
        fraud_rate = (df["Class"].sum() / len(df)) * 100
        avg_fraud_amount = df[df["Class"] == 1]["Amount"].mean()
        avg_legit_amount = df[df["Class"] == 0]["Amount"].mean()

        insights = f"""
        🤖 **AI-Generated Insights:**

        **📊 Fraud Pattern Analysis:**
        • Overall fraud rate: {fraud_rate:.2f}% (Industry average: 0.1-0.2%)
        • Average fraudulent transaction: ${avg_fraud_amount:.2f}
        • Average legitimate transaction: ${avg_legit_amount:.2f}
        • Fraud amount ratio: {avg_fraud_amount / avg_legit_amount:.1f}x higher than legitimate transactions

        **🎯 Risk Assessment:**
        • This dataset shows typical fraud patterns
        • High-value transactions have increased fraud risk
        • Time-based patterns may indicate coordinated attacks

        **💡 Recommendations:**
        • Implement real-time monitoring for transactions >${avg_fraud_amount:.0f}
        • Set up alerts for unusual time patterns
        • Consider additional verification for high-risk transactions
        """
        return insights
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"


def ai_chatbot_response(user_question: str, model_info: dict) -> str:
    """Simple AI chatbot for fraud detection questions."""
    test_metrics = model_info.get("test_metrics", {})
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")
    f1 = test_metrics.get("f1")
    roc_auc = test_metrics.get("roc_auc")

    if precision is not None:
        metrics_text = (
            f"precision {precision:.2%}, recall {recall:.2%}, "
            f"F1 {f1:.2%}, ROC-AUC {roc_auc:.4f}"
        )
    else:
        metrics_text = "metrics unavailable (run training first)"

    responses = {
        "what is fraud": "Credit card fraud is unauthorized use of someone's credit card information to make purchases or withdraw money.",
        "how does detection work": "Our model analyzes 30 different features of each transaction and assigns a fraud probability score.",
        "what features are used": "We use anonymized features (V1-V28), transaction amount, and time since first transaction.",
        "how accurate is the model": f"Our model achieves {metrics_text} on the held-out test set.",
        "what should i do if fraud is detected": "If fraud is detected, immediately block the transaction, notify the cardholder, and flag the account for review.",
        "how can i prevent fraud": "Monitor transactions regularly, use strong passwords, enable alerts, and never share card information.",
        "what is smote": "SMOTE (Synthetic Minority Oversampling Technique) helps balance our dataset by creating synthetic fraud examples.",
        "what is random forest": "Random Forest is an ensemble of decision trees that votes on the prediction, making it robust for fraud detection.",
        "what is xgboost": "XGBoost is a powerful machine learning algorithm that uses gradient boosting to make accurate predictions.",
    }

    question_lower = user_question.lower()
    for key, response in responses.items():
        if key in question_lower:
            return f"🤖 **AI Assistant:** {response}"

    return "🤖 **AI Assistant:** I can help with questions about fraud detection, model accuracy, features, and prevention strategies. Please ask a specific question!"


def generate_ai_report(df: pd.DataFrame | None, model_info: dict,
                       data_source: str | None = None) -> str:
    """Generate an AI-powered analysis report using real metrics.

    ``df`` may be None when no dataset is deployed; dataset-dependent
    stats are then reported as unavailable rather than fabricated.
    When ``data_source`` is ``"sample"``, dataset stats are labeled as
    demonstration data.
    """
    try:
        test_metrics = model_info.get("test_metrics") or {}
        val_metrics = model_info.get("validation_metrics") or {}
        model_name = model_info.get("model_name", "unknown")

        metric_keys = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-Score"),
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
        ]

        def fmt_pct(metrics: dict, key: str) -> str:
            val = metrics.get(key)
            return f"{val * 100:.2f}%" if isinstance(val, (int, float)) else "N/A"

        if test_metrics:
            perf_lines = ["### Model Performance (Test Set — final)"]
            perf_lines += [
                f"- **{label}**: {fmt_pct(test_metrics, key)}"
                for key, label in metric_keys
            ]
            if val_metrics:
                perf_lines += [
                    "",
                    "### Validation Metrics (model selection — not final test results)",
                ]
                perf_lines += [
                    f"- **{label}**: {fmt_pct(val_metrics, key)}"
                    for key, label in metric_keys
                ]
            performance_section = "\n        ".join(perf_lines)
        else:
            performance_section = (
                "### Model Performance\n"
                "        - Metrics unavailable: `artifacts/metrics.json` is missing "
                "or incomplete. Run `python -m app.ml.training` and redeploy."
            )

        if df is not None:
            total_transactions = f"{len(df):,}"
            fraud_rate = f"{(df['Class'].sum() / len(df)) * 100:.3f}%"
            high_risk_count = f"{len(df[df['Amount'] > df['Amount'].quantile(0.95)]):,}"
            avg_fraud_amount = f"${df[df['Class'] == 1]['Amount'].mean():.2f}"
            fraud_times = df.loc[df["Class"] == 1, "Time"]
            if len(fraud_times):
                peak_hour = int(((fraud_times // 3600) % 24).mode().iloc[0])
                peak_hours = f"{peak_hour:02d}:00"
            else:
                peak_hours = "N/A"
            data_note = (
                "- **Data source**: Demonstration sample (`creditcard_sample.csv`) — "
                "all dataset statistics in this report are illustrative "
                "sample-based results, not full-dataset results\n"
                if data_source == "sample" else ""
            )
            risk_note = (
                "- **Note**: dataset-dependent stats below are from the "
                "demonstration sample, not the full dataset\n"
                if data_source == "sample" else ""
            )
        else:
            total_transactions = fraud_rate = high_risk_count = avg_fraud_amount = "N/A (dataset unavailable)"
            peak_hours = "N/A (dataset unavailable)"
            data_note = ""
            risk_note = ""

        report = f"""
        # 🤖 AI-Powered Fraud Detection Analysis Report
        *Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

        ## 📊 Executive Summary
        This AI analysis reveals critical insights about credit card fraud patterns and model performance.

        ## 🎯 Key Findings

        ### Dataset Analysis
        {data_note}- **Total Transactions**: {total_transactions}
        - **Fraud Rate**: {fraud_rate}
        - **Model**: {model_name}
        - **Decision Threshold**: {model_info.get('threshold', 'N/A')}

        {performance_section}

        ### Risk Assessment
        {risk_note}- **High-Risk Transactions**: {high_risk_count}
        - **Average Fraud Amount**: {avg_fraud_amount}
        - **Peak Fraud Hours**: {peak_hours}

        ## 💡 AI Recommendations
        1. **Real-time Monitoring**: Implement 24/7 transaction monitoring
        2. **Threshold Optimization**: Adjust fraud detection thresholds based on business risk tolerance
        3. **Feature Engineering**: Consider adding merchant category and location data
        4. **Model Retraining**: Schedule regular model updates with new data

        ## 🚨 Alert Thresholds
        - **High Risk**: probability >= {model_info.get('threshold', 0.5):.2f} (model decision threshold)
        - **Medium Risk**: 0.20 - {model_info.get('threshold', 0.5):.2f}
        - **Low Risk**: < 0.20

        ---
        *This report was generated by AI and should be reviewed by fraud analysts.*
        """
        return report
    except Exception as e:
        return f"AI report generation failed: {str(e)}"


def create_feature_inputs() -> dict[str, float]:
    """Create input widgets for all features, returning a name->value dict."""
    st.subheader("📊 Transaction Details")

    col1, col2 = st.columns(2)
    values: dict[str, float] = {}

    with col1:
        values["Time"] = st.number_input(
            "Time (seconds since first transaction)",
            min_value=0.0, max_value=172792.0, value=0.0, step=1.0,
            key="feat_Time",
            help="Time in seconds elapsed between each transaction and the first transaction in the dataset",
        )
        values["Amount"] = st.number_input(
            "Amount", min_value=0.0, max_value=25691.16, value=0.0, step=0.01,
            key="feat_Amount", help="Transaction amount",
        )
        for i in range(1, 15):
            name = f"V{i}"
            values[name] = st.number_input(name, value=0.0, step=0.01, format="%.6f", key=f"feat_{name}")

    with col2:
        for i in range(15, 29):
            name = f"V{i}"
            values[name] = st.number_input(name, value=0.0, step=0.01, format="%.6f", key=f"feat_{name}")

    # Return in schema order
    return {col: values[col] for col in FEATURE_COLUMNS}


def display_prediction_result(result: dict, features: dict, feature_names: list[str],
                              importances: dict[str, float] | None) -> None:
    """Display prediction result with styling and explanation."""
    st.subheader("🔍 Prediction Result")

    fraud_prob = result["fraud_probability"] * 100
    legit_prob = (1 - result["fraud_probability"]) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        if result["predicted_class"] == 1:
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.error("🚨 FRAUD DETECTED")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="legit-alert">', unsafe_allow_html=True)
            st.success("✅ LEGITIMATE TRANSACTION")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")

    with col3:
        st.metric("Legitimate Probability", f"{legit_prob:.2f}%")

    st.subheader("📊 Probability Breakdown")
    st.progress(fraud_prob / 100)
    st.caption(f"Fraud Probability: {fraud_prob:.2f}% | Risk: {result['risk_category'].upper()} | Model: {result['model_name']}")

    if result["risk_category"] == "high":
        st.warning("⚠️ HIGH RISK: This transaction shows strong indicators of fraud.")
    elif result["risk_category"] == "medium":
        st.info("🔶 MEDIUM RISK: This transaction requires further investigation.")
    else:
        st.success("🟢 LOW RISK: This transaction appears to be legitimate.")

    st.subheader("🤖 AI Explanation")
    st.markdown(get_ai_explanation(result, features, feature_names, importances))


def create_data_visualizations(df: pd.DataFrame) -> None:
    """Create data visualizations for the dataset."""
    st.subheader("📈 Dataset Analysis")

    class_counts = df["Class"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=class_counts.values,
            names=["Legitimate", "Fraudulent"],
            title="Transaction Class Distribution",
            color_discrete_map={"Legitimate": "#2E8B57", "Fraudulent": "#DC143C"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=["Legitimate", "Fraudulent"],
            y=class_counts.values,
            title="Transaction Count by Class",
            color=["Legitimate", "Fraudulent"],
            color_discrete_map={"Legitimate": "#2E8B57", "Fraudulent": "#DC143C"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("💰 Amount Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig_amount = px.box(
            df, x="Class", y="Amount",
            title="Amount Distribution by Class",
            color="Class",
            color_discrete_map={0: "#2E8B57", 1: "#DC143C"},
        )
        fig_amount.update_layout(xaxis_title="Class (0=Legitimate, 1=Fraudulent)")
        st.plotly_chart(fig_amount, use_container_width=True)

    with col2:
        fig_time = px.histogram(df, x="Time", nbins=50, title="Time Distribution",
                                color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🎯 Feature Importance")
    correlations = df.corr()["Class"].abs().sort_values(ascending=False)[1:11]
    fig_corr = px.bar(
        x=correlations.values, y=correlations.index, orientation="h",
        title="Top 10 Features Most Correlated with Fraud",
        color=correlations.values, color_continuous_scale="Reds",
    )
    fig_corr.update_layout(xaxis_title="Absolute Correlation with Class")
    st.plotly_chart(fig_corr, use_container_width=True)


def main() -> None:
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Fraud Detection", "📊 Data Analysis", "🤖 AI Assistant", "📋 AI Report", "ℹ️ About"],
    )

    bundle = load_model()
    df, data_source = load_sample_data()

    # The model is required for all prediction pages; the dataset is optional
    # (excluded from deployment via .gitignore on Streamlit Community Cloud).
    if bundle is None:
        st.stop()

    model_info = get_model_info(bundle)
    importances = get_feature_importances(bundle)

    if page == "🔍 Fraud Detection":
        st.markdown("### Enter transaction details to detect potential fraud")

        # Sample data buttons — placed BEFORE the form so session_state
        # can be set before the input widgets are instantiated.
        # Only shown when the dataset is available.
        if df is not None:
            st.subheader("📋 Quick Sample Data")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Load Legitimate Sample", type="secondary"):
                    sample = df[df["Class"] == 0].iloc[0]
                    for col in FEATURE_COLUMNS:
                        st.session_state[f"feat_{col}"] = float(sample[col])
                    st.rerun()

            with col2:
                if st.button("🚨 Load Fraudulent Sample", type="secondary"):
                    fraud_df = df[df["Class"] == 1]
                    if len(fraud_df) > 0:
                        sample = fraud_df.iloc[0]
                        for col in FEATURE_COLUMNS:
                            st.session_state[f"feat_{col}"] = float(sample[col])
                    st.rerun()

        with st.form("fraud_detection_form"):
            features = create_feature_inputs()
            submitted = st.form_submit_button("🔍 Detect Fraud", type="primary")

            if submitted:
                try:
                    result = predict_transaction(bundle, features)
                    display_prediction_result(result, features, FEATURE_COLUMNS, importances)
                except ValueError as exc:
                    st.error(f"Prediction failed: {exc}")

    elif page == "📊 Data Analysis":
        st.markdown("### Dataset Overview and Visualizations")

        if df is None:
            st.info(
                "No dataset file is available in this deployment "
                "(`creditcard.csv` / `creditcard_sample.csv`), so dataset "
                "analysis, insights, and visualizations are disabled. "
                "Fraud Detection, AI Assistant, AI Report, and About pages "
                "work without it."
            )
        else:
            if data_source == "sample":
                st.warning(
                    "⚠️ **Demonstration data**: showing `creditcard_sample.csv` "
                    "(a small sample), not the full `creditcard.csv` dataset. "
                    "All statistics and visualizations below are illustrative "
                    "sample-based results, not full-dataset results."
                )
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Transactions", f"{len(df):,}")

            with col2:
                fraud_count = int(df["Class"].sum())
                st.metric("Fraudulent Transactions", f"{fraud_count:,}")

            with col3:
                fraud_rate = (df["Class"].sum() / len(df)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            with col4:
                avg_amount = df["Amount"].mean()
                st.metric("Average Amount", f"${avg_amount:.2f}")

            st.subheader("🤖 AI-Powered Insights")
            st.markdown(get_ai_insights(df))

            create_data_visualizations(df)

            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)

    elif page == "🤖 AI Assistant":
        st.markdown("### 🤖 AI-Powered Fraud Detection Assistant")
        st.markdown("Ask me anything about fraud detection, model performance, or prevention strategies!")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(message["content"])

        user_question = st.text_input(
            "Ask a question about fraud detection:",
            placeholder="e.g., What is fraud? How does detection work?",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary"):
                if user_question:
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    ai_response = ai_chatbot_response(user_question, model_info)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    st.rerun()

        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What is fraud?",
            "How does detection work?",
            "How accurate is the model?",
            "What should I do if fraud is detected?",
            "How can I prevent fraud?",
        ]

        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    ai_response = ai_chatbot_response(question, model_info)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    st.rerun()

    elif page == "📋 AI Report":
        st.markdown("### 📋 AI-Generated Analysis Report")

        report = generate_ai_report(df, model_info, data_source=data_source)
        st.markdown(report)

        st.download_button(
            label="📥 Download AI Report",
            data=report,
            file_name=f"fraud_detection_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    elif page == "ℹ️ About":
        test_metrics = model_info.get("test_metrics", {})

        def fmt(key: str) -> str:
            val = test_metrics.get(key)
            return f"{val:.4f}" if isinstance(val, float) else "N/A"

        st.markdown(f"""
        ## 🎯 About This Application

        This Credit Card Fraud Detection System uses a trained **{model_info.get('model_name', 'unknown')}** machine learning model to identify potentially fraudulent credit card transactions.

        ### 🔧 Model Details
        - **Algorithm**: {model_info.get('model_name', 'unknown')}
        - **Features**: 30 anonymized features (V1-V28, Time, Amount)
        - **Decision Threshold**: {model_info.get('threshold', 'N/A')}
        - **Training**: SMOTE oversampling + stratified train/validation/test split

        ### 📊 Key Features
        - **Real-time Prediction**: Instant fraud detection with explanations
        - **Probability Scores**: Risk assessment with confidence levels
        - **AI Assistant**: Interactive chatbot for fraud-related questions
        - **AI Insights**: Automated analysis and recommendations
        - **AI Reports**: Comprehensive analysis reports
        - **Data Visualization**: Comprehensive dataset analysis
        - **REST API**: FastAPI backend (`app/api/main.py`)

        ### 🛡️ Security Note
        This application is for demonstration purposes. In production environments, additional security measures and validation would be required.

        ### 📈 Model Performance (Test Set)
        - **Precision**: {fmt('precision')}
        - **Recall**: {fmt('recall')}
        - **F1-Score**: {fmt('f1')}
        - **ROC-AUC**: {fmt('roc_auc')}
        - **PR-AUC**: {fmt('pr_auc')}

        ### 🔍 How to Use
        1. **Fraud Detection**: Enter transaction details and get predictions with explanations
        2. **Data Analysis**: View AI-generated insights and comprehensive visualizations
        3. **AI Assistant**: Chat with the assistant for fraud-related questions
        4. **AI Reports**: Generate and download comprehensive analysis reports
        5. **About**: Learn about the system and model performance

        ### 📚 Technical Stack
        - **Frontend**: Streamlit
        - **Backend**: FastAPI (`app/api/main.py`)
        - **ML Pipeline**: `app/ml/` (schema, validation, preprocessing, training, predict)
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        """)


if __name__ == "__main__":
    main()
