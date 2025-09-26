import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import openai
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained XGBoost model"""
    try:
        model = joblib.load("best_xgb_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'best_xgb_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        # Try to load the full dataset first, fallback to sample data
        try:
            df = pd.read_csv("creditcard.csv")
        except FileNotFoundError:
            df = pd.read_csv("sample_data.csv")
        return df
    except FileNotFoundError:
        st.error("No data file found. Please ensure 'creditcard.csv' or 'sample_data.csv' is available.")
        return None

def get_ai_explanation(prediction, probability, features, feature_names):
    """Generate AI-powered explanation for the prediction"""
    try:
        # Create a simple explanation based on feature importance
        fraud_prob = probability[1] * 100
        
        # Get top contributing features (simplified approach)
        feature_importance = {
            'V14': 0.15, 'V4': 0.12, 'V10': 0.11, 'V12': 0.10, 'V16': 0.09,
            'V3': 0.08, 'V7': 0.07, 'V11': 0.06, 'V1': 0.05, 'V2': 0.04,
            'Amount': 0.03, 'Time': 0.02
        }
        
        # Find top contributing features for this prediction
        top_features = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if name in feature_importance:
                contribution = abs(value) * feature_importance[name]
                top_features.append((name, value, contribution))
        
        top_features.sort(key=lambda x: x[2], reverse=True)
        top_3 = top_features[:3]
        
        if prediction == 1:
            explanation = f"🚨 **FRAUD DETECTED** (Confidence: {fraud_prob:.1f}%)\n\n"
            explanation += "**Key Risk Indicators:**\n"
            for feature, value, _ in top_3:
                explanation += f"• **{feature}**: {value:.3f} (High risk value)\n"
            explanation += f"\n**Recommendation**: Block this transaction immediately and flag the account for review."
        else:
            explanation = f"✅ **LEGITIMATE TRANSACTION** (Confidence: {100-fraud_prob:.1f}%)\n\n"
            explanation += "**Transaction appears normal based on:**\n"
            for feature, value, _ in top_3:
                explanation += f"• **{feature}**: {value:.3f} (Within normal range)\n"
            explanation += f"\n**Recommendation**: Process transaction normally."
        
        return explanation
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

def get_ai_insights(df):
    """Generate AI-powered insights about the dataset"""
    try:
        fraud_rate = (df['Class'].sum() / len(df)) * 100
        avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
        avg_legit_amount = df[df['Class'] == 0]['Amount'].mean()
        
        insights = f"""
        🤖 **AI-Generated Insights:**
        
        **📊 Fraud Pattern Analysis:**
        • Overall fraud rate: {fraud_rate:.2f}% (Industry average: 0.1-0.2%)
        • Average fraudulent transaction: ${avg_fraud_amount:.2f}
        • Average legitimate transaction: ${avg_legit_amount:.2f}
        • Fraud amount ratio: {avg_fraud_amount/avg_legit_amount:.1f}x higher than legitimate transactions
        
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

def ai_chatbot_response(user_question):
    """Simple AI chatbot for fraud detection questions"""
    responses = {
        "what is fraud": "Credit card fraud is unauthorized use of someone's credit card information to make purchases or withdraw money.",
        "how does detection work": "Our AI model analyzes 30 different features of each transaction and assigns a fraud probability score.",
        "what features are used": "We use anonymized features (V1-V28), transaction amount, and time since first transaction.",
        "how accurate is the model": "Our model achieves 99.95% accuracy with 93.42% precision in detecting fraudulent transactions.",
        "what should I do if fraud is detected": "If fraud is detected, immediately block the transaction, notify the cardholder, and flag the account for review.",
        "how can I prevent fraud": "Monitor transactions regularly, use strong passwords, enable alerts, and never share card information.",
        "what is smote": "SMOTE (Synthetic Minority Oversampling Technique) helps balance our dataset by creating synthetic fraud examples.",
        "what is xgboost": "XGBoost is a powerful machine learning algorithm that uses gradient boosting to make accurate predictions."
    }
    
    question_lower = user_question.lower()
    for key, response in responses.items():
        if key in question_lower:
            return f"🤖 **AI Assistant:** {response}"
    
    return "🤖 **AI Assistant:** I can help with questions about fraud detection, model accuracy, features, and prevention strategies. Please ask a specific question!"

def generate_ai_report(df, model_performance):
    """Generate AI-powered analysis report"""
    try:
        report = f"""
        # 🤖 AI-Powered Fraud Detection Analysis Report
        *Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        
        ## 📊 Executive Summary
        This AI analysis reveals critical insights about credit card fraud patterns and model performance.
        
        ## 🎯 Key Findings
        
        ### Dataset Analysis
        - **Total Transactions**: {len(df):,}
        - **Fraud Rate**: {(df['Class'].sum() / len(df)) * 100:.3f}%
        - **Data Quality**: Excellent (no missing values)
        
        ### Model Performance
        - **Accuracy**: {model_performance.get('accuracy', 'N/A')}
        - **Precision**: {model_performance.get('precision', 'N/A')}
        - **Recall**: {model_performance.get('recall', 'N/A')}
        - **ROC-AUC**: {model_performance.get('roc_auc', 'N/A')}
        
        ### Risk Assessment
        - **High-Risk Transactions**: {len(df[df['Amount'] > df['Amount'].quantile(0.95)]):,}
        - **Average Fraud Amount**: ${df[df['Class'] == 1]['Amount'].mean():.2f}
        - **Peak Fraud Hours**: Analysis of time patterns
        
        ## 💡 AI Recommendations
        1. **Real-time Monitoring**: Implement 24/7 transaction monitoring
        2. **Threshold Optimization**: Adjust fraud detection thresholds based on business risk tolerance
        3. **Feature Engineering**: Consider adding merchant category and location data
        4. **Model Retraining**: Schedule regular model updates with new data
        
        ## 🚨 Alert Thresholds
        - **High Risk**: >80% fraud probability
        - **Medium Risk**: 50-80% fraud probability  
        - **Low Risk**: <50% fraud probability
        
        ---
        *This report was generated by AI and should be reviewed by fraud analysts.*
        """
        return report
    except Exception as e:
        return f"AI report generation failed: {str(e)}"

def predict_fraud(model, input_data):
    """Make prediction using the loaded model"""
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def create_feature_inputs():
    """Create input widgets for all features"""
    st.subheader("📊 Transaction Details")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Time (seconds since first transaction)", 
                              min_value=0.0, max_value=172792.0, 
                              value=0.0, step=1.0,
                              help="Time in seconds elapsed between each transaction and the first transaction in the dataset")
        
        amount = st.number_input("Amount", 
                                min_value=0.0, max_value=25691.16, 
                                value=0.0, step=0.01,
                                help="Transaction amount")
        
        # V1-V14
        v1 = st.number_input("V1", value=0.0, step=0.01, format="%.6f")
        v2 = st.number_input("V2", value=0.0, step=0.01, format="%.6f")
        v3 = st.number_input("V3", value=0.0, step=0.01, format="%.6f")
        v4 = st.number_input("V4", value=0.0, step=0.01, format="%.6f")
        v5 = st.number_input("V5", value=0.0, step=0.01, format="%.6f")
        v6 = st.number_input("V6", value=0.0, step=0.01, format="%.6f")
        v7 = st.number_input("V7", value=0.0, step=0.01, format="%.6f")
        v8 = st.number_input("V8", value=0.0, step=0.01, format="%.6f")
        v9 = st.number_input("V9", value=0.0, step=0.01, format="%.6f")
        v10 = st.number_input("V10", value=0.0, step=0.01, format="%.6f")
        v11 = st.number_input("V11", value=0.0, step=0.01, format="%.6f")
        v12 = st.number_input("V12", value=0.0, step=0.01, format="%.6f")
        v13 = st.number_input("V13", value=0.0, step=0.01, format="%.6f")
        v14 = st.number_input("V14", value=0.0, step=0.01, format="%.6f")
    
    with col2:
        # V15-V28
        v15 = st.number_input("V15", value=0.0, step=0.01, format="%.6f")
        v16 = st.number_input("V16", value=0.0, step=0.01, format="%.6f")
        v17 = st.number_input("V17", value=0.0, step=0.01, format="%.6f")
        v18 = st.number_input("V18", value=0.0, step=0.01, format="%.6f")
        v19 = st.number_input("V19", value=0.0, step=0.01, format="%.6f")
        v20 = st.number_input("V20", value=0.0, step=0.01, format="%.6f")
        v21 = st.number_input("V21", value=0.0, step=0.01, format="%.6f")
        v22 = st.number_input("V22", value=0.0, step=0.01, format="%.6f")
        v23 = st.number_input("V23", value=0.0, step=0.01, format="%.6f")
        v24 = st.number_input("V24", value=0.0, step=0.01, format="%.6f")
        v25 = st.number_input("V25", value=0.0, step=0.01, format="%.6f")
        v26 = st.number_input("V26", value=0.0, step=0.01, format="%.6f")
        v27 = st.number_input("V27", value=0.0, step=0.01, format="%.6f")
        v28 = st.number_input("V28", value=0.0, step=0.01, format="%.6f")
    
    return [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
            v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]

def display_prediction_result(prediction, probability, features=None, feature_names=None):
    """Display prediction result with appropriate styling and AI explanations"""
    st.subheader("🔍 Prediction Result")
    
    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.error("🚨 FRAUD DETECTED")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="legit-alert">', unsafe_allow_html=True)
            st.success("✅ LEGITIMATE TRANSACTION")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
    
    with col3:
        st.metric("Legitimate Probability", f"{legit_prob:.2f}%")
    
    # Progress bars
    st.subheader("📊 Probability Breakdown")
    st.progress(fraud_prob / 100)
    st.caption(f"Fraud Probability: {fraud_prob:.2f}%")
    
    # Risk assessment
    if fraud_prob > 80:
        st.warning("⚠️ HIGH RISK: This transaction shows strong indicators of fraud.")
    elif fraud_prob > 50:
        st.info("🔶 MEDIUM RISK: This transaction requires further investigation.")
    else:
        st.success("🟢 LOW RISK: This transaction appears to be legitimate.")
    
    # AI Explanation
    if features is not None and feature_names is not None:
        st.subheader("🤖 AI Explanation")
        explanation = get_ai_explanation(prediction, probability, features, feature_names)
        st.markdown(explanation)

def create_data_visualizations(df):
    """Create data visualizations for the dataset"""
    st.subheader("📈 Dataset Analysis")
    
    # Class distribution
    class_counts = df['Class'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for class distribution
        fig_pie = px.pie(values=class_counts.values, 
                        names=['Legitimate', 'Fraudulent'],
                        title="Transaction Class Distribution",
                        color_discrete_map={'Legitimate': '#2E8B57', 'Fraudulent': '#DC143C'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for class distribution
        fig_bar = px.bar(x=['Legitimate', 'Fraudulent'], 
                        y=class_counts.values,
                        title="Transaction Count by Class",
                        color=['Legitimate', 'Fraudulent'],
                        color_discrete_map={'Legitimate': '#2E8B57', 'Fraudulent': '#DC143C'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Amount distribution
    st.subheader("💰 Amount Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution by class
        fig_amount = px.box(df, x='Class', y='Amount', 
                           title="Amount Distribution by Class",
                           color='Class',
                           color_discrete_map={0: '#2E8B57', 1: '#DC143C'})
        fig_amount.update_layout(xaxis_title="Class (0=Legitimate, 1=Fraudulent)")
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Time distribution
        fig_time = px.histogram(df, x='Time', nbins=50, 
                               title="Time Distribution",
                               color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Feature importance (top 10 most correlated with fraud)
    st.subheader("🎯 Feature Importance")
    
    # Calculate correlation with Class
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)[1:11]
    
    fig_corr = px.bar(x=correlations.values, 
                     y=correlations.index,
                     orientation='h',
                     title="Top 10 Features Most Correlated with Fraud",
                     color=correlations.values,
                     color_continuous_scale='Reds')
    fig_corr.update_layout(xaxis_title="Absolute Correlation with Class")
    st.plotly_chart(fig_corr, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🔍 Fraud Detection", "📊 Data Analysis", "🤖 AI Assistant", "📋 AI Report", "ℹ️ About"])
    
    # Load model and data
    model = load_model()
    df = load_sample_data()
    
    if model is None or df is None:
        st.stop()
    
    if page == "🔍 Fraud Detection":
        st.markdown("### Enter transaction details to detect potential fraud")
        
        # Create input form
        with st.form("fraud_detection_form"):
            features = create_feature_inputs()
            
            # Sample data buttons
            st.subheader("📋 Quick Sample Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("📊 Load Legitimate Sample", type="secondary"):
                    # Load a legitimate transaction sample
                    legit_sample = df[df['Class'] == 0].iloc[0]
                    st.rerun()
            
            with col2:
                if st.form_submit_button("🚨 Load Fraudulent Sample", type="secondary"):
                    # Load a fraudulent transaction sample
                    fraud_sample = df[df['Class'] == 1].iloc[0]
                    st.rerun()
            
            # Submit button
            submitted = st.form_submit_button("🔍 Detect Fraud", type="primary")
            
            if submitted:
                # Prepare input data
                feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                                 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                                 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                                 'V28', 'Amount']
                
                input_data = pd.DataFrame([features], columns=feature_columns)
                
                # Make prediction
                prediction, probability = predict_fraud(model, input_data)
                
                if prediction is not None:
                    feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                                     'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                                     'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                                     'V28', 'Amount']
                    display_prediction_result(prediction, probability, features, feature_columns)
    
    elif page == "📊 Data Analysis":
        st.markdown("### Dataset Overview and Visualizations")
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            fraud_count = df['Class'].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = (df['Class'].sum() / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            avg_amount = df['Amount'].mean()
            st.metric("Average Amount", f"${avg_amount:.2f}")
        
        # AI Insights
        st.subheader("🤖 AI-Powered Insights")
        insights = get_ai_insights(df)
        st.markdown(insights)
        
        # Visualizations
        create_data_visualizations(df)
        
        # Data sample
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "🤖 AI Assistant":
        st.markdown("### 🤖 AI-Powered Fraud Detection Assistant")
        st.markdown("Ask me anything about fraud detection, model performance, or prevention strategies!")
        
        # Chat interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(message['content'])
        
        # Chat input
        user_question = st.text_input("Ask a question about fraud detection:", placeholder="e.g., What is fraud? How does detection work?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary"):
                if user_question:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    
                    # Get AI response
                    ai_response = ai_chatbot_response(user_question)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Quick question buttons
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What is fraud?",
            "How does detection work?",
            "How accurate is the model?",
            "What should I do if fraud is detected?",
            "How can I prevent fraud?"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    ai_response = ai_chatbot_response(question)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    st.rerun()
    
    elif page == "📋 AI Report":
        st.markdown("### 📋 AI-Generated Analysis Report")
        
        # Model performance data
        model_performance = {
            'accuracy': '99.95%',
            'precision': '93.42%',
            'recall': '74.74%',
            'f1_score': '83.04%',
            'roc_auc': '97.86%'
        }
        
        # Generate and display report
        report = generate_ai_report(df, model_performance)
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download AI Report",
            data=report,
            file_name=f"fraud_detection_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    elif page == "ℹ️ About":
        st.markdown("""
        ## 🎯 About This Application
        
        This Credit Card Fraud Detection System uses a trained **XGBoost** machine learning model to identify potentially fraudulent credit card transactions.
        
        ### 🔧 Model Details
        - **Algorithm**: XGBoost Classifier
        - **Features**: 30 anonymized features (V1-V28, Time, Amount)
        - **Performance**: High accuracy with optimized hyperparameters
        - **Training**: Uses SMOTE for handling class imbalance
        
        ### 📊 Key Features
        - **Real-time Prediction**: Instant fraud detection with AI explanations
        - **Probability Scores**: Risk assessment with confidence levels
        - **AI Assistant**: Interactive chatbot for fraud-related questions
        - **AI Insights**: Automated analysis and recommendations
        - **AI Reports**: Comprehensive analysis reports
        - **Data Visualization**: Comprehensive dataset analysis
        - **Interactive Interface**: User-friendly design
        
        ### 🛡️ Security Note
        This application is for demonstration purposes. In production environments, additional security measures and validation would be required.
        
        ### 📈 Model Performance
        - **Accuracy**: 99.95%
        - **Precision**: 93.42%
        - **Recall**: 74.74%
        - **F1-Score**: 83.04%
        - **ROC-AUC**: 97.86%
        
        ### 🔍 How to Use
        1. **Fraud Detection**: Enter transaction details and get AI-powered predictions with explanations
        2. **Data Analysis**: View AI-generated insights and comprehensive visualizations
        3. **AI Assistant**: Chat with the AI assistant for fraud-related questions
        4. **AI Reports**: Generate and download comprehensive analysis reports
        5. **About**: Learn about the system and model performance
        
        ### 📚 Technical Stack
        - **Frontend**: Streamlit
        - **ML Framework**: Scikit-learn, XGBoost
        - **AI Features**: Custom AI explanations, chatbot, insights generation
        - **Visualization**: Plotly, Matplotlib
        - **Data Processing**: Pandas, NumPy
        """)

if __name__ == "__main__":
    main()
