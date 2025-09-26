# 💳 Credit Card Fraud Detection System

A comprehensive machine learning application for detecting fraudulent credit card transactions using XGBoost and deployed with Streamlit.

## 🎯 Overview

This application provides an interactive web interface for real-time credit card fraud detection. It uses a trained XGBoost model that has been optimized for high accuracy and performance on imbalanced datasets.

## ✨ Features

- **🔍 Real-time Fraud Detection**: Instant prediction with probability scores
- **📊 Interactive Data Visualization**: Comprehensive dataset analysis with Plotly charts
- **🎛️ User-friendly Interface**: Clean, intuitive Streamlit interface
- **📈 Model Performance Metrics**: Detailed performance statistics
- **🛡️ Risk Assessment**: Multi-level risk categorization
- **📋 Sample Data**: Quick access to legitimate and fraudulent transaction samples

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files in your directory:
   # - app.py
   # - requirements.txt
   # - best_xgb_model.pkl
   # - creditcard.csv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically open in your default browser

## 📁 Project Structure

```
Credit Card Fraud Detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── best_xgb_model.pkl     # Trained XGBoost model
├── creditcard.csv         # Dataset
├── credit-card-fraud-detection.ipynb  # Training notebook
└── README.md             # This file
```

## 🎮 How to Use

### 1. Fraud Detection Page
- Navigate to the "🔍 Fraud Detection" tab
- Enter transaction details manually or use sample data buttons
- Click "🔍 Detect Fraud" to get prediction
- Review the risk assessment and probability scores

### 2. Data Analysis Page
- View comprehensive dataset statistics
- Explore interactive visualizations
- Analyze feature correlations
- Review sample data

### 3. About Page
- Learn about the model and its performance
- Understand the technical stack
- Get usage instructions

## 🔧 Model Details

### Algorithm
- **XGBoost Classifier** with optimized hyperparameters
- **SMOTE** for handling class imbalance
- **StandardScaler** for feature normalization

### Performance Metrics
- **Accuracy**: 99.95%
- **Precision**: 93.42%
- **Recall**: 74.74%
- **F1-Score**: 83.04%
- **ROC-AUC**: 97.86%

### Features
- **30 anonymized features** (V1-V28, Time, Amount)
- **Time**: Seconds elapsed since first transaction
- **Amount**: Transaction amount
- **V1-V28**: PCA-transformed features (anonymized for privacy)

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib

## 📊 Dataset Information

- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.17%)
- **Legitimate Transactions**: 284,315 (99.83%)
- **Features**: 30 (28 PCA components + Time + Amount)
- **No Missing Values**: Clean dataset

## 🔒 Security Considerations

⚠️ **Important**: This application is designed for demonstration and educational purposes. For production use:

- Implement proper authentication and authorization
- Add input validation and sanitization
- Use secure model serving infrastructure
- Implement logging and monitoring
- Add rate limiting and DDoS protection
- Ensure data privacy compliance (GDPR, CCPA, etc.)

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your GitHub repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku Deployment
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy to Heroku following their standard process

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**AI & Data Scientist** - Developed for educational and demonstration purposes.

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the documentation
- Review the training notebook for model details

---

**Happy Fraud Detection! 🛡️💳**
