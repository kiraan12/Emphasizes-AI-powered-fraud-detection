# 🚀 Setup Instructions for AI-Powered Fraud Detection

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager

## 🛠️ Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/kiraan12/Emphasizes-AI-powered-fraud-detection.git
cd Emphasizes-AI-powered-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Files
Since large files are not included in the repository, you need to:

#### Option A: Use Sample Data (Recommended for Demo)
The app will automatically use `sample_data.csv` for demonstration purposes.

#### Option B: Use Full Dataset
1. Download the full credit card dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Rename it to `creditcard.csv` and place it in the project directory
3. Train your own model using the provided Jupyter notebook

### 4. Train Your Model (Optional)
If you want to use your own trained model:
```bash
# Run the Jupyter notebook to train the model
jupyter notebook credit-card-fraud-detection.ipynb
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 🎯 Features Available
- ✅ AI-powered fraud detection
- ✅ Interactive chatbot assistant
- ✅ AI-generated insights and reports
- ✅ Real-time predictions with explanations
- ✅ Data visualization and analysis
- ✅ Sample data for immediate testing

## 📊 Model Performance
- **Accuracy**: 99.95%
- **Precision**: 93.42%
- **Recall**: 74.74%
- **F1-Score**: 83.04%
- **ROC-AUC**: 97.86%

## 🔧 Troubleshooting
- If you get "Model file not found" error, the app will still work with sample data
- For full functionality, ensure you have the trained model file
- Check that all dependencies are installed correctly

## 📞 Support
For issues or questions, please create an issue in the repository.
