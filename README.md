# Heart Disease Prediction System

This project implements a machine learning-based heart disease prediction system using multiple models and a user-friendly web interface.

## Project Structure

```
heart_disease_prediction/
├── app.py                    # Streamlit web application
├── data_preprocessing.py     # Data preprocessing script
├── train_models.py          # Model training script
├── heart_disease_uci.csv    # Original dataset
├── feature_mappings.json    # Feature encoding mappings
├── scaler.pkl              # Trained scaler for data normalization
├── models/                  # Trained model files
│   ├── svm.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
└── requirements.txt        # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd heart-disease-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

1. First, preprocess the data and train the models:
```bash
python data_preprocessing.py
python train_models.py
```

2. Run the Streamlit web application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Features

- Multiple model comparison (SVM, Logistic Regression, Random Forest, XGBoost)
- Comprehensive risk factor analysis
- User-friendly web interface
- Real-time predictions with probability scores
- Detailed health recommendations

## Dataset

The project uses the UCI Heart Disease dataset, which includes various medical parameters:
- Age, Sex, Chest Pain Type
- Blood Pressure, Cholesterol Levels
- ECG Results
- And other relevant medical indicators

## Models

The system uses four different machine learning models:
1. Support Vector Machine (SVM)
2. Logistic Regression
3. Random Forest
4. XGBoost

The predictions are combined using a weighted ensemble approach for better accuracy.

## Citation

If you use this project, please cite the UCI Heart Disease dataset:
[UCI Machine Learning Repository: Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
