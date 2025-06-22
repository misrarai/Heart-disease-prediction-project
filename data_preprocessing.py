import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('heart_disease_uci.csv')
    
    # Create feature mappings dictionary
    feature_mappings = {
        'sex': {'Female': 0, 'Male': 1},
        'cp': {
            'typical angina': 0,
            'atypical angina': 1,
            'non-anginal': 2,
            'asymptomatic': 3
        },
        'fbs': {'false': 0, 'true': 1},
        'restecg': {
            'normal': 0,
            'st-t abnormality': 1,
            'lv hypertrophy': 2
        },
        'exang': {'false': 0, 'true': 1},
        'slope': {
            'upsloping': 0,
            'flat': 1,
            'downsloping': 2
        },
        'thal': {
            'normal': 1,
            'fixed defect': 2,
            'reversable defect': 3
        }
    }
    
    # Save feature mappings
    with open('feature_mappings.json', 'w') as f:
        json.dump(feature_mappings, f, indent=4)
    
    # Convert boolean columns to proper format
    df['fbs'] = df['fbs'].astype(str).str.lower()
    df['exang'] = df['exang'].astype(str).str.lower()
    
    # Apply mappings
    for column, mapping in feature_mappings.items():
        df[column] = df[column].map(mapping)
    
    # Ensure all features are numeric
    numeric_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    
    # Handle missing values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Handle outliers using IQR method
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    # Standardize numerical features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Save the scaler
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
      # Split features and target
    X = df.drop(['num', 'id', 'dataset'], axis=1)  # Using 'num' as target column and dropping unnecessary columns
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary classification
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_mappings

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
