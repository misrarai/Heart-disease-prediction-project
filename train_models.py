import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

class HeartDiseaseModel:
    def __init__(self):
        self.models = {
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the data"""
        print("Loading clean dataset...")
        df = pd.read_csv('heart_disease_clean.csv')
        
        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Save the scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X, y
    
    def train_and_evaluate(self):
        """Train and evaluate all models"""
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining and evaluating models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Save model
            with open(f'{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Cross-validation mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("\nClassification Report:")
            print(self.results[name]['classification_report'])
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
    
    def plot_model_comparison(self):
        """Plot model comparison"""
        accuracies = {name: results['accuracy'] for name, results in self.results.items()}
        cv_means = {name: results['cv_mean'] for name, results in self.results.items()}
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(accuracies))
        width = 0.35
        
        plt.bar(x - width/2, list(accuracies.values()), width, label='Test Accuracy')
        plt.bar(x + width/2, list(cv_means.values()), width, label='CV Mean Accuracy')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.xticks(x, accuracies.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

def main():
    model = HeartDiseaseModel()
    model.train_and_evaluate()
    model.plot_model_comparison()
    
    # Print detailed model comparison
    print("\n=== Model Comparison ===")
    best_model = None
    best_accuracy = 0
    print("\nModel Performance Summary:")
    print("=" * 50)
    print(f"{'Model':<20} {'Accuracy':<10} {'CV Mean':<10} {'CV Std':<10}")
    print("-" * 50)
    
    for name, results in model.results.items():
        accuracy = results['accuracy']
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        print(f"{name:<20} {accuracy:.4f}    {cv_mean:.4f}    {cv_std:.4f}")
        
        # Track best model
        if cv_mean > best_accuracy:
            best_accuracy = cv_mean
            best_model = name
    
    print("\nBest Performing Model:", best_model)
    print(f"Cross-validation Score: {best_accuracy:.4f}")
    
    # Save best model name
    with open('best_model.txt', 'w') as f:
        f.write(best_model)
    
    print("\nModel training completed!")
    print("Check the generated PNG files for visualizations of the results.")

if __name__ == "__main__":
    main()
