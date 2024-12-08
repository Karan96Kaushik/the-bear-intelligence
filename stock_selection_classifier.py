import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import preprocess_trading_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_classifier(X, y, test_size=0.2, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64, 32),  # Deeper network with five hidden layers
        activation='relu',
        solver='adam',
        max_iter=100000,  # Further increased maximum iterations
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,  # Increased validation set
        n_iter_no_change=150,  # More patience before early stopping
        learning_rate_init=0.001,
        alpha=0.0005,  # Increased L2 regularization
        batch_size=64,  # Larger mini-batch size for better stability
        learning_rate='adaptive',
        momentum=0.9,  # Added momentum for faster convergence
        nesterovs_momentum=True,  # Use Nesterov's momentum
        power_t=0.5,  # Power for inverse scaling learning rate
        tol=1e-5,  # Tolerance for optimization
        verbose=True,  # Print progress messages
        warm_start=False,  # Don't reuse previous solution
        beta_1=0.9,  # Adam optimizer parameter
        beta_2=0.999,  # Adam optimizer parameter
        epsilon=1e-8,  # Adam optimizer parameter
        # class_weight='balanced',  # Automatically adjust weights based on class frequencies
        max_fun=15000  # Maximum number of loss function calls
    )
    
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Print performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot feature importance (removed as MLPClassifier doesn't have feature_importances_)
    # Instead, we can plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(clf.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    return clf, X_test, y_test

def save_model(clf, filename='stock_classifier_model.joblib'):
    """Save the trained model to a file"""
    joblib.dump(clf, filename)
    print(f"Model saved as {filename}")

def load_model(filename='stock_classifier_model.joblib'):
    """Load a trained model from a file"""
    return joblib.load(filename)

if __name__ == "__main__":
    # Load and preprocess data
    csv_path = "training.csv"
    X, y, scaler = preprocess_trading_data(csv_path)
    
    # Train the classifier
    clf, X_test, y_test = train_classifier(X, y)
    
    # Save the model
    save_model(clf)
    
    # Example of making predictions
    sample_predictions = clf.predict(X_test[:5])
    print("\nSample predictions:", sample_predictions)
    print("Actual values:", y_test[:5].values)
