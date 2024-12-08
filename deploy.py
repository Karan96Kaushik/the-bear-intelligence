import joblib
import pandas as pd
from preprocess import preprocess_trading_data  # your preprocessing module


model = joblib.load('stock_classifier_model.joblib')


csv_path = "trial.csv"
# csv_path = "training.csv"
X, y, scaler = preprocess_trading_data(csv_path)

input_df = pd.DataFrame(X)

print(input_df.head())
prediction = model.predict(input_df)

# Get indices where predictions are 1
positive_indices = [i for i, pred in enumerate(prediction) if pred == 1]
print(f"Number of 1s in prediction: {len(positive_indices)}")
print(f"Indices with positive predictions: {positive_indices}")

# Load and display original data for positive predictions
original_data = pd.read_csv(csv_path)
print("\nOriginal data for positive predictions:")
print(original_data.iloc[positive_indices])
