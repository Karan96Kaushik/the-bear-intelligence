import joblib
import pandas as pd
from preprocess import preprocess_trading_data  # your preprocessing module


# Load the model and scaler
model = joblib.load('stock_classifier_model.joblib')


csv_path = "trial.csv"
# csv_path = "training.csv"
X, y, scaler = preprocess_trading_data(csv_path)

input_df = pd.DataFrame(X)

# Get predictions and probabilities
predictions = model.predict(input_df)
prediction_probs = model.predict_proba(input_df)

# Store the original data first
original_data = pd.read_csv(csv_path)

# Ensure we only use indices that are within the range of both datasets
max_index = min(len(original_data), len(predictions))
predictions = predictions[:max_index]
prediction_probs = prediction_probs[:max_index]

pred_min = 0.75

# Get positive results within the valid range and with confidence > 0.6
positive_results = [(i, prob[1]) for i, (pred, prob) in enumerate(zip(predictions, prediction_probs)) 
                   if pred == 1 and i < max_index and prob[1] > pred_min]
positive_indices = [i for i, _ in positive_results]
positive_probs = [prob for _, prob in positive_results]

print(f"\nNumber of high-confidence positive predictions: {len(positive_indices)}")
print(f"\nPositive predictions with probabilities (confidence > {pred_min}):")
for idx, prob in positive_results:
    print(f"Index {idx}: {prob:.3f} confidence")

# Load and display original data for positive predictions
results_df = original_data.iloc[positive_indices].copy()
results_df['prediction_confidence'] = positive_probs

print("\nOriginal data for positive predictions (with confidence scores):")
print(results_df)

# Optionally, save results to CSV
results_df.to_csv('positive_predictions.csv', index=False)
