import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_trading_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Remove duplicates
    # df = df.drop_duplicates()

    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Extract time features
    # df['Hour'] = df['Timestamp'].dt.hour
    # df['Minute'] = df['Timestamp'].dt.minute

    # Remove rows where Achieved is NaN
    print('Original df shape:', df.shape)
    df = df.dropna()
    print('Df shape after dropping NaNs:', df.shape)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Candle Type'] = le.fit_transform(df['Candle Type'])
    df['MA Direction'] = le.fit_transform(df['MA Direction'])
    # print('Acheieved shape:', df['Acheieved'].shape)
    
    # Convert boolean to int
    df['Acheieved'] = df['Acheieved'].astype(int)
    
    rowwise_columns = [
        'High', 'Low', 'Open', 'Close', 
        # 'Low Day', 'High Day',
        'BB Middle', 'BB Upper', 'BB Lower'
    ]
    
    regular_scale_columns = [
        'Volume Prev Day Avg', 'Volume P Last', 'Volume P 2nd Last',
        'Volume P 3rd Last', 'MA Trend Count',
    ]

    scaler = MinMaxScaler()
    
    scaled_features = pd.DataFrame()
    
    # Add rowwise scaling
    for col in rowwise_columns:
        row_min = df[col].min(axis=0)
        row_max = df[col].max(axis=0)
        scaled_features[col] = (df[col] - row_min) / (row_max - row_min)
    
    # Add regularly scaled columns
    scaled_features[regular_scale_columns] = scaler.fit_transform(df[regular_scale_columns])
    
    # print('Scaled features shape:', scaled_features.shape)
    # print('df shape:', df[['Candle Type', 'MA Direction', 'Acheieved']].shape)
    
    final_df = pd.concat([
        scaled_features,
        df[['Candle Type', 'MA Direction', 'Acheieved']]
    ], axis=1)

    # final_df = final_df.dropna(subset=['Acheieved'])

    # Copy rows where Acheieved is True (1) and append to dataframe
    achieved_rows = final_df[final_df['Acheieved'] == 1].copy()

    final_df = pd.concat([final_df, achieved_rows], axis=0)
    final_df = pd.concat([final_df, achieved_rows], axis=0)
    # final_df = pd.concat([final_df, achieved_rows], axis=0)
    # final_df = pd.concat([final_df, achieved_rows], axis=0)

    print('Achieved rows shape:', achieved_rows.shape)
    # Reset index after concatenation
    final_df = final_df.reset_index(drop=True)
    print('Final df shape:', final_df.shape)




    # print('Final df shape:', final_df.shape)
    
    X = final_df.drop('Acheieved', axis=1)
    y = final_df['Acheieved']

    # print(X.shape)
    # print(y.shape)

    return X, y, scaler

if __name__ == "__main__":
    # Example usage
    csv_path = "training.csv"
    
    # Preprocess data
    X, y, scaler = preprocess_trading_data(csv_path)
    
    print("Input shape:", X.shape)
    print("Output shape:", y.shape)
    
    # Example of how the processed data looks
    print("\nFeature names:")
    print(X.columns.tolist())
    
    print("\nFirst few samples:")
    print(X.head())
    print("\nSample target values:")
    print(y.head())