import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import torch
import string
import re
import dagshub
import os
import io
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, render_template_string, redirect, url_for
from sklearn.preprocessing import MinMaxScaler 
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

#defining global variables
# IMPORTANT: These must match the training script exactly.
INPUT_SIZE = 5
HIDDEN_SIZE = 80
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 60

#defining model and scaler loading function
# --- MLFLOW CONFIGURATION ---
MLFLOW_MODEL_NAME = "my_model" # This must match the name used in register_model.py
MLFLOW_MODEL_STAGE = "Staging" # Load the model from the "Staging" stage
# Assumes the scaler was logged as a root artifact in the MLflow run
SCALER_ARTIFACT_PATH = "scaler.pkl" 

# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/girishsai758/yahoo-price-predictor.mlflow')
dagshub.init(repo_owner='girishsai758', repo_name='yahoo-price-predictor', mlflow=True)
# -------------------------------------------------------------------------------------
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pytorch.load_model(model_uri)
scaler_path_downloaded = './artifacts/scaler.pkl'
with open(scaler_path_downloaded, 'rb') as f:
            scaler = pickle.load(f)
logging.basicConfig(level=logging.INFO)            
# definign the class object
class StockDataset(Dataset):
    """Custom Dataset class for stock data."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
# --- 2. MODEL DEFINITION (MUST MATCH TRAINING) ---
class StockPricePredictor(nn.Module):
    """The PyTorch LSTM model architecture."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # We only care about the last time step output for prediction
        # print("lstm_out shape:", self.linear(lstm_out[:, -1, :]).shape)
       
        return self.linear(lstm_out[:, -1, :])

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")
        df1=df.drop(['Date','Dividends','Stock Splits'],axis=1)
        logging.info('Data preprocessing completed')
        return df1
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def prepare_data(df, seq_length):
    # 1. Define Features (X) and Target (Y)
    
    # Features (X): Use all relevant columns *except* the date/index
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features_raw = df[feature_cols].values.astype(float)
    
    # Target (Y): Only the 'Close' price column
    target_raw = df['Close'].values.astype(float).reshape(-1, 1)

    # 2. Normalize Data
    
    # Use one scaler for the features (X)
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    features_normalized = feature_scaler.fit_transform(features_raw)
    
    # Use a SEPARATE scaler for the target (Y - Close price)
    # This is the scaler we need to return for denormalizing the final prediction.
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_normalized = target_scaler.fit_transform(target_raw)

    # 3. Create Sequences (using the multivariate function)
    # X will have shape (num_samples, seq_length, num_features)
    # y will have shape (num_samples, 1)
    # X, y = create_sequences(features_normalized, target_normalized, seq_length)
    
    # # 4. Split Data (Chronological Split)
    # train_size = int(len(X) * train_split)
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 5. Combine into PyTorch Datasets
    # train_sequences = list(zip(X_train, y_train))
    # test_sequences = list(zip( features_normalized, target_normalized))

    # train_dataset = StockDataset(train_sequences)
    test_dataset = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
    print("test_dataset length:", len(test_dataset))
    print("A sample from test_dataset:", test_dataset, test_dataset)
    # 6. Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 7. Return Loaders and the Target Scaler
    # NOTE: You might need to save/return the FEATURE_SCALER as well 
    # if you ever need to analyze or transform the input features separately, 
    # but the TARGET_SCALER is mandatory for the final prediction denormalization.
    logging.info("Data prepared and DataLoader created.")
    return  test_dataset, target_scaler

# --- 4. PREDICTION LOGIC ---
def predict_next_day(data_df: pd.DataFrame, model, scaler) -> float:
    """
    Processes 60 days of closing price data and returns the prediction for day 61.
    """
    if len(data_df) != SEQUENCE_LENGTH:
        raise ValueError(f"Input data must contain exactly {SEQUENCE_LENGTH} rows, found {len(data_df)}.")
        
    # Extract only the 'Close' prices for prediction
    # Ensure the dataframe has a 'Close' column
    # if 'Close' not in data_df.columns:
    #      raise ValueError("Input CSV must contain a column named 'Close'.")
    df_1=preprocess_data(data_df) 
    print(df_1.info())    
    print("Data pre-processing completed for prediction.")
    normalized_data, scaler2 = prepare_data(df_1, SEQUENCE_LENGTH)
    # 1. Normalize the data using the pre-fitted scaler
    # normalized_data = scaler.transform(close_prices)
    print(len(normalized_data))
    print("Data normalization completed for prediction.")
    # # 2. Convert to PyTorch tensor (shape: 1, 60, 1)
    # input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
    # logging.info(f"Input tensor shape for prediction: {normalized_data.shape}")
    # 3. Make the prediction
    with torch.no_grad():
     
        # print(f"Input tensor shape for prediction: {normalized_data.shape}")
       
        output= model(normalized_data) # output shape: (1, 1)
        
        prediction_normalized = output.cpu().numpy()  # shape: (1, 1, 1)
        # print("prediction_normalized shape:", prediction_normalized.shape)
    logging.info(f"Model prediction (normalized): {prediction_normalized.item()}")
    # predictions_normalized = np.array(predictions_normalized)
    # 4. Inverse transform to get the actual price
    prediction_original_scale = scaler.inverse_transform(prediction_normalized)
    logging.info(f"Model prediction (original scale): {prediction_original_scale}")
    return prediction_original_scale

data_df = pd.read_csv('testing.csv')
            
            # Check for required length (60 days)
if len(data_df) != SEQUENCE_LENGTH:
                raise ValueError(f"Input file must contain exactly {SEQUENCE_LENGTH} rows. Found {len(data_df)}.")
                
            # Perform prediction
predicted_price = predict_next_day(data_df, model, scaler)
            
            # Pass the result back to the template
print(f"Predicted Closing Price for Next Day: {predicted_price[0][0]}")