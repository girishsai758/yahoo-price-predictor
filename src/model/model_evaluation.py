import setuptools
import os
import re
import string



import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import mlflow.sklearn
import dagshub
import mlflow.entities # For logging parameters
import warnings
import torch
import pickle
import pickle
 # Import your evaluation function
from sklearn.metrics import mean_squared_error # Import your evaluation function
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
project_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.logger import logging
from src.connections import s3_connections
# # from src.model.model_building import StockPricePredictor

# from src.model.model_building import create_sequences
# from src.model.model_building import StockDataset

import json
warnings.simplefilter("ignore", UserWarning)

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("YAHOO")
if not dagshub_token:
    raise EnvironmentError("yahoo environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "girishsai758"
repo_name = "yahoo-price-predictor"

#Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/girishsai758/yahoo-price-predictor.mlflow')
# dagshub.init(repo_owner='girishsai758', repo_name='yahoo-price-predictor', mlflow=True)
# -------------------------------------------------------------------------------------
class StockPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
        )
        
        # Define the fully connected layer for output
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size=1)
        print("Input x shape:", x.shape)
    # ... rest of code
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM layer
        # out shape: (batch_size, seq_length, hidden_size)
        # (hn, cn) are the final hidden and cell states
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the output of the LAST time step for prediction
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.linear(out[:, -1, :])
        
        return out
class StockDataset(Dataset):
    """Custom Dataset class for stock data."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
def create_sequences(features, target, seq_length):
   """
    Converts multi-feature time series data into sequences of length seq_length.

    Args:
        features (np.ndarray): The normalized input data (multiple columns).
        target (np.ndarray): The normalized target data (single column: 'Close').
        seq_length (int): The length of the sequence (e.g., 60 days).

    Returns:
        tuple: (X_sequences, Y_targets)
    """
   xs,ys = [],[]
    # We iterate until the number of available historical feature points 
    # allows us to define a sequence (X) AND its corresponding next-day target (Y).
   for i in range(len(features) - seq_length):
        # Input sequence (X): A block of data [seq_length x num_features]
        # Example: 60 rows of (Open, High, Low, Close, Volume)
        x = features[i:(i + seq_length)]
        
        # Target (Y): The 'Close' price for the next day.
        # This corresponds to the row index immediately following the sequence.
        y = target[i + seq_length]
        
        xs.append(x)
        ys.append(y)
        
   return np.array(xs), np.array(ys)
def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as f:
         model= pickle.load(f)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise
# ========================== load data ==========================
def load_data(file_path: str) -> pd.DataFrame:

    """Load stock data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error('No data: %s is empty', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading data: %s', e)
        raise
# ========================== save metrics ==========================
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise
# ========================== save_model_info==========================
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:

    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
# ========================== evaluate model ==========================
def evaluate_model(model, test_loader, scaler, device):
    """
    Evaluates the model and calculates RMSE on de-normalized data.
    """
    model.eval()
    predictions_normalized = []
    actuals_normalized = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            logging.info(f"Evaluating batch with sequences shape: {sequences.shape}")
            # 1. Get predictions (still normalized)
            outputs = model(sequences)
            logging.info(f"Model outputs shape: {outputs.shape}")
            predictions_normalized.extend(outputs.cpu().numpy())
            actuals_normalized.extend(labels.cpu().numpy())

    # Convert lists to NumPy arrays
    predictions_normalized = np.array(predictions_normalized)
    actuals_normalized = np.array(actuals_normalized)
    
    # --- De-Normalization ---
    # 2. Convert normalized predictions and actuals back to actual dollar values
    predictions_actual = scaler.inverse_transform(predictions_normalized)
    actuals_actual = scaler.inverse_transform(actuals_normalized)
    
    # --- Calculate Metrics ---
    
    # Calculate MSE on the actual dollar values
    mse = mean_squared_error(actuals_actual, predictions_actual)
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # Calculate MAE (Mean Absolute Error) for another common metric
    mae = np.mean(np.abs(actuals_actual - predictions_actual))
    
    print("\n--- Evaluation Results ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.4f}")

    return predictions_actual, actuals_actual, rmse
def main():
    # Define the exact parameters used during training (critical for model instantiation)
  INPUT_SIZE =5
  HIDDEN_SIZE =80
  NUM_LAYERS =3
  OUTPUT_SIZE=1
  SEQ_LENGTH =60
  

   # Must match the trained model's config
  mlflow.set_experiment("my-dvc-pipeline")
  with mlflow.start_run() as run:  # Start an MLflow run
   try:
  # 1. Load the Scaler
    load_scaler= load_model('./artifacts/scaler.pkl')
    

  # 2. Load the Model
    model = StockPricePredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load('artifacts/model_weights.pth', map_location='cpu'))
    model.eval() # Set model to evaluation mode

  # 3. Prepare Test Data (using the saved data or re-running prepare_data)
  # Assuming you saved the raw test data, load it here.
    test_df = pd.read_csv('./data/raw/test.csv') 
    train_df = pd.read_csv('./data/raw/train.csv')
    df = pd.concat([train_df, test_df], ignore_index=True)
    logging.info("Test data loaded for evaluation.")
    print(test_df.info())
    test_loader,scaler=prepare_data(df,SEQ_LENGTH) 
    print(f"Test loader has {len(test_loader)} batches.")
  # 4. Evaluate
    predictions, actuals, rmse = evaluate_model(model, test_loader,scaler,device='cpu')

    print(f"Final Evaluation RMSE: {rmse:.4f}")
    # 5. Save Metrics
    metrics = {
            'RMSE': float(rmse)
        }
    save_metrics(metrics, 'reports/metrics.json')
    # Save model info
    save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
     # Log the metrics file to MLflow
    mlflow.log_artifact('reports/metrics.json')
    # # D. Log the PyTorch Model (Correct way for PyTorch)
    mlflow.pytorch.log_model(model, "model")
    logging.info("Model logged to MLflow using mlflow.pytorch.log_model.")
            
            
    # # Save model info
    # save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
   
   except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

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
    X, y = create_sequences(features_normalized, target_normalized, seq_length)
    
    # # 4. Split Data (Chronological Split)
    # train_size = int(len(X) * train_split)
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 5. Combine into PyTorch Datasets
    # train_sequences = list(zip(X_train, y_train))
    test_sequences = list(zip(X, y))

    # train_dataset = StockDataset(train_sequences)
    test_dataset = StockDataset(test_sequences)
    
    # 6. Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 7. Return Loaders and the Target Scaler
    # NOTE: You might need to save/return the FEATURE_SCALER as well 
    # if you ever need to analyze or transform the input features separately, 
    # but the TARGET_SCALER is mandatory for the final prediction denormalization.
    logging.info("Data prepared and DataLoader created.")
    return  test_loader, target_scaler


if __name__ == '__main__':
    main()

