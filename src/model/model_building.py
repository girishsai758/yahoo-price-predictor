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
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
# ========================== creating sequences ==========================
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
# ========================== Custom Dataset Class ==========================
class StockDataset(Dataset):
    """Custom Dataset class for stock data."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# ========================== data preparing ==========================
def prepare_data(df, seq_length,train_split):
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
    
    # 4. Split Data (Chronological Split)
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 5. Combine into PyTorch Datasets
    train_sequences = list(zip(X_train, y_train))
    test_sequences = list(zip(X_test, y_test))

    train_dataset = StockDataset(train_sequences)
    test_dataset = StockDataset(test_sequences)
    
    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 7. Return Loaders and the Target Scaler
    # NOTE: You might need to save/return the FEATURE_SCALER as well 
    # if you ever need to analyze or transform the input features separately, 
    # but the TARGET_SCALER is mandatory for the final prediction denormalization.
    
    return train_loader, test_loader, target_scaler

# Example Usage (You must load your Yahoo data into a pandas DataFrame first)
# For example: df = pd.read_csv('AAPL.csv')
# train_dataset, test_dataset, scaler = prepare_data(df, seq_length=10)

# ========================== LTSM MODEL ==========================
# ========================== LTSM MODEL ==========================
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
        print("output x shape:", out.shape)
        return out
#========================= TRAINING AND EVALUATION CODE ==========================

# --- Hyperparameters ---
SEQ_LENGTH = 60
INPUT_SIZE = 5        # we are using 5 features: Open, High, Low, Close, Volume
HIDDEN_SIZE = 80
NUM_LAYERS = 3
OUTPUT_SIZE = 1       # Predicting one value: the price of the next day
#BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 100
train_split=0.8

# # --- Setup ---
# # Assume df is loaded and prepared from Section 1
# # train_dataset, test_dataset, scaler = prepare_data(df, SEQ_LENGTH) 

# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Setup ---
# Force the device to CPU. This is the only mandatory change.
device = torch.device('cpu') 
print(f"Using device: {device}")

model = StockPricePredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()  # Mean Squared Error is common for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
def train_model(model, train_loader, criterion, optimizer, epochs, device,weights_path):
    """Train the LSTM model."""
    try:
      model.train()
      for epoch in range(epochs):
        running_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # # Forward pass
            outputs = model(sequences)
            labels = labels.view_as(outputs) # Ensure shape alignment (as discussed previously)
            # Load the weights from the training process
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.mean().item()

        
        # Calculate average loss for the entire epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        # Save the model's state dictionary
      torch.save(model.state_dict(), weights_path)
      logging.info(f"New best model saved to {weights_path} with loss: {avg_train_loss:.4f}")
        
      logging.info("Training completed successfully.") 
      return weights_path 
    except Exception as e:
      logging.error(f"An error occurred during training: {e}")
      raise 
#========================= load data ==========================
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

#========================= save model==========================
def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

df1 = load_data('./data/raw/train.csv')
df2 = load_data('./data/raw/test.csv')
df = pd.concat([df1, df2], ignore_index=True)
if __name__ == "__main__":
    # Define your main directory for artifacts
    ARTIFACTS_DIR = 'artifacts'
    os.makedirs(ARTIFACTS_DIR, exist_ok=True) # Ensure the directory exists
    # 3. TRAINING
    WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, 'model_weights.pth')
    train_loader,test_loader,scaler=prepare_data(df,SEQ_LENGTH,train_split)

    best_weights_path = train_model(model, train_loader, criterion, optimizer, EPOCHS, device, WEIGHTS_FILE)
   #predictions, actuals, rmse = evaluate_model(model, test_loader, scaler, device)
  # logging.info(f"Final RMSE on test set: {rmse:.4f}")
    try:
    # A. Save the Scaler (Essential for inverse transform)
        SCALER_FILE = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
        with open(SCALER_FILE, 'wb') as f:
           pickle.dump(scaler, f)
        logging.info(f"Scaler saved to {SCALER_FILE}")
    except Exception as e:
       logging.error(f"Failed to save final artifacts: {e}")