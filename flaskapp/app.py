from flask import Flask, render_template, request
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
# --- 1. CONFIGURATION AND INITIALIZATION ---

# Define the exact model hyperparameters used during training
# IMPORTANT: These must match the training script exactly.
INPUT_SIZE = 5
HIDDEN_SIZE = 80
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 60

# --- MLFLOW CONFIGURATION ---
MLFLOW_MODEL_NAME = "my_model" # This must match the name used in register_model.py
MLFLOW_MODEL_STAGE = "Staging" # Load the model from the "Staging" stage
# Assumes the scaler was logged as a root artifact in the MLflow run
SCALER_ARTIFACT_PATH = "scaler.pkl" 
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

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' 

# Global variables to hold the loaded model and scaler
model = None
scaler = None
#pre-processing the data
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
# --- 3. ARTIFACT LOADING (MLFLOW VERSION) ---

def load_artifacts(MLFLOW_MODEL_NAME,MLFLOW_MODEL_STAGE):
    """Loads the model and scaler from the MLflow Model Registry."""
    
    # if model is not None and scaler is not None:
    #     return True

    try:
        logging.info("Attempting to load artifacts from MLflow Model Registry...")

        # Initialize MLflow Client (requires MLFLOW_TRACKING_URI to be set in environment)
        client =MlflowClient()

        # 1. Get the latest model version in the specified stage
        latest_version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE])
        if not latest_version:
            raise Exception(f"No model found in MLflow Registry for name '{MLFLOW_MODEL_NAME}' and stage '{MLFLOW_MODEL_STAGE}'.")
            
        model_version = latest_version[0]
        run_id = model_version.run_id
        
        # 2. Load the PyTorch Model using the models:/ URI
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
        model.eval()
        logging.info(f"Model loaded successfully from MLflow (Version {model_version.version}, Run ID: {run_id}).")
        
        # # 3. Download and Load the Scaler Artifact
        # # Use a temporary directory to download the artifact
        # temp_dir = "./temp_mlflow_artifacts"
        # os.makedirs(temp_dir, exist_ok=True)

        # scaler_path_downloaded = mlflow.artifacts.download_artifacts(
        #     run_id=run_id, 
        #     artifact_path=SCALER_ARTIFACT_PATH, 
        #     dst_path=temp_dir
        # )
        #will pick up the scaler from root artifact
        scaler_path_downloaded = './artifacts/scaler.pkl'
        with open(scaler_path_downloaded, 'rb') as f:
            scaler = pickle.load(f)
            
        logging.info(f"Scaler loaded successfully from MLflow run {run_id} artifact.")
        
        # # 4. Cleanup temporary files
        # os.remove(scaler_path_downloaded)
        # try:
        #     os.rmdir(temp_dir)
        # except OSError:
        #     pass 

        # return True

    except Exception as e:
        logging.error(f"Failed to load artifacts from MLflow: {e}")
        # --- Fallback to Dummy Objects ---
        model = StockPricePredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model.eval()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Fit the scaler to some dummy data so inverse_transform works
        dummy_data = np.array([100.0, 200.0, 300.0]).reshape(-1, 1)
        scaler.fit(dummy_data)
        logging.warning("Falling back to DUMMY MODEL and SCALER. Prediction will be inaccurate. Check MLflow configuration.")
        return False
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
    test_dataset = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0) #here we have to send only features normalised input size would be (1,60,5)
    
    # 6. Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 7. Return Loaders and the Target Scaler
    # NOTE: You might need to save/return the FEATURE_SCALER as well 
    # if you ever need to analyze or transform the input features separately, 
    # but the TARGET_SCALER is mandatory for the final prediction denormalization.
    logging.info("Data prepared and DataLoader created.")
    return  test_dataset, target_scaler
# # Load artifacts at startup
# artifacts_loaded = load_artifacts(MLFLOW_MODEL_NAME,MLFLOW_MODEL_STAGE)
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
    df_1=preprocess_data(data_df) #it removes unwanted columns
    
    print("Data pre-processing completed for prediction.")
    normalized_data, scaler2 = prepare_data(df_1, SEQUENCE_LENGTH) # here sequence length is 60
    # 1. Normalize the data using the pre-fitted scaler
    # normalized_data = scaler.transform(close_prices)
    print(len(normalized_data)) # normalized data will have a shape of (1,60,5)
    print("Data normalization completed for prediction.")
    # # 2. Convert to PyTorch tensor (shape: 1, 60, 1)
    # input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
    # logging.info(f"Input tensor shape for prediction: {normalized_data.shape}")
    # 3. Make the prediction
    with torch.no_grad():
     
        # print(f"Input tensor shape for prediction: {normalized_data.shape}")
       
        output= model(normalized_data) # output shape: (1, 1)
        
        prediction_normalized = output.cpu().numpy()  # will be converted to numpy array
        # print("prediction_normalized shape:", prediction_normalized.shape)
    logging.info(f"Model prediction (normalized): {prediction_normalized.item()}")
    # predictions_normalized = np.array(predictions_normalized)
    # 4. Inverse transform to get the actual price
    prediction_original_scale = scaler.inverse_transform(prediction_normalized)
    logging.info(f"Model prediction (original scale): {prediction_original_scale}")
    return prediction_original_scale.item() #.item() to return as float

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-lg bg-white shadow-2xl rounded-xl p-8 space-y-8">
        <header class="text-center">
            <h1 class="text-3xl font-bold text-gray-800">Next-Day Stock Price Prediction</h1>
            <p class="text-gray-500 mt-2">Upload a CSV file with the last 60 days of 'Close' prices.</p>
        </header>

        {% if error %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative" role="alert">
            <strong class="font-bold">Error!</strong>
            <span class="block sm:inline">{{ error }}</span>
        </div>
        {% endif %}

        {% if prediction %}
        <div class="text-center p-6 bg-green-50 rounded-lg shadow-inner">
            <h2 class="text-xl font-semibold text-green-700">Prediction Complete!</h2>
            <p class="text-4xl font-extrabold text-green-900 mt-3">
                ${{ prediction | round(4) }}
            </p>
            <p class="text-gray-600 mt-1">Predicted closing price for Day 61</p>
        </div>
        {% endif %}

        {% if message %}
        <div class="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded-lg relative" role="alert">
            <strong class="font-bold">Warning!</strong>
            <span class="block sm:inline">{{ message }}</span>
        </div>
        {% endif %}

        <form method="POST" action="{{ url_for('predict') }}" enctype="multipart/form-data" class="space-y-6">
            <div>
                <label for="csv_file" class="block text-sm font-medium text-gray-700 mb-2">Select 60-Day CSV:</label>
                <input id="csv_file" name="csv_file" type="file" required 
                       class="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 
                              text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100">
                <p class="mt-1 text-xs text-gray-500">Must be a CSV file with exactly 60 rows and a 'Close' column.</p>
            </div>
            
            <button type="submit" 
                    class="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-md text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
                Get Prediction for Day 61
            </button>
        </form>
    </div>
</body>
</html>
"""
# Flask Web Application
app = Flask(__name__)

# Below code block is for production use
# -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
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
            
@app.route('/', methods=['GET'])
def index():
    # Pass a message if artifacts failed to load
    message = None
    if model is None or scaler is None:
        message = "Artifacts (model/scaler) could not be loaded from MLflow. Running with DUMMY values. Prediction may be inaccurate."
        
    return render_template_string(HTML_TEMPLATE, message=message)

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return render_template_string(HTML_TEMPLATE, error="No file part in the request.")

    file = request.files['csv_file']
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, error="No file selected.")

    if file and file.filename.endswith('.csv'):
        try:
            # Read the file content into an in-memory string buffer
            stream = io.StringIO(file.stream.read().decode("UTF8"))
            data_df = pd.read_csv(stream)
            
            # Check for required length (60 days)
            if len(data_df) != SEQUENCE_LENGTH:
                raise ValueError(f"Input file must contain exactly {SEQUENCE_LENGTH} rows. Found {len(data_df)}.")
                
            # Perform prediction
            predicted_price = predict_next_day(data_df, model, scaler)
            
            # Pass the result back to the template
            return render_template_string(HTML_TEMPLATE, prediction=predicted_price)

        except ValueError as e:
            logging.error(f"Prediction ValueError: {e}")
            return render_template_string(HTML_TEMPLATE, error=str(e))
        except Exception as e:
            logging.error(f"Prediction General Error: {e}")
            return render_template_string(HTML_TEMPLATE, error=f"An unexpected error occurred during prediction: {e}")
    
    return render_template_string(HTML_TEMPLATE, error="Invalid file format. Please upload a CSV file.")


if __name__ == '__main__':
    # NOTE: You must have your MLFLOW_TRACKING_URI environment variable set 
    # (e.g., to a local directory or a remote server like DagsHub) 
    # for MLflowClient to connect successfully.
    # Example: os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    
    app.run(debug=True, host="0.0.0.0", port=5000)