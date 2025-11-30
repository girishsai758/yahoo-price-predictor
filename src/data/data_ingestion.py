# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


from sklearn.model_selection import train_test_split
import yaml
import logging
import sys
import os

#since src.logger is not working we are adding the project root manually
# Add project root to path
project_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
print(os.path.dirname(sys.executable))
from src.logger import logging
from src.connections import s3_connections


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        logging.info("Loading parameters from ")
        project_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        with open(params_path, 'r') as file:
            logging.info("Loading parameters from ")
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        logging.info("intializing data saving function" )
        raw_data_path = os.path.join(data_path, 'raw')
        # --- NEW DEBUG STEP ---
        
        logging.info(f"Target path being used: {raw_data_path}")
        #i added this line so that path moves to main project root
        project_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        os.makedirs(raw_data_path, exist_ok=True)
        logging.debug('Creating directory %s if it does not exist', raw_data_path)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        project_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        params = load_params(params_path='params.yaml')
        logging.info("params loading started")
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        logging.info("params loaded successfully. Test size: %s", test_size)
       # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        s3 = s3_connections.s3_operations("yahoo-price", "AKIAWS34RZ7SBDPXRUQL", "WhCGxqnNX+NU7AHxmJi/e4ni4Uz8W28A/So27hK7")
        df = s3.fetch_file_from_s3("historical_data.csv")



        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logging.info("entering data saving function")
        print(f"Train data has {len(train_data)} records and Test data has {len(test_data)} records.")
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()