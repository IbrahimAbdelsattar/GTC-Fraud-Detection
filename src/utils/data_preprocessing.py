import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Union, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config.parameters import DATA_PARAMS, ARTIFACT_PATHS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing class for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.legit_amount_mean = None
        self.amount_bin_edges = None
        self.top_corr_features = None
        self.feature_columns = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file path"""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic preprocessing steps"""
        logger.info("Starting basic preprocessing")
        
        
        missing_values = df.isnull().sum().sum()
        logger.info(f"Total missing values: {missing_values}")
        
        
        class_distribution = df['Class'].value_counts()
        logger.info(f"Class distribution:\n{class_distribution}")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets")
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATA_PARAMS['test_size'],
            stratify=y if DATA_PARAMS['stratify'] else None,
            random_state=DATA_PARAMS['random_state']
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Fraud cases: {y_train.sum()}")
        logger.info(f"Test set: {X_test.shape[0]} samples, Fraud cases: {y_test.sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessing_artifacts(self, df: pd.DataFrame, X_train: pd.DataFrame):
        """Save preprocessing artifacts for later use"""
        logger.info("Saving preprocessing artifacts")

        joblib.dump(self.scaler, ARTIFACT_PATHS['scaler'])

        joblib.dump(self.legit_amount_mean, ARTIFACT_PATHS['legit_amount_mean'])

        joblib.dump(self.amount_bin_edges, ARTIFACT_PATHS['amount_bin_edges'])

        joblib.dump(self.top_corr_features, ARTIFACT_PATHS['top_corr_features'])

        joblib.dump(X_train.columns.tolist(), ARTIFACT_PATHS['feature_columns'])

        logger.info("Preprocessing artifacts saved successfully")

    
    def load_preprocessing_artifacts(self):
        """Load preprocessing artifacts"""
        try:
            self.scaler = joblib.load(ARTIFACT_PATHS['scaler'])
            self.legit_amount_mean = joblib.load(ARTIFACT_PATHS['legit_amount_mean'])
            self.amount_bin_edges = joblib.load(ARTIFACT_PATHS['amount_bin_edges'])
            self.top_corr_features = joblib.load(ARTIFACT_PATHS['top_corr_features'])
            self.feature_columns = joblib.load(ARTIFACT_PATHS['feature_columns'])
            logger.info("Preprocessing artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessing artifacts: {str(e)}")
            raise
    
    def preprocess_new_data(self, raw_input: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess new data for prediction"""
        if isinstance(raw_input, dict):
            
            normalized = {
                k: (v if isinstance(v, (list, np.ndarray, pd.Series)) else [v])
                for k, v in raw_input.items()
            }
            df_new = pd.DataFrame(normalized)
        else:
            df_new = raw_input.copy()


        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        for col in required_columns:
            if col not in df_new.columns:
                raise ValueError(f"Missing required feature: {col}")

        from src.utils.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()

        
        fe.scaler = self.scaler
        fe.legit_amount_mean = self.legit_amount_mean
        fe.amount_bin_edges = self.amount_bin_edges
        fe.top_corr_features = self.top_corr_features

        df_processed = fe.engineer_features(df_new, is_training=False)

    
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

       
        df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

        
        df_processed = df_processed[self.feature_columns]

        if list(df_processed.columns) != list(self.feature_columns):
            raise ValueError(
                f"Feature mismatch!\nExpected: {self.feature_columns}\nGot: {df_processed.columns.tolist()}"
            )

        return df_processed






def validate_input_data(data: Union[Dict, pd.DataFrame]) -> bool:
    """Validate input data format and required fields"""
    required_fields = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    if isinstance(data, dict):
        missing_fields = [field for field in required_fields if field not in data]
    else:
        missing_fields = [field for field in required_fields if field not in data.columns]
    
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return False
    
    return True