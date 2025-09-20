import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"


for directory in [DATA_DIR, RAW_DATA_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data parameters
DATA_PARAMS = {
    'raw_data_path': RAW_DATA_DIR / 'creditcard.csv',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}


FEATURE_PARAMS = {
    'v_columns': [f'V{i}' for i in range(1, 29)],
    'time_bins': [0, 6, 12, 18, 24],
    'time_labels': ['night', 'morning', 'afternoon', 'evening'],
    'amount_quantiles': 4,
    'amount_labels': ['low', 'medium', 'high', 'very_high'],
    'top_corr_features_count': 4
}


MODEL_PARAMS = {
    'xgb': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    },
    'catboost': {
        'iterations': 100,
        'depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': 0
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'min_child_samples': 10,
        'verbose': -1
    },
    'logistic_regression': {
        'random_state': 42
    }
}


SMOTE_PARAMS = {
    'sampling_strategy': 0.1,
    'random_state': 42
}


CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

#
THRESHOLD_PARAMS = {
    'beta': 2.0  
}


MLFLOW_PARAMS = {
    'experiment_name': 'fraud_detection',
    'tracking_uri': f'file:///{str(BASE_DIR / "mlruns").replace(os.sep, "/")}',
    'artifact_location': None  
}


ARTIFACT_PATHS = {
    'model_pipeline': ARTIFACTS_DIR / 'fraud_model_pipeline.pkl',
    'scaler': ARTIFACTS_DIR / 'scaler.pkl',
    'time_scaler': ARTIFACTS_DIR / 'time_scaler.pkl',
    'legit_amount_mean': ARTIFACTS_DIR / 'legit_amount_mean.pkl',
    'amount_bin_edges': ARTIFACTS_DIR / 'amount_bin_edges.pkl',
    'top_corr_features': ARTIFACTS_DIR / 'top_corr_features.pkl',
    'optimal_threshold': ARTIFACTS_DIR / 'optimal_threshold.pkl',
    'feature_columns': ARTIFACTS_DIR / 'feature_columns.pkl'
}


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'fraud_detection.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}