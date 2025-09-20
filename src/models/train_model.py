import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.config.parameters import (
    DATA_PARAMS, MODEL_PARAMS, SMOTE_PARAMS, 
    ARTIFACT_PATHS, RAW_DATA_DIR
)
from src.utils.data_preprocessing import DataPreprocessor
from src.utils.feature_engineering import FeatureEngineer
from src.utils.model_utils import ModelTrainer, save_model_artifacts
from src.utils.monitoring import setup_logging, log_model_performance

warnings.filterwarnings('ignore')

def main():
    """Main training pipeline"""
    
    logger = setup_logging()
    logger.info("="*50)
    logger.info("FRAUD DETECTION MODEL TRAINING STARTED")
    logger.info("="*50)
    
    try:
        
        logger.info("Step 1: Data Loading and Preprocessing")
        preprocessor = DataPreprocessor()
        
        
        df = preprocessor.load_data(DATA_PARAMS['raw_data_path'])
        
        
        df = preprocessor.basic_preprocessing(df)
        
        
        logger.info("Step 2: Feature Engineering")
        feature_engineer = FeatureEngineer()
        df_processed = feature_engineer.engineer_features(df, is_training=True)
        
        
        preprocessor.scaler = feature_engineer.scaler
        preprocessor.legit_amount_mean = feature_engineer.legit_amount_mean
        preprocessor.amount_bin_edges = feature_engineer.amount_bin_edges
        preprocessor.top_corr_features = feature_engineer.top_corr_features
        
        
        if hasattr(feature_engineer, 'time_scaler'):
            preprocessor.time_scaler = feature_engineer.time_scaler
        
        
        logger.info("Step 3: Train-Test Split")
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        
        preprocessor.save_preprocessing_artifacts(df_processed, X_train)
        
        
        logger.info("Step 4: Model Training Setup")
        trainer = ModelTrainer()
        
        
        base_models = [
            ('xgb', xgb.XGBClassifier(**MODEL_PARAMS['xgb'])),
            ('cat', CatBoostClassifier(**MODEL_PARAMS['catboost'])),
            ('lgb', lgb.LGBMClassifier(**MODEL_PARAMS['lightgbm']))
        ]
        
        meta_model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
        stacking_classifier = StackingClassifier(
            estimators=base_models, 
            final_estimator=meta_model, 
            cv=5
        )
        
       
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', SMOTE(**SMOTE_PARAMS)),
            ('classifier', stacking_classifier)
        ])
        
       
        logger.info("Step 5: Model Training and Evaluation")
        
        
        results = trainer.train_and_log_model(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name="Stacking_Ensemble_SMOTE",
            parameters={
                **MODEL_PARAMS,
                **SMOTE_PARAMS,
                'pipeline_steps': ['imputer', 'smote', 'stacking_classifier']
            }
        )
        
        
        logger.info("Step 6: Saving Model Artifacts")
        save_model_artifacts(
            model=results['model'],
            optimal_threshold=results['optimal_threshold']
        )
        
        
        logger.info("Step 7: Final Results Summary")
        log_model_performance(
            metrics={k: v for k, v in results['test_metrics'].items() 
                     if isinstance(v, (int, float))},
            model_name="Final Stacking Ensemble"
        )
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
       
        print("\n" + "="*60)
        print("FRAUD DETECTION MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset Shape: {df.shape}")
        print(f"Features after Engineering: {X_train.shape[1]}")
        print(f"Training Samples: {X_train.shape[0]}")
        print(f"Test Samples: {X_test.shape[0]}")
        print(f"Fraud Cases (Train): {y_train.sum()}")
        print(f"Fraud Cases (Test): {y_test.sum()}")
        print("\nKey Metrics (Test Set):")
        print(f"  Accuracy:  {results['test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['test_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['test_metrics']['f1']:.4f}")
        print(f"  ROC-AUC:   {results['test_metrics']['roc_auc']:.4f}")
        print(f"  PR-AUC:    {results['test_metrics']['pr_auc']:.4f}")
        print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
        print(f"\nModel saved to: {ARTIFACT_PATHS['model_pipeline']}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results = main()