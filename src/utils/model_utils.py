import pandas as pd
import numpy as np
import joblib
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, f1_score, precision_score, 
    recall_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.parameters import MODEL_PARAMS, CV_PARAMS, THRESHOLD_PARAMS, MLFLOW_PARAMS, ARTIFACT_PATHS

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training and evaluation class with MLflow integration"""
    
    def __init__(self):
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(MLFLOW_PARAMS['tracking_uri'])
        
        try:
            experiment = mlflow.get_experiment_by_name(MLFLOW_PARAMS['experiment_name'])
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    MLFLOW_PARAMS['experiment_name']
                    
                )
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_id=experiment_id)
            logger.info(f"MLflow experiment set: {MLFLOW_PARAMS['experiment_name']}")
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            logger.warning("Continuing without MLflow logging")
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating model: {model_name}")
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        logger.info(f"Evaluation results for {model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        
        return {
            **metrics,
            'y_proba': y_proba,
            'y_pred': y_pred,
            'threshold': threshold
        }
    
    def _plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legit', 'Fraud'], 
                   yticklabels=['Legit', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_path = f"artifacts/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def find_optimal_threshold(self, y_true: pd.Series, y_proba: np.ndarray, 
                             beta: float = None) -> Tuple[float, float]:
        """Find optimal threshold using F-beta score"""
        if beta is None:
            beta = THRESHOLD_PARAMS['beta']
            
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        
        optimal_idx = np.argmax(f_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_score = f_scores[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F{beta}-score: {optimal_score:.4f})")
        return optimal_threshold, optimal_score
    
    def cross_validate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                           model_name: str) -> Dict[str, List[float]]:
        """Perform cross-validation with MLflow logging"""
        logger.info(f"Performing {CV_PARAMS['n_splits']}-fold cross-validation for {model_name}")
        
        cv_results = {
            'accuracy': [],
            'roc_auc': [],
            'pr_auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'threshold': []
        }
        
        skf = StratifiedKFold(
            n_splits=CV_PARAMS['n_splits'],
            shuffle=CV_PARAMS['shuffle'],
            random_state=CV_PARAMS['random_state']
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            logger.info(f"Processing fold {fold}/{CV_PARAMS['n_splits']}")
            
            # Split data
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            try:
                
                fold_model = clone(model)
            except Exception as e:
                logger.warning(f"Could not clone model: {e}. Creating fresh instance.")
                
                if hasattr(model, '__class__'):
                    try:
                        
                        if not hasattr(model, 'steps'):  
                            fold_model = model.__class__(**model.get_params())
                        else:
                            
                            steps = [(name, clone(estimator)) for name, estimator in model.steps]
                            fold_model = Pipeline(steps)
                    except Exception as inner_e:
                        logger.error(f"Failed to create model copy: {inner_e}")
                        fold_model = model
            
            fold_model.fit(X_tr, y_tr)
            
            y_proba = fold_model.predict_proba(X_val)[:, 1]
            

            opt_thresh, _ = self.find_optimal_threshold(y_val, y_proba)
            y_pred = (y_proba >= opt_thresh).astype(int)

            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            fold_metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_proba),
                'pr_auc': pr_auc,
                'f1': f1_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'threshold': opt_thresh
            }

            for metric, value in fold_metrics.items():
                cv_results[metric].append(value)
            
            logger.info(f"Fold {fold} metrics: F1={fold_metrics['f1']:.4f}, "
                       f"PR-AUC={fold_metrics['pr_auc']:.4f}, "
                       f"Threshold={fold_metrics['threshold']:.4f}")
        
        return cv_results
    
    def train_and_log_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series, 
                           model_name: str, parameters: Dict = None) -> Dict[str, Any]:
        """Train model and log everything to MLflow"""
        try:
            with mlflow.start_run(run_name=model_name):
                logger.info(f"Training {model_name}")

                if parameters:
                    mlflow.log_params(parameters)

                cv_results = self.cross_validate_model(model, X_train, y_train, model_name)

                for metric, values in cv_results.items():
                    mlflow.log_metric(f"cv_{metric}_mean", np.mean(values))
                    mlflow.log_metric(f"cv_{metric}_std", np.std(values))
                
                model.fit(X_train, y_train)
                

                test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
                

                optimal_threshold, _ = self.find_optimal_threshold(
                    y_test, test_metrics['y_proba']
                )
                
                optimized_metrics = self.evaluate_model(
                    model, X_test, y_test, f"{model_name} (Optimized)", optimal_threshold
                )
                
                for metric, value in optimized_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"test_{metric}", value)
                
                try:
                    sample_input = X_test.iloc[:5]  
                    
                    if hasattr(model, 'booster'):  
                        mlflow.xgboost.log_model(
                            model, 
                            "model",
                            input_example=sample_input,
                            signature=mlflow.models.infer_signature(sample_input, model.predict(sample_input))
                        )
                    else:  
                        mlflow.sklearn.log_model(
                            model, 
                            "model",
                            input_example=sample_input,
                            signature=mlflow.models.infer_signature(sample_input, model.predict(sample_input))
                        )
                except Exception as model_log_error:
                    logger.warning(f"Could not log model to MLflow: {model_log_error}")
                    logger.info("Continuing without model logging to MLflow")
                
                try:
                    confusion_matrix_path = self._plot_confusion_matrix(
                        y_test, optimized_metrics['y_pred'], model_name
                    )
                    mlflow.log_artifact(confusion_matrix_path)
                except Exception as artifact_error:
                    logger.warning(f"Could not log confusion matrix: {artifact_error}")
                
                return {
                    'model': model,
                    'cv_results': cv_results,
                    'test_metrics': optimized_metrics,
                    'optimal_threshold': optimal_threshold
                }
                
        except Exception as e:
            logger.error(f"MLflow logging failed: {str(e)}")
            logger.info("Training will continue without MLflow logging")

            logger.info(f"Training {model_name} without MLflow")

            cv_results = self.cross_validate_model(model, X_train, y_train, model_name)

            model.fit(X_train, y_train)

            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)

            optimal_threshold, _ = self.find_optimal_threshold(
                y_test, test_metrics['y_proba']
            )

            optimized_metrics = self.evaluate_model(
                model, X_test, y_test, f"{model_name} (Optimized)", optimal_threshold
            )
            
            return {
                'model': model,
                'cv_results': cv_results,
                'test_metrics': optimized_metrics,
                'optimal_threshold': optimal_threshold
            }


class ModelPredictor:
    """Model prediction class"""
    
    def __init__(self, model_path: str = None, threshold_path: str = None):
        self.model = None
        self.optimal_threshold = None
        
        if model_path:
            self.load_model(model_path)
        if threshold_path:
            self.load_threshold(threshold_path)
    
    def load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            model_path = ARTIFACT_PATHS['model_pipeline']
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_threshold(self, threshold_path: str = None):
        """Load optimal threshold"""
        if threshold_path is None:
            threshold_path = ARTIFACT_PATHS['optimal_threshold']
        
        try:
            self.optimal_threshold = joblib.load(threshold_path)
            logger.info(f"Optimal threshold loaded: {self.optimal_threshold}")
        except Exception as e:
            logger.error(f"Error loading threshold: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame, use_optimal_threshold: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        y_proba = self.model.predict_proba(X)[:, 1]

        if use_optimal_threshold and self.optimal_threshold is not None:
            threshold = self.optimal_threshold
        else:
            threshold = 0.5
        
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred, y_proba
    
    def predict_single(self, sample: Dict, use_optimal_threshold: bool = True) -> Tuple[int, float]:
        """Predict single sample"""
        from src.utils.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessing_artifacts()
        
        X_processed = preprocessor.preprocess_new_data(sample)
        
        y_pred, y_proba = self.predict(X_processed, use_optimal_threshold)
        
        return int(y_pred[0]), float(y_proba[0])


def save_model_artifacts(model, optimal_threshold: float):
    """Save model and related artifacts"""
    logger.info("Saving model artifacts")

    joblib.dump(model, ARTIFACT_PATHS['model_pipeline'])

    joblib.dump(optimal_threshold, ARTIFACT_PATHS['optimal_threshold'])
    
    logger.info("Model artifacts saved successfully")