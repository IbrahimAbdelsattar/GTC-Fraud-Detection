import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from src.config.parameters import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.legit_amount_mean = None
        self.amount_bin_edges = None
        self.top_corr_features = None
        
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering")
        
        df_processed = df.copy()
        
        df_processed = self._scale_features(df_processed, is_training)

        df_processed = self._create_temporal_features(df_processed)
 
        df_processed = self._create_amount_features(df_processed, is_training)

        df_processed = self._create_v_aggregated_features(df_processed)

        df_processed = self._create_feature_interactions(df_processed, is_training)

        df_processed = self._cleanup_columns(df_processed)

        df_processed = self._reorder_columns(df_processed)
        
        logger.info(f"Feature engineering completed. New shape: {df_processed.shape}")
        return df_processed
    
    def _scale_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scale Amount and Time features with one scaler"""
        if is_training:

            df[['scaled_amount', 'scaled_time']] = self.scaler.fit_transform(df[['Amount', 'Time']])
        else:
           
            df[['scaled_amount', 'scaled_time']] = self.scaler.transform(df[['Amount', 'Time']])

        return df

    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from Time"""
       
        df['hour_of_day'] = (df['Time'] % 86400) // 3600
        
        df['time_bin'] = pd.cut(
            df['hour_of_day'], 
            bins=FEATURE_PARAMS['time_bins'],
            labels=FEATURE_PARAMS['time_labels'],
            include_lowest=True
        )

        df = pd.get_dummies(df, columns=['time_bin'], drop_first=True)
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Create amount-based features"""
        if is_training:

            if 'Class' in df.columns:
                self.legit_amount_mean = df[df['Class'] == 0]['scaled_amount'].mean()
            else:
                self.legit_amount_mean = df['scaled_amount'].mean()
                
            df['amount_bin'] = pd.qcut(
                df['scaled_amount'], 
                q=FEATURE_PARAMS['amount_quantiles'],
                labels=FEATURE_PARAMS['amount_labels']
            )
            
            
            self.amount_bin_edges = df['scaled_amount'].quantile([0, 0.25, 0.5, 0.75, 1]).values
        else:
            
            df['amount_bin'] = pd.cut(
                df['scaled_amount'],
                bins=self.amount_bin_edges,
                labels=FEATURE_PARAMS['amount_labels'],
                include_lowest=True
            )

        df['amount_deviation'] = df['scaled_amount'] - self.legit_amount_mean
        
        df = pd.get_dummies(df, columns=['amount_bin'], drop_first=True)
        
        return df
    
    def _create_v_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from V1-V28"""
        v_columns = FEATURE_PARAMS['v_columns']
        
        df['mean_V'] = df[v_columns].mean(axis=1)
        df['std_V'] = df[v_columns].std(axis=1)
        
        return df
    
    def _create_feature_interactions(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Create feature interactions based on correlation with target"""
        if is_training and 'Class' in df.columns:
            numerical_cols = ['scaled_time', 'scaled_amount', 'hour_of_day', 
                            'amount_deviation', 'mean_V', 'std_V'] + FEATURE_PARAMS['v_columns']
            
            corr = df[numerical_cols + ['Class']].corr()['Class'].abs().sort_values(ascending=False)
            self.top_corr_features = corr[1:FEATURE_PARAMS['top_corr_features_count']+1].index.tolist()
        
   
            if self.top_corr_features:
                for i, f1 in enumerate(self.top_corr_features[:2]):
                    for f2 in self.top_corr_features[i+1:3]:
                        col_name = f'{f1}_{f2}_interaction'
                        if col_name not in df.columns:  
                            df[col_name] = df[f1] * df[f2]

        
        return df
    
    def _cleanup_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove original Time and Amount columns"""
        columns_to_drop = ['Time', 'Amount']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df.drop(existing_columns_to_drop, axis=1, inplace=True)
        return df
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns in a logical order"""
        # Base features
        base_cols = ['scaled_time', 'scaled_amount', 'hour_of_day', 'amount_deviation', 'mean_V', 'std_V']
        
        # V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        # Categorical dummy columns
        categorical_cols = [col for col in df.columns if col.startswith('time_bin_') or col.startswith('amount_bin_')]
        
        # Interaction columns
        interaction_cols = [col for col in df.columns if '_interaction' in col]
        
        # Target column (if exists)
        target_col = ['Class'] if 'Class' in df.columns else []
        
        # Reorder
        column_order = base_cols + v_cols + categorical_cols + interaction_cols + target_col
        df = df.loc[:, ~df.columns.duplicated()]
        
        existing_columns = [col for col in column_order if col in df.columns]
        
        return df[existing_columns]


def create_feature_summary(df: pd.DataFrame):
    """Create a summary of engineered features"""
    summary = {
        'total_features': len(df.columns),
        'scaled_features': len([col for col in df.columns if col.startswith('scaled_')]),
        'temporal_features': len([col for col in df.columns if 'time' in col or 'hour' in col]),
        'amount_features': len([col for col in df.columns if 'amount' in col]),
        'v_aggregated_features': len([col for col in df.columns if col in ['mean_V', 'std_V']]),
        'interaction_features': len([col for col in df.columns if '_interaction' in col]),
        'categorical_dummies': len([col for col in df.columns if col.endswith('_')]),
        'original_v_features': len([col for col in df.columns if col.startswith('V') and col[1:].isdigit()])
    }
    
    return summary