
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly detection methods (IQR, Z-score, rule-based)
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Utility class for statistical and rule-based anomaly checks.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.anomalies = []
        
    def check_z_score(self, column: str, threshold: float = 3.0) -> List[int]:
        """
        Detect outliers using Z-score method.
        Returns indices of outliers.
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found for Z-score check")
            return []
            
        data = self.df[column].dropna()
        if len(data) < 2:
            return []
            
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return []
            
        z_scores = (data - mean) / std
        outliers = data[abs(z_scores) > threshold]
        
        if not outliers.empty:
            logger.info(f"Found {len(outliers)} Z-score outliers in {column}")
            
        return outliers.index.tolist()

    def check_iqr(self, column: str, multiplier: float = 1.5) -> List[int]:
        """
        Detect outliers using IQR method.
        Returns indices of outliers.
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found for IQR check")
            return []
            
        data = self.df[column].dropna()
        if len(data) < 2:
            return []
            
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        if not outliers.empty:
            logger.info(f"Found {len(outliers)} IQR outliers in {column}")
            
        return outliers.index.tolist()

    def check_rule(self, column: str, rule_func, rule_name: str) -> List[int]:
        """
        Apply a custom rule to a column.

        `rule_func` should return True for valid values and False for anomalies.
        """
        if column not in self.df.columns:
            return []
            
        mask = ~self.df[column].apply(rule_func)
        outliers = self.df[mask]
        
        if not outliers.empty:
            logger.info(f"Found {len(outliers)} anomalies for rule '{rule_name}' in {column}")
            
        return outliers.index.tolist()
