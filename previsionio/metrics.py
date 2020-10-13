from enum import Enum


class Regression(Enum):
    """
    Metrics for regression projects
    Available metrics in prevision:
            rmse, mape, rmsle, mse, mae
    """
    RMSE = 'rmse'
    """Root Mean Squared Error"""
    RMSLE = 'rmsle'
    """Root Mean Squared Logarithmic Error"""
    MAPE = 'mape'
    """Mean Average Percentage Error"""
    MAE = 'mae'
    """Mean Average Error"""
    MSE = 'mse'
    """Mean squared Error"""


class Classification(Enum):
    """
    Metrics for classification projects
    Available metrics in prevision:
        auc, log_loss, error_rate_binary
    """
    AUC = 'auc'
    """Area Under ROC Curve"""
    log_loss = 'log_loss'
    """Logarithmic Loss"""
    error_rate = 'error_rate_binary'
    """Error rate"""


class MultiClassification(Enum):
    """
    Metrics for multiclassification projects
    """
    log_loss = 'log_loss'
    """Logarithmic Loss"""
    error_rate = 'error_rate_multi'
    """Multi-class Error rate"""


class Clustering(Enum):
    """
    Metrics for clustering projects
    """
    silhouette = 'silhouette'
    """Clustering silhouette metric"""
    calinski_harabaz = 'calinski_harabaz'
    """Clustering calinski_harabaz metric"""
