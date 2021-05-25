from enum import Enum


class Regression(Enum):
    """
    Metrics for regression projects
    Available metrics in prevision: rmse, mape, rmsle, mse, mae
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
    RMSPE = 'rmspe'
    """Root Mean Squared Percentage Error"""
    MER = 'mer'
    """Median Absolute Error"""
    R2 = 'R2'
    """R2 Error"""
    SMAPE = 'smape'
    """Symmetric Mean Absolute Percentage Error"""


class Classification(Enum):
    """
    Metrics for classification projects
    Available metrics in prevision: auc, log_loss, error_rate_binary
    """
    AUC = 'auc'
    """Area Under ROC Curve"""
    log_loss = 'log_loss'
    """Logarithmic Loss"""
    error_rate = 'error_rate_binary'
    """Error rate"""
    accuracy = 'accuracy'
    """Accuracy"""
    F05 = 'F05'
    """F05 Score"""
    F1 = 'F1'
    """Balanced F-score"""
    F2 = 'F2'
    """F2 Score"""
    F3 = 'F3'
    """F3 Score"""
    F4 = 'F4'
    """F4 Score"""
    MCC = 'mcc'
    """Matthews correlation coefficient"""
    gini = 'gini'
    """Gini score"""
    AUCPR = 'aucpr'
    """precision recall area under the curve score"""
    Lift01 = 'lift_at_0.1'
    """lift at ratio 0.1"""
    Lift02 = 'lift_at_0.2'
    """lift at ratio 0.2"""
    Lift03 = 'lift_at_0.3'
    """lift at ratio 0.3"""
    Lift04 = 'lift_at_0.4'
    """lift at ratio 0.4"""
    Lift05 = 'lift_at_0.5'
    """lift at ratio 0.5"""
    Lift06 = 'lift_at_0.6'
    """lift at ratio 0.6"""
    Lift07 = 'lift_at_0.7'
    """lift at ratio 0.7"""
    Lift08 = 'lift_at_0.8'
    """lift at ratio 0.8"""
    Lift09 = 'lift_at_0.9'
    """lift at ratio 0.9"""


class MultiClassification(Enum):
    """
    Metrics for multiclassification projects
    """
    log_loss = 'log_loss'
    """Logarithmic Loss"""
    error_rate = 'error_rate_multi'
    """Multi-class Error rate"""
    macroF1 = 'macroF1'
    """balanced F-score"""
    AUC = 'auc'
    """Area Under ROC Curve"""
    accuracy = 'accuracy'
    """accuracy"""
    qkappa = 'qkappa'
    """quadratic weighted kappa"""
    MAP3 = 'map_at_3'
    """qmean average precision @3"""
    MAP5 = 'map_at_5'
    """qmean average precision @5"""
    MAP10 = 'map_at_10'
    """qmean average precision @10"""


class Clustering(Enum):
    """
    Metrics for clustering projects
    """
    silhouette = 'silhouette'
    """Clustering silhouette metric"""
    calinski_harabaz = 'calinski_harabaz'
    """Clustering calinski_harabaz metric"""


class TextSimilarity(Enum):
    """
    Metrics for text similarity projects
    """
    accuracy_at_k = 'accuracy_at_k'
    """Accuracy at K"""
    mrr_at_k = 'mrr_at_k'
    """Mean Reciprocal Rank at K"""
