from typing import Union
from previsionio.usecase_version import ClassicUsecaseVersion
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score, precision_score


def cv_classif_analysis(usecase: ClassicUsecaseVersion, thresh: float = None, step: Union[int, float] = 1000):
    '''Get metrics on a CV file retrieved from the platform for a binary classification usecase

    Args:
        usecase (Usecase): usecase to analyze
        thresh (float, optional): threshold to use (if none is provided, optimal threshold will be computed
            given F1 score)
        step (int): number of iterations required to find optimal threshold (1000 by default = 0.1% resolution per fold)
    Returns:
        metrics computed for the CV (Dataframe)
    '''
    step = float(step)

    # get cv
    cv = usecase.get_cv()

    if '__fold__' not in cv.columns:
        raise Exception('The fold column is missing in the cv dataframe')

    fold = cv['__fold__'].unique()
    AUC = []
    f1 = []
    f1_fold = []
    precision_fold = []
    recall_fold = []
    threshold = []

    target_col_name = usecase.column_config.target_column
    assert target_col_name
    pred_target_col_name = 'pred_' + target_col_name

    for i in sorted(fold):
        f = cv['__fold__'] == i
        actual = cv[target_col_name][f].values
        predicted = cv[pred_target_col_name][f].values

        # compute auc per fold
        AUC.append(roc_auc_score(actual, predicted))

        # find optimal threshold for F1 score per fold
        f1 = []
        for j in range(1, int(step) + 1):
            idx = predicted >= j / step
            p = np.zeros(len(predicted))
            p[idx] = 1
            f1.append((j, fbeta_score(actual, p, beta=1)))
        f1 = pd.DataFrame(f1, columns=['j', 'f1'])

        # best threshold is the one that maximises f1 (if not provided by user)
        if thresh is None:
            threshold.append(f1['j'][f1['f1'].idxmax()] / step)
        else:
            threshold = np.repeat(thresh, len(fold))

        # store f1 score/recall/precision/specifity for optimal thresh per fold
        f1_fold.append(fbeta_score(cv[target_col_name][f].values,
                                   (cv[pred_target_col_name][f] >= threshold[i - 1]).astype('int'),
                                   beta=1))
        recall_fold.append(recall_score(cv[target_col_name][f].values,
                                        (cv[pred_target_col_name][f] >= threshold[i - 1]).astype('int')))
        precision_fold.append(precision_score(cv[target_col_name][f].values,
                                              (cv[pred_target_col_name][f] >= threshold[i - 1]).astype('int')))

    print('F1 by fold:', f1_fold)
    print('Precision by fold:', precision_fold)
    print('Recall by fold:', recall_fold)
    print('Threshold by fold:', threshold)

    return {
        'auc': np.round(np.mean(AUC), 4),
        'f1': np.round(np.mean(f1_fold), 4),
        'precision': np.round(np.mean(precision_fold), 4),
        'recall': np.round(np.mean(recall_fold), 4),
        'threshold': np.round(np.mean(threshold), 4)
    }
