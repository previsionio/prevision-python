import itertools
from previsionio.usecase_version import ClassicUsecaseVersion
from previsionio.usecase_config import TypeProblem
import numpy as np
import os

try:
    # the package is running inside the prevision.io notebook
    local_url = os.environ['PREVISION_WIDGET_URL']
except KeyError:
    # the package is running inside the prevision.io notebook
    from matplotlib import pyplot as plt
else:
    import IPython

from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, f1_score, precision_score


class Plotter:
    def __init__(self):
        pass


class PrevisionioPlotter(Plotter):
    prevision_widgets = ['featdetail', 'featimp', 'regscores', 'multiclassifRoc', 'multiclassifConfusion',
                         'multiclassifScore', 'decisionChart', 'classifMetrics', 'classifDensity', 'classifRoc',
                         'clusterStats', 'costMatrix', 'confusionMatrix', 'regvs', 'regdisp', 'repartition',
                         'classifLift', 'classifPerBinLift']

    def __init__(self, usecase):
        super().__init__()

        def add_widget_method(widget):
            def f(plotter):
                return plotter.__plot(widget, usecase)

            setattr(PrevisionioPlotter, widget, f)

        for method in self.prevision_widgets:
            add_widget_method(method)

    def __plot(self, widget, usecase, width=1000, height=600):
        iframe_url = '<iframe src={}/widgets/{}/{} width={} height={}></iframe>'.format(local_url,
                                                                                        widget,
                                                                                        usecase.usecase_name,
                                                                                        height,
                                                                                        width)
        return IPython.display.HTML(iframe_url)


class PlotlyPlotter(Plotter):
    def __init__(self, usecase: ClassicUsecaseVersion):
        self.usecase = usecase
        super().__init__()

    def plot_roc(self):
        """
        Plot a ROC curve. Needs a prediction made with the truth present in the dataset.

        Only available for binary classification

        """

        import plotly.graph_objs as go
        from plotly.offline import iplot, init_notebook_mode
        from plotly import tools

        init_notebook_mode()

        if self.usecase.training_type not in [TypeProblem.Classification, TypeProblem.MultiClassification]:
            raise Exception(
                'ROC curve only available for classification or multiclassif, '
                'not ' + self.usecase.training_type.value)

        preds = self.usecase.get_cv()

        target_col_name = self.usecase.column_config.target_column
        assert target_col_name

        if self.usecase.training_type == TypeProblem.Classification:
            fpr, tpr, _ = roc_curve(preds[target_col_name], preds['pred_' + target_col_name])
            roc_auc = auc(fpr, tpr)

            lw = 1

            trace1 = go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                line=dict(color='darkorange', width=lw),
                                name='ROC curve (area = %0.2f)' % roc_auc
                                )

            trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                line=dict(color='navy', width=.2, dash='dash'),
                                showlegend=False)

            layout = go.Layout(title='Receiver operating characteristic example',
                               xaxis=dict(title='False Positive Rate'),
                               yaxis=dict(title='True Positive Rate'))

            fig = go.Figure(data=[trace1, trace2], layout=layout)
            iplot(fig)

        else:
            max_cols = 3
            max_rows = 3

            labels = sorted(preds[target_col_name].unique())

            if len(labels) > 12:
                raise Exception('cannot plot roc curves for more than 12 classes')

            n_cols = min(max_cols, len(labels))
            n_rows = (len(labels) // (max_rows - 1))

            fig = tools.make_subplots(rows=n_rows,
                                      cols=n_cols,
                                      subplot_titles=['Class: {}'.format(label) for label in labels])

            for i, label in enumerate(labels):
                truth = preds[target_col_name] == label

                fpr, tpr, _ = roc_curve(truth, preds['pred_{}_{}'.format(target_col_name, label)])
                roc_auc = auc(fpr, tpr)

                col = i % max_cols
                row = i // max_rows

                lw = 1

                trace1 = go.Scatter(x=fpr, y=tpr,
                                    mode='lines',
                                    line=dict(color='darkorange', width=lw),
                                    name='ROC curve (area = {:0.2f})'.format(roc_auc))

                trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    line=dict(color='navy', width=.2, dash='dash'),
                                    showlegend=False)

                fig.add_trace(trace1, row + 1, col + 1)
                fig.add_trace(trace2, row + 1, col + 1)

            fig['layout'].update(height=500 * n_rows, width=500 * n_cols, title='Multiclass ROC')
            iplot(fig)


class MatplotlibPlotter(Plotter):
    def __init__(self, usecase):
        self.usecase = usecase
        super().__init__()

    def featdetail(self):
        raise NotImplementedError

    def featimp(self):
        raise NotImplementedError

    def regscores(self):
        raise NotImplementedError

    def multiclassifRoc(self):
        raise NotImplementedError

    def multiclassifConfusion(self):
        raise NotImplementedError

    def multiclassifScore(self):
        raise NotImplementedError

    def decisionChart(self):
        raise NotImplementedError

    def classifMetrics(self):
        raise NotImplementedError

    def classifDensity(self):
        raise NotImplementedError

    def classifRoc(self):
        raise NotImplementedError

    def clusterStats(self):
        raise NotImplementedError

    def costMatrix(self):
        raise NotImplementedError

    def confusionMatrix(self):
        raise NotImplementedError

    def regvs(self):
        raise NotImplementedError

    def regdisp(self):
        raise NotImplementedError

    def repartition(self):
        raise NotImplementedError

    def classifLift(self):
        raise NotImplementedError

    def classifPerBinLift(self):
        raise NotImplementedError

    def plot_roc(self, predict_id=None):
        """
        Plot a ROC curve. Needs a prediction made with the truth present in the dataset.

        Only available for binary classification

        Args:
            usecase:
            predict_id (str): ID of the prediction
        """
        if self.usecase.training_type not in [TypeProblem.Classification, TypeProblem.MultiClassification]:
            raise Exception(
                'ROC curve only available for classification or multiclassif, '
                'not ' + self.usecase.training_type)

        if predict_id:
            raise NotImplementedError
            # if predict_8id not in self.usecase.predictions:
            #     raise KeyError('No such prediction')
            #
            # prediction = self.usecase.predictions[predict_id]
            #
            # if prediction.truth is None:
            #     raise ValueError('Truth not set for this prediction')
            # else:
            #     truth, pred = prediction.truth, prediction.data[self.usecase.usecase_params['target_column']]
        else:
            preds = self.usecase.get_cv()

        target_col_name = self.usecase.column_config.target_column

        if not plt:
            raise Exception('matplotlib not installed. please install to plot curves')

        lw = 1

        if self.usecase.training_type == TypeProblem.Classification:
            fpr, tpr, _ = roc_curve(preds[target_col_name], preds['pred_' + target_col_name])
            roc_auc = auc(fpr, tpr)
            lw = 2
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = {})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc='lower right')

        else:
            max_cols = 3
            max_rows = 3

            labels = preds[target_col_name].unique()

            if len(labels) > 9:
                raise Exception('cannot plot roc curves for more than 9 classes')

            n_cols = min(max_cols, len(labels))
            n_rows = (len(labels) // (max_rows - 1))

            f, axarr = plt.subplots(n_rows, n_cols)

            for i, label in enumerate(labels):
                truth = preds[target_col_name] == label

                fpr, tpr, _ = roc_curve(truth, preds['pred_{}_{}'.format(target_col_name, label)])
                roc_auc = auc(fpr, tpr)

                col = i % max_cols
                row = i // max_rows

                ax = axarr[row, col] if n_rows > 1 else axarr[col]

                ax.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = {})'.format(roc_auc))
                ax.plot([0, 1], [0, 1], color='navy', lw=lw)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc='lower right')

            plt.title('ROC curve')

        plt.show()

    def plot_conf_mat(self, predict_id):
        """
        Plot a confusion matrix. Needs a prediction made with the truth present in the dataset.

        Only available for multi-classification.

        Args:
            predict_id (str): ID of the prediction

        """
        if self.usecase.training_type != TypeProblem.MultiClassification:
            raise Exception('Confusion matrices only available for multiclassification, not {}'.format(
                            self.usecase.training_type))

        # retrieve current list of predictions if necessary
        if len(self.usecase.predictions) == 0:
            self.usecase.predictions = self.usecase.get_predictions(full=True)

        if predict_id not in self.usecase.predictions:
            raise KeyError('No such prediction')

        prediction = self.usecase.predictions[predict_id]

        # if prediction.truth is None:
        #     raise ValueError('Truth not set for this prediction')

        if not plt:
            raise Exception('matplotlib not installed. please install to plot curves')

        id_col_name = self.usecase.column_config.id_column
        target_col_name = self.usecase.column_config.target_column
        pred_target_col_name = 'pred_' + target_col_name if target_col_name else 'pred_TARGET'
        pred_cols = prediction.drop([id_col_name or 'ID',
                                     target_col_name or 'TARGET',
                                     pred_target_col_name], axis=1)

        pred_labels = pred_cols. \
            idxmax(axis=1). \
            apply(lambda s: s.replace(target_col_name + '_', ''))

        cnf_matrix = confusion_matrix(prediction[target_col_name],
                                      prediction[pred_target_col_name])

        cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        classes = pred_labels.unique()

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.show()

    def plot_classif_analysis(self, predict_id, top):
        """
        Plot a classification analysis. Needs a prediction made with the truth present in the dataset.

        Only available for classification.

        Args:
            predict_id (str): ID of the prediction
            top (int): top individuals to analyze

        """
        if self.usecase.training_type != TypeProblem.Classification:
            raise Exception('Classification analysis plots only available for classification, not {}'.format(
                            self.usecase.training_type))

        # retrieve current list of predictions if necessary
        if len(self.usecase.predictions) == 0:
            self.usecase.predictions = self.usecase.get_predictions(full=True)

        if predict_id not in self.usecase.predictions:
            raise KeyError('No such prediction')

        prediction = self.usecase.predictions[predict_id]

        target_col_name = self.usecase.column_config.target_column
        pred_target_col_name = 'pred_' + target_col_name if target_col_name else 'pred_TARGET'

        # sort by predicted values (decreasing)
        prediction.sort_values(pred_target_col_name, ascending=False, inplace=True)
        actual = prediction[target_col_name].values
        predicted = prediction[pred_target_col_name].values

        rec = []
        prec = []
        f1 = []

        # compute indicators
        for i in range(1, top + 1):
            predicted_flag = (predicted >= predicted[i]).astype('int')
            rec.append(recall_score(actual, predicted_flag))
            prec.append(precision_score(actual, predicted_flag))
            f1.append(f1_score(actual, predicted_flag))

        top = np.argmax(f1)

        # plot results
        if not plt:
            raise Exception('matplotlib not installed. please install to plot curves')

        lw = 2

        f, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(rec, 'b-', lw=lw)
        ax1.vlines(x=[top], ymin=0, ymax=1, colors='r', linestyles='solid')
        ax1.set_xlabel('Top individuals')
        ax1.set_ylabel('Recall')
        ax1.set_title('Recall with respect to top n individuals')

        ax2.plot(prec, 'b-', lw=lw)
        ax2.vlines(x=[top], ymin=0, ymax=1, colors='r', linestyles='solid')
        ax2.set_xlabel('Top individuals')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision with respect to top n individuals')

        ax3.plot(f1, 'b-', lw=lw)
        ax3.vlines(x=[top], ymin=0, ymax=1, colors='r', linestyles='solid')
        ax3.set_xlabel('Top individuals')
        ax3.set_ylabel('F1')
        ax3.set_title('F1 score with respect to top n individuals')

        plt.tight_layout()
        plt.show()

        return {
            'f1': np.round(f1[top], 4),
            'precision': np.round(prec[top], 4),
            'recall': np.round(rec[top], 4),
            'top': (top, 4)
        }
