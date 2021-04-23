# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from . import TrainingConfig
from . import metrics
from .usecase import BaseUsecase
from .model import Model, RegressionModel, \
    ClassificationModel, MultiClassificationModel
from .utils import PrevisionException

MODEL_CLASS_DICT = {
    'regression': RegressionModel,
    'classification': ClassificationModel,
    'multiclassification': MultiClassificationModel
}


class Supervised(BaseUsecase):

    """ A supervised usecase. """

    type_problem = 'nan'
    start_command = 'focus'
    default_metric = 'NA'
    data_type = 'tabular'

    # model_class = Model

    def __init__(self, **usecase_info):
        self.dataset = usecase_info.get('dataset_id')
        if usecase_info.get('holdout_dataset_id'):
            self.holdout_dataset = usecase_info.get('holdout_dataset_id')
        else:
            self.holdout_dataset = None

        super().__init__(**usecase_info)
        self.model_class = MODEL_CLASS_DICT.get(self._status['training_type'], Model)

    @classmethod
    def from_name(cls, name, raise_if_non_unique=False, partial_match=False):
        """Get a supervised usecase from the platform by its name.

        Args:
            name (str): Name of the usecase to retrieve
            raise_if_non_unique (bool, optional): Whether or not to raise an error if
                duplicates are found (default: ``False``)
            partial_match (bool, optional): If true, usecases with a name containing
                the requested name will also be returned; else, only perfect matches
                will be found (default: ``False``)

        Raises:
            PrevisionException: Error if duplicates are found and
                the ``raise_if_non_unique`` is enabled

        Returns:
            :class:`.Supervised`: Fetched usecase
        """
        instance = super(BaseUsecase, cls).from_name(name, raise_if_non_unique, partial_match)
        type_problem = instance._status['training_type']
        if cls.type_problem != 'nan' and type_problem != cls.type_problem:
            raise PrevisionException('Invalid problem type: should be "{}" but is "{}".'.format(cls.type_problem,
                                                                                                type_problem))
        return instance

    @classmethod
    def from_id(cls, _id, version=1):
        """Get a supervised usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase to retrieve
            version (int, optional): Specific version of the usecase to retrieve
                (default: 1)

        Returns:
            :class:`.Supervised`: Fetched usecase

        Raises:
            PrevisionException: Invalid problem type or any error while fetching
                data from the platform or parsing result
        """
        instance = super().from_id(_id, version=version)
        type_problem = instance._status['training_type']
        if cls.type_problem != 'nan' and type_problem != cls.type_problem:
            raise PrevisionException('Invalid problem type: should be "{}" but is "{}".'.format(cls.type_problem,
                                                                                                type_problem))
        return instance

    @classmethod
    def fit(cls, name, dataset, column_config, metric=None, holdout_dataset=None,
            training_config=TrainingConfig(), type_problem=None, **kwargs):
        """ Start a supervised usecase training with a specific training configuration
        (on the platform).

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
            type_problem (str, optional): Specific problem type to train (default: ``None``)

        Returns:
            :class:`.Supervised`: Newly created supervised usecase object
        """

        config_args = training_config.to_kwargs()
        column_args = column_config.to_kwargs()
        training_args = dict(config_args + column_args)
        training_args.update(kwargs)

        if holdout_dataset:
            if isinstance(holdout_dataset, str):
                training_args['holdout_dataset_id'] = holdout_dataset
            else:
                training_args['holdout_dataset_id'] = holdout_dataset.id

        if not metric:
            metric = cls.default_metric

        if isinstance(dataset, str):
            dataset_id = dataset
        elif isinstance(dataset, tuple):
            dataset_id = [d.id for d in dataset]
        else:
            dataset_id = dataset.id

        return cls._start_usecase(name,
                                  dataset_id=dataset_id,
                                  data_type=cls.data_type,
                                  type_problem=type_problem if type_problem else cls.type_problem,
                                  metric=metric if isinstance(metric, str) else metric.value,
                                  **training_args)

    def new_version(self, **fit_params):
        """ Start a supervised usecase training to create a new version of the usecase (on the
        platform): the training config is copied from the current version and then overridden
        for the given parameters.

        Args:
            fit_params (kwargs): Training config parameters to change for the new version
                (compared to the current version)

        Returns:
            :class:`.Supervised`: Newly created supervised usecase object (new version)
        """

        params = {'name': self.name,
                  'dataset': self.dataset,
                  'column_config': self.column_config,
                  'metric': self.metric,
                  'holdout_dataset': self.holdout_dataset,
                  'training_config': self.training_config,
                  'type_problem': self._status['training_type'],
                  'usecase_id': self._id,
                  'parent_version': self.version,
                  'nextVersion': max([v['version'] for v in self.versions]) + 1}

        params.update(fit_params)
        return self.fit(**params)

    def _save_json(self):
        json_dict = {
            '_id': self.id,
            'usecase_version_params': self._status.get('usecase_version_params', {}),
            'dataset_id': self._status['dataset_id'],
            'type_problem': self.type_problem,
            'data_type': self.data_type,
        }
        if self.holdout_dataset:
            json_dict['holdout_id'] = self.holdout_dataset
        return json_dict


class SupervisedImages(Supervised):

    """ A supervised usecase with an image dataset. """

    type_problem = 'nan'
    start_command = 'image_focus'
    default_metric = 'NA'
    data_type = 'images'


class Regression(Supervised):

    """ A regression usecase for a numerical target using a basic dataset. """

    type_problem = 'regression'
    default_metric = metrics.Regression.RMSE
    # model_class = RegressionModel


class Classification(Supervised):

    """ A (binary) classification usecase for a categorical target with
    exactly 2 modalities using a basic dataset. """

    type_problem = 'classification'
    default_metric = metrics.Classification.AUC
    # model_class = ClassificationModel

    def predict_proba(self, df,
                      use_best_single=False,
                      confidence=False) -> pd.DataFrame:
        if use_best_single:
            best = self.best_single
        else:
            best = self.best_model

        return best.predict_proba(df=df, confidence=confidence)


class MultiClassification(Supervised):

    """ A multiclassification usecase for a categorical target with
    strictly more than 2 modalities using a basic dataset. """

    type_problem = 'multiclassification'
    default_metric = metrics.MultiClassification.log_loss
    # model_class = MultiClassificationModel


class RegressionImages(SupervisedImages):

    """ A regression usecase for a numerical target using an image dataset. """

    type_problem = 'regression'
    default_metric = metrics.Regression.RMSE


class ClassificationImages(SupervisedImages):

    """ A (binary) classification usecase for a categorical target with
    exactly 2 modalities using an image dataset. """

    type_problem = 'classification'
    default_metric = metrics.Classification.AUC


class MultiClassificationImages(SupervisedImages):

    """ A multiclassification usecase for a categorical target with
    strictly more than 2 modalities using an image dataset. """

    type_problem = 'multiclassification'
    default_metric = metrics.MultiClassification.log_loss
