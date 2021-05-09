# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Tuple, Union
from previsionio.usecase_config import ColumnConfig, DataType, TypeProblem
from previsionio.dataset import Dataset, DatasetImages
import pandas as pd
from . import TrainingConfig
from . import metrics
from .usecase_version import ClassicUsecaseVersion
from .model import Model, RegressionModel, \
    ClassificationModel, MultiClassificationModel
from .utils import PrevisionException
from .usecase_version import BaseUsecaseVersion

MODEL_CLASS_DICT = {
    TypeProblem.Regression: RegressionModel,
    TypeProblem.Classification: ClassificationModel,
    TypeProblem.MultiClassification: MultiClassificationModel
}


class Supervised(ClassicUsecaseVersion):

    """ A supervised usecase. """

    start_command = 'focus'
    data_type = DataType.Tabular
    metric_type = metrics.Enum
    default_metric = metrics.Enum

    # model_class = Model

    def __init__(self, **usecase_info):
        if usecase_info.get('holdout_dataset_id'):
            self.holdout_dataset_id = usecase_info.get('holdout_dataset_id')
        else:
            self.holdout_dataset_id = None

        super().__init__(**usecase_info)
        self.model_class = MODEL_CLASS_DICT.get(self.training_type, Model)

    @classmethod
    def from_id(cls, _id) -> 'Supervised':
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
        instance = super().from_id(_id)
        return instance

    @classmethod
    def fit(cls, project_id: str, name: str, dataset: Union[Dataset, Tuple[Dataset, DatasetImages]], column_config: ColumnConfig, metric, holdout_dataset: Dataset = None,
            training_config: TrainingConfig = TrainingConfig(), **kwargs):
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
        return cls._start_usecase(project_id,
                                  name,
                                  dataset_id=dataset_id,
                                  data_type=cls.data_type,
                                  type_problem=cls.type_problem,
                                  metric=metric if isinstance(metric, str) else metric.value,
                                  **training_args)

    def new_version(self, name: str, dataset: Union[Dataset, Tuple[Dataset, DatasetImages]] = None, column_config: ColumnConfig = None, metric: metrics.Enum = None, holdout_dataset: Dataset = None,
                    training_config: TrainingConfig = None, **fit_params):
        """ Start a supervised usecase training to create a new version of the usecase (on the
        platform): the training config is copied from the current version and then overridden
        for the given parameters.

        Args:
            name (str): Name of the usecase version to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`, optional): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`, optional): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
        Returns:
            :class:`.Supervised`: Newly created supervised usecase object (new version)
        """

        if column_config is None:
            column_config = self.column_config

        if training_config is None:
            training_config = self.training_config

        if metric is None:
            metric = self.metric_type(self.metric)

        if dataset is None:
            dataset_ids = self.dataset_id
        else:
            if isinstance(dataset, Dataset):
                dataset_ids = dataset.id
            elif isinstance(dataset, tuple):
                dataset_ids = [d.id for d in dataset]

        if holdout_dataset is None:
            holdout_dataset_id = self.holdout_dataset_id
        else:
            holdout_dataset_id = holdout_dataset.id

        params = {'name': name,
                  'dataset': dataset_ids,
                  'metric': metric.value,
                  'holdout_dataset': holdout_dataset_id,
                  'type_problem': self.type_problem,
                  'usecase_id': self._id,
                  'parent_version': self.version,
                  # 'nextVersion': max([v['version'] for v in self.versions]) + 1  FA: wait what ?
                  }

        params.update(
            dict(column_config.to_kwargs() + training_config.to_kwargs()))

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
        if self.holdout_dataset_id:
            json_dict['holdout_id'] = self.holdout_dataset_id
        return json_dict


class SupervisedImages(Supervised):

    """ A supervised usecase with an image dataset. """

    start_command = 'image_focus'
    default_metric = 'NA'
    data_type = DataType.Images


class Regression(Supervised):

    """ A regression usecase for a numerical target using a basic dataset. """

    type_problem = TypeProblem.Regression
    metric_type = metrics.Regression
    default_metric = metrics.Regression.RMSE
    # model_class = RegressionModel


class Classification(Supervised):

    """ A (binary) classification usecase for a categorical target with
    exactly 2 modalities using a basic dataset. """

    type_problem = TypeProblem.Classification
    metric_type = metrics.Classification
    default_metric = metrics.Classification.AUC
    # model_class = ClassificationModel


class MultiClassification(Supervised):

    """ A multiclassification usecase for a categorical target with
    strictly more than 2 modalities using a basic dataset. """

    type_problem = TypeProblem.Classification
    metric_type = metrics.MultiClassification
    default_metric = metrics.MultiClassification.log_loss
    # model_class = MultiClassificationModel


class RegressionImages(SupervisedImages):

    """ A regression usecase for a numerical target using an image dataset. """

    type_problem = TypeProblem.Regression
    metric_type = metrics.Regression
    default_metric = metrics.Regression.RMSE


class ClassificationImages(SupervisedImages):

    """ A (binary) classification usecase for a categorical target with
    exactly 2 modalities using an image dataset. """

    type_problem = TypeProblem.Classification
    metric_type = metrics.Classification
    default_metric = metrics.Classification.AUC


class MultiClassificationImages(SupervisedImages):

    """ A multiclassification usecase for a categorical target with
    strictly more than 2 modalities using an image dataset. """

    type_problem = TypeProblem.MultiClassification
    metric_type = metrics.MultiClassification
    default_metric = metrics.MultiClassification.log_loss
