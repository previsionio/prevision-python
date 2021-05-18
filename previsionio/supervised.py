# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Tuple, Union

import requests
from previsionio.usecase_config import ColumnConfig, DataType, TypeProblem
from previsionio.dataset import Dataset, DatasetImages
from . import TrainingConfig
from . import metrics
from .usecase_version import ClassicUsecaseVersion
from .model import RegressionModel, \
    ClassificationModel, MultiClassificationModel
from .utils import EventTuple, handle_error_response, parse_json
from .prevision_client import client
import previsionio as pio

MODEL_CLASS_DICT = {
    TypeProblem.Regression: RegressionModel,
    TypeProblem.Classification: ClassificationModel,
    TypeProblem.MultiClassification: MultiClassificationModel
}


class Supervised(ClassicUsecaseVersion):

    """ A supervised usecase. """

    data_type = DataType.Tabular
    # model_class = Model

    def __init__(self, **usecase_info):
        self.holdout_dataset_id = usecase_info.get('holdout_dataset_id', None)

        print(usecase_info)
        super().__init__(**usecase_info)
        self.model_class = MODEL_CLASS_DICT.get(self.training_type)

    @classmethod
    def from_id(cls, _id: str) -> 'Supervised':
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
        return cls(**super()._from_id(_id))

    @classmethod
    def load(cls, pio_file: str) -> 'Supervised':
        return cls(**super()._load(pio_file))

    @classmethod
    def _fit(cls, project_id: str, name: str, data_type: str, type_problem: str,
             dataset: Union[Dataset, Tuple[Dataset, DatasetImages]], column_config: ColumnConfig,
             metric: metrics.Enum, holdout_dataset: Dataset = None,
             training_config: TrainingConfig = TrainingConfig(), **kwargs) -> 'Supervised':
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

        assert metric

        if isinstance(dataset, str):
            dataset_id = dataset
        elif isinstance(dataset, tuple):
            dataset_id = [d.id for d in dataset]
        else:
            dataset_id = dataset.id
        start_response = cls._start_usecase(project_id,
                                            name,
                                            dataset_id=dataset_id,
                                            data_type=data_type,
                                            type_problem=type_problem,
                                            metric=metric if isinstance(metric, str) else metric.value,
                                            **training_args)
        usecase = cls.from_id(start_response['_id'])
        usecase.data_type = data_type
        usecase.type_problem = type_problem
        events_url = '/{}/{}'.format(cls.resource, start_response['_id'])
        pio.client.event_manager.wait_for_event(usecase.resource_id,
                                                cls.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', 'state', 'running',
                                                           [('state', 'failed')]),
                                                specific_url=events_url)

        return usecase

    def new_version(self, description: str = None, dataset: Union[Dataset, Tuple[Dataset, DatasetImages]] = None, column_config: ColumnConfig = None, metric: metrics.Enum = None, holdout_dataset: Dataset = None,
                    training_config: TrainingConfig = None, **fit_params):
        """ Start a supervised usecase training to create a new version of the usecase (on the
        platform): the training configs are copied from the current version and then overridden
        for the given parameters.

        Args:
            description (str, optional): additional description of the version
            dataset (:class:`.Dataset`, :class:`.DatasetImages`, optional): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`, optional): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (metrics.Enum, optional): Specific metric to use for the usecase (default: ``None``)
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

        metric_str: str = self.metric
        if metric is not None:
            metric_str = metric.value

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

        params = {
            'dataset_id': dataset_ids,
            'metric': metric_str,
            'holdout_dataset': holdout_dataset_id,
            'type_problem': self.type_problem,
            'usecase_id': self._id,
            'parent_version': self.version,
            # 'nextVersion': max([v['version'] for v in self.versions]) + 1  FA: wait what ?
        }

        if description:
            params["description"] = description

        params.update(
            dict(column_config.to_kwargs() + training_config.to_kwargs()))

        params.update(fit_params)
        endpoint = "/usecases/{}/versions".format(self.usecase_id)
        resp = client.request(endpoint=endpoint, data=params, method=requests.post)
        handle_error_response(resp, endpoint, params)
        json = parse_json(resp)

        usecase = type(self).from_id(json["_id"])
        usecase.data_type = self.data_type
        usecase.type_problem = self.type_problem

        events_url = '/{}/{}'.format(self.resource, json['_id'])
        client.event_manager.wait_for_event(usecase.resource_id,
                                            self.resource,
                                            EventTuple('USECASE_VERSION_UPDATE', 'state', 'running',
                                                       [('state', 'failed')]),
                                            specific_url=events_url)

        return usecase

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
