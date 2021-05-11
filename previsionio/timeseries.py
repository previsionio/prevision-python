# -*- coding: utf-8 -*-
from __future__ import print_function

import requests
from previsionio.utils import EventTuple, handle_error_response, parse_json
from . import TrainingConfig
from .usecase_config import DataType, UsecaseConfig, ColumnConfig, TypeProblem
from .usecase_version import ClassicUsecaseVersion
from .metrics import Regression
from .model import RegressionModel
from .dataset import Dataset
from .prevision_client import client


class TimeWindowException(Exception):
    pass


class TimeWindow(UsecaseConfig):
    """
    A time window object for representing either feature derivation window periods or
    forecast window periods
    """
    config = {
        'derivation_start': 'start_dw',
        'derivation_end': 'end_dw',
        'forecast_start': 'start_fw',
        'forecast_end': 'end_fw',
    }

    def __init__(self, derivation_start: int, derivation_end: int, forecast_start: int, forecast_end: int):
        """Instantiate a new :class:`.TimeWindow` util object.

        Args:
            derivation_start (int): Start of the derivation window (must be < 0)
            derivation_end (int): End of the derivation window (must be < 0)
            forecast_start (int): Start of the forecast window (must be > 0)
            forecast_end (int): End of the forecast window (must be > 0)
        """

        if not derivation_start < derivation_end or not forecast_start < forecast_end:
            raise TimeWindowException('start must be smaller than end')

        if derivation_start > 0 or derivation_end > 0:
            raise TimeWindowException('derivation window bounds must be negative')

        if forecast_start < 0 or forecast_end < 0:
            raise TimeWindowException('forecast window bounds must be positive')

        if not(derivation_end <= ((2 * forecast_start) - forecast_end)):
            raise TimeWindowException('derivation_end must be smaller than (2 * forecast_start) - forecast_end')
        # Not valid anymore :
        # if not abs(derivation_end) >= (forecast_end - forecast_start):
        #     raise TimeWindowException('end of derivation window must be smaller than end of forecast window')

        self.derivation_start = derivation_start
        self.forecast_start = forecast_start
        self.derivation_end = derivation_end
        self.forecast_end = forecast_end


class TimeSeries(ClassicUsecaseVersion):
    """
    A TimeSeries usecase.
    """
    type_problem = TypeProblem.Regression
    metric_type = Regression
    default_metric = Regression.RMSE
    data_type = DataType.TimeSeries
    model_class = RegressionModel

    def __init__(self, **usecase_info):
        self.holdout_dataset_id = usecase_info.get('holdout_dataset_id', None)
        self.time_window = TimeWindow.from_dict(usecase_info.get('time_window'))

        super().__init__(**usecase_info)

    @classmethod
    def fit(cls, project_id: str, name: str, dataset: Dataset, column_config: ColumnConfig, time_window: TimeWindow,
            metric: Regression = None, holdout_dataset: Dataset = None, training_config: TrainingConfig = TrainingConfig()):
        config_args = training_config.to_kwargs()
        column_args = column_config.to_kwargs()
        time_window_args = time_window.to_kwargs()
        training_args = dict(config_args + column_args + time_window_args)

        if holdout_dataset:
            training_args['holdout_dataset_id'] = holdout_dataset.id

        if not metric:
            metric = cls.default_metric

        return cls._start_usecase(project_id=project_id,
                                  name=name,
                                  dataset_id=dataset.id,
                                  data_type=cls.data_type,
                                  type_problem=cls.type_problem,
                                  metric=metric.value,
                                  **training_args)

    def new_version(self, name: str, dataset: Dataset = None, column_config: ColumnConfig = None, time_window: TimeWindow = None,
                    metric: Regression = None, holdout_dataset: Dataset = None, training_config: TrainingConfig = TrainingConfig()):

        if not dataset:
            dataset_id = self.dataset_id
        else:
            dataset_id = dataset.id

        if not column_config:
            column_config = self.column_config

        if not time_window:
            time_window = self.time_window

        if not metric:
            metric = Regression(self.metric)

        if not holdout_dataset:
            holdout_dataset_id = self.holdout_dataset_id
        else:
            holdout_dataset_id = holdout_dataset.id

        if not training_config:
            training_config = self.training_config

        config_args = training_config.to_kwargs()
        column_args = column_config.to_kwargs()
        time_window_args = time_window.to_kwargs()
        training_args = dict(config_args + column_args + time_window_args)

        params = {'name': name,
                  'dataset_id': dataset_id,
                  'metric': metric.value,
                  'holdout_dataset': holdout_dataset_id,
                  'type_problem': self.type_problem,
                  'usecase_id': self._id,
                  'parent_version': self.version,
                  # 'nextVersion': max([v['version'] for v in self.versions]) + 1  FA: wait what ?
                  }

        params.update(training_args)

        endpoint = "/usecases/{}/versions".format(self.usecase_id)
        
        resp = client.request(endpoint=endpoint, data=params, method=requests.post)
        handle_error_response(resp, endpoint, params)
        json = parse_json(resp)

        usecase = self.from_id(json["_id"])
        usecase.type_problem = TypeProblem.Regression
        usecase.metric_type = Regression
        usecase.default_metric = Regression.RMSE
        usecase.data_type = DataType.TimeSeries
        usecase.model_class = RegressionModel

        events_url = '/{}/{}'.format(self.resource, json['_id'])
        client.event_manager.wait_for_event(usecase.resource_id,
                                            self.resource,
                                            EventTuple('USECASE_VERSION_UPDATE', 'state', 'running',
                                                       [('state', 'failed')]),
                                            specific_url=events_url)

        return usecase

