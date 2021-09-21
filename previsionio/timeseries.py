# -*- coding: utf-8 -*-
from typing import Dict, Union

import requests
from previsionio.utils import EventTuple, parse_json, to_json
from . import TrainingConfig
from .usecase_config import DataType, UsecaseConfig, ColumnConfig, TypeProblem
from .usecase_version import ClassicUsecaseVersion
from .metrics import Regression
from .model import RegressionModel
from .dataset import Dataset
from .prevision_client import client
import previsionio as pio


class TimeWindowException(Exception):
    pass


class TimeWindow(UsecaseConfig):
    """
    A time window object for representing either feature derivation window periods or
    forecast window periods.

    Args:
        derivation_start (int): Start of the derivation window (must be < 0)
        derivation_end (int): End of the derivation window (must be < 0)
        forecast_start (int): Start of the forecast window (must be > 0)
        forecast_end (int): End of the forecast window (must be > 0)
    """
    config = {
        'derivation_start': 'start_dw',
        'derivation_end': 'end_dw',
        'forecast_start': 'start_fw',
        'forecast_end': 'end_fw',
    }

    def __init__(self, derivation_start: int, derivation_end: int, forecast_start: int, forecast_end: int):
        """Instantiate a new :class:`.TimeWindow` util object."""

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
    """ A supervised usecase version, for timeseries data """

    training_type = TypeProblem.Regression
    metric_type = Regression
    default_metric = Regression.RMSE
    data_type = DataType.TimeSeries
    model_class = RegressionModel

    def __init__(self, **usecase_info):
        self.holdout_dataset_id: Union[str, None] = usecase_info.get('holdout_dataset_id', None)
        self.time_window = TimeWindow.from_dict(usecase_info['usecase_version_params']['timeseries_values'])

        super().__init__(**usecase_info)

    @classmethod
    def from_id(cls, _id: str) -> 'TimeSeries':
        return cls(**super()._from_id(_id))

    @classmethod
    def load(cls, pio_file: str) -> 'TimeSeries':
        return cls(**super()._load(pio_file))

    @classmethod
    def _fit(cls, project_id: str,
             name: str,
             dataset: Dataset,
             column_config: ColumnConfig,
             time_window: TimeWindow,
             metric: Regression = None,
             holdout_dataset: Dataset = None,
             training_config: TrainingConfig = TrainingConfig()) -> 'TimeSeries':
        training_args = to_json(training_config)
        assert isinstance(training_args, Dict)

        training_args.update(to_json(column_config))
        training_args.update(to_json(time_window))

        if holdout_dataset:
            training_args['holdout_dataset_id'] = holdout_dataset.id

        if not metric:
            metric = cls.default_metric

        start_response = cls._start_usecase(project_id=project_id,
                                            name=name,
                                            dataset_id=dataset.id,
                                            data_type=cls.data_type,
                                            training_type=cls.training_type,
                                            metric=metric.value,
                                            **training_args)

        usecase = cls.from_id(start_response['_id'])
        events_url = '/{}/{}'.format(cls.resource, start_response['_id'])
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(usecase.resource_id,
                                                cls.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', ('state', 'running')),
                                                specific_url=events_url)
        return usecase

    def new_version(
        self,
        description: str = None,
        dataset: Dataset = None,
        column_config: ColumnConfig = None,
        time_window: TimeWindow = None,
        metric: Regression = None,
        holdout_dataset: Dataset = None,
        training_config: TrainingConfig = TrainingConfig(),
    ):
        """ Start a time series usecase training to create a new version of the usecase (on the
        platform): the training configs are copied from the current version and then overridden
        for the given parameters.

        Args:
            description (str, optional): additional description of the version
            dataset (:class:`.Dataset`, :class:`.DatasetImages`, optional): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`, optional): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            time_window (:class: `.TimeWindow`, optional): a time window object for representing either feature
                derivation window periods or forecast window periods
            metric (metrics.Regression, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
        Returns:
            :class:`.TimeSeries`: Newly created text similarity usecase version object (new version)
        """

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

        training_args = to_json(training_config)
        assert isinstance(training_args, Dict)
        training_args.update(to_json(column_config))
        training_args.update(to_json(time_window))

        params = {
            'dataset_id': dataset_id,
            'metric': metric.value,
            'holdout_dataset': holdout_dataset_id,
            'training_type': self.training_type.value,
            'usecase_id': self._id,
            'parent_version': self.version,
            # 'nextVersion': max([v['version'] for v in self.versions]) + 1  FA: wait what ?
        }

        if description:
            params["description"] = description

        params.update(training_args)

        endpoint = "/usecases/{}/versions".format(self.usecase_id)

        resp = client.request(endpoint=endpoint,
                              data=params,
                              method=requests.post,
                              content_type='application/json',
                              message_prefix='Time series usecase start')
        json = parse_json(resp)

        usecase = self.from_id(json["_id"])

        events_url = '/{}/{}'.format(self.resource, json['_id'])
        assert client.event_manager is not None
        client.event_manager.wait_for_event(usecase.resource_id,
                                            self.resource,
                                            EventTuple('USECASE_VERSION_UPDATE', ('state', 'running')),
                                            specific_url=events_url)

        return usecase
