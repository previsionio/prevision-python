# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Dict

from previsionio.utils import to_json
from . import TrainingConfig
from .usecase_config import DataType, UsecaseConfig, ColumnConfig, TypeProblem
from .usecase_version import ClassicUsecaseVersion
from .metrics import Regression
from .model import RegressionModel
from .dataset import Dataset


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

    def __init__(self, **usecase_version_info):
        super().__init__(**usecase_version_info)
        self._update_from_dict(**usecase_version_info)

    def _update_from_dict(self, **usecase_version_info):
        super()._update_from_dict(**usecase_version_info)
        self.time_window = TimeWindow.from_dict(usecase_version_info['usecase_version_params']['timeseries_values'])

    @classmethod
    def from_id(cls, _id: str) -> 'TimeSeries':
        return cls(**super()._from_id(_id))

    @classmethod
    def load(cls, pio_file: str) -> 'TimeSeries':
        return cls(**super()._load(pio_file))

    @staticmethod
    def _build_usecase_version_creation_data(description, dataset, column_config, time_window, metric,
                                             holdout_dataset, training_config,
                                             parent_version=None) -> Dict:
        data = super(TimeSeries, TimeSeries)._build_usecase_version_creation_data(
            description,
            parent_version=parent_version,
        )

        data['dataset_id'] = dataset.id
        data.update(to_json(column_config))
        data.update(to_json(time_window))
        data['metric'] = metric if isinstance(metric, str) else metric.value
        data['holdout_dataset_id'] = holdout_dataset.id if holdout_dataset is not None else None
        data.update(to_json(training_config))

        return data

    @classmethod
    def _fit(cls,
             usecase_id: str,
             dataset: Dataset,
             column_config: ColumnConfig,
             time_window: TimeWindow,
             metric: Regression = None,
             holdout_dataset: Dataset = None,
             training_config: TrainingConfig = TrainingConfig(),
             description: str = None) -> 'TimeSeries':

        return super()._fit(
            usecase_id,
            description=description,
            dataset=dataset,
            column_config=column_config,
            time_window=time_window,
            metric=metric,
            holdout_dataset=holdout_dataset,
            training_config=training_config,
        )

    def new_version(
        self,
        dataset: Dataset = None,
        column_config: ColumnConfig = None,
        time_window: TimeWindow = None,
        metric: Regression = None,
        holdout_dataset: Dataset = None,
        training_config: TrainingConfig = None,
        description: str = None,
    ) -> 'TimeSeries':
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
            :class:`.TimeSeries`: Newly created TimeSeries usecase version object (new version)
        """
        return TimeSeries._fit(
                               self.usecase_id,
                               dataset if dataset is not None else self.dataset,
                               column_config if column_config is not None else self.column_config,
                               time_window if time_window is not None else self.time_window,
                               metric if metric is not None else self.metric,
                               holdout_dataset=holdout_dataset if holdout_dataset is not None else self.holdout_dataset,
                               training_config=training_config if training_config is not None else self.training_config,
                               description=description,
        )
