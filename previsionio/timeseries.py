# -*- coding: utf-8 -*-
from __future__ import print_function

from typing import Dict
from . import TrainingConfig
from .experiment_config import DataType, ExperimentConfig, ColumnConfig, TypeProblem
from .experiment_version import ClassicExperimentVersion
from .metrics import Regression
from .model import RegressionModel
from .dataset import Dataset
from .utils import to_json


class TimeWindowException(Exception):
    pass


class TimeWindow(ExperimentConfig):
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


class TimeSeries(ClassicExperimentVersion):
    """ A supervised experiment version, for timeseries data """

    training_type = TypeProblem.Regression
    metric_type = Regression
    default_metric = Regression.RMSE
    data_type = DataType.TimeSeries
    model_class = RegressionModel

    def __init__(self, **experiment_version_info):
        super().__init__(**experiment_version_info)

    def _update_from_dict(self, **experiment_version_info):
        super()._update_from_dict(**experiment_version_info)
        timeseries_values = experiment_version_info['experiment_version_params']['timeseries_values']
        self.time_window = TimeWindow.from_dict(timeseries_values)

    @classmethod
    def from_id(cls, _id: str) -> 'TimeSeries':
        return cls(**super()._from_id(_id))

    @staticmethod
    def _build_experiment_version_creation_data(description, dataset, column_config, time_window, metric,
                                                holdout_dataset, training_config,
                                                parent_version=None) -> Dict:
        data = super(TimeSeries, TimeSeries)._build_experiment_version_creation_data(
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
    def _fit(
        cls,
        experiment_id: str,
        dataset: Dataset,
        column_config: ColumnConfig,
        time_window: TimeWindow,
        metric: Regression = None,
        holdout_dataset: Dataset = None,
        training_config: TrainingConfig = TrainingConfig(),
        description: str = None,
        parent_version: str = None,
    ) -> 'TimeSeries':
        """ Start a timeseries experiment version training with a specific configuration (on the platform).

        Args:
            experiment_id (str): The id of the experiment from which this version is created
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the experiment
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            time_window (:class:`.TimeWindow`): Time configuration
                (see the documentation of the :class:`.TimeWindow` resource for more details)
            metric (metrics.Regression): Specific metric to use for the experiment version
                (default ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
            description (str, optional): The description of this experiment version (default: ``None``)
            parent_version (str, optional): The parent version of this experiment_version (default: ``None``)


        Returns:
            :class:`.TimeSeries`: Newly created supervised experiment version object
        """

        return super()._fit(
            experiment_id,
            description=description,
            parent_version=parent_version,
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
        """
        Start a new timeseries experiment version training from this version (on the platform).
        The training parameters are copied from the current version and then overridden for those provided.

        Args:
            dataset (:class:`.Dataset`, optional): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`, optional): Column configuration for the experiment
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (metrics.Regression, optional): Specific metric to use for the experiment version
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
            description (str, optional): The description of this experiment version (default: ``None``)
        Returns:
            :class:`.TimeSeries`: Newly created timeSeries experiment version object (new version)
        """
        return TimeSeries._fit(
            self.experiment_id,
            dataset if dataset is not None else self.dataset,
            column_config if column_config is not None else self.column_config,
            time_window if time_window is not None else self.time_window,
            metric if metric is not None else self.metric,
            holdout_dataset=holdout_dataset if holdout_dataset is not None else self.holdout_dataset,
            training_config=training_config if training_config is not None else self.training_config,
            description=description,
            parent_version=self.version,
        )
