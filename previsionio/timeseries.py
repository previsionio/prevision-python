# -*- coding: utf-8 -*-
from __future__ import print_function
from . import TrainingConfig
from .usecase_config import UsecaseConfig
from .usecase import BaseUsecase
from .metrics import Regression
from .model import RegressionModel


class TimeWindowException(Exception):
    pass


class TimeWindow(UsecaseConfig):
    """
    A time window object for representing either feature derivation window periods or
    forecast window periods
    """
    config = {
        'derivation_start': 'startDW',
        'derivation_end': 'endDW',
        'forecast_start': 'startFW',
        'forecast_end': 'endFW',
    }

    def __init__(self, derivation_start, derivation_end, forecast_start, forecast_end):
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

        if forecast_start < 0 or forecast_start < 0:
            raise TimeWindowException('forecast window bounds must be positive')

        # Not valid anymore :
        # if not abs(derivation_end) >= (forecast_end - forecast_start):
        #     raise TimeWindowException('end of derivation window must be smaller than end of forecast window')

        self.derivation_start = derivation_start
        self.forecast_start = forecast_start
        self.derivation_end = derivation_end
        self.forecast_end = forecast_end


class TimeSeries(BaseUsecase):
    """
    A TimeSeries usecase.
    """
    type_problem = 'regression'
    default_metric = Regression.RMSE
    data_type = 'timeseries'
    model_class = RegressionModel

    @classmethod
    def fit(cls, name, dataset, column_config, time_window,
            metric=None, training_config=TrainingConfig()):
        config_args = training_config.to_kwargs()
        column_args = column_config.to_kwargs()
        time_window_args = time_window.to_kwargs()
        training_args = dict(config_args + column_args + time_window_args)
        # TimeSeries experimental (ALN) is no longer available
        #if aln:
        #    training_args['experimentalTimeseries'] = 'true'

        if not metric:
            metric = cls.default_metric

        return cls._start_usecase(name,
                                  dataset_id=dataset.id,
                                  data_type=cls.data_type,
                                  type_problem=cls.type_problem,
                                  metric=metric.value,
                                  **training_args)
