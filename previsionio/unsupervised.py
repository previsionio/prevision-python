# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from . import TypeProblem, clustering_base_config
from .usecase_version import BaseUsecaseVersion


class UnSupervised(BaseUsecaseVersion):
    specific_required_params = {
        'type_problem': 'clustering',
    }

    start_command = 'focus'

    def __init__(self, usecase_params):
        super(UnSupervised, self).__init__(usecase_params)

    @classmethod
    def unsupervised_from_dataset(cls, dataset, use_case, type_problem,
                                  metric=None, training_config=clustering_base_config):
        """
        Creates an unsupervised usecase from a pio.Dataset.

        Args:
            dataset: A pio.Dataset
            use_case: str: Usecase name
            type_problem: pio.TypeProblem
            metric: pio.metrics
            training_config: pio.TrainingConfig (default base_config)

        Returns:
            pio.UnSupervised, started with defined params

        """
        usecase_params = {'use_case': use_case,
                          'type_problem': type_problem}
        if metric:
            usecase_params['metric'] = metric.value

        for param in ['id_column', 'fold_column', 'delimiter', 'drop_list', 'weight_column']:
            value = dataset.__getattribute__(param)
            if value:
                usecase_params[param] = value

        usecase_params['profile'] = training_config.profile
        usecase_params['CLUSTER_magnitude'] = training_config.cluster_magnitude

        if isinstance(dataset.data, pd.DataFrame):
            return cls.from_dataframe(dataset.data, **usecase_params)
        elif isinstance(dataset.data, str):
            return cls.from_filename(dataset.data, **usecase_params)


class Clustering(UnSupervised):
    """
    A Clustering usecase.
    Use Clustering.from_dataset
    """

    specific_optional_params = {
        'CLUSTER_magnitude': 'CLUSTER_magnitude',
    }

    @classmethod
    def from_dataset(cls, dataset, use_case, metric=None,
                     training_config=clustering_base_config):
        """
        Creates a Clustering usecase from a pio.Dataset.

        Args:
            dataset: A pio.Dataset
            use_case: str: Usecase name
            metric: pio.metrics
            training_config: pio.TrainingConfig (default base_config)

        Returns:
            pio.Clustering, started with defined params

        """
        return cls.unsupervised_from_dataset(dataset, use_case, TypeProblem.Clustering,
                                             metric=metric,
                                             training_config=training_config)
