# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from typing import Dict, List, Tuple, Union
import requests
import pandas as pd

from . import metrics
from .experiment_config import TypeProblem
from .experiment_version import ClassicExperimentVersion
from .model import ExternalRegressionModel, ExternalClassificationModel, ExternalMultiClassificationModel
from .prevision_client import client
from .utils import parse_json
from .dataset import Dataset


# NOTE: We inherit from ClassicExperimentVersion because it contains a lot of methods we need here, but not all,
#       we could make a big refactor with intermediate Base classes
class ExternalExperimentVersion(ClassicExperimentVersion):

    def __get_model_class(self):
        if self.training_type is None:
            model_class = None
        elif self.training_type == TypeProblem.Regression:
            model_class = ExternalRegressionModel
        elif self.training_type == TypeProblem.Classification:
            model_class = ExternalClassificationModel
        elif self.training_type == TypeProblem.MultiClassification:
            model_class = ExternalMultiClassificationModel
        else:
            raise ValueError(f'Unknown training_type for ExternalExperimentVersion: {self.training_type}')
        return model_class

    def __init__(self, **experiment_version_info):
        super().__init__(**experiment_version_info)

    def _update_from_dict(self, **experiment_version_info):
        # we don't want to inherit from ClassicExperimentVersion._update_from_dict but from its mother...
        super(ClassicExperimentVersion, self)._update_from_dict(**experiment_version_info)

        experiment_version_params = experiment_version_info['experiment_version_params']

        holdout_dataset_id: str = experiment_version_info['holdout_dataset_id']
        self.holdout_dataset: Dataset = Dataset.from_id(holdout_dataset_id)
        self.target_column = experiment_version_params['target_column']

        # this is dict, maybe we should parse it in a tuple like in creation
        self.external_models = experiment_version_info.get('external_models')

        self.metric = experiment_version_info.get('metric')
        dataset_id: Union[str, None] = experiment_version_info.get('dataset_id')
        self.dataset: Union[str, None] = Dataset.from_id(dataset_id) if dataset_id is not None else None

        self.metric: str = experiment_version_params['metric']

        self.model_class = self.__get_model_class()

    def _update_draft(self, external_models, **kwargs):
        self.__add_external_models(external_models)

    @staticmethod
    def _build_experiment_version_creation_data(description, holdout_dataset, target_column,
                                                metric, dataset, parent_version=None,
                                                **kwargs) -> Dict:
        data = super(ExternalExperimentVersion, ExternalExperimentVersion)._build_experiment_version_creation_data(
            description,
            parent_version=parent_version,
        )

        data['holdout_dataset_id'] = holdout_dataset.id
        data['target_column'] = target_column
        data['metric'] = metric if isinstance(metric, str) else metric.value
        data['dataset_id'] = dataset.id if dataset is not None else None

        return data

    @classmethod
    def _fit(cls,
             experiment_id: str,
             holdout_dataset: Dataset,
             target_column: str,
             external_models: List[Tuple],
             metric: metrics.Enum,
             dataset: Dataset = None,
             description: str = None,
             parent_version: str = None) -> 'ExternalExperimentVersion':
        return super()._fit(
            experiment_id,
            description=description,
            parent_version=parent_version,
            holdout_dataset=holdout_dataset,
            target_column=target_column,
            external_models=external_models,
            metric=metric,
            dataset=dataset,
        )

    def new_version(self,
                    external_models: List[Tuple],
                    holdout_dataset: Dataset = None,
                    target_column: str = None,
                    metric: metrics.Enum = None,
                    dataset: Dataset = None,
                    description: str = None) -> 'ExternalExperimentVersion':
        return ExternalExperimentVersion._fit(
            self.experiment_id,
            holdout_dataset if holdout_dataset is not None else self.holdout_dataset,
            target_column if target_column is not None else self.target_column,
            external_models,
            metric if metric is not None else self.metric,
            dataset=dataset if dataset is not None else self.dataset,
            description=description,
            parent_version=self.version,
        )

    def __add_external_model(self, external_model: Tuple) -> None:
        external_model_upload_endpoint = f'/experiment-versions/{self._id}/external-models'
        external_model_upload_method = requests.post
        external_model_message_prefix = 'External model uploading'
        name_key = 'name'
        onnx_key = 'onnx_file'
        yaml_key = 'yaml_file'
        onnx_content_type = 'application/octet-stream'
        yaml_content_type = 'text/x-yaml'

        name, onnx_file, yaml_file = external_model
        onnx_filename = os.path.basename(onnx_file)
        yaml_filename = os.path.basename(yaml_file)
        with open(onnx_file, 'rb') as onnx_fd, open(yaml_file, 'rb') as yaml_fd:
            external_model_upload_files = [
                (name_key, (None, name)),
                (onnx_key, (onnx_filename, onnx_fd, onnx_content_type)),
                (yaml_key, (yaml_filename, yaml_fd, yaml_content_type)),
            ]
            external_model_upload_response = client.request(external_model_upload_endpoint,
                                                            method=external_model_upload_method,
                                                            files=external_model_upload_files,
                                                            message_prefix=external_model_message_prefix,
                                                            )
        experiment_version_info = parse_json(external_model_upload_response)
        self._update_from_dict(**experiment_version_info)

    def __add_external_models(self, external_models: List[Tuple]) -> None:
        for external_model in external_models:
            self.__add_external_model(external_model)

    @property
    def dropped_features(self):
        raise NotImplementedError

    @property
    def drop_list(self) -> List[str]:
        raise NotImplementedError

    @property
    def feature_list(self):
        raise NotImplementedError

    def get_cv(self):
        raise NotImplementedError

    def predict_single(self, data) -> Dict:
        """ Get a prediction on a single instance using the best model of the experiment.

        Args:

        Returns:
            dict: Dictionary containing the prediction.

            .. note::

                The format of the predictions dictionary depends on the problem type
                (regression, classification...)
        """
        return super().predict_single(data,
                                      confidence=False,
                                      explain=False)

    def predict_from_dataset(self,
                             dataset,
                             dataset_folder=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the experiment.

        Arguments:
            dataset (:class:`.Dataset`): Reference to the dataset object to make
                predictions for
            dataset_folder (:class:`.Dataset`): Matching folder dataset for the
                predictions, if necessary

        Returns:
            ``pd.DataFrame``: Predictions as a ``pandas`` dataframe
        """
        return super().predict_from_dataset(dataset, confidence=False, dataset_folder=dataset_folder)

    def predict(self, df, prediction_dataset_name=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the experiment with a Scikit-learn style blocking prediction mode.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours).
            Prefer using this for simple models and small dataframes, or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): ``pandas`` DataFrame containing the test data

        Returns:
            tuple(pd.DataFrame, str): Prediction data (as ``pandas`` dataframe) and prediction job ID.
        """
        return super().predict(df=df, confidence=False, prediction_dataset_name=prediction_dataset_name)
