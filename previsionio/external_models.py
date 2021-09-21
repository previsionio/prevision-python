# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from typing import Dict, List, Tuple, Union
import requests

from . import metrics
from .usecase_version import BaseUsecaseVersion
from .prevision_client import client
from .utils import parse_json
from .dataset import Dataset


class ExternalUsecaseVersion(BaseUsecaseVersion):

    def __init__(self, **usecase_version_info):
        super().__init__(**usecase_version_info)
        self._update_from_dict(**usecase_version_info)

    def _update_from_dict(self, **usecase_version_info):
        super()._update_from_dict(**usecase_version_info)
        holdout_dataset_id: str = usecase_version_info['holdout_dataset_id']
        self.holdout_dataset: Dataset = Dataset.from_id(holdout_dataset_id)
        self.target_column = usecase_version_info.get('target_column')

        self.metric = usecase_version_info.get('metric')
        dataset_id: Union[str, None] = usecase_version_info.get('dataset_id')
        self.dataset: Union[str, None] = Dataset.from_id(dataset_id) if dataset_id is not None else None

        usecase_version_params = usecase_version_info['usecase_version_params']
        self.metric: str = usecase_version_params['metric']

    def _update_draft(self, external_models, **kwargs):
        self.__add_external_models(external_models)

    @staticmethod
    def _build_usecase_version_creation_data(description, holdout_dataset, target_column,
                                             metric, dataset, parent_version=None,
                                             **kwargs) -> Dict:
        data = super(ExternalUsecaseVersion, ExternalUsecaseVersion)._build_usecase_version_creation_data(
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
             usecase_id: str,
             holdout_dataset: Dataset,
             target_column: str,
             external_models: List[Tuple],
             metric: metrics.Enum,
             dataset: Dataset = None,
             description: str = None) -> 'ExternalUsecaseVersion':
        return super()._fit(
            usecase_id,
            description=description,
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
                    description: str = None) -> 'ExternalUsecaseVersion':
        return ExternalUsecaseVersion._fit(
            self.usecase_id,
            holdout_dataset if holdout_dataset is not None else self.holdout_dataset,
            target_column if target_column is not None else self.target_column,
            external_models,
            metric if metric is not None else self.metric,
            dataset=dataset if dataset is not None else self.dataset,
            description=description,
        )

    def __add_external_model(self, external_model: Tuple) -> None:
        external_model_upload_endpoint = f'/usecase-versions/{self._id}/external-models'
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
            print(f'\ncall to {external_model_upload_endpoint}:\nfiles={external_model_upload_files}')
            external_model_upload_response = client.request(external_model_upload_endpoint,
                                                            method=external_model_upload_method,
                                                            files=external_model_upload_files,
                                                            message_prefix=external_model_message_prefix,
                                                            )
        usecase_version_info = parse_json(external_model_upload_response)
        self._update_from_dict(**usecase_version_info)

    def __add_external_models(self, external_models: List[Tuple]) -> None:
        for external_model in external_models:
            self.__add_external_model(external_model)
