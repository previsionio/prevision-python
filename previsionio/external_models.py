# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from typing import List, Tuple, Union
import requests
from dateutil import parser

from . import metrics
from .usecase_version import BaseUsecaseVersion
from .usecase_config import DataType, TypeProblem
from .logger import logger
from .prevision_client import client
from .utils import parse_json
from .dataset import Dataset


class ExternalUsecaseVersion(BaseUsecaseVersion):

    def __init__(self, **usecase_version_info):
        print("\nExternalUsecaseVersion__init__")
        print("usecase_version_info:")
        import pprint
        pprint.pprint(usecase_version_info)

        # super().__init__(**usecase_info)
        # NOTE: almost like BaseUsecaseVersion.__init__ but dataset_id is not mandatory here and
        # holdout_dataset_id is not optional.

        # But we need the following call to the grandparent ApiResource.__init__ that set self.resource_id = _id
        super(BaseUsecaseVersion, self).__init__(**usecase_version_info)

        # NOTE: why take the name of the usecase in the usecase version ?
        # self.name: str = usecase_info.get('name', usecase_info['usecase'].get('name'))

        self.description: str = usecase_version_info['description']
        self._id: str = usecase_version_info['_id']
        self.usecase_id: str = usecase_version_info['usecase_id']
        self.project_id: str = usecase_version_info['project_id']
        self.holdout_dataset_id: str = usecase_version_info['holdout_dataset_id']
        self.dataset_id: Union[str, None] = usecase_version_info.get('dataset_id', None)
        self.data_type: DataType = DataType(usecase_version_info['usecase'].get('data_type'))
        self.training_type: TypeProblem = TypeProblem(usecase_version_info['usecase'].get('training_type'))
        self.created_at = parser.parse(usecase_version_info["created_at"])
        self.version: int = int(usecase_version_info.get('version', 1))

        usecase_version_params = usecase_version_info['usecase_version_params']
        self.metric: str = usecase_version_params['metric']

        self._models = {}

    @classmethod
    def _fit(cls, project_id: str, name: str, data_type: DataType, training_type: TypeProblem,
             holdout_dataset: Dataset, target_column: str, external_models: List[Tuple],
             metric: metrics.Enum, dataset: Dataset = None,
             usecase_version_description: str = None) -> 'ExternalUsecaseVersion': # NOTE: in fact it does not return a string...

        holdout_dataset_id = holdout_dataset.id
        js_usecase_version = cls._start_usecase(project_id,
                                                name,
                                                data_type,
                                                training_type,
                                                holdout_dataset_id,
                                                target_column,
                                                external_models,
                                                metric if isinstance(metric, str) else metric.value,
                                                dataset=dataset,
                                                usecase_version_description=usecase_version_description)

        # NOTE: we already have all the info about the usecase_version from the response of confirm api
        #       so we can build the class directly from it and don't call the method _from_id
        # usecase_version = cls.from_id(usecase_version_id)
        usecase_version = cls(**js_usecase_version)

        # NOTE: why wait for usecase_version running ?
        """
        events_url = '/{}/{}'.format(cls.resource, usecase_version_id)
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(usecase_version.resource_id,
                                                cls.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', ('state', 'running')),
                                                specific_url=events_url)

        """

        return usecase_version

    @classmethod
    def _start_usecase(cls, project_id: str, name: str, data_type: DataType, training_type: TypeProblem,
                       holdout_dataset_id: Dataset,
                       target_column: str,
                       external_models: List[Tuple],
                       metric: str,
                       dataset: Dataset = None,
                       usecase_version_description: str = None):
        """ Start a usecase of the given data type and problem type with a specific
        training configuration (on the platform).

        Args:
            name (str): Registration name for the usecase to create
            holdout_datset_id (str|tuple(str, str)): Unique id of the training dataset resource or a tuple of csv and folder id
            data_type (str): Type of data used in the usecase (among "tabular", "images"
                and "timeseries")
            training_type: Type of problem to compute with the usecase (among "regression",
                "classification", "multiclassification" and "object-detection")
            **kwargs:

        Returns:
            :class:`.BaseUsecaseVersion`: Newly created usecase object
        """
        logger.info('[Usecase] Starting usecase')

        usecase_creation_data = {
            'data_type': data_type.value,
            'training_type': training_type.value,
            'name': name,
            'provider': 'external',
        }
        usecase_creation_endpoint = '/projects/{}/{}'.format(project_id, 'usecases')
        print(f'\ncall to {usecase_creation_endpoint}:\ndata={usecase_creation_data}')
        usecase_creation_response = client.request(usecase_creation_endpoint,
                                                   method=requests.post,
                                                   data=usecase_creation_data,
                                                   content_type='application/json',
                                                   message_prefix='Usecase creation')
        usecase_creation_response = parse_json(usecase_creation_response)
        print("\nusecase_creation_response:")
        print(usecase_creation_response)

        usecase_id = usecase_creation_response['_id']

        usecase_version_creation_data = {
            'description': usecase_version_description,
            'holdout_dataset_id': holdout_dataset_id,
            'metric': metric,
            'target_column': target_column,
        }
        usecase_version_creation_endpoint = f'/usecases/{usecase_id}/versions'
        print(f'\ncall to {usecase_version_creation_endpoint}:\ndata={usecase_version_creation_data}')
        usecase_version_creation_response = client.request(usecase_version_creation_endpoint,
                                                           method=requests.post,
                                                           data=usecase_version_creation_data,
                                                           content_type='application/json',
                                                           message_prefix='Usecase version creation')
        usecase_version_creation_response = parse_json(usecase_version_creation_response)
        print("\nusecase_version_creation_response:")
        print(usecase_version_creation_response)

        usecase_version_id = usecase_version_creation_response['_id']

        external_model_upload_endpoint = f'/usecase-versions/{usecase_version_id}/external-models'
        external_model_upload_method = requests.post
        external_model_message_prefix = 'External model uploading'
        onnx_key = 'onnx_file'
        yaml_key = 'yaml_file'
        onnx_content_type = 'application/octet-stream'
        yaml_content_type = 'text/x-yaml'
        for external_model in external_models:
            name, onnx_file, yaml_file = external_model
            external_model_upload_data = {
                'name': name,
            }
            onnx_filename = os.path.basename(onnx_file)
            yaml_filename = os.path.basename(yaml_file)
            with open(onnx_file, 'rb') as onnx_fd, open(yaml_file, 'rb') as yaml_fd:
                external_model_upload_files = [
                    ('name', (None, name)),
                    (onnx_key, (onnx_filename, onnx_fd, onnx_content_type)),
                    (yaml_key, (yaml_filename, yaml_fd, yaml_content_type)),
                ]
                print(f'\ncall to {external_model_upload_endpoint}:\ndata={external_model_upload_data}\nfiles={external_model_upload_files}')
                external_model_upload_response = client.request(external_model_upload_endpoint,
                                                                method=external_model_upload_method,
                                                                files=external_model_upload_files,
                                                                message_prefix=external_model_message_prefix,
                                                                )
            external_model_upload_response = parse_json(external_model_upload_response)
            print("\nexternal_model_upload_response:")
            print(external_model_upload_response)

        usecase_version_confirm_endpoint = f'/usecase-versions/{usecase_version_id}/confirm'
        print(f'\ncall to {usecase_version_confirm_endpoint}...')
        usecase_version_confirm_response = client.request(usecase_version_confirm_endpoint,
                                                          method=requests.put,
                                                          content_type='application/json',
                                                          message_prefix='Usecase version confirmation')
        usecase_version_confirm_response = parse_json(usecase_version_confirm_response)
        print("\nusecase_version_confirm_response:")
        print(usecase_version_confirm_response)

        return usecase_version_confirm_response
