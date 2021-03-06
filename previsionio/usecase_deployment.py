from typing import List
import requests

from .api_resource import ApiResource
from . import client
from .usecase_version import BaseUsecaseVersion
from .utils import parse_json, PrevisionException


class UsecaseDeployment(ApiResource):
    """ UsecaseDeployment objects represent usecase deployment resource that will be explored by Prevision.io platform.

    """

    resource = 'model-deployments'

    def __init__(self, _id: str, name: str, usecase_id, usecase_version_id, current_version,
                 versions, deploy_state, run_state, access_type, project_id, training_type, models, url=None,
                  **kwargs):

        self.name = name
        self._id = _id
        self.usecase_version_id = usecase_version_id
        self.current_version = current_version
        self.versions = versions
        self.deploy_state = deploy_state
        self.run_state = run_state
        self.access_type = access_type
        self.project_id = project_id
        self.training_type = training_type
        self.models = models
        self.url = url

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str):
        url = '/{}/{}'.format('deployments', _id)
        return cls(**super()._from_id(_id=_id, specific_url=url))

    @classmethod
    def list(cls, project_id: str, all: bool = True) -> List['UsecaseDeployment']:
        """ List all the available usecase in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Usecase` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Usecase`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**usecase_deployment) for usecase_deployment in resources]

    @classmethod
    def _new(cls, project_id: str, name: str, main_model, challenger_model=None, access_type: str = 'public'):
        """ Create a new usecase deployement object on the platform.

        Args:
            project_id (str): project id
            name (str): usecase deployement name
            main_model: main model
            challenger_model (optional): challenger model. main and challenger models should be in the same usecase
            access_type (str, optional): public/ fine_grained/ private

        Returns:
            :class:`.UsecaseDeployment`: The registered usecase deployment object in the current project

        Raises:
            PrevisionException: Any error while creating usecase deployment to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        if access_type not in ['public', 'fine_grained', 'private']:
            raise PrevisionException('access type must be public, fine_grained or private')
        main_model_usecase_version_id = main_model.usecase_version_id
        main_usecase = BaseUsecaseVersion._from_id(main_model_usecase_version_id)
        main_usecase_id = main_usecase['usecase_id']
        data = {
            'name': name,
            'usecase_id': main_usecase_id,
            'main_model_usecase_version_id': main_model_usecase_version_id,
            'main_model_id': main_model._id,
            'access_type': access_type
        }

        if challenger_model:
            challenger_model_usecase_version_id = challenger_model.usecase_version_id
            challenger_usecase = BaseUsecaseVersion._from_id(main_model_usecase_version_id)
            challenger_usecase_id = challenger_usecase['usecase_id']
            if main_usecase_id != challenger_usecase_id:
                raise PrevisionException('main and challenger models must be from the same usecase')
            data['challenger_model_usecase_version_id'] = challenger_model_usecase_version_id
            data['challenger_model_id'] = challenger_model._id

        url = '/projects/{}/model-deployments'.format(project_id)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='UsecaseDeployment creation')
        json_resp = parse_json(resp)
        usecase_deployment = cls.from_id(json_resp['_id'])
        return usecase_deployment

    def new_version(self, name: str, main_model, challenger_model=None):
        """ Create a new usecase deployement version.

        Args:
            name (str): usecase deployement name
            main_model: main model
            challenger_model (optional): challenger model. main and challenger models should be in the same usecase

        Returns:
            :class:`.UsecaseDeployment`: The registered usecase deployment object in the current project

        Raises:
            PrevisionException: Any error while creating usecase deployment to the platform
                or parsing the result
            Exception: For any other unknown error
        """
        main_model_usecase_version_id = main_model.usecase_version_id
        main_usecase = BaseUsecaseVersion._from_id(main_model_usecase_version_id)
        main_usecase_id = main_usecase['usecase_id']
        data = {
            'name': name,
            'main_model_usecase_version_id': main_model_usecase_version_id,
            'main_model_id': main_model._id,
        }
        if challenger_model:
            challenger_model_usecase_version_id = challenger_model.usecase_version_id
            challenger_usecase = BaseUsecaseVersion._from_id(main_model_usecase_version_id)
            challenger_usecase_id = challenger_usecase['usecase_id']
            if main_usecase_id != challenger_usecase_id:
                raise PrevisionException('main and challenger models must be from the same usecase')
            data['challenger_model_usecase_version_id'] = challenger_model_usecase_version_id
            data['challenger_model_id'] = challenger_model._id

        url = '/deployments/{}/versions'.format(self._id)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='UsecaseDeployment creation new version')
        json_resp = parse_json(resp)
        usecase_deployment = self.from_id(json_resp['_id'])
        return usecase_deployment

    def delete(self):
        """Delete a usecase deployement from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/deployments/{}'.format(self.id),
                              method=requests.delete,
                              message_prefix='UsecaseDeployment delete')
        return resp

    def create_api_key(self):
        """Get run logs of usecase deployement from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        print('/deployments/{}/api-keys'.format(self.id))
        resp = client.request(endpoint='/deployments/{}/api-keys'.format(self.id),
                              method=requests.post,
                              message_prefix='UsecaseDeployment create api key')
        resp = parse_json(resp)
        return resp

    def get_api_keys(self):
        """Get run logs of usecase deployement from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        print('/deployments/{}/api-keys'.format(self.id))
        resp = client.request(endpoint='/deployments/{}/api-keys'.format(self.id),
                              method=requests.get,
                              message_prefix='UsecaseDeployment get api key')
        resp = parse_json(resp)
        return resp
