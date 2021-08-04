from typing import List
import time
import requests

from . import config
from .logger import logger
from .api_resource import ApiResource
from . import client
from .usecase_version import BaseUsecaseVersion
from .utils import parse_json, PrevisionException, get_all_results
from .prediction import DeploymentPrediction
from .dataset import Dataset


class UsecaseDeployment(ApiResource):
    """ UsecaseDeployment objects represent usecase deployment resource that will be explored by Prevision.io platform.

    """

    resource = 'model-deployments'

    def __init__(self, _id: str, name: str, usecase_id, current_version,
                 versions, deploy_state, access_type, project_id, training_type, models, url=None,
                 **kwargs):

        self.name = name
        self._id = _id
        #self.usecase_version_id = usecase_version_id
        self.usecase_id = usecase_id
        self.current_version = current_version
        self.versions = versions
        self._deploy_state = deploy_state
        self.access_type = access_type
        self.project_id = project_id
        self.training_type = training_type
        self.models = models
        self.url = url

        self._run_state = kwargs.pop("main_model_run_state", kwargs.pop("run_state", "error"))
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str):
        url = '/{}/{}'.format('deployments', _id)
        result = super()._from_id(_id=_id, specific_url=url)
        logger.debug(result)
        return cls(**result)

    @property
    def deploy_state(self):
        usecase_deployment = self.from_id(self._id)
        return usecase_deployment._deploy_state

    @property
    def run_state(self):
        usecase_deployment = self.from_id(self._id)
        return usecase_deployment._run_state

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
        return [UsecaseDeployment(**usecase_deployment) for usecase_deployment in resources]

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

    def wait_until(self, condition, timeout: float = config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseUsecaseVersion`) -> bool.): Function to use to check the
                break condition
            raise_on_error (bool, optional): If true then the function will stop on error,
                otherwise it will continue waiting (default: ``True``)
            timeout (float, optional): Maximal amount of time to wait before forcing exit

        Example::

            usecase.wait_until(lambda usecasev: len(usecasev.models) > 3)

        Raises:
            PrevisionException: If the resource could not be fetched or there was a timeout.
        """
        t0 = time.time()
        while True:
            if timeout is not None and time.time() - t0 > timeout:
                raise PrevisionException('timeout while waiting on {}'.format(condition))
            try:
                if condition(self):
                    break
                elif self.deploy_state == 'failed' or self.run_state == 'failed':
                    raise PrevisionException('Resource failed while waiting')
            except PrevisionException as e:
                logger.warning(e.__repr__())
                raise

            time.sleep(config.scheduler_refresh_rate)

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

    def predict_from_dataset(self, dataset: Dataset) -> DeploymentPrediction:
        """ Make a prediction for a dataset stored in the current active [client]
        workspace (using the current SDK dataset object).

        Args:
            dataset (:class:`.Dataset`): Dataset resource to make a prediction for

        Returns:
            ``pd.DataFrame``: Prediction object
        """
        if self.training_type not in ['regression', 'classification', 'multiclassification']:
            PrevisionException('Prediction not supported yet for training type {}', self.training_type)
        data = {
            'dataset_id': dataset._id,
        }

        predict_start = client.request('/deployments/{}/deployment-predictions'.format(self._id),
                                       method=requests.post,
                                       data=data,
                                       message_prefix='Bulk predict')
        predict_start_parsed = parse_json(predict_start)
        return DeploymentPrediction(**predict_start_parsed)

    def list_predictions(self) -> List[DeploymentPrediction]:

        end_point = '/deployments/{}/deployment-predictions'.format(self._id)
        predictions = get_all_results(client, end_point, method=requests.get)
        return [DeploymentPrediction(**prediction) for prediction in predictions]
