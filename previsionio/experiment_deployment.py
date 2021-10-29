from typing import List
import time
import requests

from . import config
from .logger import logger
from .api_resource import ApiResource
from . import client
from .experiment_version import BaseExperimentVersion
from .utils import parse_json, PrevisionException, get_all_results
from .prediction import DeploymentPrediction
from .dataset import Dataset
from .model import Model


class ExperimentDeployment(ApiResource):
    """ ExperimentDeployment objects represent experiment deployment
    resource that will be explored by Prevision.io platform.
    """

    resource = 'model-deployments'

    def __init__(self, _id: str, name: str, experiment_id, current_version,
                 versions, deploy_state, access_type, project_id, training_type, models, url=None,
                 **kwargs):

        self.name = name
        self._id = _id
        # self.experiment_version_id = experiment_version_id
        self.experiment_id = experiment_id
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
        """Get a deployed experiment from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment version to retrieve

        Returns:
            :class:`.ExperimentDeployment`: Fetched deployed experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        url = '/{}/{}'.format('deployments', _id)
        result = super()._from_id(_id=_id, specific_url=url)
        logger.debug(result)
        return cls(**result)

    @property
    def deploy_state(self):
        experiment_deployment = self.from_id(self._id)
        return experiment_deployment._deploy_state

    @property
    def run_state(self):
        experiment_deployment = self.from_id(self._id)
        return experiment_deployment._run_state

    @classmethod
    def list(cls, project_id: str, all: bool = True) -> List['ExperimentDeployment']:
        """ List all the available experiment in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.ExperimentDeployment` objects rather
            than plain dictionaries with the corresponding data.

        Args:
            project_id (str): project id
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.ExperimentDeployment`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [ExperimentDeployment(**experiment_deployment) for experiment_deployment in resources]

    @classmethod
    def _new(
        cls,
        project_id: str,
        name: str,
        main_model: Model,
        challenger_model: Model = None,
        access_type: str = 'public',
    ):
        """ Create a new experiment deployment object on the platform.

        Args:
            project_id (str): project id
            name (str): experiment deployment name
            main_model (:class:`.Model`): main model
            challenger_model (:class:`.Model`, optional): challenger model (main and challenger
                models should come from the same experiment)
            access_type (str, optional): public/ fine_grained/ private

        Returns:
            :class:`.ExperimentDeployment`: The registered experiment deployment object in the current project

        Raises:
            PrevisionException: Any error while creating experiment deployment to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        if access_type not in ['public', 'fine_grained', 'private']:
            raise PrevisionException('access type must be public, fine_grained or private')
        main_model_experiment_version_id = main_model.experiment_version_id
        main_experiment = BaseExperimentVersion._from_id(main_model_experiment_version_id)
        main_experiment_id = main_experiment['experiment_id']
        data = {
            'name': name,
            'experiment_id': main_experiment_id,
            'main_model_experiment_version_id': main_model_experiment_version_id,
            'main_model_id': main_model._id,
            'access_type': access_type
        }

        if challenger_model:
            challenger_model_experiment_version_id = challenger_model.experiment_version_id
            challenger_experiment = BaseExperimentVersion._from_id(main_model_experiment_version_id)
            challenger_experiment_id = challenger_experiment['experiment_id']
            if main_experiment_id != challenger_experiment_id:
                raise PrevisionException('main and challenger models must be from the same experiment')
            data['challenger_model_experiment_version_id'] = challenger_model_experiment_version_id
            data['challenger_model_id'] = challenger_model._id

        url = '/projects/{}/model-deployments'.format(project_id)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='ExperimentDeployment creation')
        json_resp = parse_json(resp)
        experiment_deployment = cls.from_id(json_resp['_id'])
        return experiment_deployment

    def new_version(self, name: str, main_model, challenger_model=None):
        """ Create a new experiment deployment version.

        Args:
            name (str): experiment deployment name
            main_model: main model
            challenger_model (optional): challenger model. main and challenger models should be in the same experiment

        Returns:
            :class:`.ExperimentDeployment`: The registered experiment deployment object in the current project

        Raises:
            PrevisionException: Any error while creating experiment deployment to the platform
                or parsing the result
            Exception: For any other unknown error
        """
        main_model_experiment_version_id = main_model.experiment_version_id
        main_experiment = BaseExperimentVersion._from_id(main_model_experiment_version_id)
        main_experiment_id = main_experiment['experiment_id']
        data = {
            'name': name,
            'main_model_experiment_version_id': main_model_experiment_version_id,
            'main_model_id': main_model._id,
        }
        if challenger_model:
            challenger_model_experiment_version_id = challenger_model.experiment_version_id
            challenger_experiment = BaseExperimentVersion._from_id(main_model_experiment_version_id)
            challenger_experiment_id = challenger_experiment['experiment_id']
            if main_experiment_id != challenger_experiment_id:
                raise PrevisionException('main and challenger models must be from the same experiment')
            data['challenger_model_experiment_version_id'] = challenger_model_experiment_version_id
            data['challenger_model_id'] = challenger_model._id

        url = '/deployments/{}/versions'.format(self._id)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='ExperimentDeployment creation new version')
        json_resp = parse_json(resp)
        experiment_deployment = self.from_id(json_resp['_id'])
        return experiment_deployment

    def delete(self):
        """Delete an experiment deployment from the actual [client] workspace.

        Raises:
            PrevisionException: If the experiment deployment does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    def wait_until(self, condition, timeout: float = config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseExperimentVersion`) -> bool.): Function to use to check the
                break condition
            raise_on_error (bool, optional): If true then the function will stop on error,
                otherwise it will continue waiting (default: ``True``)
            timeout (float, optional): Maximal amount of time to wait before forcing exit

        Example::

            experiment.wait_until(lambda experimentv: len(experimentv.models) > 3)

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
        """Create an api key of the experiment deployment from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        print('/deployments/{}/api-keys'.format(self.id))
        resp = client.request(endpoint='/deployments/{}/api-keys'.format(self.id),
                              method=requests.post,
                              message_prefix='ExperimentDeployment create api key')
        resp = parse_json(resp)
        return resp

    def get_api_keys(self):
        """Fetch the api keys client id and cient secret of the
        experiment deployment from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/deployments/{}/api-keys'.format(self.id),
                              method=requests.get,
                              message_prefix='ExperimentDeployment get api key _id')
        resp = parse_json(resp)
        api_keys_ids = [item['_id'] for item in resp['items']]
        res = []
        for api_keys_id in api_keys_ids:
            resp = client.request(endpoint='/api-keys/{}/secret'.format(api_keys_id),
                                  method=requests.get,
                                  message_prefix='ExperimentDeployment get api key client_id and secret')
            resp = parse_json(resp)
            res.append({'client_id': resp['service_account_client_id'],
                        'client_secret': resp['client_secret']})
        return res

    def predict_from_dataset(self, dataset: Dataset) -> DeploymentPrediction:
        """ Make a prediction for a dataset stored in the current active [client]
        workspace (using the current SDK dataset object).

        Args:
            dataset (:class:`.Dataset`): Dataset resource to make a prediction for

        Returns:
            :class:`.DeploymentPrediction`: The registered prediction object in the current workspace
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
        """ List all the available predictions in the current active [client] workspace.

        Returns:
            list(:class:`.DeploymentPrediction`): Fetched deployed predictions objects
        """
        end_point = '/deployments/{}/deployment-predictions'.format(self._id)
        predictions = get_all_results(client, end_point, method=requests.get)
        return [DeploymentPrediction(**prediction) for prediction in predictions]
