import os
from typing import Dict, List
import time
import requests
from requests_toolbelt import MultipartEncoder

from . import config
from .logger import logger
from .api_resource import ApiResource
from . import client
from .experiment_version import BaseExperimentVersion
from .utils import parse_json, PrevisionException, get_all_results, EventTuple
from .prediction import DeploymentPrediction
from .dataset import Dataset
from .model import Model


class BaseExperimentDeployment(ApiResource):
    """ ExperimentDeployment objects represent experiment deployment
    resource that will be explored by Prevision.io platform.
    """

    resource = 'model-deployments'

    def __init__(
        self,
        _id: str,
        name: str,
        experiment_id: str,
        current_version: int,
        versions: List[Dict],
        deploy_state: str,
        project_id: str,
        training_type: str,
        models: List[Dict],
        current_type_violation_policy: str,
        **kwargs,
    ):
        self.name = name
        self._id = _id
        # self.experiment_version_id = experiment_version_id
        self.experiment_id = experiment_id
        self.current_version = current_version
        self.versions = versions
        self._deploy_state = deploy_state
        self.project_id = project_id
        self.training_type = training_type
        self.models = models
        self.type_violation_policy = current_type_violation_policy

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str) -> 'BaseExperimentDeployment':
        """Get a deployed experiment from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment version to retrieve

        Returns:
            :class:`.BaseExperimentDeployment`: Fetched deployed experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        url = '/{}/{}'.format('deployments', _id)
        result = super()._from_id(_id=_id, specific_url=url)
        logger.debug(result)
        return cls(**result)

    @property
    def deploy_state(self) -> str:
        experiment_deployment = self.from_id(self._id)
        return experiment_deployment._deploy_state

    @classmethod
    def list(cls, project_id: str, all: bool = True) -> List['ExperimentDeployment']:
        """ List all the available experiment in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.ExperimentDeployment` objects rather
            than plain dictionaries with the corresponding data.

        Args:
            project_id (str): project id
            all (bool, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.BaseExperimentDeployment`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**experiment_deployment) for experiment_deployment in resources]

    @classmethod
    def _new(
        cls,
        project_id: str,
        name: str,
        main_model: Model,
        challenger_model: Model = None,
        type_violation_policy: str = 'best_effort',
        access_type: str = None,
    ) -> 'BaseExperimentDeployment':
        """ Create a new experiment deployment object on the platform.

        Args:
            project_id (str): project id
            name (str): experiment deployment name
            main_model (:class:`.Model`): main model
            challenger_model (:class:`.Model`, optional): challenger model (main and challenger
                models should come from the same experiment)
            type_violation_policy (str, optional): best_effort/ strict
            access_type (str, optional): public/ fine_grained/ private

        Returns:
            :class:`.BaseExperimentDeployment`: The registered experiment deployment object in the current project

        Raises:
            PrevisionException: Any error while creating experiment deployment to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        if access_type and access_type not in ['public', 'fine_grained', 'private']:
            raise PrevisionException('access type must be public, fine_grained or private')
        if type_violation_policy not in ['best_effort', 'strict']:
            raise PrevisionException('type_violation_policy must be best_effort or strict')
        main_model_experiment_version_id = main_model.experiment_version_id
        main_experiment = BaseExperimentVersion._from_id(main_model_experiment_version_id)
        main_experiment_id = main_experiment['experiment_id']
        data = {
            'name': name,
            'experiment_id': main_experiment_id,
            'main_model_experiment_version_id': main_model_experiment_version_id,
            'main_model_id': main_model._id,
            'type_violation_policy': type_violation_policy
        }
        if access_type:
            data['access_type'] = access_type
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

    def new_version(
        self,
        name: str,
        main_model: Model,
        challenger_model: Model = None,
    ) -> 'BaseExperimentDeployment':
        """ Create a new experiment deployment version.

        Args:
            name (str): experiment deployment name
            main_model(:class:`.Model`): main model
            challenger_model (:class:`.Model`, optional): challenger model.
                Main and challenger models should be in the same experiment

        Returns:
            :class:`.BaseExperimentDeployment`: The registered experiment deployment object in the current project

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
        url = '/{}/{}'.format('deployments', self._id)
        client.request(endpoint=url,
                       method=requests.delete,
                       message_prefix='Delete deployment')

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


class ExperimentDeployment(BaseExperimentDeployment):

    def __init__(
        self,
        _id: str,
        name: str,
        experiment_id: str,
        current_version: int,
        versions: List[Dict],
        deploy_state: str,
        current_type_violation_policy: str,
        access_type: str,
        project_id: str,
        training_type: str,
        models: List[Dict],
        url: str = None,
        **kwargs,
    ):
        super().__init__(_id, name, experiment_id, current_version, versions,
                         deploy_state, project_id, training_type, models,
                         current_type_violation_policy, **kwargs)

        self.access_type = access_type
        self.training_type = training_type
        self.models = models
        self._run_state = kwargs.pop("main_model_run_state", kwargs.pop("run_state", "error"))

    @property
    def run_state(self) -> str:
        experiment_deployment = self.from_id(self._id)
        return experiment_deployment._run_state

    def create_api_key(self) -> Dict:
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

    def get_api_keys(self) -> List[Dict]:
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
        return DeploymentPrediction.from_id(predict_start_parsed['_id'])

    def list_predictions(self) -> List[DeploymentPrediction]:
        """ List all the available predictions in the current active [client] workspace.

        Returns:
            list(:class:`.DeploymentPrediction`): Fetched deployed predictions objects
        """
        end_point = '/deployments/{}/deployment-predictions'.format(self._id)
        predictions = get_all_results(client, end_point, method=requests.get)
        return [DeploymentPrediction(**prediction) for prediction in predictions]


class ExternallyHostedModelDeployment(BaseExperimentDeployment):

    def log_bulk_prediction(
        self,
        input_file_path: str,
        output_file_path: str,
        model_role: str = 'main',
    ) -> Dict:
        """ Log bulk prediction from local parquet files.

        Args:
            input_file_path (str): Path to an input parquet file
            output_file_path (str): Path to an ouput parquet file
            model_role (str, optional): main / challenger

        Raises:
            PrevisionException: If error while logging bulk prediction
            requests.exceptions.ConnectionError: Error processing the request
        """
        request_url = '/deployments/{}/log-bulk-predictions'.format(self._id)
        data = {'type_model': model_role}
        with open(input_file_path, 'rb') as f_input, open(output_file_path, 'rb') as f_output:
            data['pred_input'] = (os.path.basename(input_file_path), f_input, '')
            data['pred_output'] = (os.path.basename(output_file_path), f_output, '')
            encoder = MultipartEncoder(fields=data)
            create_resp = client.request(request_url,
                                         content_type=encoder.content_type,
                                         data=encoder,
                                         is_json=False,
                                         method=requests.post,
                                         message_prefix='log bulk prediction')
            create_resp_parsed = parse_json(create_resp)
            log_bulk_prediction_id = create_resp_parsed['_id']
            specific_url = "/log-bulk-predictions/{}".format(log_bulk_prediction_id)
            client.event_manager.wait_for_event(log_bulk_prediction_id,
                                                specific_url,
                                                EventTuple(
                                                    'DEPLOYMENT_PREDICTION_UPDATE',
                                                    ('state', 'done'),
                                                    [('state', 'failed')]),
                                                specific_url=specific_url)
            url = '/{}/{}'.format('log-bulk-predictions', log_bulk_prediction_id)
            resp = client.request(endpoint=url,
                                  method=requests.get,
                                  message_prefix='Get log bulk prediction')
            create_resp_parsed = parse_json(resp)
            return create_resp_parsed

    def get_log_bulk_prediction_by_id(id):
        url = '/{}/{}'.format('log-bulk-predictions', id)
        resp = client.request(endpoint=url,
                              method=requests.get,
                              message_prefix='Get log bulk prediction')
        create_resp_parsed = parse_json(resp)
        return create_resp_parsed

    def log_unit_prediction(
        self,
        _input: Dict,
        output: Dict,
        model_role: str = 'main',
        deployment_version: int = None,
    ) -> Dict:
        """ Log unit prediction.

        Args:
            input (dict): input prediction data
            output (dict): output prediction data
            model_role (str, optional): main / challenger
            deployment_version (int, optional): deployment version to use.
                Last version is used by default

        Raises:
            PrevisionException: If error while logging unit prediction
            requests.exceptions.ConnectionError: Error processing the request
        """
        request_url = '/deployments/{}/log-unit-prediction'.format(self._id)
        data = {'input': _input,
                'output': output,
                'model_role': model_role,
                'deployment_version': deployment_version}
        create_resp = client.request(request_url,
                                     data=data,
                                     method=requests.post,
                                     message_prefix='log unit prediction')
        create_resp_parsed = parse_json(create_resp)
        return create_resp_parsed

    def list_log_bulk_predictions(self) -> List[Dict]:
        """ List all the available log bulk predictions.

        Returns:
            list(dict): Fetched log bulk predictions
        """
        end_point = '/deployments/{}/log-bulk-predictions'.format(self._id)
        log_bulk_predictions = get_all_results(client, end_point, method=requests.get)
        return log_bulk_predictions
