from typing import List
import requests

from .api_resource import ApiResource
from . import client
from .utils import PrevisionException, EventTuple, zip_to_pandas


class ValidationPrediction(ApiResource):
    """ Prediction object represent usecase prediction bulk resource that will be explored by Prevision.io platform.

    """

    resource = 'validation-prediction'

    def __init__(self, _id: str,usecase_id: str, usecase_version_id: str, project_id: str,  state='running' , model_id=None, model_name=None,
                 dataset_id=None, download_available=False, score=None, duration=None, predictions_count=None, **kwargs):

        self._id = _id
        self.usecase_id = usecase_id
        self.usecase_version_id = usecase_version_id
        self.project_id = project_id
        self.model_id = model_id
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.download_available = download_available
        self.score = score
        self.duration = duration
        self.predictions_count = predictions_count
        self._state = state

    @classmethod
    def from_id(cls, _id: str) -> 'ValidationPrediction':
        """Get a prediction from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase prediction bulk to retrieve

        Returns:
            :class:`.Prediction`: Fetched prediction

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id)))

    @property
    def status(self):
        status = self._status
        return status['state']

    def delete(self):
        """Delete a prediction from the platform by its unique id.

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        response = client.request(endpoint='/{}/{}'.format(self.resource, self._id),
                                  method=requests.delete,
                                  message_prefix='Prediction deletion')
        return response

    def get_result(self):
        """Delete a prediction from the platform by its unique id.

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        self._wait_for_prediction()
        url = '/{}/{}/download'.format(self.resource, self._id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)
        return pred_response

    def _wait_for_prediction(self):
        """ Wait for a specific prediction to finish.

        Args:
            predict_id (str): Unique id of the prediction to wait for
        """
        specific_url = '/{}/{}'.format(self.resource, self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple('PREDICTION_UPDATE', 'state', 'done', [('state', 'failed')]),
                                            specific_url=specific_url)


class DeploymentPrediction(ApiResource):
    """ Prediction object represent usecase prediction bulk resource that will be explored by Prevision.io platform.

    """

    resource = 'deployment-predictions'

    def __init__(self, _id: str, project_id: str, deployment_id: str, state='running', main_model_id=None,
                 challenger_model_id=None, **kwargs):

        self._id = _id
        self.project_id = project_id
        self.deployment_id = deployment_id
        self.main_model_id = main_model_id
        self.challenger_model_id = challenger_model_id
        self._state = state
        for k, v in kwargs.items():
            #print("k============", k)
            #print("v============", v)
            self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str) -> 'DeploymentPrediction':
        """Get a prediction from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase prediction bulk to retrieve

        Returns:
            :class:`.DeploymentPrediction`: Fetched prediction

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id)))

    def get_result(self):
        specific_url = '/{}/{}'.format(self.resource, self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple('DEPLOYMENT_PREDICTION_UPDATE', 'main_model_prediction_state', 'done',
                                                       [('main_model_prediction_state', 'failed')]),
                                            specific_url=specific_url)
        #
        url = '/{}/{}/download'.format(self.resource, self._id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)

    def get_challenger_result(self):
        if self.challenger_model_id is None:
            PrevisionException('Challenger data not availbale for this prediction')
        specific_url = '/{}/{}'.format(self.resource, self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple('DEPLOYMENT_PREDICTION_UPDATE', 'challenger_model_prediction_state', 'done',
                                                       [('challenger_model_prediction_state', 'failed')]),
                                            specific_url=specific_url)
        #
        url = '/{}/{}/download'.format(self.resource, self._id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)
