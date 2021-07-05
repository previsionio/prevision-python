from typing import List
import requests

from .api_resource import ApiResource
from . import client
from .utils import parse_json, PrevisionException, EventTuple, zip_to_pandas
from .usecase_config import UsecaseState


class Prediction(ApiResource):
    """ Prediction object represent usecase prediction bulk resource that will be explored by Prevision.io platform.

    """

    resource = 'predictions'

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


        # for k, v in kwargs.items():
        #     print("k============", k)
        #     print("v============", v)
        #     self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str) -> 'Prediction':
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

    @classmethod
    def delete(self):
        """Delete a prediction from the platform by its unique id.

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        response = client.request(endpoint='/predictions/{}'.format(self._id),
                                  method=requests.delete,
                                  message_prefix='Prediction deletion')
        return response


    def get_data(self):
        """Delete a prediction from the platform by its unique id.

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        self._wait_for_prediction()
        url = '/predictions/{}/download'.format(self._id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)
        return response


    def _wait_for_prediction(self):
        """ Wait for a specific prediction to finish.

        Args:
            predict_id (str): Unique id of the prediction to wait for
        """
        specific_url = '/predictions/{}'.format(self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple('PREDICTION_UPDATE', 'state', 'done', [('state', 'failed')]),
                                            specific_url=specific_url)
