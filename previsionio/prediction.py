import os
import requests

from .api_resource import ApiResource
from . import client
from .utils import PrevisionException, EventTuple, zip_to_pandas


class ValidationPrediction(ApiResource):
    """ A prediction object that represents an experiment bulk prediction resource
    which can be explored on the Prevision.io platform.
    """

    resource = 'validation-predictions'

    def __init__(self, _id: str, experiment_id: str, experiment_version_id: str, project_id: str, state='running',
                 model_id=None, dataset_id=None, filename=None, download_available=False, score=None, duration=None,
                 predictions_count=None, **kwargs):
        self._id = _id
        self.experiment_id = experiment_id
        self.experiment_version_id = experiment_version_id
        self.project_id = project_id
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.filename = filename
        self.download_available = download_available
        self.score = score
        self.duration = duration
        self.predictions_count = predictions_count
        self._state = state

    @classmethod
    def from_id(cls, _id: str) -> 'ValidationPrediction':
        """Get a prediction from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment bulk prediction to retrieve

        Returns:
            :class:`.ValidationPrediction`: The fetched prediction

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
            PrevisionException: If the prediction images does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    def get_result(self):
        """Get the prediction result.

        Returns:
            ``pd.DataFrame``: Prediction results dataframe

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

    def download(self, path: str = None, extension="zip"):
        """Download validation prediction file.

        Args:
            path (str, optional): Target local path
                (if none is provided, the current working directory is
                used)
            extension(str, optional): possible extensions: zip, parquet, csv
        Returns:
            str: Path the data was downloaded to

        Raises:
            PrevisionException: If prediction does not exist or if there
                was another error fetching or parsing data
        """
        endpoint = '/{}/{}/download'.format(self.resource, self.id)
        if extension:
            endpoint += "?extension={}".format(extension)
        resp = client.request(endpoint=endpoint,
                              method=requests.get,
                              stream=True,
                              message_prefix='validation Prediction download')
        if not path:
            download_path = os.getcwd()
            prediction_file_name = 'prediction_{}.{}'.format(self.filename.replace(' ', '_'), extension)
            path = os.path.join(download_path, prediction_file_name)

        with open(path, "wb") as file:
            for chunk in resp.iter_content(chunk_size=100_000_000):
                if chunk:
                    file.write(chunk)
            file.seek(0)

        return path

    def _wait_for_prediction(self):
        """ Wait for a specific prediction to finish.

        Args:
            predict_id (str): Unique id of the prediction to wait for
        """
        specific_url = '/{}/{}'.format(self.resource, self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple('PREDICTION_UPDATE'),
                                            specific_url=specific_url)


class DeploymentPrediction(ApiResource):
    """ A prediction object that represents a deployed experiment bulk prediction resource which can be explored on the
    Prevision.io platform.
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
            self.__setattr__(k, v)

    @classmethod
    def from_id(cls, _id: str) -> 'DeploymentPrediction':
        """Get a prediction from the platform by its unique id.

        Args:
            _id (str): Unique id of the deployed experiment bulk prediction to retrieve

        Returns:
            :class:`.DeploymentPrediction`: The fetched prediction

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id)))

    def get_result(self):
        """Get the prediction result of the main model.

        Returns:
            ``pd.DataFrame``: Prediction results dataframe

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        specific_url = '/{}/{}'.format(self.resource, self._id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple(
                                                'DEPLOYMENT_PREDICTION_UPDATE',
                                                ('main_model_prediction_state', 'done'),
                                                [('main_model_prediction_state', 'failed')]),
                                            specific_url=specific_url)

        url = '/{}/{}/download'.format(self.resource, self._id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)

    def download(self, path: str = None, extension="zip"):
        """Download deployment prediction file.

        Args:
            path (str, optional): Target local path
                (if none is provided, the current working directory is
                used)
            extension(str, optional): possible extensions: zip, parquet, csv
        Returns:
            str: Path the data was downloaded to

        Raises:
            PrevisionException: If prediction does not exist or if there
                was another error fetching or parsing data
        """
        endpoint = '/{}/{}/download'.format(self.resource, self.id)
        if extension:
            endpoint += "?extension={}".format(extension)
        resp = client.request(endpoint=endpoint,
                              method=requests.get,
                              stream=True,
                              message_prefix='Deployment Prediction download')

        if not path:
            download_path = os.getcwd()
            prediction_file_name = 'prediction_{}.{}'.format(self.filename.replace(' ', '_'), extension)
            path = os.path.join(download_path, prediction_file_name)

        with open(path, "wb") as file:
            for chunk in resp.iter_content(chunk_size=100_000_000):
                if chunk:
                    file.write(chunk)
            file.seek(0)

        return path

    def get_challenger_result(self):
        """Get the prediction result of the challenger model.

        Returns:
            ``pd.DataFrame``: Prediction results dataframe

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        if self.challenger_model_id is None:
            PrevisionException('Challenger data not availbale for this prediction')
        specific_url = '/{}/{}'.format(self.resource, self.challenger_model_id)
        client.event_manager.wait_for_event(self._id,
                                            specific_url,
                                            EventTuple(
                                                'DEPLOYMENT_PREDICTION_UPDATE',
                                                ('challenger_model_prediction_state', 'done'),
                                                [('challenger_model_prediction_state', 'failed')]),
                                            specific_url=specific_url)

        url = '/{}/{}/download'.format(self.resource, self.challenger_model_id)
        pred_response = client.request(url,
                                       method=requests.get,
                                       message_prefix='Predictions download')

        return zip_to_pandas(pred_response)
