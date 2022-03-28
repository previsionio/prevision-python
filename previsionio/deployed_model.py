import os
from typing import Dict
from requests.models import Response
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
from previsionio.utils import NpEncoder
import json
import time
import requests
from . import logger
from . import config
from .utils import handle_error_response, parse_json, PrevisionException
from .prevision_client import client


class DeployedModel(object):

    """
    DeployedModel class to interact with a deployed model.

    Args:
        prevision_app_url (str): URL of the App. Can be retrieved on your app dashbord.
        client_id (str): Your app client id. Can be retrieved on your app dashbord.
        client_secret (str): Your app client secret. Can be retrieved on your app dashbord.
        prevision_token_url (str): URL to get the OAuth2 token of the deployed model. Required
            only if working on-premise (custom IP address) otherwise it is retrieved automatically.
    """

    def __init__(
        self,
        prevision_app_url: str,
        client_id: str,
        client_secret: str,
        prevision_token_url: str = None,
    ):
        """Init DeployedModel (and check that the connection is valid)."""
        self.prevision_app_url = prevision_app_url
        self.client_id = client_id
        self.client_secret = client_secret

        if prevision_token_url:
            self.prevision_token_url = prevision_token_url
        else:
            try:
                version_resp = client.request(
                    '/version',
                    method=requests.get,
                    message_prefix='Getting deployed model oidc_url',
                )
                version_resp = parse_json(version_resp)
                self.prevision_token_url = version_resp['oidc_url']
                self.prevision_token_url += '/auth/realms/prevision.io/protocol/openid-connect/token'
            except Exception as e:
                logger.error(e)
                raise PrevisionException(f'Cannot get prevision_token_url: {e}')

        self.problem_type = None
        self.token = None

        try:
            about_resp = self.request('/about', method=requests.get)
            app_info = parse_json(about_resp)
            self.problem_type = app_info['problem_type']
            self.provider = app_info['provider']
            inputs_resp = self.request('/inputs', method=requests.get)
            self.inputs = parse_json(inputs_resp)
            outputs_resp = self.request('/outputs', method=requests.get)
            self.outputs = parse_json(outputs_resp)
        except Exception as e:
            logger.error(e)
            raise PrevisionException(f'Cannot connect: {e}')

    def _generate_token(self):
        client = BackendApplicationClient(client_id=self.client_id)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(token_url=self.prevision_token_url,
                                  client_id=self.client_id,
                                  client_secret=self.client_secret)
        self.token = token
        return token

    def _get_token(self):
        while self.token is None or time.time() > self.token['expires_at'] - 60:
            try:
                self._generate_token()
            except Exception as e:
                logger.warning(f'failed to generate token with error {e.__repr__()}')
                time.sleep(.5)

    def _check_token_url_app(self):
        if not self.prevision_app_url:
            raise PrevisionException('No url configured. Call client_app.init_client() to initialize')
        if not self.client_id:
            raise PrevisionException('No client id configured. Call client_app.init_client() to initialize')
        if not self.client_secret:
            raise PrevisionException('No client secret configured. Call client_app.init_client() to initialize')

    def request(
        self,
        endpoint: str,
        method,
        files: Dict = None,
        data: Dict = None,
        allow_redirects: bool = True,
        content_type: str = None,
        check_response: bool = True,
        message_prefix: str = None,
        **requests_kwargs,
    ):
        """
        Make a request on the desired endpoint with the specified method & data.

        Requires initialization.

        Args:
            endpoint: (str): api endpoint (e.g. /experiments, /prediction/file)
            method (requests.{get,post,delete}): requests method
            files (dict): files dict
            data (dict): for single predict
            content_type (str): force request content-type
            allow_redirects (bool): passed to requests method

        Returns:
            request response

        Raises:
            Exception: Error if url/token not configured
        """
        self._check_token_url_app()

        url = self.prevision_app_url + endpoint

        status_code = 502
        retries = config.request_retries
        n_tries = 0
        resp = None
        while (n_tries < retries) and (status_code in config.retry_codes):
            n_tries += 1
            try:
                self._get_token()
                assert self.token is not None
                headers = {
                    "Authorization": "Bearer " + self.token['access_token'],
                }
                if content_type:
                    headers['content-type'] = content_type

                resp = method(url,
                              headers=headers,
                              files=files,
                              allow_redirects=allow_redirects,
                              data=data,
                              **requests_kwargs)
                status_code = resp.status_code

            except Exception as e:
                raise PrevisionException(f'Error requesting: {url} with error {e.__repr__()}')

            if status_code in config.retry_codes:
                logger.warning(f'Failed to request {url} with status code {status_code}.'
                               f' Retrying {retries - n_tries} times')
                time.sleep(config.request_retry_time)

        assert isinstance(resp, Response)
        if check_response:
            handle_error_response(resp, url, data, message_prefix=message_prefix, n_tries=n_tries)

        return resp

    def predict(
        self,
        predict_data: Dict = None,
        use_confidence: bool = False,
        explain: bool = False,
        top_k: int = None,
        image_path: str = None,
        threshold: float = None,
    ) -> Dict:
        """ Get a prediction on a single instance using the best model of the experiment.

        Args:
            predict_data (dict, optional): input data for prediction
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            explain (bool, optional): Whether to explain prediction (default: ``False``)
            top_k (int, optional): Number of closest items to return for text-similarity
            image_path (str, optional): Image path for object detection
            threshold (float, optional): prediction threshold for object detection

        Returns:
            dict: prediction result
        """

        predict_url = '/predict'
        files = None
        if use_confidence:
            if self.problem_type not in ['regression', 'classification', 'multiclassification']:
                raise PrevisionException(f'Confidence not available for {self.problem_type}')
            if self.provider == 'external':
                raise PrevisionException('Confidence not available for external models')

        if explain:
            if self.problem_type not in ['regression', 'classification', 'multiclassification']:
                raise PrevisionException(f'Explain not available for {self.problem_type}')

        if top_k is not None:
            if self.problem_type != 'text_similarity':
                raise PrevisionException(f'`top_k` not available for {self.problem_type}')

        if image_path is not None:
            if self.problem_type != 'object_detection':
                raise PrevisionException(f'`image_path` not available for {self.problem_type}')

        if threshold is not None:
            if self.problem_type != 'object_detection':
                raise PrevisionException(f'`threshold` not available for {self.problem_type}')

        if explain or use_confidence:
            predict_url += '?'
            if explain:
                predict_url += 'explain=true&'
            if use_confidence:
                predict_url += 'confidence=true'

        if self.problem_type == 'text_similarity':
            top_k = 10 if top_k is None else top_k
            if not isinstance(top_k, int) or top_k <= 0:
                raise PrevisionException(f'`top_k` should be a strictly positive integer not {top_k}')
            predict_data['top_k'] = top_k

        if self.problem_type == 'object_detection':
            if image_path is None:
                raise PrevisionException('`image_path` is required for object-detector')
            predict_url = '/model/predict'
            if threshold:
                predict_url += '?threshold={}'.format(threshold)
            predict_data = {}
            files = [('image', (os.path.basename(image_path), open(image_path, 'rb'), None))]

        predict_url = predict_url.rstrip('&')
        if predict_data:
            predict_data = json.dumps(predict_data, cls=NpEncoder)
        resp = self.request(predict_url,
                            files=files,
                            data=predict_data,
                            method=requests.post,
                            message_prefix='Deployed model predict')

        return resp.json()
