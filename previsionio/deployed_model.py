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

PREVISION_TOKEN_URL = 'https://accounts.prevision.io/auth/realms/prevision.io/protocol/openid-connect/token'


class DeployedModel(object):

    """
    DeployedModel class to interact with a deployed model.

    Args:
        prevision_app_url (str): URL of the App. Can be retrieved on your app dashbord.
        client_id (str): Your app client id. Can be retrieved on your app dashbord.
        client_secret (str): Your app client secret. Can be retrieved on your app dashbord.
        prevision_token_url (str): URL of get token. Should be
            https://accounts.prevision.io/auth/realms/prevision.io/protocol/openid-connect/token
            if you're in the cloud, or a custom IP address if installed on-premise.
    """

    def __init__(self, prevision_app_url: str, client_id: str, client_secret: str, prevision_token_url: str = None):
        """Init DeployedModel (and check that the connection is valid)."""
        self.prevision_app_url = prevision_app_url
        self.client_id = client_id
        self.client_secret = client_secret
        if prevision_token_url:
            self.prevision_token_url = prevision_token_url
        else:
            self.prevision_token_url = PREVISION_TOKEN_URL
        self.problem_type = None

        self.token = None
        self.url = None

        self.access_token = None

        try:
            about_resp = self.request('/about', method=requests.get)
            app_info = parse_json(about_resp)
            self.problem_type = app_info['problem_type']
            inputs_resp = self.request('/inputs', method=requests.get)
            self.inputs = parse_json(inputs_resp)
            outputs_resp = self.request('/outputs', method=requests.get)
            self.outputs = parse_json(outputs_resp)
        except Exception as e:
            logger.error(e)
            raise PrevisionException('Cannot connect: {}'.format(e))

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

    def check_types(self, features):
        for feature, value in features:
            pass

    def _check_token_url_app(self):

        if not self.prevision_app_url:
            raise PrevisionException('No url configured. Call client_app.init_client() to initialize')

        if not self.client_id:
            raise PrevisionException('No client id configured. Call client_app.init_client() to initialize')

        if not self.client_secret:
            raise PrevisionException('No client secret configured. Call client_app.init_client() to initialize')

    def request(self, endpoint, method, files=None, data=None, allow_redirects=True, content_type=None,
                check_response=True, message_prefix=None, **requests_kwargs):
        """
        Make a request on the desired endpoint with the specified method & data.

        Requires initialization.

        Args:
            endpoint: (str): api endpoint (e.g. /usecases, /prediction/file)
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

    def predict(self, predict_data: Dict, use_confidence: bool = False, explain: bool = False):
        """ Get a prediction on a single instance using the best model of the usecase.

        Args:
            predict_data (dictionary): input data for prediction
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            explain (bool): Whether to explain prediction (default: ``False``)

        Returns:
            tuple(float, float, dict): Tuple containing the prediction value, confidence and explain.
            In case of regression problem type, confidence format is a list.
            In case of multiclassification problem type, prediction value format is a string.

        """
        # FIXME add some checks for feature name with input api
        features = [{'name': feature, 'value': value}
                    for feature, value in predict_data.items()]

        predict_url = '/predict'

        if explain or use_confidence:
            predict_url += '?'

        if explain:
            predict_url += 'explain=true&'

        if use_confidence:
            predict_url += 'confidence=true'

        predict_url = predict_url.rstrip('&')
        resp = self.request(predict_url,
                            data=json.dumps(features, cls=NpEncoder),
                            method=requests.post,
                            message_prefix='Deployed model predict')

        pred_response = resp.json()
        target_name = self.outputs[0]['keyName']
        preds = pred_response['response']['predictions']
        prediction = preds[target_name]
        if use_confidence:
            if self.problem_type == 'regression':
                confidance_resp = [{key: value} for key, value in preds.items() if 'TARGET_quantile=' in key]
            elif 'confidence' in preds:
                confidance_resp = preds['confidence']
            else:
                confidance_resp = None
        else:
            confidance_resp = None

        if explain and 'explanation' in preds:
            explain_resp = preds['explanation']
        else:
            explain_resp = None

        return prediction, confidance_resp, explain_resp
