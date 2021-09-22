import logging

import os
import copy
from typing import Dict, Union
import requests
import time
import json
import threading
from datetime import datetime

from requests.models import Response

import previsionio.utils
from .logger import logger, event_logger
from . import config
from .utils import handle_error_response, parse_json, PrevisionException


class EventManager:
    def __init__(self, event_endpoint: str, auth_headers: Dict[str, str], client: 'Client'):
        self.event_endpoint = event_endpoint
        auth_headers = copy.deepcopy(auth_headers)
        self.headers = auth_headers
        self.client = client
        self.t = threading.Thread(target=self.update_events, daemon=True)
        self.t.start()

        self._events = (threading.Semaphore(1), {})

    @property
    def events(self):
        events_dict_copy = {}
        semd, event_dict = self._events

        with semd:
            for key, (semi, event_list) in event_dict.items():
                with semi:
                    events_dict_copy[key] = copy.deepcopy(event_list)

        return events_dict_copy

    def wait_for_event(self,
                       resource_id: str,
                       resource_type: str,
                       event_tuple: previsionio.utils.EventTuple,
                       specific_url: str):
        # ignore invalid resource ids
        if not isinstance(resource_id, str):
            return
        self.register_resource(resource_id)
        t0 = time.time()
        while time.time() < t0 + config.default_timeout:
            reconnect_start = time.time()
            semd, event_dict = self._events
            semd.acquire()
            if self.check_resource_event(resource_id, specific_url, event_tuple, semd):
                return
            semd.release()
            time.sleep(0.1)
            while time.time() < reconnect_start + 300:
                semd, event_dict = self._events
                semd.acquire()
                semi, event_list = event_dict[resource_id]
                remaining_events = []
                with semi:
                    for event in event_list:
                        name = event.get('event')
                        if name == event_tuple.name:
                            if self.check_resource_event(resource_id, specific_url, event_tuple, semd):
                                return
                        else:
                            remaining_events.append(event)

                event_dict[resource_id] = semi, remaining_events
                semd.release()
                time.sleep(0.1)
        else:
            raise TimeoutError('Failed to get status {} on {} {}'.format(event_tuple,
                                                                         resource_type,
                                                                         resource_id))

    def check_resource_event(self, resource_id: str, endpoint: str, event_tuple: previsionio.utils.EventTuple,
                             semd: threading.Semaphore):
        resp = self.client.request(endpoint=endpoint, method=requests.get, check_response=False)
        json_response = parse_json(resp)
        logging.debug("{} - endpoint='{}' -> '{}".format(datetime.now(), endpoint, json_response))
        if event_tuple.is_failure(json=json_response):
            semd.release()
            msg = 'Error on resource {}: {}\n{}'.format(resource_id,
                                                        json_response.get('errorMessage', ''),
                                                        json_response)
            raise PrevisionException(msg)
        if event_tuple.is_success(json=json_response):
            semd.release()
            return True
        return False

    def register_resource(self, resource_id: str):
        event_logger.debug('Registering resource with id {}'.format(resource_id))
        payload = {'event': 'REGISTER', '_id': resource_id}
        self.add_event(resource_id, payload)

    def update_events(self):
        sse_timeout = 300
        event_data = None
        event_name = None

        while True:
            sse = requests.get(self.event_endpoint,
                               stream=True,
                               headers=self.headers,
                               timeout=sse_timeout)

            try:
                for line in sse.iter_lines(chunk_size=None):
                    if not line:
                        # filter out keep-alive new lines
                        continue

                    event_logger.debug('url: {} -- data: {}'.format(self.event_endpoint, line))
                    line = line.decode()
                    if line.startswith(':'):
                        # SSE comments can start with ":" character
                        event_logger.debug('sse comment{}'.format(line))
                        continue
                    elif line.startswith('id '):
                        # skip id message information
                        continue
                    elif line.startswith('event: '):
                        # get event name
                        event_name = line[len('event: '):]
                    elif line.startswith('data: '):
                        if event_name is None:
                            # ignore event with unknown name
                            continue

                        # get event data
                        try:
                            event_data = json.loads(line[len('data: '):])
                        except Exception as e:
                            # skip malformed event data
                            event_logger.warning('failed to parse json: "{}" -- error: {}'.format(line, e.__repr__()))
                            continue

                        resource_id = event_data.get('_id', None)

                        if not isinstance(resource_id, str):
                            # ignore event with invalid resource id
                            continue

                        # add event only if monitored resource
                        semd, event_dict = self._events
                        if resource_id in event_dict:
                            payload = {'event': event_name, 'id': resource_id}
                            event_logger.debug('url: {} -- event: {} payload: {}'.format(self.event_endpoint,
                                                                                         event_name,
                                                                                         payload))
                            self.add_event(resource_id, payload)
            except requests.exceptions.ChunkedEncodingError:
                event_logger.warning('closing connection to endpoint: "{}"'.format(self.event_endpoint))
            except requests.exceptions.ConnectionError:
                logger.warning('{}: no messages in {} seconds. reconnecting'.format(self.event_endpoint, sse_timeout))
            except Exception as e:
                logger.error(e)
            finally:
                sse.close()

    def add_event(self, resource_id: str, payload: Union[None, Dict]):
        event_logger.debug('adding event for {}:\npayload = {}'.format(resource_id, payload))
        semd, event_dict = self._events

        if payload and isinstance(payload, dict):
            with semd:
                if resource_id in event_dict:
                    semi, event_list = event_dict[resource_id]
                else:
                    semi, event_list = (threading.Semaphore(1), [])

                with semi:
                    event_list.append(payload)

                event_dict[resource_id] = semi, event_list

                self._events = semd, event_dict


class Client(object):

    """Client class to interact with the Prevision.io platform and manage authentication."""

    def __init__(self):
        self.token = None
        self.prevision_url = None
        self.user_info = None
        self.headers = {
            'accept-charset': 'UTF-8',
            'cache-control': 'no-cache',
            'accept': 'application/json',
        }

        self.api_version = '/ext/v1'

        self.url = None

        self.event_manager = None

    def _check_token_url(self):

        if not self.token:
            raise PrevisionException('No token configured. Call client.init_client() to initialize')

        if not self.prevision_url:
            raise PrevisionException('No url configured. Call client.init_client() to initialize')

    def request(self, endpoint: str, method, files: Dict = None, data: Dict = None,
                format: Dict = None, allow_redirects: bool = True,
                content_type: str = None, check_response: bool = True,
                message_prefix: str = None, **requests_kwargs) -> Response:
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
            check_response (bool): wether to handle error or not
            message_prefix (str): prefix message in error logs

        Returns:
            request response

        Raises:
            Exception: Error if url/token not configured
        """
        self._check_token_url()

        headers = copy.deepcopy(self.headers)
        if content_type:
            headers['content-type'] = content_type

        if self.url is None:
            raise RuntimeError("client.url not properly initialized")

        if format and len(format):
            def format_bool(val):
                if isinstance(val, bool):
                    if val:
                        return "true"
                    return "false"
                return val
            endpoint += '?' + '&'.join(["{}={}".format(key, format_bool(val)) for key, val in format.items()])

        url = self.url + endpoint

        status_code = 502
        retries = config.request_retries
        n_tries = 0
        resp = None
        while (n_tries < retries) and (status_code in config.retry_codes):
            n_tries += 1

            try:
                resp = method(url,
                              headers=headers,
                              files=files,
                              allow_redirects=allow_redirects,
                              json=data,
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
            handle_error_response(resp=resp,
                                  url=url,
                                  data=data,
                                  files=files,
                                  message_prefix=message_prefix,
                                  n_tries=n_tries)

        return resp

    def update_user_info(self):
        assert self.url is not None
        user_info_response = requests.get(self.url + '/profile',
                                          headers=self.headers)
        result = parse_json(user_info_response)

        if 'err_code' in result and result['err_code'] == 'E_UNK':
            raise ValueError('Wrong token ' + str(result))
        self.user_info = result

    def init_client(self, prevision_url: str, token: str):
        """
        Init the client (and check that the connection is valid).

        Args:
            prevision_url (str): URL of the Prevision.io platform. Should be of the form
                https://<instance_name>.prevision.io, or a custom IP address if working on-premise.

            token (str): Your Prevision.io master token. Can be retrieved on
                /dashboard/infos on the web interface or obtained programmatically through:

                .. code-block:: python

                    client.init_client(prevision_url, token)
        """
        self.prevision_url = prevision_url
        self.url = self.prevision_url + self.api_version
        self.token = token
        self.headers['Authorization'] = self.token

        # check for correct connection
        resp = self.request('/version', requests.get)
        handle_error_response(resp, '/version')

        logger.debug('subscribing to events manager')
        self.event_manager = EventManager(self.url + '/events', auth_headers=self.headers, client=self)


client = Client()


if os.getenv('PREVISION_URL') and os.getenv('PREVISION_MASTER_TOKEN'):
    logger.info('Initializing Prevision.io client using environment variables')
    logger.debug('PREVISION_URL:' + os.getenv('PREVISION_URL', ""))
    logger.debug('PREVISION_MASTER_TOKEN:' + os.getenv('PREVISION_MASTER_TOKEN', ""))
    client.init_client(os.getenv('PREVISION_URL', ""), os.getenv('PREVISION_MASTER_TOKEN', ""))
