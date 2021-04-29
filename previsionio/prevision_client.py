from __future__ import print_function

import os
import copy
import requests
import time
import json
import threading

import previsionio.utils
from .logger import logger, event_logger
from . import config
from .utils import parse_json, PrevisionException

EVENT_TIMEOUT = int(os.environ.get('EVENT_TIMEOUT', 600))


class EventManager:
    def __init__(self, event_endpoint, auth_headers, client):
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
                       resource_id,
                       resource_type,
                       event_tuple: previsionio.utils.EventTuple,
                       specific_url=None):
        # ignore invalid resource ids
        if not isinstance(resource_id, str):
            return
        self.register_resource(resource_id)
        t0 = time.time()
        print("event_tuple.name", event_tuple.name)
        while time.time() < t0 + config.default_timeout:
            reconnect_start = time.time()
            while time.time() < reconnect_start + 60:
                semd, event_dict = self._events

                semd.acquire()
                semi, event_list = event_dict[resource_id]

                remaining_events = []
                with semi:
                    for event in event_list:
                        if event.get('event') == event_tuple.name:
                            print("event========", event)
                            print("specific_url=======", specific_url)
                            resp = self.client.request(endpoint=specific_url, method=requests.get)
                            print("resp=====", resp)
                            print("resp.text==========",resp.text)
                            json_response = parse_json(resp)
                            for k, v in event_tuple.fail_checks:
                                if json_response.get(k) == v:
                                    semd.release()
                                    msg = 'Error on resource {}: {}\n{}'.format(resource_id,
                                                                                json_response.get('errorMessage', ''),
                                                                                json_response)
                                    raise PrevisionException(msg)
                            if json_response.get(event_tuple.key) == event_tuple.value:
                                semd.release()
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

    def register_resource(self, resource_id):
        event_logger.debug('Registering resource with id {}'.format(resource_id))
        payload = {'event': 'REGISTER', '_id': resource_id}
        self.add_event(resource_id, payload)

    def update_events(self):
        sse_timeout = 300
        while True:
            print("self.event_endpoint",self.event_endpoint)
            sse = requests.get(self.event_endpoint,
                               stream=True,
                               headers=self.headers,
                               timeout=sse_timeout)

            try:
                for msg in sse.iter_content(chunk_size=None):
                    event_logger.debug('url: {} -- data: {}'.format(self.event_endpoint, msg))
                    msg = msg.decode()
                    # SSE comments can start with ":" character
                    if msg[0] == ':':
                        event_logger.debug('sse comment{}'.format(msg))
                        continue
                    try:
                        if len(msg.split('\n'))>=3:
                            _, event_name, event_data, *rest = msg.split('\n')
                            event_name = event_name.replace('event: ', '')
                            event_data = json.loads(event_data.replace('data: ', '').strip())
                    except json.JSONDecodeError as e:
                        event_logger.warning('failed to parse json: "{}" -- error: {}'.format(msg, e.__repr__()))
                    except requests.exceptions.ChunkedEncodingError:
                        event_logger.warning('closing connection to endpoint: "{}"'.format(self.event_endpoint))
                        sse.close()
                        return
                    else:
                        resource_id = event_data.get('_id', None)
                        # ignore invalid resource ids
                        if not isinstance(resource_id, str):
                            continue
                        payload = {'event': event_name, 'id': resource_id}
                        event_logger.debug('url: {} -- event: {} payload: {}'.format(self.event_endpoint,
                                                                                     event_name,
                                                                                     payload))
                        # add event only if monitored resource
                        semd, event_dict = self._events
                        if resource_id in event_dict:
                            self.add_event(resource_id, payload)
            except requests.exceptions.ConnectionError:
                logger.warning('{}: no messages in {} seconds. reconnecting'.format(self.event_endpoint, sse_timeout))
            except Exception as e:
                logger.error(e)
                raise
            finally:
                sse.close()

    def add_event(self, resource_id, payload):
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

    def request(self, endpoint, method, files=None, data=None, allow_redirects=True, content_type=None,
                no_retries=False, **requests_kwargs):
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
            no_retries (bool): force request to run the first time, or exit directly

        Returns:
            request response

        Raises:
            Exception: Error if url/token not configured
        """
        self._check_token_url()

        headers = copy.deepcopy(self.headers)
        if content_type:
            headers['content-type'] = content_type

        url = self.url + endpoint

        retries = 1 if no_retries else config.request_retries

        for i in range(retries):
            try:
                req = method(url,
                             headers=headers,
                             files=files,
                             allow_redirects=allow_redirects,
                             json=data,
                             **requests_kwargs)
            except Exception as e:
                logger.warning('failed to request ' + url + ' retrying ' + str(retries - i) + ' times: ' + e.__repr__())
                if no_retries:
                    raise PrevisionException('error requesting: {} (no retry allowed)'.format(url)) from None
                time.sleep(config.request_retry_time)
            else:
                break
        else:
            raise Exception('failed to request')

        return req

    def update_user_info(self):
        user_info_response = requests.get(self.url + '/profile',
                                          headers=self.headers)
        result = parse_json(user_info_response)

        if 'err_code' in result and result['err_code'] == 'E_UNK':
            raise ValueError('Wrong token ' + str(result))
        self.user_info = result

    def init_client(self, prevision_url, token):
        """
        Init the client (and check that the connection is valid).

        Args:
            prevision_url (str): URL of the Prevision.io platform. Should be
                https://cloud.prevision.io if you're in the cloud, or a custom
                IP address if installed on-premise.

            token (str): Your Prevision.io master token. Can be retrieved on
                /dashboard/infos on the web interface or obtained programmatically through:

                .. code-block:: python

                    client.init_client_with_login(prevision_url, email, password)
        """
        self.prevision_url = prevision_url
        self.url = self.prevision_url + self.api_version
        self.token = token
        self.headers['Authorization'] = self.token

        # check for correct connection
        try:
            resp = self.request('/version', requests.get, no_retries=True)
        except Exception as e:
            logger.error(e)
            raise PrevisionException('Cannot connect: check your instance url')
        if resp.status_code == 401:
            msg = 'Cannot connect: check your master token'
            logger.error(msg)
            raise PrevisionException(msg)

        logger.debug('subscribing to events manager')
        self.event_manager = EventManager(self.url + '/events', auth_headers=self.headers, client=self)


client = Client()


if os.getenv('PREVISION_URL') and os.getenv('PREVISION_MASTER_TOKEN'):
    logger.info('Initializing Prevision.io client using environment variables')
    logger.debug('PREVISION_URL:' + os.getenv('PREVISION_URL'))
    logger.debug('PREVISION_MASTER_TOKEN:' + os.getenv('PREVISION_MASTER_TOKEN'))
    client.init_client(os.getenv('PREVISION_URL'), os.getenv('PREVISION_MASTER_TOKEN'))
