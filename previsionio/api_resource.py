from typing import Optional
import datetime
import requests
from .utils import parse_json, PrevisionException, get_all_results
from .prevision_client import client, EventManager
from . import logger
from enum import Enum


class ApiResourceType(Enum):

    "All the different resource types and matching API endpoints."

    ApiResource = 'api_resource'
    Dataset = 'files/dataset'
    Folder = 'files/folder'
    Usecase = 'usecases'
    Connector = 'connectors'
    DataSource = 'datasources'
    Project = 'projects'


class UniqueResourceMixin:
    @classmethod
    def from_name(cls, name):
        resources = cls.list()
        resources_match = [c for c in resources if c.name == name]
        if resources_match:
            return resources_match[0]
        else:
            raise Exception('No such {}: {}'.format(cls.resource, name))


class ApiResource:

    """Base parent class for all SDK resource objects."""

    resource = ApiResourceType.ApiResource
    resource_params = []
    id_key = '_id'

    def __init__(self, *args, **params):
        self._id = params.get('_id')
        self.resource_id = self._id
        self.event_manager: Optional[EventManager] = None

    def update_status(self, specific_url=None):
        """Get an update on the status of a resource.

        Args:
            specific_url (str, optional): Specific (already parametrized) url to
                fetch the resource from (otherwise the url is built from the
                resource type and unique ``_id``)

        Returns:
            dict: Updated status info
        """
        if specific_url is None:
            url = '/{}/{}'.format(self.resource, self._id)
        else:
            url = specific_url

        # call api, add event to eventmanager events if available and return
        resource_status = client.request(url, method=requests.get)
        resource_status_dict = parse_json(resource_status)
        resource_status_dict['event_type'] = 'update'
        resource_status_dict['event_name'] = 'update'

        if self.event_manager:
            self.event_manager.add_event(self.resource_id, resource_status_dict)

        return resource_status_dict

    @property
    def _status(self):
        if self.event_manager:
            events = self.event_manager.events
            if self.resource_id in events:
                return sorted(events[self.resource_id],
                              key=lambda k: datetime.datetime.strptime(k['end'], '%Y-%m-%dT%H:%M:%S.%fZ'))[-1]

        resource_status_dict = self.update_status()
        return resource_status_dict

    @property
    def id(self):
        return self._id

    @classmethod
    def from_name(cls, name, raise_if_non_unique=False, partial_match=False):
        """Get a resource from the platform by its name.

        Args:
            name (str): Name of the resource to retrieve
            raise_if_non_unique (bool, optional): Whether or not to raise an error if
                duplicates are found (default: ``False``)
            partial_match (bool, optional): If true, resources with a name containing
                the requested name will also be returned; else, only perfect matches
                will be found (default: ``False``)

        Raises:
            PrevisionException: Error if duplicates are found and
                the ``raise_if_non_unique`` is enabled

        Returns:
            :class:`.ApiResource`: Fetched resource
        """
        resources = cls.list()

        if partial_match:
            matches = []
            for i in resources:
                if (isinstance(i, dict) and name in i['name']) or \
                   (isinstance(i, ApiResource) and name in i.name):
                    matches.append(i)
        else:
            matches = []
            for i in resources:
                if (isinstance(i, dict) and name == i['name']) or \
                   (isinstance(i, ApiResource) and name in i.name):
                    matches.append(i)

        if len(matches) == 1 or len(matches) > 1 and not raise_if_non_unique:
            if isinstance(matches[0], dict):
                result = cls.from_id(matches[0][cls.id_key])
            else:
                result = cls.from_id(matches[0]._id)
            logger.info('[Fetch {} OK] by name: "{}"'.format(cls.__name__, name))
            return result
        elif len(matches) > 1 and raise_if_non_unique:
            msg = 'Ambiguous {} name: {}, found {} resources matching: {}'
            msg = msg.format(cls.resource,
                             name,
                             len(matches),
                             ', '.join([m['name'] for m in matches]))
            raise PrevisionException(msg)
        else:
            raise PrevisionException('No such {}: {}'.format(cls.resource, name))

    def delete(self):
        """Delete a resource from the actual [client] workspace.

        Raises:
            PrevisionException: Any error while deleting data from the platform
        """
        resp = client.request('/{}/{}'.format(self.resource, self._id), method=requests.delete)
        resp_json = parse_json(resp)
        if resp.status_code == 200:
            logger.info('[Delete {} OK] {}'.format(self.resource, str(resp_json)))
            return
        else:
            raise PrevisionException('[Delete {}] {} Error'.format(self.resource, str(resp_json)))

    @classmethod
    def from_id(cls, _id=None, specific_url=None):
        """Get a resource from the platform by its unique id.
        You must provide either an ``_id`` or a ``specific_url``.

        Args:
            _id (str, optional): Unique id of the resource to retrieve
            specific_url (str, optional): Specific (already parametrized) url to
                fetch the resource from

        Returns:
            :class:`.ApiResource`: Fetched resource

        Raises:
            Exception: If neither an ``_id`` nor a ``specific_url`` was provided
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        if specific_url is None:
            if _id is None:
                raise Exception('[{}] Provide an _id or a specific url for "from_id" method'.format(cls.resource))
            url = '/{}/{}'.format(cls.resource, _id)
        else:
            url = specific_url
        resp = client.request(url, method=requests.get)

        if resp.status_code != 200:
            logger.error('[{}] {}'.format(cls.resource, resp.text))
            raise PrevisionException('[{}] {}'.format(cls.resource, resp.text))
        resp_json = parse_json(resp)
        if _id is not None:
            logger.info('[Fetch {} OK] by id: "{}"'.format(cls.__name__, _id))
        else:
            logger.info('[Fetch {} OK] by url: "{}"'.format(cls.__name__, specific_url))
        return cls(**resp_json)

    @classmethod
    def list(cls, all=False, project_id=None):
        """List all available instances of this resource type on the platform.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            dict: Fetched resources
        """
        if project_id:
            url = '/projects/{}/{}'.format(project_id, cls.resource)
        else:
            url = '/{}'.format(cls.resource)
        if all:
            return get_all_results(client, url, method=requests.get)
        else:
            resources = client.request(url, method=requests.get)
            return parse_json(resources)['items']

    def edit(self, **kwargs):
        """Edit a resource on the platform. You simply pass the function a
        dictionary of all the fields you want to update (as kwargs), with
        the name of the field as key and the new data for the field as value.

        .. note::

            The parameters you can update can be listed by calling:

            .. code-block:: python

                print(my_resource.resource_params)

        Returns:
            dict: Updated resource data

        Raises:
            PrevisionException: Any error while updating data on the platform
                or parsing result
        """
        update_fields = {
            'uid': self._id
        }

        for param in self.resource_params:
            if kwargs.get(param):
                update_fields[param] = kwargs[param]

        resp = client.request('/{}'.format(self.resource),
                              body=update_fields,
                              method=requests.put)

        resp_json = parse_json(resp)

        if resp_json['status'] != 200:
            logger.error('[{}] {}'.format(self.resource, resp_json['message']))
            raise PrevisionException('[{}] {}'.format(self.resource, resp_json['message']))

        logger.debug('[{}] {}'.format(self.resource, resp_json['message']))

        return resp_json['status'] == 200
