from typing import Dict
import requests
from .utils import parse_json, get_all_results
from .prevision_client import client
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
    def from_name(cls, name: str):
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

    def __init__(self, **params):
        self._id: str = params.get('_id', "")
        if self._id == "":
            raise RuntimeError("Invalid _id received from {}".format(str(params)))
        self.resource_id = self._id
        # self.event_manager: Optional[EventManager] = None

    def update_status(self, specific_url: str = None) -> Dict:
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
        resource_status = client.request(url, method=requests.get,
                                         message_prefix='Update status {}'.format(self.resource))
        resource_status_dict = parse_json(resource_status)
        resource_status_dict['event_type'] = 'update'
        resource_status_dict['event_name'] = 'update'

        # if self.event_manager:
        #    self.event_manager.add_event(self.resource_id, resource_status_dict)

        return resource_status_dict

    @property
    def _status(self) -> Dict:

        # if self.event_manager:
        #     events = self.event_manager.events
        #     if self.resource_id in events:
        #         return sorted(events[self.resource_id],
        #                       key=lambda k: datetime.datetime.strptime(k['end'], '%Y-%m-%dT%H:%M:%S.%fZ'))[-1]
        #
        resource_status_dict = self.update_status()
        return resource_status_dict

    @property
    def id(self) -> str:
        return self._id

    def delete(self):
        """Delete a resource from the actual [client] workspace.

        Raises:
            PrevisionException: Any error while deleting data from the platform
        """
        _ = client.request('/{}/{}'.format(self.resource, self._id),
                           method=requests.delete,
                           message_prefix='Delete {}'.format(self.resource))
        logger.info('[Delete {} OK]'.format(self.resource))

    @classmethod
    def _from_id(cls, _id: str = None, specific_url: str = None) -> Dict:
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
        resp = client.request(url, method=requests.get,
                              message_prefix='From id {}'.format(cls.resource))

        resp_json = parse_json(resp)
        if _id is not None:
            logger.info('[Fetch {} OK] by id: "{}"'.format(cls.__name__, _id))
        else:
            logger.info('[Fetch {} OK] by url: "{}"'.format(cls.__name__, specific_url))

        return resp_json

    @classmethod
    def _list(cls, all: bool = True, project_id: str = None):
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
            resources = client.request(url, method=requests.get,
                                       message_prefix='List {}'.format(cls.resource))
            return parse_json(resources)['items']

    # def edit(self, **kwargs):
    #     """Edit a resource on the platform. You simply pass the function a
    #     dictionary of all the fields you want to update (as kwargs), with
    #     the name of the field as key and the new data for the field as value.

    #     .. note::

    #         The parameters you can update can be listed by calling:

    #         .. code-block:: python

    #             print(my_resource.resource_params)

    #     Returns:
    #         dict: Updated resource data

    #     Raises:
    #         PrevisionException: Any error while updating data on the platform
    #             or parsing result
    #     """
    #     update_fields = {
    #         'uid': self._id
    #     }

    #     for param in self.resource_params:
    #         if kwargs.get(param):
    #             update_fields[param] = kwargs[param]
    #     url = '/{}'.format(self.resource)
    #     resp = client.request(url,
    #                           body=update_fields,
    #                           method=requests.put)

    #     resp_json = parse_json(resp)

    #     logger.debug('[{}] {}'.format(self.resource, resp_json['message']))

    #     return resp_json['status'] == 200
