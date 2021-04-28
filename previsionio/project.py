# -*- coding: utf-8 -*-
from __future__ import print_function
import requests

from . import client
from .utils import parse_json, PrevisionException
from . import logger
from .api_resource import ApiResource, UniqueResourceMixin


class Project(ApiResource, UniqueResourceMixin):

    """ A Project

    Args:
        _id (str): Unique id of the datasource
        name (str): Name of the project
        description(str, optional): Description of the project
        color (str, optional): Color of the project

    """

    resource = '/projects'

    def __init__(self, _id: str, name: str, description: str = None, color: str = None, created_by: str = None, admins=[], contributors=[], viewers=[],
                 pipelines_count: int = 0, usecases_count: int = 0, dataset_count: int = 0, users=[], **kwargs):
        """ Instantiate a new :class:`.DataSource` object to manipulate a datasource resource
        on the platform. """
        super().__init__(_id=_id,
                         name=name,
                         description=description,
                         color=color)

        self._id = _id
        self.name = name
        self.description = description
        self.color = color
        self.created_by = created_by
        self.admins = admins
        self.contributors = contributors
        self.viewers = viewers
        self.pipelines_count = pipelines_count
        self.usecases_count = usecases_count
        self.dataset_count = dataset_count
        #self.users = users

    @classmethod
    def list(cls, all=False):
        """ List all the available project in the current active [client] workspace.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.
        Returns:
            list(:class:`.Project`): Fetched project objects
        """
        # FIXME : get /resource return type should be consistent
        resources = super().list(all=all)
        return [cls(**source_data) for source_data in resources]

    @classmethod
    def from_id(cls, _id):
        """Get a project from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.Project`: The fetched datasource

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        # FIXME GET datasource should not return a dict with a "data" key
        resp = client.request('/{}/{}'.format(cls.resource, _id), method=requests.get)
        resp_json = parse_json(resp)

        if resp.status_code != 200:
            logger.error('[{}] {}'.format(cls.resource, resp_json['list']))
            raise PrevisionException('[{}] {}'.format(cls.resource, resp_json['list']))

        return cls(**resp_json)

    def get_id(self):
        return self._id

    def users(self):
        """Get a project from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.Project`: The fetched datasource

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """

        end_point = '/{}/{}/users'.format(self.resource, self._id)
        response = client.request(endpoint=end_point, method=requests.get)
        if response.status_code != 200:
            logger.error('cannot get users for project id {}'.format(self._id))
            raise PrevisionException('[{}] {}'.format(self.resource, response.status_code))

        res = parse_json(response)
        return res

    def add_user(self, email, project_role):
        """Get a project from the instance by its unique id.

        Args:
            email (str): new user email
            project_role (str): user project role. Possible project role: admin, contributor, viewer
        Returns:
            :class:`.Project`: The fetched project

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        if project_role not in ['admin', 'contributor', 'viewer']:
            PrevisionException("Possible project role: admin, contributor, viewer ")
        data = {"email": email, "projectRole": project_role}
        end_point = '/{}/{}/users'.format(self.resource, self._id)
        response = client.request(endpoint=end_point, data=data, method=requests.post)
        if response.status_code != 200:
            logger.error('cannot get users for project id {}'.format(self._id))
            raise PrevisionException('[{}] {}'.format(self.resource, response.status_code))

        res = parse_json(response)
        return res

    def info(self):
        """Get a datasource from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.DataSource`: The fetched datasource

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        project_info = {"_id": self._id,
                        "name": self.name,
                        "description": self.description,
                        "color": self.color,
                        "created_by": self.created_by,
                        "admins": self.admins,
                        "contributors": self.contributors,
                        "viewers": self.viewers,
                        "pipelines_count": self.pipelines_count,
                        "usecases_count": self.usecases_count,
                        "dataset_count": self.dataset_count,
                        "users": self.users}
        return project_info

    @classmethod
    def new(cls, name, description=None, color=None):
        """ Create a new datasource object on the platform.

        Args:

            name (str): Name of the datasource
            name (str): Name of the project
            description(str, optional): Description of the project
            color (str, optional): Color of the project

        Returns:
            :class:`.Project`: The registered project object in the current workspace

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        data = {
            'name': name,
            'description': description,
            'color': color
        }

        resp = client.request('/{}'.format(cls.resource),
                              data=data,
                              method=requests.post)

        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))

        return cls(json['_id'], name, description, color, json['created_by'], json['admins'], json['contributors'], json['pipelines_count'])

    def delete(self):
        """Delete a project from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'
                              .format(self.resource, self.id),
                              method=requests.delete)
        return resp
