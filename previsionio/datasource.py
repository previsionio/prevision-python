# -*- coding: utf-8 -*-
from __future__ import print_function
import requests

from . import client
from .utils import parse_json, PrevisionException
from .api_resource import ApiResource, UniqueResourceMixin


class DataSource(ApiResource, UniqueResourceMixin):

    """ A datasource to access a distant data pool and create or fetch data easily. This
    resource is linked to a :class:`.Connector` resource that represents the connection to
    the distant data source.

    Args:
        _id (str): Unique id of the datasource
        connector (:class:`.Connector`): Reference to the associated connector (the resource
            to go through to get a data snapshot)
        name (str): Name of the datasource
        path (str, optional): Path to the file to fetch via the connector
        database (str, optional): Name of the database to fetch data from via the
            connector
        table (str, optional): Name of the table  to fetch data from via the connector
        request (str, optional): Direct SQL request to use with the connector to fetch data
    """

    resource = 'data-sources'

    def __init__(self, _id, connector_id: str, name: str, path: str = None, database: str = None,
                 table: str = None, request: str = None, gCloud=None, **kwargs):
        """ Instantiate a new :class:`.DataSource` object to manipulate a datasource resource
        on the platform. """
        super().__init__(_id=_id,
                         connector=connector_id,
                         name=name,
                         path=path,
                         database=database,
                         table=table,
                         request=request,
                         gCloud=gCloud)

        self._id = _id
        self.connector = connector_id

        self.name = name
        self.path = path
        self.database = database
        self.table = table
        self.request = request
        self.gCloud = gCloud

        self.other_params = kwargs

    @classmethod
    def list(cls, project_id: str, all: bool = False):
        """ List all the available datasources in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.DataSource` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.DataSource`): Fetched datasource objects
        """
        # FIXME : get /resource return type should be consistent
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**source_data) for source_data in resources]

    @classmethod
    def from_id(cls, _id: str):
        """Get a datasource from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.DataSource`: The fetched datasource

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        # FIXME GET datasource should not return a dict with a "data" key
        url = '/{}/{}'.format(cls.resource, _id)
        resp = client.request(url, method=requests.get, message_prefix='From id data source')
        resp_json = parse_json(resp)

        return cls(**resp_json)

    @classmethod
    def _new(cls, project_id: str, connector, name: str, path: str = None, database: str = None, table: str = None,
             bucket=None, request=None, gCloud=None):
        """ Create a new datasource object on the platform.

        Args:
            connector (:class:`.Connector`): Reference to the associated connector (the resource
                to go through to get a data snapshot)
            name (str): Name of the datasource
            path (str, optional): Path to the file to fetch via the connector
            database (str, optional): Name of the database to fetch data from via the
                connector
            table (str, optional): Name of the table  to fetch data from via the connector
            request (str, optional): Direct SQL request to use with the connector to fetch data

        Returns:
            :class:`.DataSource`: The registered datasource object in the current workspace

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        data = {
            'connector_id': connector._id,
            'name': name,
            'path': path,
            'database': database,
            'bucket': bucket,
            'table': table,
            'request': request
        }
        if gCloud:
            data['g_cloud'] = gCloud

        url = '/projects/{}/{}'.format(project_id, cls.resource)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='Datasource creation')
        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))
        return cls(json['_id'], connector, name, path, database, table, request)
