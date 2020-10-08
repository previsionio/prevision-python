# -*- coding: utf-8 -*-
from __future__ import print_function
import requests

from . import client
from .utils import parse_json, PrevisionException
from . import logger
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

    resource = 'datasources'

    def __init__(self, _id, connector, name, path=None, database=None, table=None, request=None, gCloud=None, **kwargs):
        """ Instantiate a new :class:`.DataSource` object to manipulate a datasource resource
        on the platform. """
        super().__init__(_id, name,
                         connector=connector,
                         name=name,
                         path=path,
                         database=database,
                         table=table,
                         request=request,
                         gCloud=gCloud)

        self._id = _id
        self.connector = connector

        self.name = name
        self.path = path
        self.database = database
        self.table = table
        self.request = request
        self.gCloud = gCloud

        self.other_params = kwargs

    @classmethod
    def list(cls):
        """ List all the available datasources in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.DataSource` objects rather than
            plain dictionaries with the corresponding data.

        Returns:
            list(:class:`.DataSource`): Fetched datasource objects
        """
        # FIXME : get /resource return type should be consistent
        resp = client.request('/{}'.format(cls.resource), method=requests.get)
        resp = parse_json(resp)['items']
        return [cls(**source_data) for source_data in resp]

    @classmethod
    def from_id(cls, _id):
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
        resp = client.request('/{}/{}'.format(cls.resource, _id), method=requests.get)
        resp_json = parse_json(resp)

        if resp.status_code != 200:
            logger.error('[{}] {}'.format(cls.resource, resp_json['list']))
            raise PrevisionException('[{}] {}'.format(cls.resource, resp_json['list']))

        return cls(**resp_json)

    @classmethod
    def new(cls, connector, name, path=None, database=None, table=None, bucket=None, request=None, gCloud=None):
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
            'connectorId': connector._id,
            'name': name,
            'path': path,
            'database': database,
            'bucket': bucket,
            'table': table,
            'request': request
        }
        if gCloud:
            data['gCloud'] = gCloud

        resp = client.request('/{}'.format(cls.resource),
                              data=data,
                              method=requests.post)

        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))

        return cls(json['_id'], connector, name, path, database, table, request)
