import requests
from . import client
from .utils import parse_json
from .api_resource import ApiResource, UniqueResourceMixin
import json


class Connector(ApiResource, UniqueResourceMixin):
    """ A connector to interact with a distant source of data (and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        conn_type (str): Type of the connector, among "FTP", "SFTP", "SQL", "S3",
            "HIVE", "HBASE", "GCP"
        username (str, optional): Username to use connect to the remote data source
        password (str, optional): Password to use connect to the remote data source
    """

    resource = 'connectors'
    conn_type = 'connector'

    def __init__(self, _id, name, host=None, port=None, type=None,
                 username='', password='', googleCredentials=None, **kwargs):
        super().__init__(_id, name, host=host, port=port, conn_type=type,
                         username=username, password=password, googleCredentials=googleCredentials)
        self._id = _id
        self.name = name
        self.host = host
        self.port = port
        self.type = type
        self.username = username
        self.password = password
        self.googleCredentials = googleCredentials

        self.other_params = kwargs

    @classmethod
    def list(cls):
        """ List all the available connectors in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Connector` objects rather than
            plain dictionaries with the corresponding data.

        Returns:
            list(:class:`.Connector`): Fetched connector objects
        """
        resp = client.request('/{}'.format(cls.resource),
                              data={'rowsPerPage': -1},
                              method=requests.get)
        resp = parse_json(resp)
        items = resp['items']

        return [cls(**conn_data) for conn_data in items
                if conn_data['type'] == cls.conn_type or cls.conn_type == 'connector']

    @classmethod
    def _new(cls, name, host, port, conn_type, username=None, password=None, googleCredentials=None):
        """ Create a new connector object on the platform.

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            conn_type (str): Type of the connector, among "FTP", "SFTP", "SQL", "S3",
                "HIVE", "HBASE" or "GCP"
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source

        Returns:
            :class:`.Connector`: Newly create connector object
        """
        data = {
            'name': name,
            'host': host,
            'port': port,
            'type': conn_type
        }
        if username:
            data['username'] = username
        if password:
            data['password'] = password
        if googleCredentials:
            data['googleCredentials'] = googleCredentials
            content_type = 'application/json'
            resp = client.request('/{}'.format(cls.resource), data=json.dumps(data),
                                  method=requests.post, content_type=content_type)
        else:
            resp = client.request('/{}'.format(cls.resource), data=data, method=requests.post)

        resp_json = parse_json(resp)
        if '_id' not in resp_json:
            if 'message' in resp_json:
                raise Exception('Prevision.io error: {}'.format(' '.join(resp_json['message'])))
            else:
                raise Exception('unknown error:{}'.format(resp_json))
        data['_id'] = resp_json['_id']

        return cls(**data)

    def test(self):
        """ Test a connector already uploaded on the platform.

        Returns:
            dict: Test results
        """
        resp = client.request('/connectors/{}/test'.format(self.id), method=requests.post)
        resp_json = parse_json(resp)
        return resp_json['message'] == 'Connection successful'


class DataBaseConnector(Connector):

    """ A specific type of connector to interact with a database client (containing databases and tables). """

    def list_databases(self):
        """ List all available databases for the client.

        Returns:
            dict: Databases information
        """
        print("self.resource", self.resource)
        print("self._id", self._id)
        resp = client.request('/{}/{}/data-bases'.format(self.resource, self._id), requests.get)
        resp_json = parse_json(resp)
        return resp_json

    def list_tables(self, database):
        """ List all available tables in a specific database for the client.

        Args:
            database (str): Name of the database to find tables for

        Returns:
            dict: Tables information
        """
        resp = client.request('/{}/{}/data-bases/{}'.format(self.resource, self._id, database), requests.get)
        resp_json = parse_json(resp)
        return resp_json


class FTPConnector(Connector):

    """ A specific type of connector to interact with a FTP client (containing files). """

    conn_type = 'FTP'

    @classmethod
    def new(cls, name, host, port=21, username='', password=''):
        return cls._new(name=name, host=host, conn_type='FTP', port=port, username=username, password=password)


class SFTPConnector(Connector):

    """ A specific type of connector to interact with a secured FTP client (containing files). """

    conn_type = 'SFTP'

    @classmethod
    def new(cls, name, host, port=23, username='', password=''):
        return cls._new(name=name, host=host, conn_type='SFTP', port=port, username=username, password=password)


class SQLConnector(DataBaseConnector):

    """ A specific type of connector to interact with a SQL database client (containing databases and tables). """

    conn_type = 'SQL'

    @classmethod
    def new(cls, name, host, port=3306, username='', password=''):
        return cls._new(name=name, host=host, conn_type='SQL', port=port, username=username, password=password)


class HiveConnector(DataBaseConnector):

    """ A specific type of connector to interact with a Hive database client (containing databases and tables). """

    conn_type = 'HIVE'

    @classmethod
    def new(cls, name, host, port=10000, username='', password=''):
        return cls._new(name=name, host=host, conn_type='HIVE', port=port, username=username, password=password)


# class HBaseConnector(DataBaseConnector):
#
#     """ A specific type of connector to interact with a HBase database client (containing databases and tables). """
#
#     conn_type = 'HBASE'
#
#     @classmethod
#     def new(cls, name, host, port=9090, username='', password=''):
#         return cls._new(name=name, host=host, conn_type='HBASE', port=port, username=username, password=password)


class S3Connector(Connector):

    """ A specific type of connector to interact with an Amazon S3 client (containing buckets with files). """

    conn_type = 'S3'

    @classmethod
    def new(cls, name, host='', port='', username='', password=''):
        return cls._new(name=name, host=host, conn_type='S3', port=port, username=username, password=password)


class GCPConnector(Connector):

    """ A specific type of connector to interact with a GCP database client
        (containing databases and tables or buckets)."""

    conn_type = 'GCP'

    @classmethod
    def new(cls, name, host='', port='', username='', password='', googleCredentials=''):
        return cls._new(name=name, host=host, conn_type='GCP', port=port,
                        username=username, password=password, googleCredentials=googleCredentials)

#
# class HDFSConnector(Connector):
#
#     """ A specific type of connector to interact with a HDFS client. """
#
#     conn_type = 'HDFS'
#
#     @classmethod
#     def new(cls, name, host, port=50070, username='', password=''):
#         return cls._new(name=name, host=host, conn_type='HDFS', port=port, username=username, password=password)


connectors_names = {
    'SQL': SQLConnector,
    'FTP': FTPConnector,
    'SFTP': SFTPConnector,
    'S3': S3Connector,
    'HIVE': HiveConnector,
    'GCP': GCPConnector
}
