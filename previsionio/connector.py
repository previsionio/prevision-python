from enum import Enum
from typing import Union
import requests
from . import client
from .utils import parse_json
from .api_resource import ApiResource, UniqueResourceMixin


class GCloud(Enum):
    """
    Google services."""
    big_query = 'BigQuery'
    """Google BigQuery"""
    storage = 'Storage'
    """Google Storage"""


class Connector(ApiResource, UniqueResourceMixin):
    """ A connector to interact with a distant source of data (and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        conn_type (str): Type of the connector, among "FTP", "SFTP", "SQL", "S3", "GCP"
        username (str, optional): Username to use connect to the remote data source
        password (str, optional): Password to use connect to the remote data source
    """

    resource = 'connectors'
    conn_type = 'connector'

    def __init__(self, _id: str, name: str, host: str = None, port: int = None, type: str = None,
                 username: str = '', password: str = '', googleCredentials: str = None, **kwargs):
        super().__init__(_id=_id, name=name, host=host, port=port, conn_type=type,
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
    def list(cls, project_id: str, all: bool = False):
        """ List all the available connectors in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Connector` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Connector`): Fetched connector objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources
                if conn_data['type'] == cls.conn_type or cls.conn_type == 'connector']

    @classmethod
    def _new(cls, project_id: str, name: str, host: str, port: Union[int, None], conn_type: str, username: str = None,
             password: str = None, googleCredentials: str = None):
        """ Create a new connector object on the platform.

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            conn_type (str): Type of the connector, among "FTP", "SFTP", "SQL", "S3", "GCP"
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
        message_prefix = 'New connector creation'
        if username:
            data['username'] = username
        if password:
            data['password'] = password
        if googleCredentials:
            data['googleCredentials'] = googleCredentials
            content_type = 'application/json'
            resp = client.request('/projects/{}/{}'.format(project_id, cls.resource), data=data, method=requests.post,
                                  content_type=content_type, message_prefix=message_prefix)
        else:
            resp = client.request('/projects/{}/{}'.format(project_id, cls.resource), data=data, method=requests.post,
                                  message_prefix=message_prefix)

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
        resp = client.request('/connectors/{}/test'.format(self.id), method=requests.post, check_response=False)
        if resp.status_code == 200:
            return True
        else:
            return False

    def delete(self):
        """Delete a connector from the actual [client] workspace.

        Raises:
            PrevisionException: If the connector does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()


class DataTableBaseConnector(Connector):

    """ A specific type of connector to interact with a database client (containing databases and tables). """

    def list_databases(self):
        """ List all available databases for the client.

        Returns:
            dict: Databases information
        """
        url = '/{}/{}/databases'.format(self.resource, self._id)
        resp = client.request(url, requests.get, message_prefix='Databases listing')
        resp_json = parse_json(resp)
        return resp_json['items']

    def list_tables(self, database: str):
        """ List all available tables in a specific database for the client.

        Args:
            database (str): Name of the database to find tables for

        Returns:
            dict: Tables information
        """
        url = '/{}/{}/databases/{}/tables'.format(self.resource, self._id, database)
        resp = client.request(url, requests.get, message_prefix='Tables listing')
        resp_json = parse_json(resp)
        return resp_json['items']


class DataFileBaseConnector(Connector):
    """ A specific type of connector to interact with a database client (containing files). """

    def list_files(self):
        """ List all available tables in a specific database for the client.

        Args:
            database (str): Name of the database to find tables for

        Returns:
            dict: files information
        """
        url = '/{}/{}/paths'.format(self.resource, self._id)
        resp = client.request(url, requests.get, message_prefix='Datasource files listing')
        resp_json = parse_json(resp)
        return resp_json['items']


class FTPConnector(DataFileBaseConnector):

    """ A specific type of connector to interact with a FTP client (containing files). """

    conn_type = 'FTP'


class SFTPConnector(DataFileBaseConnector):

    """ A specific type of connector to interact with a secured FTP client (containing files). """

    conn_type = 'SFTP'


class SQLConnector(DataTableBaseConnector):

    """ A specific type of connector to interact with a SQL database client (containing databases and tables). """

    conn_type = 'SQL'


# class HiveConnector(DataTableBaseConnector):

#     """ A specific type of connector to interact with a Hive database client (containing databases and tables). """

#     conn_type = 'HIVE'


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


class GCPConnector(Connector):

    """ A specific type of connector to interact with a GCP database client
        (containing databases and tables or buckets)."""

    conn_type = 'GCP'

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
    'GCP': GCPConnector,
}
