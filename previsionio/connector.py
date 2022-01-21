from enum import Enum
from typing import Dict
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
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
    """

    resource = 'connectors'
    conn_type = 'connector'

    def __init__(self, _id: str, project_id: str, name: str):
        super().__init__(_id=_id)
        self.project_id = project_id
        self.name = name

    @classmethod
    def list(cls, project_id: str, all: bool = False):
        """ List all the available connectors in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Connector` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            project_id (str): Unique reference of the project id on the platform
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Connector`): Fetched connector objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [connectors_names.get(conn_data['type']).from_dict(conn_data) for conn_data in resources
                if conn_data['type'] == cls.conn_type or cls.conn_type == 'connector']

    @classmethod
    def _create_connector(cls, project_id: str, data: Dict, content_type: str = None):
        data['type'] = cls.conn_type
        resp = client.request('/projects/{}/{}'.format(project_id, cls.resource),
                              data=data,
                              method=requests.post,
                              message_prefix='New connector creation',
                              content_type=content_type)

        connector_info = parse_json(resp)
        if '_id' not in connector_info:
            if 'message' in connector_info:
                raise Exception('Prevision.io error: {}'.format(' '.join(connector_info['message'])))
            else:
                raise Exception('unknown error:{}'.format(connector_info))

        return connector_info

    @classmethod
    def _new(cls, project_id: str, name: str, host: str, port: int, username: str, password: str):
        """ Create a new connector object on the platform.

        Args:
            project_id (str): Unique reference of the project id on the platform
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str): Username to use connect to the remote data source
            password (str): Password to use connect to the remote data source

        Returns:
            :class:`.Connector`: Newly create connector object
        """
        data = {
            'name': name,
            'host': host,
            'port': port,
            'username': username,
            'password': password,
        }
        connector_info = cls._create_connector(project_id, data)
        return cls.from_dict(connector_info)

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
    """ A specific type of connector to interact with a database client (containing databases and tables), and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    def __init__(self, _id: str, project_id: str, name: str, host: str, port: int, username: str):
        super().__init__(_id=_id, project_id=project_id, name=name)
        self.host = host
        self.port = port
        self.username = username

    @classmethod
    def from_dict(cls, connector_info: Dict):
        connector = cls(
            connector_info['_id'],
            connector_info['project_id'],
            connector_info['name'],
            connector_info['host'],
            connector_info['port'],
            connector_info['username'],
        )
        return connector

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
    """ A specific type of connector to interact with a database client (containing files), and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    def __init__(self, _id: str, project_id: str, name: str, host: str, port: int, username: str):
        super().__init__(_id=_id, project_id=project_id, name=name)
        self.host = host
        self.port = port
        self.username = username

    @classmethod
    def from_dict(cls, connector_info: Dict):
        connector = cls(
            connector_info['_id'],
            connector_info['project_id'],
            connector_info['name'],
            connector_info['host'],
            connector_info['port'],
            connector_info['username'],
        )
        return connector

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
    """ A specific type of connector to interact with a FTP client (containing files), and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    conn_type = 'FTP'


class SFTPConnector(DataFileBaseConnector):
    """ A specific type of connector to interact with a secured FTP client (containing files), and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    conn_type = 'SFTP'


class SQLConnector(DataTableBaseConnector):
    """ A specific type of connector to interact with a SQL database client (containing databases and tables), and
    easily get data snapshots using an associated :class:`.DataSource`
    resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    conn_type = 'SQL'


class S3Connector(Connector):
    """ A specific type of connector to interact with an Amazon S3 client (containing buckets with files),
    and easily get data snapshots using an associated :class:`.DataSource` resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
        host (str): Url of the connector
        port (int): Port of the connector
        username (str): Username to use connect to the remote data source
    """

    conn_type = 'S3'

    def __init__(self, _id: str, project_id: str, name: str, username: str):
        super().__init__(_id=_id, project_id=project_id, name=name)
        self.username = username

    @classmethod
    def from_dict(cls, connector_info: Dict):
        connector = cls(
            connector_info['_id'],
            connector_info['project_id'],
            connector_info['name'],
            connector_info['username'],
        )
        return connector

    @classmethod
    def _new(cls, project_id: str, name: str, username: str, password: str):
        """ Create a new connector object on the platform.

        Args:
            project_id (str): Unique reference of the project id on the platform
            name (str): Name of the connector
            username (str): Username to use connect to the remote data source
            password (str): Password to use connect to the remote data source

        Returns:
            :class:`.S3Connector`: Newly create connector object
        """
        data = {
            'name': name,
            'username': username,
            'password': password,
        }
        connector_info = cls._create_connector(project_id, data)
        return cls.from_dict(connector_info)


class GCPConnector(Connector):
    """ A specific type of connector to interact with a GCP database client
    (containing databases and tables or buckets), and easily get data snapshots
    using an associated :class:`.DataSource` resource).

    Args:
        _id (str): Unique reference of the connector on the platform
        project_id (str): Unique reference of the project id on the platform
        name (str): Name of the connector
    """

    conn_type = 'GCP'

    @classmethod
    def from_dict(cls, connector_info: Dict):
        connector = cls(
            connector_info['_id'],
            connector_info['project_id'],
            connector_info['name'],
        )
        return connector

    @classmethod
    def _new(cls, project_id: str, name: str, googleCredentials: Dict):
        """ Create a new connector object on the platform.

        Args:
            project_id (str): Unique reference of the project id on the platform
            name (str): Name of the connector
            googleCredentials(dict): google credentials

        Returns:
            :class:`.GCPConnector`: Newly create connector object
        """
        data = {
            'name': name,
            'googleCredentials': googleCredentials,
        }
        connector_info = cls._create_connector(project_id, data, content_type='application/json')
        return cls.from_dict(connector_info)


connectors_names = {
    'SQL': SQLConnector,
    'FTP': FTPConnector,
    'SFTP': SFTPConnector,
    'S3': S3Connector,
    'GCP': GCPConnector,
}
