import os
import requests
from enum import Enum

from . import client
from .utils import parse_json, PrevisionException
from .api_resource import ApiResource, UniqueResourceMixin
from .connector import Connector, GCloud
from .export import Export
from .dataset import Dataset
from .prediction import DeploymentPrediction


class ExporterWriteMode(Enum):
    """ Write mode for exporters.
    """
    safe = 'safe'
    """Fail if file already exists."""
    replace = 'replace'
    """Replace existing file/table."""
    append = 'append'
    """Append to existing table."""
    timestamp = 'timestamp'
    """Append timestamp to the output filename."""


class Exporter(ApiResource, UniqueResourceMixin):

    """ An exporter to access a distant data pool and upload data easily. This
    resource is linked to a :class:`.Connector` resource that represents the connection to
    the distant data source.

    Args:
        _id (str): Unique id of the exporter
        connector_id (str): Reference to the associated connector (the resource
            to go through to get a data snapshot)
        name (str): Name of the exporter
        description (str, optional): Description of the exporter
        path (str, optional): Path to the file to write on via the exporter
        bucket (str, optional): Bucket of the file to write on via the exporter
        database (str, optional): Name of the database to write on via the exporter
        table (str, optional): Name of the table to write on via the exporter
        g_cloud (:class:`.GCloud`, optional): Type of google cloud service
        write_mode (:class:`ExporterWriteMode`, optional): Write mode
    """

    resource = 'exporters'

    def __init__(self, _id, connector_id: str, name: str, description: str = None, path: str = None,
                 bucket: str = None, database: str = None, table: str = None, g_cloud: GCloud = None,
                 write_mode: ExporterWriteMode = ExporterWriteMode.safe, **kwargs):
        """ Instantiate a new :class:`.Exporter` object to manipulate an exporter resource
        on the platform. """
        super().__init__(_id=_id)

        self._id = _id
        self.connector_id = connector_id

        self.name = name
        self.description = description
        self.path = path
        self.bucket = bucket
        self.database = database
        self.table = table
        self.g_cloud = g_cloud
        self.write_mode = write_mode

        self.other_params = kwargs



    @classmethod
    def list(cls, project_id: str, all: bool = False):
        """ List all the available exporters in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Exporter` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Exporter`): fetched exporter objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**source_data) for source_data in resources]

    @classmethod
    def from_id(cls, _id: str):
        """Get an exporter from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.Exporter`: the fetched exporter

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        url = '/{}/{}'.format(cls.resource, _id)
        resp = client.request(url, method=requests.get, message_prefix='From id exporter')
        resp_json = parse_json(resp)

        return cls(**resp_json)

    @classmethod
    def _new(cls, project_id: str, connector: Connector, name: str, description: str = None, path: str = None,
             bucket: str = None, database: str = None, table: str = None, g_cloud: GCloud = None,
             write_mode: ExporterWriteMode = ExporterWriteMode.safe):
        """ Create a new exporter object on the platform.

        Args:
            project_id (str): Unique project id on which to create the exporter
            connector (:class:`.Connector`): Reference to the associated connector (the resource
                to go through to get a data snapshot)
            name (str): Name of the exporter
            description (str, optional): Description of the exporter
            path (str, optional): Path to the file to write on via the exporter
            bucket (str, optional): Bucket of the file to write on via the exporter
            database (str, optional): Name of the database to write on via the exporter
            table (str, optional): Name of the table to write on via the exporter
            g_cloud (:class:`.GCloud`, optional): Type of google cloud service
            write_mode (:class:`ExporterWriteMode`, optional): Write mode

        Returns:
            :class:`.Exporter`: The registered exporter object in the current workspace

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        data = {
            'connector_id': connector._id,
            'name': name,
            'description': description,
            'filepath': path,
            'bucket': bucket,
            'database': database,
            'g_cloud': g_cloud,
            'table': table,
        }

        if database is not None:
            data['database_write_mode'] = write_mode.value
        else:
            data['file_write_mode'] = write_mode.value

        url = '/projects/{}/{}'.format(project_id, cls.resource)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='Exporter creation')
        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))

        return cls(json['_id'], connector._id, name, description=description, path=path,
                   bucket=bucket, database=database, table=table, write_mode=write_mode,
                   g_cloud=g_cloud)

    def delete(self):
        """Delete an exporter from the actual [client] workspace.

        Raises:
            PrevisionException: If the exporter does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    def apply_file(self, file_path: str, name: str, origin: str = None,
                   pipeline_scheduled_run_id: str = None, encoding: str = None,
                   separator: str = None, decimal: str = None, thousands: str = None):
        return Export.apply_file(self._id, file_path=file_path, name=name, origin=origin,
                                 pipeline_scheduled_run_id=pipeline_scheduled_run_id,
                                 encoding=encoding, separator=separator, decimal=decimal,
                                 thousands=thousands)

    def apply_dataset(self, dataset: Dataset):
        return Export.apply_dataset(exporter_id=self._id, dataset=dataset)

    def apply_prediction(self, prediction: DeploymentPrediction):
        return Export.apply_prediction(self._id, prediction=prediction)

    """
    def apply_file(self, file_path, name):
        data = {
            'name': name,
        }
        files = {}
        with open(file_path, 'r') as f:
            files['file'] = (os.path.basename(file_path), f, 'text/csv')
        request_url = '/exporters/{}/file'.format(self._id)
        create_resp = client.request(request_url,
                                     data=data,
                                     files=files,
                                     method=requests.post,
                                     message_prefix='Export file')
        create_json = parse_json(create_resp)
        print("create_json===", create_json)
    
    def apply_dataset(self, dataset_id):

        request_url = '/exporters/{}/dataset/{}'.format(self._id, dataset_id)
        create_resp = client.request(request_url,
                                     method=requests.post,
                                     message_prefix='Export dataset')
        create_json = parse_json(create_resp)
        print("create_json===", create_json)
    """