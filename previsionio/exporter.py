import requests
from enum import Enum
from typing import Union

from . import client
from .utils import parse_json, PrevisionException
from .api_resource import ApiResource, UniqueResourceMixin
from .connector import Connector, GCloud
from .export import Export
from .dataset import Dataset
from .prediction import DeploymentPrediction, ValidationPrediction


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

        Args:
            project_id (str): Unique reference of the project id on the platform
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Exporter`): The fetched exporter objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**source_data) for source_data in resources]

    @classmethod
    def from_id(cls, _id: str):
        """Get an exporter from the instance by its unique id.

        Args:
            _id (str): Unique id of the exporter to retrieve

        Returns:
            :class:`.Exporter`: The fetched exporter

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        resp_json = super()._from_id(_id=_id)
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

        write_mode = write_mode.value
        if database is not None:
            if write_mode not in ['replace', 'append']:
                raise PrevisionException(f'Write mode "{write_mode}" is not compatible with database connector')
            data['database_write_mode'] = write_mode
        else:
            if write_mode not in ['timestamp', 'safe', 'replace']:
                raise PrevisionException(f'Write mode "{write_mode}" is not compatible with file connectors')
            data['file_write_mode'] = write_mode

        url = '/projects/{}/{}'.format(project_id, cls.resource)
        resp = client.request(url,
                              data=data,
                              method=requests.post,
                              message_prefix='Exporter creation')
        json = parse_json(resp)

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

    def export_file(self, file_path: str, encoding: str = None, separator: str = None, decimal: str = None,
                    thousands: str = None, wait_for_export: bool = False, **kwargs):
        """ Upload a CSV file using the exporter.

        Args:
            file_path (str): Path of the file to upload
            encoding (str, optional): Encoding of the file to upload
            separator (str, optional): Separator of the file to upload
            decimal (str, optional): Decimal of the file to upload
            thousands (str, optional): Thousands of the file to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return Export.export_file(self._id, file_path=file_path, encoding=encoding, separator=separator,
                                  decimal=decimal, thousands=thousands, wait_for_export=wait_for_export,
                                  **kwargs)

    def export_dataset(self, dataset: Dataset, wait_for_export: bool = False):
        """ Upload a :class:`.Dataset` from the current active project using the exporter.

        Args:
            dataset (:class:`.Dataset`): dataset to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return Export.export_dataset(exporter_id=self._id, dataset=dataset, wait_for_export=wait_for_export)

    def export_prediction(self, prediction: Union[DeploymentPrediction, ValidationPrediction],
                          wait_for_export: bool = False):
        """ Upload a :class:`.DeploymentPrediction` or a :class:`.ValidationPrediction`
        from the current active project using the exporter.

        Args:
            dataset (:class:`.DeploymentPrediction`|:class:`.ValidationPrediction`): prediction to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return Export.export_prediction(self._id, prediction=prediction, wait_for_export=wait_for_export)

    def list_exports(self):
        """ List all the available exports given the exporter id.

        Returns:
            list(:class:`.Export`): The fetched export objects
        """
        return Export.list(self._id)
