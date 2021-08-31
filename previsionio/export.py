import os
import requests

from . import client
from .utils import parse_json, PrevisionException, get_all_results, EventTuple
from .api_resource import ApiResource, UniqueResourceMixin
from .dataset import Dataset
from .prediction import DeploymentPrediction
from .connector import Connector, GCloud


class Export(ApiResource, UniqueResourceMixin):

    """ An export

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

    resource = 'exports'

    def __init__(self, _id, origin: str = None, status: str = None, created_by: dict = {},
                 created_at: str = None, exporter_id: str = None, dataset_id: str = None, **kwargs):
        """ Instantiate a new :class:`.Exporter` object to manipulate an exporter resource
        on the platform. """
        super().__init__(_id=_id)

        self._id = _id
        self.origin = origin
        self.state = status
        self.created_by = created_by
        self.created_at = created_at
        self.exporter_id = exporter_id
        self.dataset_id = dataset_id
        self.other_params = kwargs

    @classmethod
    def list(cls, exporter_id: str):
        """ List all the available exports given an exporter id.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Export` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            exporter_id: exporter id

        Returns:
            list(:class:`.Export`): fetched export objects
        """
        end_point = '/exporters/{}/exports'.format(exporter_id)
        exports = get_all_results(client, end_point, method=requests.get)
        return [cls(**data) for data in exports]

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
        resp = client.request(url, method=requests.get, message_prefix='From id export')
        resp_json = parse_json(resp)

        return cls(**resp_json)

    @classmethod
    def _new(cls, exporter_id: str, dataset: Dataset = None, prediction: DeploymentPrediction = None,
             file_path: str = None, name: str = None, origin: str = None, pipeline_scheduled_run_id: str = None,
             encoding: str = None, separator: str = None, decimal: str = None, thousands: str = None):
        """ Create a new exporter object on the platform.

        Args:
            exporter_id (str): Unique exporter id on which to create the export

        Returns:
            :class:`.Export`: The registered export object

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        if dataset is not None:
            request_url = '/exporters/{}/dataset/{}'.format(exporter_id, dataset._id)
            create_resp = client.request(request_url,
                                         method=requests.post,
                                         message_prefix='Export dataset')
        elif prediction is not None:
            request_url = '/exporters/{}/prediction/{}'.format(exporter_id, prediction._id)
            create_resp = client.request(request_url,
                                         method=requests.post,
                                         message_prefix='Export prediction')
        elif file_path is not None:
            data = {
                'name': name,
            }
            if origin is not None:
                data['origin'] = origin
            if pipeline_scheduled_run_id is not None:
                data['pipeline_scheduled_run_id'] = pipeline_scheduled_run_id
            if encoding is not None:
                data['encoding'] = encoding
            if separator is not None:
                data['separator'] = separator
            if decimal is not None:
                data['decimal'] = decimal
            if thousands is not None:
                data['thousands'] = thousands
            files = {}
            with open(file_path, 'r') as f:
                files['file'] = (os.path.basename(file_path), f, 'text/csv')
                request_url = '/exporters/{}/file'.format(exporter_id)
                create_resp = client.request(request_url,
                                             data=data,
                                             files=files,
                                             method=requests.post,
                                             message_prefix='Export file')
        else:
            print("raise error")
        create_resp = parse_json(create_resp)

        if '_id' not in create_resp:
            if 'message' in create_resp:
                raise PrevisionException(create_resp['message'])
            else:
                raise Exception('unknown error: {}'.format(create_resp))
        #specific_url = '/{}/{}'.format('exports', create_resp['_id'])
        #client.event_manager.wait_for_event(create_resp['_id'],
        #                                    specific_url,
        #                                    EventTuple(
        #                                        'EXPORT_UPDATE',
        #                                        ('status', 'done'),
        #                                        [('status', 'failed')]),
        #                                    specific_url=specific_url)
        return cls.from_id(create_resp['_id'])

    @classmethod
    def apply_file(cls, exporter_id, file_path: str, name: str, origin: str = None,
                   pipeline_scheduled_run_id: str = None, encoding: str = None,
                   separator: str = None, decimal: str = None, thousands: str = None):
        return cls._new(exporter_id, file_path=file_path, name=name, origin=origin,
                        pipeline_scheduled_run_id=pipeline_scheduled_run_id, encoding=encoding,
                        separator=separator, decimal=decimal, thousands=thousands)

    @classmethod
    def apply_dataset(cls, exporter_id, dataset: Dataset):
        return cls._new(exporter_id, dataset=dataset)

    @classmethod
    def apply_prediction(cls, exporter_id, prediction: DeploymentPrediction):
        return cls._new(exporter_id, prediction=prediction)


