import os
import requests
from typing import Union

from . import client
from .utils import parse_json, get_all_results, PrevisionException, EventTuple
from .api_resource import ApiResource, UniqueResourceMixin
from .dataset import Dataset
from .prediction import DeploymentPrediction, ValidationPrediction


class Export(ApiResource, UniqueResourceMixin):

    """ An export

    Args:
        _id (str): Unique id of the export
        exporter_id (str): Unique exporter id on which to create the export
        dataset_id (str, optional): Unique dataset id on which to create the export
        prediction_id (str, optional): Unique prediction id on which to create the export
    """

    resource = 'exports'

    def __init__(self, _id, exporter_id: str, dataset_id: str = None, prediction_id: str = None,
                 status: str = None, **kwargs):
        """ Instantiate a new :class:`.Export` object to manipulate an export resource
        on the platform. """
        super().__init__(_id=_id)

        self._id = _id
        self.exporter_id = exporter_id
        self.dataset_id = dataset_id
        self.state = status
        self.other_params = kwargs

    @classmethod
    def list(cls, exporter_id: str):
        """ List all the available exports given an exporter id.

        Args:
            exporter_id: exporter id

        Returns:
            list(:class:`.Export`): The fetched export objects
        """
        end_point = '/exporters/{}/exports'.format(exporter_id)
        exports = get_all_results(client, end_point, method=requests.get)
        return [cls(**data) for data in exports]

    @classmethod
    def from_id(cls, _id: str):
        """Get an export from the instance by its unique id.

        Args:
            _id (str): Unique id of the export to retrieve

        Returns:
            :class:`.Export`: The fetched export

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        resp_json = super()._from_id(_id=_id)
        return cls(**resp_json)

    @classmethod
    def _new(cls, exporter_id: str, prediction: Union[DeploymentPrediction, ValidationPrediction] = None,
             dataset: Dataset = None, file_path: str = None, encoding: str = None, separator: str = None,
             decimal: str = None, thousands: str = None, wait_for_export: bool = False, **kwargs):
        """ Create a new exporter object on the platform.

        Args:
            exporter_id (str): Unique exporter id on which to create the export
            prediction (:class:`.DeploymentPrediction`|:class:`.ValidationPrediction`, optional): prediction to upload
            dataset (:class:`.Dataset`, optional): Dataset to upload
            file_path (str, optional): Path of the file to upload
            encoding (str, optional): Encoding of the file to upload
            separator (str, optional): Separator of the file to upload
            decimal (str, optional): Decimal of the file to upload
            thousands (str, optional): Thousands of the file to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """
        data = {}

        if dataset is not None:
            request_url = '/exporters/{}/dataset/{}'.format(exporter_id, dataset._id)
            create_resp = client.request(request_url,
                                         method=requests.post,
                                         message_prefix='Export dataset')

        elif prediction is not None:
            request_url = '/exporters/{}/prediction/{}'.format(exporter_id, prediction._id)
            if isinstance(prediction, DeploymentPrediction):
                data['prediction_type'] = 'deployment'
            elif isinstance(prediction, ValidationPrediction):
                data['prediction_type'] = 'usecase'
            else:
                msg = 'prediction must be of type DeploymentPrediction or ValidationPrediction,'
                msg += f' got: {type(prediction)}'
                raise PrevisionException(msg)
            create_resp = client.request(request_url,
                                         data=data,
                                         method=requests.post,
                                         message_prefix='Export prediction')

        elif file_path is not None:
            if 'origin' in kwargs:
                data['origin'] = kwargs['origin']
            if 'pipeline_scheduled_run_id' in kwargs:
                data['pipeline_scheduled_run_id'] = kwargs['pipeline_scheduled_run_id']
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
                for k, v in data.items():
                    files[k] = (None, v)
                files['file'] = (os.path.basename(file_path), f, 'text/csv')
                request_url = '/exporters/{}/file'.format(exporter_id)
                create_resp = client.request(request_url,
                                             data=data,
                                             files=files,
                                             method=requests.post,
                                             message_prefix='Export file')

        else:
            raise PrevisionException('Need to specify one of: dataset, prediction or file_path')

        create_resp = parse_json(create_resp)

        if wait_for_export:
            try:
                specific_url = '/{}/{}'.format('exports', create_resp['_id'])
                client.event_manager.wait_for_event(create_resp['_id'],
                                                    cls.resource,
                                                    EventTuple(
                                                        'EXPORT_UPDATE',
                                                        ('status', 'done'),
                                                        [('status', 'failed')]),
                                                    specific_url=specific_url)

            except PrevisionException:
                # Extract log if failure
                resp_json = super()._from_id(_id=create_resp['_id'])
                msg = f'Export failure.\nLogs: {resp_json["logs"]}\nurl: {request_url}'
                if data:
                    msg += f'\ndata: {data}'
                raise PrevisionException(msg)

            except Exception:
                raise

        return cls.from_id(create_resp['_id'])

    @classmethod
    def export_file(cls, exporter_id: str, file_path: str, encoding: str = None, separator: str = None,
                    decimal: str = None, thousands: str = None, wait_for_export: bool = False, **kwargs):
        """ Upload a CSV file using an exporter.

        Args:
            exporter_id (str): Unique exporter id on which to create the export
            file_path (str): Path of the file to upload
            encoding (str, optional): Encoding of the file to upload
            separator (str, optional): Separator of the file to upload
            decimal (str, optional): Decimal of the file to upload
            thousands (str, optional): Thousands of the file to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return cls._new(exporter_id, file_path=file_path, encoding=encoding, separator=separator,
                        decimal=decimal, thousands=thousands, wait_for_export=wait_for_export, **kwargs)

    @classmethod
    def export_dataset(cls, exporter_id, dataset: Dataset, wait_for_export: bool = False):
        """ Upload a :class:`.Dataset` from the current active project using an exporter.

        Args:
            exporter_id (str): Unique exporter id on which to create the export
            dataset (:class:`.Dataset`): Dataset to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return cls._new(exporter_id, dataset=dataset, wait_for_export=wait_for_export)

    @classmethod
    def export_prediction(cls, exporter_id, prediction: Union[DeploymentPrediction, ValidationPrediction],
                          wait_for_export: bool = False):
        """ Upload a :class:`.DeploymentPrediction` or a :class:`.ValidationPrediction`
        from the current active project using an exporter.

        Args:
            exporter_id (str): Unique exporter id on which to create the export
            prediction (:class:`.DeploymentPrediction`|:class:`.ValidationPrediction`): Prediction to upload
            wait_for_export (bool, optional): Wether to wait until the export is complete or not

        Returns:
            :class:`.Export`: The registered export object
        """
        return cls._new(exporter_id, prediction=prediction, wait_for_export=wait_for_export)
