# -*- coding: utf-8 -*-
from __future__ import print_function
from itertools import combinations
from io import BytesIO, StringIO
from previsionio.usecase_config import UsecaseState
import numpy as np
import pandas as pd
import os
import tempfile
import zipfile
import previsionio.utils
from . import client
from .utils import parse_json, zip_to_pandas, PrevisionException
from .api_resource import ApiResource
from . import logger
import previsionio as pio
import requests
from .datasource import DataSource

from typing import Dict, Union
from pandas import DataFrame, Series
FrameOrSeriesUnion = Union["DataFrame", "Series"]


class Dataset(ApiResource):

    """ Dataset objects represent data resources that will be explored by Prevision.io platform.

        In order to launch an auto ml process (see :class:`.BaseUsecase` class), we need to have
        the matching dataset stored in the related workspace.

        Within the platform they are stored in tabular form and are derived:

        - from files (CSV, ZIP)
        - or from a Data Source at a given time (snapshot) """

    resource = 'datasets'

    def __init__(self, _id: str, name: str, datasource: DataSource = None, _data: DataFrame = None,
                 describe_state: Dict = None, drift_state=None, embeddings_state=None, separator=',', **kwargs):

        super().__init__(_id=_id, datasource=datasource)
        self.name = name
        self._id = _id
        self.separator = separator
        self.datasource = datasource
        self.describe_state = describe_state
        self.drift_state = drift_state
        self.embeddings_state = embeddings_state
        self.other_params = kwargs
        self._data = _data

    def to_pandas(self) -> pd.DataFrame:
        """Load in memory the data content of the current dataset into a pandas DataFrame.

        Returns:
            ``pd.DataFrame``: Dataframe for the data object

        Raises:
            PrevisionException: Any error while fetching or parsing the data
        """
        if self._data is not None:
            return self._data
        else:
            response = client.request(endpoint='/{}/{}/download'.format(self.resource, self.id),
                                      method=requests.get,
                                      message_prefix='Dataset download')

            self._data = zip_to_pandas(response, separator=self.separator)

            return self._data

    data = property(to_pandas)

    @classmethod
    def from_id(cls, _id: str) -> 'Dataset':
        return cls(**super()._from_id(_id=_id))

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available datasets in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Dataset` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Dataset`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    def update_status(self):
        url = '/{}/{}'.format(self.resource, self._id)
        dset_resp = client.request(url, method=requests.get, message_prefix='Dataset status update')
        dset_json = parse_json(dset_resp)
        self.describe_state = dset_json['describe_state']
        self.drift_state = dset_json['drift_state']
        self.embeddings_state = dset_json['embeddings_state']

    def get_describe_status(self):
        return UsecaseState(self.describe_state)

    def get_drift_status(self):
        return UsecaseState(self.drift_state)

    def get_embedding_status(self):
        return UsecaseState(self.embeddings_state)

    def delete(self):
        """Delete a dataset from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'.format(self.resource, self.id),
                              method=requests.delete,
                              message_prefix='Dataset delete')
        return resp

    def start_embedding(self):
        """Starts the embeddings analysis of the dataset from the actual [client] workspace

        Raises:
            PrevisionException: DatasetNotFoundError
            requests.exceptions.ConnectionError: request error
        """

        resp = client.request(endpoint='/{}/{}/analysis'.format(self.resource, self.id),
                              method=requests.post,
                              message_prefix='Dataset start embedding')

        logger.info('Started embedding for dataset {}'.format(self.name))
        resp = parse_json(resp)
        return resp

    def get_embedding(self) -> Dict:
        """Gets the embeddings analysis of the dataset from the actual [client] workspace

        Raises:
            PrevisionException: DatasetNotFoundError
            requests.exceptions.ConnectionError: request error
        """
        resp = client.request(endpoint='/{}/{}/explorer'.format(self.resource, self.id),
                              method=requests.get,
                              message_prefix='Dataset get embedding')

        tensors_shape = parse_json(resp)['embeddings'][0]['tensorShape']
        labels_resp = client.request(endpoint='/{}/{}/explorer/labels.bytes'.format(self.resource, self.id),
                                     method=requests.get,
                                     message_prefix='Dataset labels bytes')
        tensors_resp = client.request(endpoint='/{}/{}/explorer/tensors.bytes'.format(self.resource, self.id),
                                      method=requests.get,
                                      message_prefix='Dataset tensors bytes')

        labels = pd.read_csv(StringIO(labels_resp.text), sep="\t")
        tensors = np.frombuffer(BytesIO(tensors_resp.content).read(),
                                dtype="float32").reshape(*tensors_shape)
        return {'labels': labels, 'tensors': tensors}

    def download(self, download_path: str = None):
        """Download the dataset from the platform locally.

        Args:
            download_path (str, optional): Target local directory path
                (if none is provided, the current working directory is
                used)

        Returns:
            str: Path the data was downloaded to

        Raises:
            PrevisionException: If dataset does not exist or if there
                was another error fetching or parsing data
        """
        endpoint = '/{}/{}/download'.format(self.resource, self.id)
        resp = client.request(endpoint=endpoint,
                              method=requests.get,
                              message_prefix='Dataset download')

        if resp._content is None:
            raise PrevisionException('could not download dataset')

        if not download_path:
            download_path = os.getcwd()
        path = os.path.join(download_path, self.name + ".zip")

        with open(path, "wb") as file:
            file.write(resp._content)

        return path

    @classmethod
    def _new(cls, project_id: str, name: str, datasource: DataSource = None,
             file_name: str = None, dataframe: DataFrame = None, **kwargs):
        """ Register a new dataset in the workspace for further processing.
        You need to provide either a datasource, a file name or a dataframe
        (only one can be specified).

        .. note::

            To start a new use case on a dataset, it has to be already
            registred in your workspace.

        Args:
            project_id (str): project id
            name (str): Registration name for the dataset
            datasource (:class:`.DataSource`, optional): A DataSource object used
                to import a remote dataset (if you want to import a specific dataset
                from an existent database, you need a datasource connector
                (:class:`.Connector` object) designed to point to the related data source)
            file_name (str, optional): Path to a file to upload as dataset
            dataframe (pd.DataFrame, optional): A ``pandas`` dataframe containing the
                data to upload

        Raises:
            Exception: If more than one of the keyword arguments ``datasource``, ``file_name``,
                ``dataframe`` was specified
            PrevisionException: Error while creating the dataset on the platform

        Returns:
            :class:`.Dataset`: The registered dataset object in the current workspace.
        """
        if any(a is not None and b is not None for a, b in combinations([datasource, file_name, dataframe], 2)):
            raise Exception('only one of [datasource, file_handle, data] must be specified')

        if not any([datasource is not None, file_name is not None, dataframe is not None]):
            raise Exception('at least one of [datasource, file_handle, data] must be specified')

        data = {
            'name': name,
        }

        if 'origin' in kwargs:
            valid_origin = ["pipeline_output", "pipeline_intermediate_file"]
            origin = kwargs.get('origin')
            if not isinstance(origin, str) or origin not in valid_origin:
                raise RuntimeError(f"invalid origin: {origin}")
            data['origin'] = origin

        files = {

        }
        request_url = '/projects/{}/{}/file'.format(project_id, cls.resource)
        create_resp = None
        if datasource is not None:
            request_url = '/projects/{}/{}/data-sources'.format(project_id, cls.resource)
            data['datasource_id'] = datasource.id
            create_resp = client.request(request_url,
                                         data=data,
                                         method=requests.post,
                                         message_prefix='Dataset upload from datasource')

        elif dataframe is not None:
            file_ext = '.zip'

            with tempfile.NamedTemporaryFile(prefix=name, suffix='.csv') as temp:
                dataframe.to_csv(temp.name, index=False)

                file_name = temp.name.replace('.csv', file_ext)
                with zipfile.ZipFile(file_name, 'w') as zip_file:
                    zip_file.write(temp.name, arcname=name + '.csv')
                assert zip_file.filename is not None

                with open(zip_file.filename, 'rb') as f:
                    files['file'] = (os.path.basename(file_name), f, 'application/zip')
                    for k, v in data.items():
                        files[k] = (None, v)

                    create_resp = client.request(request_url,
                                                 data=data,
                                                 files=files,
                                                 method=requests.post,
                                                 message_prefix='Dataset upload from dataframe')

        elif file_name is not None:

            if zipfile.is_zipfile(file_name):
                with open(file_name, 'rb') as f:
                    files['file'] = (os.path.basename(file_name), f, 'application/zip')

                    for k, v in data.items():
                        files[k] = (None, v)

                    create_resp = client.request(request_url,
                                                 data=data,
                                                 files=files,
                                                 method=requests.post,
                                                 message_prefix='Dataset upload from zip file')

            # If not a zip, assert it is a CSV
            else:
                with open(file_name, 'r') as f:
                    files['file'] = (os.path.basename(file_name), f, 'text/csv')

                    for k, v in data.items():
                        files[k] = (None, v)

                    create_resp = client.request(request_url,
                                                 data=data,
                                                 files=files,
                                                 method=requests.post,
                                                 message_prefix='Dataset upload from csv file')

        if create_resp is None:
            raise PrevisionException('[Dataset] Unexpected case in dataset creation')

        create_json = parse_json(create_resp)
        url = '/{}/{}'.format(cls.resource, create_json['_id'])
        event_tuple = previsionio.utils.EventTuple('DATASET_UPDATE', 'describe_state', 'done',
                                                   [('ready', 'failed'), ('drift', 'failed')])
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(create_json['_id'],
                                                cls.resource,
                                                event_tuple,
                                                specific_url=url)

        dset_resp = client.request(url, method=requests.get, message_prefix='Dataset loading')
        dset_json = parse_json(dset_resp)

        if dataframe is not None and file_name is not None:
            os.remove(file_name)

        return cls(**dset_json)


class DatasetImages(ApiResource):

    """ DatasetImages objects represent image data resources that will be used by
        Prevision.io's platform.

        In order to launch an auto ml process (see :class:`.BaseUsecase` class), we need to have
        the matching dataset stored in the related workspace.

        Within the platform, image folder datasets are stored as ZIP files and are copied from
        ZIP files. """

    resource = 'image-folders'

    def __init__(self, _id: str, name: str, project_id: str, copy_state, **kwargs):
        super().__init__(_id=_id)
        self.name = name
        self._id = _id
        self.project_id = project_id
        self.copy_state = copy_state

        self.other_params = kwargs

    @classmethod
    def from_id(cls, _id: str) -> 'DatasetImages':
        return cls(**super()._from_id(_id=_id))

    def delete(self):
        """Delete a DatasetImages from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset images does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'.format(self.resource, self.id),
                              method=requests.delete,
                              message_prefix='Image folder delete')
        return resp

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available dataset image in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.DatasetImages` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.DatasetImages`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    @classmethod
    def _new(cls, project_id: str, name: str, file_name: str):
        """ Register a new image dataset in the workspace for further processing
        (in the image folders group).

        .. note::

            To start a new use case on a dataset image, it has to be already
            registred in your workspace.

        Args:
            name (str): Registration name for the dataset
            file_name (str): Path to the zip file to upload as image dataset

        Raises:
            PrevisionException: Error while creating the dataset on the platform

        Returns:
            :class:`.DatasetImages`: The registered dataset object in the current workspace.
        """
        request_url = '/projects/{}/{}'.format(project_id, cls.resource)
        source = open(file_name, 'rb')

        files = {
            'name': (None, name),
            'file': (name, source)
        }

        create_resp = client.request(request_url,
                                     files=files,
                                     method=requests.post,
                                     message_prefix='Image folder upload')
        source.close()

        create_json = parse_json(create_resp)
        url = '/{}/{}'.format(cls.resource, create_json['_id'])
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(create_json['_id'],
                                                cls.resource,
                                                previsionio.utils.EventTuple(
                                                    'FOLDER_UPDATE', 'state', 'done', [('state', 'failed')]),
                                                specific_url=url)

        dset_resp = client.request(url, method=requests.get, message_prefix='Image folder loading')
        dset_json = parse_json(dset_resp)
        return cls(**dset_json)

    def download(self, download_path: str = None):
        """Download the dataset from the platform locally.

        Args:
            download_path (str, optional): Target local directory path
                (if none is provided, the current working directory is
                used)

        Returns:
            str: Path the data was downloaded to

        Raises:
            PrevisionException: If dataset does not exist or if there
                was another error fetching or parsing data
        """
        endpoint = '/{}/{}/download'.format(self.resource, self.id)
        resp = client.request(endpoint=endpoint,
                              method=requests.get,
                              message_prefix='Image folder download')
        if resp._content is None:
            raise PrevisionException('could not download dataset')
        if not download_path:
            download_path = os.getcwd()
        path = os.path.join(download_path, self.name + ".zip")
        with open(path, "wb") as file:
            file.write(resp._content)
        return path
