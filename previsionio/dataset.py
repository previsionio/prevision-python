# -*- coding: utf-8 -*-
from __future__ import print_function
from itertools import combinations
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
from pandas._typing import FrameOrSeriesUnion
import os
import tempfile
from zipfile import ZipFile
import previsionio.utils
from . import client
from .utils import parse_json, zip_to_pandas, PrevisionException
from .api_resource import ApiResource
from . import logger
import previsionio as pio
import requests
from .datasource import DataSource


class Dataset(ApiResource):

    """ Dataset objects represent data resources that will be explored by Prevision.io platform.

        In order to launch an auto ml process (see :class:`.BaseUsecase` class), we need to have
        the matching dataset stored in the related workspace.

        Within the platform they are stored in tabular form and are derived:

        - from files (CSV, ZIP)
        - or from a Data Source at a given time (snapshot) """

    resource = 'datasets'

    def __init__(self, _id: str, name: str, datasource: DataSource = None, _data=None, describe_state=None,
                 drift_state=None, embeddings_state=None, **kwargs):
        super().__init__(_id=_id, datasource=datasource)
        self.name = name
        self._id = _id

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
            response = client.request(endpoint='/{}/{}/download'
                                      .format(self.resource, self.id),
                                      method=requests.get)

            if response.ok:
                pd_df = zip_to_pandas(response)
                self._data = pd_df
            else:
                msg = 'Failed to download dataset {}: {}'.format(self.id, response.text)
                raise PrevisionException(msg)
            return self._data

    data = property(to_pandas)

    @classmethod
    def list(cls, project_id, all=all):
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
        resources = super().list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    @classmethod
    def getid_from_name(cls, project_id, name, version='last'):
        """ Return the dataset id corresponding to a given name.

        Args:
            name (str): Name of the dataset
            version (int, str, optional): Specific version of the
                dataset (can be an int, or 'last' - default - to
                get the latest version of the dataset)

        Raises:
            PrevisionException: If dataset does not exist, version
                number is out of range or there is another error
                fetching or parsing data
        """
        # use prevision api to search by name
        # FIXME useless version

        if version != 'last':
            if not isinstance(version, type(7)):
                raise TypeError("version argument takes as values positive int or 'last' ")

        datasets = cls.list(all=True, project_id=project_id)
        datasets = list(filter(lambda d: d.name == name, datasets))
        if len(datasets) > 0:
            # get the corresponding id
            if version == 'last':
                version = len(datasets)
            if (len(datasets) < version) or (version <= 0):
                raise PrevisionException('list index out of range')
            return datasets[(version - 1)]._id

        msg_error = 'DatasetNotFoundError: No such dataset name : {}'.format(name)
        raise PrevisionException(msg_error)

    def update_status(self):
        url = '/{}/{}'.format(self.resource, self._id)
        dset_resp = client.request(url, method=requests.get)
        dset_json = parse_json(dset_resp)
        self.describe_state = dset_json['describe_state']
        self.drift_state = dset_json['drift_state']
        self.embeddings_state = dset_json['embeddings_state']

    def get_describe_status(self):
        return self.describe_state

    def get_drift_status(self):
        return self.drift_state

    def get_embedding_status(self):
        return self.embeddings_state

    def delete(self):
        """Delete a dataset from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'
                              .format(self.resource, self.id),
                              method=requests.delete)
        return resp

    def start_embedding(self):
        """Starts the embeddings analysis of the dataset from the actual [client] workspace

        Raises:
            PrevisionException: DatasetNotFoundError
            requests.exceptions.ConnectionError: request error
        """

        resp = client.request(endpoint='/{}/{}/analysis'
                              .format(self.resource, self.id),
                              method=requests.post)

        logger.info('Started embedding for dataset {}'.format(self.name))
        resp = parse_json(resp)
        return resp

    def get_embedding(self):
        """Gets the embeddings analysis of the dataset from the actual [client] workspace

        Raises:
            PrevisionException: DatasetNotFoundError
            requests.exceptions.ConnectionError: request error
        """
        resp = client.request(endpoint='/{}/{}/explorer'
                              .format(self.resource, self.id),
                              method=requests.get)
        if resp.status_code == 404 or resp.status_code == 501 or resp.status_code == 502:
            logger.warning('Dataset {} has not been analyzed yet. '
                           'Cannot retrieve embedding.'.format(self.name))
        elif resp.status_code == 400:
            raise PrevisionException(parse_json(resp)['message'])
        else:
            tensors_shape = parse_json(resp)['embeddings'][0]['tensorShape']
            labels_resp = client.request(endpoint='/{}/{}/explorer/labels.bytes'
                                         .format(self.resource, self.id),
                                         method=requests.get)
            tensors_resp = client.request(endpoint='/{}/{}/explorer/tensors.bytes'
                                          .format(self.resource, self.id),
                                          method=requests.get)

            labels = pd.read_csv(StringIO(labels_resp.text), sep="\t")
            tensors = np.frombuffer(BytesIO(tensors_resp.content).read(),
                                    dtype="float32").reshape(*tensors_shape)
            return {'labels': labels, 'tensors': tensors}

    def download(self, download_path=None):
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
        resp = client.request(endpoint='/{}/{}/download'.format(self.resource, self.id),
                              method=requests.get)
        if resp.status_code == 200 and resp._content is not None:
            if not download_path:
                download_path = os.getcwd()
            path = os.path.join(download_path, self.name + ".zip")
            with open(path, "wb") as file:
                file.write(resp._content)
            return path
        else:
            raise PrevisionException('could not download dataset')

    @classmethod
    def get_by_name(cls, project_id, name, version='last'):
        """Get an already registered dataset from the platform (using its registration
        name).

        Args:
            name (str): Name of the dataset to retrieve
            version (int, str, optional): Specific version of the
                dataset (can be an int, or 'last' - default - to
                get the latest version of the dataset)

        Raises:
            AttributeError: if dataset_name is not given
            PrevisionException: If dataset does not exist or if there
                was another error fetching or parsing data

        Returns:
            :class:`.Dataset`: Fetched dataset
        """
        dataset_id = cls.getid_from_name(project_id, name, version=version)
        return cls.from_id(dataset_id)

    @classmethod
    def new(cls, project_id: str, name: str, datasource: DataSource = None, file_name: str = None, dataframe: FrameOrSeriesUnion = None):
        """ Register a new dataset in the workspace for further processing.
        You need to provide either a datasource, a file name or a dataframe
        (only one can be specified).

        .. note::

            To start a new use case on a dataset, it has to be already
            registred in your workspace.

        Args:
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
            'name': name
        }

        files = {

        }
        request_url = '/projects/{}/{}/file'.format(project_id, cls.resource)
        create_resp = None
        if datasource is not None:
            request_url = '/projects/{}/{}/data-sources'.format(project_id, cls.resource)
            data['datasource_id'] = datasource.id
            create_resp = client.request(request_url,
                                         data=data,
                                         method=requests.post)

        elif dataframe is not None:
            file_ext = '.zip'

            with tempfile.NamedTemporaryFile(prefix=name, suffix='.csv') as temp:
                dataframe.to_csv(temp.name, index=False)

                file_name = temp.name.replace('.csv', file_ext)
                with ZipFile(file_name, 'w') as zip_file:
                    zip_file.write(temp.name, arcname=name + '.csv')

                with open(temp.name, 'rb') as f:
                    files['file'] = (os.path.basename(file_name), f)

                    for k, v in data.items():
                        files[k] = (None, v)
                    create_resp = client.request(request_url,
                                                data=data,
                                                files=files,
                                                method=requests.post)

        elif file_name is not None:
            with open(file_name, 'r') as f:
                files['file'] = (os.path.basename(file_name), f)

                for k, v in data.items():
                    files[k] = (None, v)

                create_resp = client.request(request_url,
                                             data=data,
                                             files=files,
                                             method=requests.post)

        if create_resp is None:
            raise PrevisionException('[Dataset] Uexpected case in dataset creation')
            
        create_json = parse_json(create_resp)
        if create_resp.status_code == 200:
            url = '/{}/{}'.format(cls.resource, create_json['_id'])
            event_tuple = previsionio.utils.EventTuple('DATASET_UPDATE', 'describe_state', 'done',
                                                       [('ready', 'failed'), ('drift', 'failed')])
            pio.client.event_manager.wait_for_event(create_json['_id'],
                                                    cls.resource,
                                                    event_tuple,
                                                    specific_url=url)

            dset_resp = client.request(url, method=requests.get)
            dset_json = parse_json(dset_resp)
            return cls(**dset_json)

        else:
            raise PrevisionException('[Dataset] Error: {}'.format(create_json))


class DatasetImages(ApiResource):

    """ DatasetImages objects represent image data resources that will be used by
        Prevision.io's platform.

        In order to launch an auto ml process (see :class:`.BaseUsecase` class), we need to have
        the matching dataset stored in the related workspace.

        Within the platform, image folder datasets are stored as ZIP files and are copied from
        ZIP files. """

    resource = 'image-folders'

    def __init__(self, _id, name, project_id, copy_state, **kwargs):
        super().__init__(_id=_id)
        self.name = name
        self._id = _id
        self.project_id = project_id
        self.copy_state = copy_state

        self.other_params = kwargs

    def to_pandas(self) -> pd.DataFrame:
        """ Invalid method for a :class:`.DatasetImages` object.

        Raises:
            ValueError: Folder datasets cannot be converted to a ``pandas`` dataframe
        """
        raise ValueError("Cannot convert a folder dataset to pandas.")

    def delete(self):
        """Delete a DatasetImages from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset images does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'
                              .format(self.resource, self.id),
                              method=requests.delete)
        return resp

    @classmethod
    def list(cls, project_id, all=all):
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
        resources = super().list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    @classmethod
    def new(cls, project_id, name, file_name):
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
                                     method=requests.post)
        source.close()

        create_json = parse_json(create_resp)

        if create_resp.status_code == 200:
            url = '/{}/{}'.format(cls.resource, create_json['_id'])
            pio.client.event_manager.wait_for_event(create_json['_id'],
                                                    cls.resource,
                                                    previsionio.utils.EventTuple('FOLDER_UPDATE', 'copy_state', 'done'),
                                                    specific_url=url)

            dset_resp = client.request(url, method=requests.get)
            dset_json = parse_json(dset_resp)
            return cls(**dset_json)

        else:
            raise PrevisionException('[Dataset] Error: {}'.format(create_json))

    def download(self, download_path=None):
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
        resp = client.request(endpoint='/{}/{}/download'.format(self.resource, self.id),
                              method=requests.get)
        if resp.status_code == 200 and resp._content is not None:
            if not download_path:
                download_path = os.getcwd()
            path = os.path.join(download_path, self.name + ".zip")
            with open(path, "wb") as file:
                file.write(resp._content)
            return path
        else:
            raise PrevisionException('could not download dataset')
