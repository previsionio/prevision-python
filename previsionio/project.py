# -*- coding: utf-8 -*-
from __future__ import print_function
from enum import Enum
from typing import Dict, Tuple, Union
from previsionio import metrics
from previsionio.usecase_config import ColumnConfig
import requests

from . import client
from .utils import parse_json, PrevisionException
from . import logger
from . import TrainingConfig
from .api_resource import ApiResource, UniqueResourceMixin
from .datasource import DataSource
from .dataset import Dataset, DatasetImages
from .connector import Connector, SQLConnector, FTPConnector, \
    SFTPConnector, S3Connector, HiveConnector, GCPConnector
from .supervised import Regression, Classification, MultiClassification, \
    RegressionImages, ClassificationImages, MultiClassificationImages
from .timeseries import TimeSeries, TimeWindow
from .text_similarity import DescriptionsColumnConfig, ListModelsParameters, TextSimilarity
from .usecase import Usecase
from pandas import DataFrame


class ProjectColor(Enum):
    greenLighter = 'greenLighter'
    green = 'green'
    greenDarker = 'greenDarker'
    greenDarkest = 'greenDarkest'
    purpleLighter = 'purpleLighter'
    purple = 'purple'
    purpleDarker = 'purpleDarker'
    purpleDarkest = 'purpleDarkest'
    blueLighter = 'blueLighter'
    blue = 'blue'
    blueDarker = 'blueDarker'
    blueDarkest = 'blueDarkest'
    yellowLighter = 'yellowLighter'
    yellow = 'yellow'
    yellowDarker = 'yellowDarker'
    yellowDarkest = 'yellowDarkest'


class Project(ApiResource, UniqueResourceMixin):

    """ A Project

    Args:
        _id (str): Unique id of the project
        name (str): Name of the project
        description(str, optional): Description of the project
        color (str, optional): Color of the project

    """

    resource = 'projects'

    def __init__(self, _id: str, name: str, description: str = None, color: str = None, created_by: str = None,
                 admins=[], contributors=[], viewers=[], pipelines_count: int = 0, usecases_count: int = 0,
                 dataset_count: int = 0, **kwargs):
        """ Instantiate a new :class:`.Project` object to manipulate a project resource
        on the platform. """
        super().__init__(_id=_id,
                         name=name,
                         description=description,
                         color=color)

        self._id = _id
        self.name = name
        self.description = description
        self.color = color
        self.created_by = created_by
        self.admins = admins
        self.contributors = contributors
        self.viewers = viewers
        self.pipelines_count = pipelines_count
        self.usecases_count = usecases_count
        self.dataset_count = dataset_count

    @classmethod
    def list(cls, all: bool = False):
        """ List all the available project in the current active [client] workspace.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.
        Returns:
            list(:class:`.Project`): Fetched project objects
        """
        # FIXME : get /resource return type should be consistent
        resources = super().list(all=all)
        return [cls(**source_data) for source_data in resources]

    @classmethod
    def from_id(cls, _id: str):
        """Get a project from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.Project`: The fetched datasource

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

    @property
    def id(self):
        return self._id

    def users(self):
        """Get a project from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            :class:`.Project`: The fetched datasource

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """

        end_point = '/{}/{}/users'.format(self.resource, self._id)
        response = client.request(endpoint=end_point, method=requests.get)
        if response.status_code != 200:
            logger.error('cannot get users for project id {}'.format(self._id))
            raise PrevisionException('[{}] {}'.format(self.resource, response.status_code))

        res = parse_json(response)
        return res

    def info(self) -> Dict:
        """Get a datasource from the instance by its unique id.

        Args:
            _id (str): Unique id of the resource to retrieve

        Returns:
            Dict: Information about the Project with these entries:
                "_id"
                "name"
                "description"
                "color"
                "created_by"
                "admins"
                "contributors"
                "viewers"
                "pipelines_count"
                "usecases_count"
                "dataset_count"
                "users"

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing the result
        """
        project_info = {"_id": self._id,
                        "name": self.name,
                        "description": self.description,
                        "color": self.color,
                        "created_by": self.created_by,
                        "admins": self.admins,
                        "contributors": self.contributors,
                        "viewers": self.viewers,
                        "pipelines_count": self.pipelines_count,
                        "usecases_count": self.usecases_count,
                        "dataset_count": self.dataset_count,
                        "users": self.users}
        return project_info

    @classmethod
    def new(cls, name: str, description: str = None, color: ProjectColor = None) -> 'Project':
        """ Create a new datasource object on the platform.

        Args:
            name (str): Name of the project
            description(str, optional): Description of the project
            color (str, optional): Color of the project

        Returns:
            :class:`.Project`: The registered project object in the current workspace

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """

        data = {
            'name': name,
            'description': description,
            'color': color
        }

        url = '/{}'.format(cls.resource)
        resp = client.request(url,
                              data=data,
                              method=requests.post)

        if resp.status_code != 200:
            message = "Error {} reaching url: {} with data: {}".format(resp.status_code, cls.resource, data)
            raise PrevisionException(message)
        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))

        return cls(json['_id'], name, description, color, json['created_by'],
                   json['admins'], json['contributors'], json['pipelines_count'])

    def delete(self):
        """Delete a project from the actual [client] workspace.

        Raises:
            PrevisionException: If the dataset does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        resp = client.request(endpoint='/{}/{}'
                              .format(self.resource, self.id),
                              method=requests.delete)
        return resp

    def create_dataset(self, name: str, datasource: DataSource = None, file_name: str = None, dataframe: DataFrame = None):
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
        return Dataset._new(self._id, name, datasource=datasource, file_name=file_name, dataframe=dataframe)

    def list_datasets(self, all: bool = True):
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
        return Dataset.list(self._id, all=all)

    def create_image_folder(self, name: str, file_name: str):
        return DatasetImages._new(self._id, name, file_name)

    def list_image_folders(self, all: bool = True):
        return DatasetImages.list(self._id, all=all)

    def create_sql_connector(self, name: str, host: str, port: int = 3306, username: str = '', password: str = ''):
        return SQLConnector._new(self._id, name, host, port, 'SQL', username=username, password=password)

    def create_ftp_connector(self, name: str, host: str, port: int = 21, username: str = '', password: str = ''):
        return FTPConnector._new(self._id, name, host, port, 'FTP', username=username, password=password)

    def create_sftp_connector(self, name, host, port=23, username='', password=''):
        return SFTPConnector._new(self._id, name, host, port, 'SFTP', username=username, password=password)

    def create_s3_connector(self, name, host='', port='', username='', password=''):
        return S3Connector._new(self._id, name, host, port, 'S3', username=username, password=password)

    def create_hive_connector(self, name, host, port=10000, username='', password=''):
        return HiveConnector._new(self._id, name, host, port, 'HIVE', username=username, password=password)

    def create_gcp_connector(self, name, host='', port='', username='', password='', googleCredentials=''):
        return GCPConnector._new(self._id, name, host, port, 'GCP', username=username, password=password,
                                 googleCredentials=googleCredentials)

    def list_connectors(self, all=all):
        return Connector.list(self._id, all=all)

    def create_datasource(self, connector, name, path=None, database=None,
                          table=None, bucket=None, request=None, gCloud=None):
        return DataSource._new(self._id, connector, name, path=path, database=database,
                               table=table, bucket=bucket, request=request, gCloud=gCloud)

    def list_datasource(self, all=all):
        return DataSource.list(self._id, all=all)

    def fit_regression(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.Regression = metrics.Regression.RMSE, holdout_dataset=None,
                       training_config=TrainingConfig(), **kwargs):
        return Regression.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                              training_config=training_config, **kwargs)

    def fit_classification(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.Classification = None, holdout_dataset=None,
                           training_config=TrainingConfig(), **kwargs):
        return Classification.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                  training_config=training_config, **kwargs)

    def fit_multiclassification(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.MultiClassification = None, holdout_dataset=None,
                                training_config=TrainingConfig(), **kwargs):
        return MultiClassification.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                       training_config=training_config, **kwargs)

    def fit_image_regression(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.Regression = None, holdout_dataset=None,
                             training_config=TrainingConfig(), **kwargs):
        return RegressionImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                    training_config=training_config, **kwargs)

    def fit_image_classification(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.Classification = None, holdout_dataset=None,
                                 training_config=TrainingConfig(), **kwargs):
        return ClassificationImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                        training_config=training_config, **kwargs)

    def fit_image_multiclassification(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.MultiClassification = None, holdout_dataset=None,
                                      training_config=TrainingConfig(), **kwargs):
        return MultiClassificationImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                             training_config=training_config, **kwargs)

    def fit_timeseries_regression(self, name: str, dataset: Dataset, column_config: ColumnConfig, time_window: TimeWindow, metric: metrics.Regression = None, holdout_dataset=None,
                                  training_config=TrainingConfig()):
        return TimeSeries.fit(self._id, name, dataset, column_config, time_window, metric=metric, holdout_dataset=holdout_dataset,
                              training_config=training_config)

    def fit_text_similarity(self, name: str, dataset: Dataset, description_column_config: DescriptionsColumnConfig, metric: metrics.TextSimilarity = None, top_k=None, lang: str = 'auto',
                            queries_dataset=None, queries_column_config=None, models_parameters=ListModelsParameters()):
        return TextSimilarity.fit(self._id, name, dataset, description_column_config, metric=metric, top_k=top_k, lang=lang,
                                  queries_dataset=queries_dataset, queries_column_config=queries_column_config, models_parameters=models_parameters)

    def list_usecases(self, all: bool = True):
        return Usecase.list(self._id, all=all)


connectors_names = {
    'SQL': "create_sql_connector",
    'FTP': "create_ftp_connector",
    'SFTP': "create_sftp_connector",
    'S3': "create_s3_connector",
    'HIVE': "create_hive_connector",
    'GCP': "create_gcp_connector"
}
