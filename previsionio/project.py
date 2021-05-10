# -*- coding: utf-8 -*-
from __future__ import print_function
from enum import Enum
from typing import Dict, Tuple, Union

from requests.models import Response
from previsionio import metrics
from previsionio.usecase_config import ColumnConfig
import requests

from . import client
from .utils import handle_error_response, parse_json, PrevisionException
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
    greenLighter = '#16D92C'
    green = '#29CC3C'
    greenDarker = '#47B354'
    greenDarkest = '#5C9963'
    purpleLighter = '#8E00E6'
    purple = '#8D29CC'
    purpleDarker = '#8947B3'
    purpleDarkest = '#825C99'
    blueLighter = '#0099E6'
    blue = '#2996CC'
    blueDarker = '#478FB3'
    blueDarkest = '#5C8599'
    yellowLighter = '#E6E600'
    yellow = '#CCCC29'
    yellowDarker = '#B3B347'
    yellowDarkest = '#99995C'


class Project(ApiResource, UniqueResourceMixin):

    """ A Project

    Args:
        _id (str): Unique id of the project
        name (str): Name of the project
        description(str, optional): Description of the project
        color (ProjectColor, optional): Color of the project

    """

    resource = 'projects'

    def __init__(self, _id: str, name: str, description: str = None, color: ProjectColor = None, created_by: str = None,
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
        url = '/{}/{}'.format(cls.resource, _id)
        resp = client.request(url, method=requests.get)
        resp_json = parse_json(resp)

        handle_error_response(resp, url)

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
        handle_error_response(response, end_point,
                              message_prefix="Error while fetching user for project id {}".format(self._id))

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
            'description': description
        }
        if color:
            data['color'] = color.value

        url = '/{}'.format(cls.resource)
        resp = client.request(url,
                              data=data,
                              method=requests.post)

        handle_error_response(resp, url, data)
        json = parse_json(resp)

        if '_id' not in json:
            if 'message' in json:
                raise PrevisionException(json['message'])
            else:
                raise Exception('unknown error: {}'.format(json))

        return cls(json['_id'], name, description, color, json['created_by'],
                   json['admins'], json['contributors'], json['pipelines_count'])

    def delete(self) -> Response:
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
        """ List all the available datasets in the current active project.

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
        return DatasetImages._new(self._id, name, file_name)

    def list_image_folders(self, all: bool = True):
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
        return DatasetImages.list(self._id, all=all)

    def create_sql_connector(self, name: str, host: str, port: int = 3306, username: str = '', password: str = ''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source
        Returns:
            :class:`.SQLConnector`: The registered connector object in the current project.
        """
        return SQLConnector._new(self._id, name, host, port, 'SQL', username=username, password=password)

    def create_ftp_connector(self, name: str, host: str, port: int = 21, username: str = '', password: str = ''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source
        Returns:
            :class:`.FTPConnector`: The registered connector object in the current project.
        """
        return FTPConnector._new(self._id, name, host, port, 'FTP', username=username, password=password)

    def create_sftp_connector(self, name: str, host: str, port: int = 23, username: str = '', password: str = ''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source
        Returns:
            :class:`.SFTPConnector`: The registered connector object in the current project.
        """
        return SFTPConnector._new(self._id, name, host, port, 'SFTP', username=username, password=password)

    def create_s3_connector(self, name: str, host: str = '', port: int = None, username: str = '', password: str = ''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source
        Returns:
            :class:`.S3Connector`: The registered connector object in the current project.
        """
        return S3Connector._new(self._id, name, host, port, 'S3', username=username, password=password)

    def create_hive_connector(self, name: str, host: str, port: int = 10000, username: str = '', password: str = ''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            host (str): Url of the connector
            port (int): Port of the connector
            username (str, optional): Username to use connect to the remote data source
            password (str, optional): Password to use connect to the remote data source
        Returns:
            :class:`.HiveConnector`: The registered connector object in the current project.
        """
        return HiveConnector._new(self._id, name, host, port, 'HIVE', username=username, password=password)

    def create_gcp_connector(self, name: str = '', host: str = '', port=None, username: str = '', password: str = '', googleCredentials=''):
        """ A connector to interact with a distant source of data (and
        easily get data snapshots using an associated :class:`.DataSource`
        resource).

        Args:
            name (str): Name of the connector
            googleCredentials(str): google credentials
        Returns:
            :class:`.GCPConnector`: The registered connector object in the current project.
        """
        return GCPConnector._new(self._id, name, host, port, 'GCP', username=username, password=password,
                                 googleCredentials=googleCredentials)

    def list_connectors(self, all: bool = True):
        """ List all the available connectors in the current active project.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Connector` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Connector`): Fetched dataset objects
        """
        return Connector.list(self._id, all=all)

    def create_datasource(self, connector, name, path=None, database=None,
                          table=None, bucket=None, request=None, gCloud=None):
        """ Create a new datasource object on the platform.

        Args:
            connector (:class:`.Connector`): Reference to the associated connector (the resource
                to go through to get a data snapshot)
            name (str): Name of the datasource
            path (str, optional): Path to the file to fetch via the connector
            database (str, optional): Name of the database to fetch data from via the
                connector
            table (str, optional): Name of the table to fetch data from via the connector
            bucket (str, optional): Name of the bucket to fetch data from via the connector
            gCloud (str, optional): gCloud
            request (str, optional): Direct SQL request to use with the connector to fetch data
        Returns:
            :class:`.DataSource`: The registered datasource object in the current project

        Raises:
            PrevisionException: Any error while uploading data to the platform
                or parsing the result
            Exception: For any other unknown error
        """
        return DataSource._new(self._id, connector, name, path=path, database=database,
                               table=table, bucket=bucket, request=request, gCloud=gCloud)

    def list_datasource(self, all=all):
        """ List all the available datasources in the current active project.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.DataSource` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.DataSource`): Fetched dataset objects
        """
        return DataSource.list(self._id, all=all)

    def fit_regression(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.Regression = metrics.Regression.RMSE, holdout_dataset=None,
                       training_config=TrainingConfig(), **kwargs):
        """ Start a tabular regression usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.Regression`: Newly created Regression usecase version object
        """
        return Regression.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                              training_config=training_config, **kwargs)

    def fit_classification(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.Classification = None, holdout_dataset=None,
                           training_config=TrainingConfig(), **kwargs):
        """ Start a tabular classification usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.Classification`: Newly created Classification usecase version object
        """
        return Classification.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                  training_config=training_config, **kwargs)

    def fit_multiclassification(self, name: str, dataset: Dataset, column_config: ColumnConfig, metric: metrics.MultiClassification = None, holdout_dataset=None,
                                training_config=TrainingConfig(), **kwargs):
        """ Start a tabular multiclassification usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.MultiClassification`: Newly created MultiClassification usecase version object
        """
        return MultiClassification.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                       training_config=training_config, **kwargs)

    def fit_image_regression(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.Regression = None, holdout_dataset=None,
                             training_config=TrainingConfig(), **kwargs):
        """ Start an image regression usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.RegressionImages`: Newly created RegressionImages usecase version object
        """
        return RegressionImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                    training_config=training_config, **kwargs)

    def fit_image_classification(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.Classification = None, holdout_dataset=None,
                                 training_config=TrainingConfig(), **kwargs):
        """ Start an image classification usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.ClassificationImages`: Newly created ClassificationImages usecase version object
        """
        return ClassificationImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                        training_config=training_config, **kwargs)

    def fit_image_multiclassification(self, name: str, dataset: Tuple[Dataset, DatasetImages], column_config: ColumnConfig, metric: metrics.MultiClassification = None, holdout_dataset=None,
                                      training_config=TrainingConfig(), **kwargs) -> MultiClassificationImages:
        """ Start an image multiclassification usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.supervised.MultiClassificationImages`: Newly created MultiClassificationImages usecase version object
        """
        return MultiClassificationImages.fit(self._id, name, dataset, column_config, metric=metric, holdout_dataset=holdout_dataset,
                                             training_config=training_config, **kwargs)

    def fit_timeseries_regression(self, name: str, dataset: Dataset, column_config: ColumnConfig, time_window: TimeWindow, metric: metrics.Regression = None, holdout_dataset=None,
                                  training_config=TrainingConfig()) -> TimeSeries:
        """ Start a timeseries regression usecase version training

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`): Column configuration for the usecase version
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            time_window (:class:`.TimeWindow`): Time configuration
                (see the documentation of the :class:`.TimeWindow` resource for more details)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)

        Returns:
            :class:`.TimeSeries`: Newly created TimeSeries usecase version object
        """
        return TimeSeries.fit(self._id, name, dataset, column_config, time_window, metric=metric, holdout_dataset=holdout_dataset,
                              training_config=training_config)

    def fit_text_similarity(self, name: str, dataset: Dataset, description_column_config: DescriptionsColumnConfig, metric: metrics.TextSimilarity = None, top_k=None, lang: str = 'auto',
                            queries_dataset=None, queries_column_config=None, models_parameters=ListModelsParameters()):
        """ Start a text similarity usecase training with a specific training configuration.

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            description_column_config (:class:`.DescriptionsColumnConfig`): Description column configuration
                (see the documentation of the :class:`.DescriptionsColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            top_k (str, optional): top_k
            queries_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a queries dataset (default: ``None``)
            queries_column_config (:class:`.QueriesColumnConfig`): Queries column configuration
                (see the documentation of the :class:`.QueriesColumnConfig` resource for more details
                on each possible column types)
            models_parameters (:class:`.ListModelsParameters`): Specific training configuration
                (see the documentation of the :class:`.ListModelsParameters` resource for more details
                on all the parameters)

        Returns:
            :class:`.TextSimilarity`: Newly created TextSimilarity usecase version object
        """
        return TextSimilarity.fit(self._id, name, dataset, description_column_config, metric=metric, top_k=top_k, lang=lang,
                                  queries_dataset=queries_dataset, queries_column_config=queries_column_config, models_parameters=models_parameters)

    def list_usecases(self, all: bool = True):
        """ List all the available usecase in the current project.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.
        Returns:
            list(:class:`.Usecase`): Fetched usecase objects
        """
        return Usecase.list(self._id, all=all)


connectors_names = {
    'SQL': "create_sql_connector",
    'FTP': "create_ftp_connector",
    'SFTP': "create_sftp_connector",
    'S3': "create_s3_connector",
    'HIVE': "create_hive_connector",
    'GCP': "create_gcp_connector"
}
