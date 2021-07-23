# -*- coding: utf-8 -*-
from __future__ import print_function
import json
from previsionio import metrics
from previsionio.model import Model
from typing import Dict, List, Union
import pandas as pd
import requests
import time
import previsionio as pio
import os
from functools import lru_cache
from dateutil import parser

from . import config
from .usecase_config import (AdvancedModel, DataType, Feature, NormalModel, Profile, SimpleModel,
                             TrainingConfig, ColumnConfig, TypeProblem, UsecaseState)
from .logger import logger
from .prevision_client import client
from .utils import parse_json, EventTuple, PrevisionException, zip_to_pandas, get_all_results
from .api_resource import ApiResource
from .dataset import Dataset
# from .usecase import Usecase


class BaseUsecaseVersion(ApiResource):

    """Base parent class for all usecases objects."""

    training_config: TrainingConfig
    column_config: ColumnConfig
    id_key = 'usecase_id'

    resource = 'usecase-versions'
    training_type: TypeProblem
    data_type: DataType
    model_class: Model

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        self.name: str = usecase_info.get('name', usecase_info['usecase'].get('name'))
        self._id: str = usecase_info['_id']
        self.usecase_id: str = usecase_info['usecase_id']
        self.project_id: str = usecase_info['project_id']
        self.dataset_id: str = usecase_info['dataset_id']
        self.holdout_dataset_id: Union[str, None] = usecase_info.get('holdout_dataset_id', None)
        self.created_at = parser.parse(usecase_info["created_at"])
        self._models = {}
        self.version = 1

    def __len__(self):
        return len(self.models)

    def update_status(self):
        return super().update_status(specific_url='/{}/{}'.format(self.resource,
                                                                  self._id))

    @classmethod
    def _from_id(cls, _id: str) -> Dict:
        """Get a usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase to retrieve
            version (int, optional): Specific version of the usecase to retrieve
                (default: 1)

        Returns:
            :class:`.BaseUsecaseVersion`: Fetched usecase

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id))

    # @property
    # def usecase(self) -> 'Usecase':
    #     """Get a usecase of current usecase version.

    #     Returns:
    #         :class:`.Usecase`: Fetched usecase

    #     Raises:
    #         PrevisionException: Any error while fetching data from the platform
    #             or parsing result
    #     """
    #     return Usecase.from_id(self.usecase_id)

    @property
    def models(self):
        """Get the list of models generated for the current use case. Only the models that
        are done training are retrieved.

        Returns:
            list(:class:`.Model`): List of models found by the platform for the usecase
        """
        end_point = '/{}/{}/models'.format(self.resource, self._id)
        models = get_all_results(client, end_point, method=requests.get)
        for model in models:
            if model['_id'] not in self._models:
                self._models[model['_id']] = self.model_class(**model)
        return list(self._models.values())

    @property
    @lru_cache()
    def train_dataset(self):
        """ Get the :class:`.Dataset` object corresponding to the training dataset
        of the usecase.

        Returns:
            :class:`.Dataset`: Associated training dataset
        """
        return Dataset.from_id(_id=self.dataset_id)

    @property
    @lru_cache()
    def schema(self) -> dict:
        """ Get the data schema of the usecase.

        Returns:
            dict: Usecase schema
        """
        end_point = '/{}/{}/graph'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Usecase schema')
        uc_schema = json.loads(response.content.decode('utf-8'))
        return uc_schema

    @property
    def best_model(self):
        """ Get the model with the best predictive performance over all models (including
        Blend models), where the best performance corresponds to a minimal loss.

        Returns:
            (:class:`.Model`, None): Model with the best performance in the usecase, or
            ``None`` if no model matched the search filter.
        """
        best_model = None
        if len(self.models) == 0:
            raise PrevisionException('models not ready yet')
        for model in self.models:
            if 'best' in model.tags:
                best_model = model

        if best_model is None:
            best_model = self.models[0]
        return best_model

    @property
    def fastest_model(self):
        """Returns the model that predicts with the lowest response time

        Returns:
            Model object -- corresponding to the fastest model
        """
        fastest_model = None
        if len(self.models) == 0:
            raise PrevisionException('models not ready yet')
        for model in self.models:
            if 'fastest' in model.tags:
                fastest_model = model

        if fastest_model is None:
            fastest_model = self.models[0]

        return fastest_model

    @property
    def done(self) -> bool:
        """ Get a flag indicating whether or not the usecase is currently done.

        Returns:
            bool: done status
        """
        status = self._status
        return status['state'] == UsecaseState.Done.value

    @property
    def running(self) -> bool:
        """ Get a flag indicating whether or not the usecase is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return status['state'] == UsecaseState.Running.value

    @property
    def status(self) -> UsecaseState:
        """ Get a flag indicating whether or not the usecase is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return UsecaseState(status['state'])

    @property
    def advanced_models_list(self) -> List[AdvancedModel]:
        """ Get the list of selected advanced models in the usecase.

        Returns:
            list(AdvancedModel): Names of the normal models selected for the usecase
        """
        return [AdvancedModel(f) for f in self._status['usecase_version_params'].get('normal_models', [])]

    @property
    def normal_models_list(self) -> List[NormalModel]:
        """ Get the list of selected normal models in the usecase.

        Returns:
            list(NormalModel): Names of the normal models selected for the usecase
        """
        return [NormalModel(f) for f in self._status['usecase_version_params'].get('lite_models', [])]

    @property
    def simple_models_list(self) -> List[SimpleModel]:
        """ Get the list of selected simple models in the usecase.

        Returns:
            list(SimpleModel): Names of the simple models selected for the usecase
        """
        return [SimpleModel(f) for f in self._status['usecase_version_params'].get('simple_models', [])]

    def stop(self):
        """ Stop a usecase (stopping all nodes currently in progress). """
        logger.info('[Usecase] stopping usecase')
        end_point = '/{}/{}/stop'.format(self.resource, self._id)
        response = client.request(end_point,
                                  requests.put,
                                  message_prefix='Usecase stop')
        events_url = '/{}/{}'.format(self.resource, self._id)
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(self.resource_id,
                                                self.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', 'state',
                                                           'done', [('state', 'failed')]),
                                                specific_url=events_url)
        logger.info('[Usecase] stopping:' + '  '.join(str(k) + ': ' + str(v)
                                                      for k, v in parse_json(response).items()))

    # def delete(self):
    #     """ Delete a usecase from the actual [client] workspace.
    #
    #     Returns:
    #         dict: Deletion process results
    #     """
    #     response = client.request(endpoint='/usecases/{}'.format(self._id),
    #                               method=requests.delete)
    #     return (json.loads(response.content.decode('utf-8')))

    def wait_until(self, condition, raise_on_error: bool = True, timeout: float = config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseUsecaseVersion`) -> bool.): Function to use to check the
                break condition
            raise_on_error (bool, optional): If true then the function will stop on error,
                otherwise it will continue waiting (default: ``True``)
            timeout (float, optional): Maximal amount of time to wait before forcing exit

        Example::

            usecase.wait_until(lambda usecasev: len(usecasev.models) > 3)

        Raises:
            PrevisionException: If the resource could not be fetched or there was a timeout.
        """
        t0 = time.time()
        while True:
            if timeout is not None and time.time() - t0 > timeout:
                raise PrevisionException('timeout while waiting on {}'.format(condition))

            try:
                if condition(self):
                    break
                elif self._status['state'] == 'failed':
                    raise PrevisionException('Resource failed while waiting')
            except PrevisionException as e:
                logger.warning(e.__repr__())
                if raise_on_error:
                    raise

            time.sleep(config.scheduler_refresh_rate)

    def get_holdout_predictions(self, full: bool = False):
        """
        Retrieves the list of holdout predictions for the current usecase from client workspace
        (with the full predictions object if necessary)
        Args:
            full (boolean): If true, return full holdout prediction objects (else only metadata)
        """
        end_point = '/usecase-versions/{}/holdout-predictions'.format(self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Holdout predictions listing')
        preds_list = (json.loads(response.content.decode('utf-8')))['items']
        preds_dict = {}
        for pred in preds_list:
            _id = pred.pop('_id')
            if full:
                end_point = '/predictions/{}/download'.format(_id)
                response = client.request(endpoint=end_point,
                                          method=requests.get,
                                          message_prefix='Predictions fetching')
                preds_dict[_id] = zip_to_pandas(response)
            else:
                preds_dict[_id] = pred
        return preds_dict

    def get_predictions(self, full: bool = False):
        """
        Retrieves the list of predictions for the current usecase from client workspace
        (with the full predictions object if necessary)
        Args:
            full (boolean): If true, return full prediction objects (else only metadata)
        """
        response = client.request(endpoint='/usecases/{}/predictions'.format(self._id),
                                  method=requests.get,
                                  message_prefix='Predictions listing')
        preds_list = (json.loads(response.content.decode('utf-8')))['items']
        preds_dict = {}
        for pred in preds_list:
            _id = pred.pop('_id')
            if full:
                response = client.request(endpoint='/predictions/{}/download'.format(_id),
                                          method=requests.get,
                                          message_prefix='Predictions fetching')
                preds_dict[_id] = zip_to_pandas(response)
            else:
                preds_dict[_id] = pred
        return preds_dict

    def delete_prediction(self, prediction_id: str):
        """ Delete a prediction in the list for the current usecase from the actual [client] workspace.

        Args:
            prediction_id (str): Unique id of the prediction to delete

        Returns:
            dict: Deletion process results
        """
        endpoint = '/usecases/{}/versions/{}/predictions/{}'.format(self._id, self.version, prediction_id)
        response = client.request(endpoint=endpoint,
                                  method=requests.delete,
                                  message_prefix='Prediction delete')
        return (json.loads(response.content.decode('utf-8')))

    def delete_predictions(self):
        """ Delete all predictions in the list for the current usecase from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}/versions/{}/predictions'.format(self._id, self.version),
                                  method=requests.delete,
                                  message_prefix='Predictions delete')
        return (json.loads(response.content.decode('utf-8')))

    @property
    def score(self) -> float:
        """ Get the current score of the usecase (i.e. the score of the model that is
        currently considered the best performance-wise for this usecase).

        Returns:
            float: Usecase score (or infinity if not available).
        """
        try:
            return self._status['score']
        except KeyError:
            return float('inf')

    def _save_json(self):
        raise NotImplementedError

    def save(self, directory: str = '.'):
        version_dict = self._save_json()
        with open(os.path.join(directory, self.name) + '.pio', 'w') as f:
            json.dump(version_dict, f)

    @classmethod
    def _load(cls, pio_file: str) -> Dict:
        with open(pio_file, 'r') as f:
            mdl = json.load(f)
        uc = cls._from_id(mdl['_id'])
        # TODO check holdout_dataset in usecase_version_params
        # if mdl['usecase_version_params'].get('holdout_dataset_id'):
        #     uc.holdout_dataset = mdl['usecase_version_params'].get('holdout_dataset_id')[0]
        return uc


class ClassicUsecaseVersion(BaseUsecaseVersion):

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        usecase_params = usecase_info['usecase_version_params']
        self.metric: str = usecase_params['metric']
        self.column_config = ColumnConfig(target_column=usecase_params.get('target_column'),
                                          fold_column=usecase_params.get('fold_column'),
                                          id_column=usecase_params.get('id_column'),
                                          weight_column=usecase_params.get('weight_column'),
                                          time_column=usecase_params.get('time_column', None),
                                          group_columns=usecase_params.get('group_columns'),
                                          apriori_columns=usecase_params.get('apriori_columns'),
                                          drop_list=usecase_params.get('drop_list'))

        self.training_config = TrainingConfig(profile=Profile(usecase_params.get('profile')),
                                              features=[Feature(f) for f in usecase_params.get(
                                                  'features_engineering_selected_list', [])],
                                              advanced_models=[
                                                  AdvancedModel(f) for f in usecase_params.get('normal_models', [])],
                                              normal_models=[NormalModel(f)
                                                             for f in usecase_params.get('lite_models', [])],
                                              simple_models=[SimpleModel(f)
                                                             for f in usecase_params.get('simple_models', [])],
                                              feature_time_seconds=usecase_params.get('features_selection_time', 3600),
                                              feature_number_kept=usecase_params.get('features_selection_count', None))

        self._id: str = usecase_info['_id']
        self.usecase_id: str = usecase_info['usecase_id']
        self.project_id: str = usecase_info['project_id']
        self.version = usecase_info.get('version', 1)
        self._usecase_info = usecase_info
        self.data_type: DataType = DataType(usecase_info['usecase'].get('data_type'))
        self.training_type: TypeProblem = TypeProblem(usecase_info['usecase'].get('training_type'))
        self.dataset_id: str = usecase_info['dataset_id']
        self.predictions = {}
        self.predict_token = None
        self._models = {}

    def print_info(self):
        """ Print all info on the usecase. """
        for k, v in self._usecase_info.items():
            print(str(k) + ': ' + str(v))

    @property
    @lru_cache()
    def correlation_matrix(self) -> pd.DataFrame:
        """ Get the correlation matrix of the features (those constitute the dataset
        on which the usecase was trained).

        Returns:
            ``pd.DataFrame``: Correlation matrix as a ``pandas`` dataframe
        """
        end_point = '/{}/{}/correlation-matrix'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Correlation matrix fetching')
        corr = json.loads(response.content.decode('utf-8'))
        var_names = [d['name'] for d in corr]
        matrix = pd.DataFrame(0, index=var_names, columns=var_names)
        for i, name in enumerate(var_names):
            matrix[name] = [d['score'] for d in corr[i]['correlation']]
        return matrix

    @property
    @lru_cache()
    def features(self) -> List[dict]:
        """ Get the general description of the usecase's features, such as:

        - feature types distribution
        - feature information list
        - list of dropped features

        Returns:
            dict: General features information
        """
        # todo pagination
        url = '/{}/{}/features'.format(self.resource, self._id)
        return get_all_results(client, url, method=requests.get)

    @property
    @lru_cache()
    def features_stats(self):
        """ Get the general description of the usecase's features, such as:

        - feature types distribution
        - feature information list
        - list of dropped features

        Returns:
            dict: General features information
        """
        # todo pagination
        end_point = '/{}/{}/features-stats'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Features stats fetching')
        return (json.loads(response.content.decode('utf-8')))

    @property
    @lru_cache()
    def dropped_features(self):
        """ Get dropped features

        Returns:
            dict: Dropped features
        """
        # todo pagination
        end_point = '/{}/{}/dropped-features'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Dropped features fetching')
        return (json.loads(response.content.decode('utf-8')))

    def get_feature_info(self, feature_name: str) -> Dict:
        """ Return some information about the given feature, such as:

        - name: the name of the feature as it was given in the ``feature_name`` parameter
        - type: linear, categorical, ordinal...
        - stats: some basic statistics such as number of missing values, (non missing) values
          count, plus additional information depending on the feature type:

            - for a linear feature: min, max, mean, std and median
            - for a categorical/textual feature: modalities/words frequencies,
              list of the most frequent tokens

        - role: whether or not the feature is a target/fold/weight or id feature (and for time
          series usecases, whether or not it is a group/apriori feature - check the
          `Prevision.io's timeseries documentation
          <https://previsionio.readthedocs.io/fr/latest/projects_timeseries.html>`_)
        - importance_value: scores reflecting the importance of the given feature

        Args:
            feature_name (str): Name of the feature to get informations about

            .. warning::

                The ``feature_name`` is case-sensitive, so "age" and "Age" are different
                features!

        Returns:
            dict: Dictionary containing the feature information

        Raises:
            PrevisionException: If the given feature name does not match any feaure
        """
        endpoint = '/{}/{}/features/{}'.format(self.resource, self._id, feature_name)
        response = client.request(endpoint=endpoint,
                                  method=requests.get,
                                  message_prefix='Features info fetching')

        result = (json.loads(response.content.decode('utf-8')))

        # drop chart-related informations
        keep_list = list(filter(lambda x: 'chart' not in x.lower(),
                                result.keys())
                         )
        return {k: result[k] for k in keep_list}

    @property
    def drop_list(self) -> List[str]:
        """ Get the list of drop columns in the usecase.

        Returns:
            list(str): Names of the columns dropped from the dataset
        """
        return self._status['usecase_version_params'].get('drop_list', [])

    @property
    def feature_list(self) -> List[Feature]:
        """ Get the list of selected feature engineering modules in the usecase.

        Returns:
            list(str): Names of the feature engineering modules selected for the usecase
        """
        res = [Feature(f) for f in self._status['usecase_version_params'].get('features_engineering_selected_list', [])]
        return res

    def get_cv(self) -> pd.DataFrame:
        """ Get the cross validation dataset from the best model of the usecase.

        Returns:
            ``pd.DataFrame``: Cross validation dataset
        """

        best = self.best_model

        return best.cross_validation

    @classmethod
    def _start_usecase(cls, project_id: str, name: str,
                       dataset_id: Union[str, List[str]], data_type: DataType, training_type: TypeProblem, **kwargs):
        """ Start a usecase of the given data type and problem type with a specific
        training configuration (on the platform).

        Args:
            name (str): Registration name for the usecase to create
            dataset_id (str|tuple(str, str)): Unique id of the training dataset resource or a tuple of csv and folder id
            data_type (str): Type of data used in the usecase (among "tabular", "images"
                and "timeseries")
            training_type: Type of problem to compute with the usecase (among "regression",
                "classification", "multiclassification" and "object-detection")
            **kwargs:

        Returns:
            :class:`.BaseUsecaseVersion`: Newly created usecase object
        """
        logger.info('[Usecase] Starting usecase')

        if data_type == DataType.Tabular or data_type == DataType.TimeSeries:
            data = dict(name=name, dataset_id=dataset_id, **kwargs)
        elif data_type == DataType.Images:
            csv_id, folder_id = dataset_id
            data = dict(name=name, dataset_id=csv_id, folder_dataset_id=folder_id, **kwargs)
        else:
            raise PrevisionException('invalid data type: {}'.format(data_type))
        endpoint = '/projects/{}/{}/{}/{}'.format(project_id, 'usecases', data_type.value, training_type.value)
        start = client.request(endpoint,
                               method=requests.post,
                               data=data,
                               content_type='application/json',
                               message_prefix='Usecase start')
        return parse_json(start)

    def predict_single(self,
                       data,
                       confidence=False,
                       explain=False):
        """ Get a prediction on a single instance using the best model of the usecase.

        Args:
            use_best_single (bool, optional): Whether to use the best single model
                instead of the best model overall (default: ``False``)
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            explain (bool): Whether to explain prediction (default: ``False``)

        Returns:
            dict: Dictionary containing the prediction.

            .. note::

                The format of the predictions dictionary depends on the problem type
                (regression, classification...)
        """

        best = self.best_model
        return best.predict_single(data,
                                   confidence=confidence,
                                   explain=explain)

    def predict_from_dataset(self,
                             dataset,
                             confidence=False,
                             dataset_folder=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the usecase.

        Arguments:
            dataset (:class:`.Dataset`): Reference to the dataset object to make
                predictions for
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            dataset_folder (:class:`.Dataset`): Matching folder dataset for the
                predictions, if necessary

        Returns:
            ``pd.DataFrame``: Predictions as a ``pandas`` dataframe
        """

        best = self.best_model

        return best.predict_from_dataset(dataset, confidence=confidence, dataset_folder=dataset_folder)

    def predict(self, df, confidence=False, prediction_dataset_name=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the usecase with a Scikit-learn style blocking prediction mode.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours).
            Prefer using this for simple models and small dataframes, or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): ``pandas`` DataFrame containing the test data
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)

        Returns:
            tuple(pd.DataFrame, str): Prediction data (as ``pandas`` dataframe) and prediction job ID.
        """

        best = self.best_model

        return best.predict(df=df, confidence=confidence, prediction_dataset_name=prediction_dataset_name)
