# -*- coding: utf-8 -*-
import json
from previsionio.model import Model
from typing import Dict, List, Union
import pandas as pd
import requests
import time
import datetime
import previsionio as pio
from functools import lru_cache
from dateutil import parser

from . import config
from .experiment_config import (AdvancedModel, DataType, Feature, NormalModel, Profile, SimpleModel,
                                TrainingConfig, ColumnConfig, TypeProblem, ExperimentState)
from .logger import logger
from .prevision_client import client
from .utils import parse_json, EventTuple, PrevisionException, zip_to_pandas, get_all_results
from .api_resource import ApiResource
from .dataset import Dataset


class BaseExperimentVersion(ApiResource):

    """Base parent class for all experiments objects."""

    training_config: TrainingConfig
    column_config: ColumnConfig
    id_key = 'experiment_id'

    resource = 'experiment-versions'
    training_type: TypeProblem
    data_type: DataType
    model_class: Model

    def __init__(self, **experiment_version_info):
        super().__init__(experiment_version_info['_id'])
        self._update_from_dict(**experiment_version_info)

        self._models = {}

    def _update_from_dict(self, **experiment_version_info):
        # we keep the raw entire dict, atm used only in print_info...
        self._experiment_version_info = experiment_version_info

        self.project_id: str = experiment_version_info.get('project_id')
        self.experiment_id: str = experiment_version_info.get('experiment_id')
        self.description: str = experiment_version_info.get('description')
        self.version = experiment_version_info['version']
        self.parent_version: str = experiment_version_info.get('parent_version')
        if 'experiment' in experiment_version_info and 'data_type' in experiment_version_info['experiment']:
            self.data_type: DataType = DataType(experiment_version_info['experiment']['data_type'])
        else:
            self.data_type = None
        if 'experiment' in experiment_version_info and 'training_type' in experiment_version_info['experiment']:
            self.training_type: TypeProblem = TypeProblem(experiment_version_info['experiment']['training_type'])
        else:
            self.training_type = None

        self.created_at: datetime.datetime = parser.parse(experiment_version_info.get('created_at'))

    def print_info(self):
        """ Print all info on the experiment. """
        # NOTE: maybe not set self._experiment_version_info and print each object attribut
        for k, v in self._experiment_version_info.items():
            print(str(k) + ': ' + str(v))

    # NOTE: this method is just here to parse raw_data (objects) and build the corresponding data (strings)
    #       that can be sent directly to the endpoint
    @staticmethod
    def _build_experiment_version_creation_data(description, parent_version=None) -> Dict:
        data = {
            'description': description,
            'parent_version': parent_version,
        }
        return data

    @classmethod
    def _new(cls, experiment_id, data) -> 'BaseExperimentVersion':
        endpoint = f'/experiments/{experiment_id}/versions'
        response = client.request(endpoint,
                                  method=requests.post,
                                  data=data,
                                  message_prefix='Experiment version creation')
        experiment_version_info = parse_json(response)
        experiment_version = cls(**experiment_version_info)
        return experiment_version

    def _update_draft(self, **kwargs):
        return self

    def _confirm(self) -> 'BaseExperimentVersion':
        endpoint = f'/experiment-versions/{self._id}/confirm'
        response = client.request(endpoint,
                                  method=requests.put,
                                  message_prefix='Experiment version confirmation')
        experiment_version_info = parse_json(response)
        self._update_from_dict(**experiment_version_info)
        return self

    @classmethod
    def _fit(
        cls,
        experiment_id: str,
        description: str = None,
        parent_version: str = None,
        **kwargs,
    ) -> 'BaseExperimentVersion':

        experiment_version_creation_data = cls._build_experiment_version_creation_data(
            description,
            parent_version=parent_version,
            **kwargs,
        )
        experiment_version_draft = cls._new(experiment_id, experiment_version_creation_data)
        experiment_version_draft._update_draft(**kwargs)
        experiment_version = experiment_version_draft._confirm()

        # NOTE: maybe update like that to be sure to have all the correct info of the resource
        # experiment_version._update_from_dict(**cls._from_id(experiment_version._id))

        return experiment_version

    def new_version(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.models)

    def update_status(self):
        return super().update_status(specific_url='/{}/{}'.format(self.resource,
                                                                  self._id))

    @classmethod
    def _from_id(cls, _id: str) -> Dict:
        """Get an experiment from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment to retrieve
            version (int, optional): Specific version of the experiment to retrieve
                (default: 1)

        Returns:
            :class:`.BaseExperimentVersion`: Fetched experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id))

    # @property
    # def experiment(self) -> 'Experiment':
    #     """Get an experiment of current experiment version.

    #     Returns:
    #         :class:`.Experiment`: Fetched experiment

    #     Raises:
    #         PrevisionException: Any error while fetching data from the platform
    #             or parsing result
    #     """
    #     return Experiment.from_id(self.experiment_id)

    @property
    def models(self):
        """Get the list of models generated for the current experiment version. Only the models that
        are done training are retrieved.

        Returns:
            list(:class:`.Model`): List of models found by the platform for the experiment
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
        of the experiment.

        Returns:
            :class:`.Dataset`: Associated training dataset
        """
        return self.dataset

    @property
    @lru_cache()
    def schema(self) -> dict:
        """ Get the data schema of the experiment.

        Returns:
            dict: Experiment schema
        """
        end_point = '/{}/{}/graph'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Experiment schema')
        schema = json.loads(response.content.decode('utf-8'))
        return schema

    @property
    def best_model(self):
        """ Get the model with the best predictive performance over all models (including
        Blend models), where the best performance corresponds to a minimal loss.

        Returns:
            (:class:`.Model`, None): Model with the best performance in the experiment, or
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
        """ Get a flag indicating whether or not the experiment is currently done.

        Returns:
            bool: done status
        """
        status = self._status
        return status['state'] == ExperimentState.Done.value

    @property
    def running(self) -> bool:
        """ Get a flag indicating whether or not the experiment is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return status['state'] == ExperimentState.Running.value

    @property
    def status(self) -> ExperimentState:
        """ Get a flag indicating whether or not the experiment is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return ExperimentState(status['state'])

    def stop(self):
        """ Stop an experiment (stopping all nodes currently in progress). """
        logger.info('[Experiment] stopping experiment')
        end_point = '/{}/{}/stop'.format(self.resource, self._id)
        response = client.request(end_point,
                                  requests.put,
                                  message_prefix='Experiment stop')
        events_url = '/{}/{}'.format(self.resource, self._id)
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(self.resource_id,
                                                self.resource,
                                                EventTuple('EXPERIMENT_VERSION_UPDATE'),
                                                specific_url=events_url)
        logger.info('[Experiment] stopping:' + '  '.join(str(k) + ': ' + str(v)
                                                         for k, v in parse_json(response).items()))

    def delete(self):
        """Delete an experiment version from the actual [client] workspace.

        Raises:
            PrevisionException: If the experiment version does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    def wait_until(self, condition, raise_on_error: bool = True, timeout: float = config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseExperimentVersion`) -> bool.): Function to use to check the
                break condition
            raise_on_error (bool, optional): If true then the function will stop on error,
                otherwise it will continue waiting (default: ``True``)
            timeout (float, optional): Maximal amount of time to wait before forcing exit

        Example::

            experiment.wait_until(lambda experimentv: len(experimentv.models) > 3)

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
        Retrieves the list of holdout predictions for the current experiment from client workspace
        (with the full predictions object if necessary)
        Args:
            full (boolean): If true, return full holdout prediction objects (else only metadata)
        """
        end_point = '/experiment-versions/{}/holdout-predictions'.format(self._id)
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
        Retrieves the list of predictions for the current experiment from client workspace
        (with the full predictions object if necessary)
        Args:
            full (boolean): If true, return full prediction objects (else only metadata)
        """
        response = client.request(endpoint='/experiments/{}/predictions'.format(self._id),
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
        """ Delete a prediction in the list for the current experiment from the actual [client] workspace.

        Args:
            prediction_id (str): Unique id of the prediction to delete

        Returns:
            dict: Deletion process results
        """
        endpoint = '/experiments/{}/versions/{}/predictions/{}'.format(self._id, self.version, prediction_id)
        response = client.request(endpoint=endpoint,
                                  method=requests.delete,
                                  message_prefix='Prediction delete')
        return (json.loads(response.content.decode('utf-8')))

    def delete_predictions(self):
        """ Delete all predictions in the list for the current experiment from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/experiments/{}/versions/{}/predictions'.format(self._id, self.version),
                                  method=requests.delete,
                                  message_prefix='Predictions delete')
        return (json.loads(response.content.decode('utf-8')))

    @property
    def score(self) -> float:
        """ Get the current score of the experiment (i.e. the score of the model that is
        currently considered the best performance-wise for this experiment).

        Returns:
            float: Experiment score (or infinity if not available).
        """
        try:
            return self._status['best_model']['metric']['value']
        except KeyError:
            PrevisionException('Score not available yet')


class ClassicExperimentVersion(BaseExperimentVersion):

    def __init__(self, **experiment_version_info):
        super().__init__(**experiment_version_info)

        self.predictions = {}
        self.predict_token = None

    def _update_from_dict(self, **experiment_version_info):
        super()._update_from_dict(**experiment_version_info)

        self.dataset_id: str = experiment_version_info['dataset_id']

        experiment_params = experiment_version_info['experiment_version_params']
        self.column_config = ColumnConfig(target_column=experiment_params.get('target_column'),
                                          fold_column=experiment_params.get('fold_column'),
                                          id_column=experiment_params.get('id_column'),
                                          weight_column=experiment_params.get('weight_column'),
                                          time_column=experiment_params.get('time_column', None),
                                          group_columns=experiment_params.get('group_columns'),
                                          apriori_columns=experiment_params.get('apriori_columns'),
                                          drop_list=experiment_params.get('drop_list'))

        self.metric: str = experiment_params['metric']

        self.holdout_dataset_id: Union[str, None] = experiment_version_info.get('holdout_dataset_id', None)

        self.training_config = TrainingConfig(
            profile=Profile(experiment_params.get('profile')),
            features=[Feature(f) for f in experiment_params.get('features_engineering_selected_list', [])],
            advanced_models=[AdvancedModel(f) for f in experiment_params.get('normal_models', [])],
            normal_models=[NormalModel(f) for f in experiment_params.get('lite_models', [])],
            simple_models=[SimpleModel(f) for f in experiment_params.get('simple_models', [])],
            feature_time_seconds=experiment_params.get('features_selection_time', 3600),
            feature_number_kept=experiment_params.get('features_selection_count', None))

    @property
    def dataset(self) -> Dataset:
        """ Get the :class:`.Dataset` object corresponding to the training dataset of this experiment version.

        Returns:
            :class:`.Dataset`: Associated training dataset
        """
        return Dataset.from_id(self.dataset_id)

    @property
    def holdout_dataset(self) -> Union[Dataset, None]:
        """ Get the :class:`.Dataset` object corresponding to the holdout dataset of this experiment version.

        Returns:
            :class:`.Dataset`: Associated holdout dataset
        """
        return Dataset.from_id(self.holdout_dataset_id) if self.holdout_dataset_id is not None else None

    @property
    def advanced_models_list(self) -> List[AdvancedModel]:
        """ Get the list of selected advanced models in the experiment.

        Returns:
            list(AdvancedModel): Names of the normal models selected for the experiment
        """
        return [AdvancedModel(f) for f in self._status['experiment_version_params'].get('normal_models', [])]

    @property
    def normal_models_list(self) -> List[NormalModel]:
        """ Get the list of selected normal models in the experiment.

        Returns:
            list(NormalModel): Names of the normal models selected for the experiment
        """
        return [NormalModel(f) for f in self._status['experiment_version_params'].get('lite_models', [])]

    @property
    def simple_models_list(self) -> List[SimpleModel]:
        """ Get the list of selected simple models in the experiment.

        Returns:
            list(SimpleModel): Names of the simple models selected for the experiment
        """
        return [SimpleModel(f) for f in self._status['experiment_version_params'].get('simple_models', [])]

    @property
    @lru_cache()
    def correlation_matrix(self) -> pd.DataFrame:
        """ Get the correlation matrix of the features (those constitute the dataset
        on which the experiment was trained).

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
        """ Get the general description of the experiment's features, such as:

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
        """ Get the general description of the experiment's features, such as:

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
          series experiments, whether or not it is a group/apriori feature - check the
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
        """ Get the list of drop columns in the experiment.

        Returns:
            list(str): Names of the columns dropped from the dataset
        """
        return self._status['experiment_version_params'].get('drop_list', [])

    @property
    def feature_list(self) -> List[Feature]:
        """ Get the list of selected feature engineering modules in the experiment.

        Returns:
            list(str): Names of the feature engineering modules selected for the experiment
        """
        res = [Feature(f) for f in self._status['experiment_version_params'].get(
            'features_engineering_selected_list', [])]
        return res

    def get_cv(self) -> pd.DataFrame:
        """ Get the cross validation dataset from the best model of the experiment.

        Returns:
            ``pd.DataFrame``: Cross validation dataset
        """

        best = self.best_model

        return best.cross_validation

    def predict_from_dataset(self,
                             dataset,
                             confidence=False,
                             dataset_folder=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the experiment.

        Arguments:
            dataset (:class:`.Dataset`): Reference to the dataset object to make
                predictions for
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            dataset_folder (:class:`.Dataset`): Matching folder dataset for the
                predictions, if necessary

        Returns:
            :class:`previsionio.prediction.ValidationPrediction`: The registered prediction object in the current
            workspace
        """

        best = self.best_model

        return best.predict_from_dataset(dataset, confidence=confidence, dataset_folder=dataset_folder)

    def predict(self, df, confidence=False, prediction_dataset_name=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the experiment with a Scikit-learn style blocking prediction mode.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours).
            Prefer using this for simple models and small dataframes, or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): ``pandas`` DataFrame containing the test data
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)

        Returns:
            ``pd.DataFrame``: Prediction results dataframe
        """

        best = self.best_model

        return best.predict(df=df, confidence=confidence, prediction_dataset_name=prediction_dataset_name)
