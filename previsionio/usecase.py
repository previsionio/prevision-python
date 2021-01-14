# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import pandas as pd
import requests
import time
import previsionio as pio
import os
from functools import lru_cache

from . import config
from .usecase_config import TrainingConfig, ColumnConfig
from .logger import logger
from .prevision_client import client
from .utils import parse_json, EventTuple, PrevisionException, zip_to_pandas
from .api_resource import ApiResource
from .dataset import Dataset


class BaseUsecase(ApiResource):

    """Base parent class for all usecases objects."""

    training_config: TrainingConfig
    column_config: ColumnConfig
    id_key = 'usecaseId'

    resource = 'usecases'
    type_problem = 'nan'
    data_type = 'nan'

    def __init__(self, **usecase_info):
        super().__init__()
        self.name: str = usecase_info.get('name')
        self.metric = usecase_info.get('metric')
        usecase_params = usecase_info['usecaseParameters']
        self.column_config = ColumnConfig(target_column=usecase_params.get('targetColumn'),
                                          fold_column=usecase_params.get('foldColumn'),
                                          id_column=usecase_params.get('idColumn'),
                                          weight_column=usecase_params.get('weightColumn'))

        self.training_config = TrainingConfig(profile=usecase_params.get('profile'),
                                              fe_selected_list=usecase_params.get(
                                                  'featuresEngineeringSelectedList'),
                                              normal_models=usecase_params.get('normalModels'),
                                              lite_models=usecase_params.get('liteModels'),
                                              simple_models=usecase_params.get('simpleModels'))

        self._id = usecase_info.get('usecaseId')
        self.resource_id = usecase_info.get('_id')
        self.version = usecase_info.get('version', 1)
        self.shared_users = usecase_info.get('shareList', [])
        self._usecase_info = usecase_info

        self.predictions = {}
        self.predict_token = None

        self._models = {}

    def __len__(self):
        return len([m for m in self._status['modelsList']
                    if m['status'] == 'done'
                    # Ignoring simple models for now since we can't predict with them
                    ])

    def update_status(self):
        return super().update_status(specific_url='/{}/{}/versions/{}'.format(self.resource,
                                                                              self._id,
                                                                              self.version))

    @classmethod
    def from_id(cls, _id, version=1):
        """Get a usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase to retrieve
            version (int, optional): Specific version of the usecase to retrieve
                (default: 1)

        Returns:
            :class:`.BaseUsecase`: Fetched usecase

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return super().from_id(specific_url='/{}/{}/versions/{}'.format(cls.resource, _id, version))

    @property
    def models(self):
        """Get the list of models generated for the current use case. Only the models that
        are done training are retrieved.

        Returns:
            list(:class:`.Model`): List of models found by the platform for the usecase
        """
        done = [m for m in self._status['modelsList'] if (m['status'] == 'done')]

        for done_model in done:
            if done_model['_id'] not in self._models:
                self._models[done_model['_id']] = self.model_class(uc_id=self._id,
                                                                   uc_version=self.version,
                                                                   **done_model)

        return list(self._models.values())

    @property
    def versions(self):
        """Get the list of all versions for the current use case.

        Returns:
            list(dict): List of the usecase versions (as JSON metadata)
        """
        end_point = '/usecases/{}/versions'.format(self._id)
        response = client.request(endpoint=end_point, method=requests.get)
        res = parse_json(response)
        return res['items']

    @property
    @lru_cache()
    def train_dataset(self):
        """ Get the :class:`.Dataset` object corresponding to the training dataset
        of the usecase.

        Returns:
            :class:`.Dataset`: Associated training dataset
        """
        return Dataset.from_id(_id=self._status['datasetId'])

    @property
    @lru_cache()
    def correlation_matrix(self) -> pd.DataFrame:
        """ Get the correlation matrix of the features (those constitute the dataset
        on which the usecase was trained).

        Returns:
            ``pd.DataFrame``: Correlation matrix as a ``pandas`` dataframe
        """
        end_point = '/usecases/{}/versions/{}/correlation-matrix'.format(self._id, self.version)
        response = client.request(endpoint=end_point,
                                  method=requests.get)
        corr = json.loads(response.content.decode('utf-8'))
        var_names = [d['name'] for d in corr]
        matrix = pd.DataFrame(0, index=var_names, columns=var_names)
        for i, name in enumerate(var_names):
            matrix[name] = [d['score'] for d in corr[i]['correlation']]
        return matrix

    @property
    @lru_cache()
    def schema(self):
        """ Get the data schema of the usecase.

        Returns:
            ``pd.DataFrame``: Usecase schema
        """
        end_point = '/usecases/{}/versions/{}/schema'.format(self._id, self.version)
        response = client.request(endpoint=end_point,
                                  method=requests.get)
        uc_schema = json.loads(response.content.decode('utf-8'))
        return pd.DataFrame(uc_schema)

    @property
    @lru_cache()
    def feature_stats(self):
        """ Get the general description of the usecase's features, such as:

        - feature types distribution
        - feature information list
        - list of dropped features

        Returns:
            dict: General features information
        """
        end_point = '/usecases/{}/versions/{}/features'.format(self._id, self.version)
        response = client.request(endpoint=end_point,
                                  method=requests.get)
        return (json.loads(response.content.decode('utf-8')))

    def get_feature_info(self, feature_name=None):
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
        response = client.request(
            endpoint='/usecases/{}/versions/{}/features/{}'.format(self._id, self.version, feature_name),
            method=requests.get)
        result = (json.loads(response.content.decode('utf-8')))
        if result.get('status', 200) != 200:
            msg = result['message']
            logger.error(msg)
            raise PrevisionException(msg)
        # drop chart-related informations
        keep_list = list(filter(lambda x: 'chart' not in x.lower(),
                                result.keys())
                         )

        return {k: result[k] for k in keep_list}

    def get_model_from_id(self, id):
        """ Get a model of the usecase by its unique id.

        .. note::

            The model is only searched through the models that are
            done training.

        Args:
            id (str): Unique id of the model resource to search for

        Returns:
            (:class:`.Model`, None): Matching model resource, or ``None``
            if none with the given id could be found
        """
        for m in self.models:
            if m._id == id:
                return m
        return None

    def get_model_from_name(self, model_name=None):
        """ Get a model of the usecase by its name.

        .. note::

            The model is only searched through the models that are
            done training.

        Args:
            name (str): Name of the model resource to search for

        Returns:
            (:class:`.Model`, None): Matching model resource, or ``None``
            if none with the given name could be found
        """
        if model_name:
            for m in self.models:
                if m.name == model_name.upper():
                    return m
        return None

    def _get_best(self, models_list=[], by='loss', default_cost_value=1):
        """ (Util function) Find out the element having the minimal value
        of the attribute defined by the parameter 'by'.

        Args:
            models_list (list(:class:`.Model`)): List of models to search through
            by (str, optional): Key to sort by - the function will return the item
                with the minimal value for this key (default: "loss")
            default_cost_value (_any_, optional): Default value to input for a model
                if the sorting key was not found

        Returns:
            (:class:`.Model`, None): Model with the minimal cost in the given list, or
            ``None`` the list was empty.
        """
        if len(models_list) == 0:
            return None
        for m in models_list:
            if m[by] is None:
                m[by] = default_cost_value

        best = min(models_list, key=lambda m: m[by])
        best = self.model_class(uc_id=self._id, uc_version=self.version, **best)
        return best

    @property
    def best_model(self):
        """ Get the model with the best predictive performance over all models (including
        Blend models), where the best performance corresponds to a minimal loss.

        Returns:
            (:class:`.Model`, None): Model with the best performance in the usecase, or
            ``None`` if no model matched the search filter.
        """
        filter_list = list(filter(lambda m: not (m['tags'].get('simple')),
                                  self._status['modelsList'])
                           )

        return self._get_best(models_list=filter_list)

    @property
    def best_single(self):
        """ Get the model with the best predictive performance (the minimal loss)
        over single models (excluding Blend models), where the best performance
        corresponds to a minimal loss.

        Returns:
            :class:`.Model`: Single (non-blend) model with the best performance in the
            usecase, or ``None`` if no model matched the search filter.
        """
        filter_list = list(filter(lambda m: not (m['tags'].get('simple') or
                                                 m['tags'].get('blend') or
                                                 m['tags'].get('mean')),
                                  self._status['modelsList'])
                           )
        return self._get_best(models_list=filter_list)

    @property
    def fastest_model(self):
        """Returns the model that predicts with the lowest response time

        Returns:
            Model object -- corresponding to the fastest model
        """
        models = self._status['modelsList']
        fastest_model = [m for m in models if m['tags'].get('fastest')]
        fastest_model = self.model_class(uc_id=self._id, uc_version=self.version, **fastest_model[0])
        return fastest_model

    @property
    def running(self):
        """ Get a flag indicating whether or not the usecase is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return status['status'] == 'running'

    @property
    def drop_list(self):
        """ Get the list of drop columns in the usecase.

        Returns:
            list(str): Names of the columns dropped from the dataset
        """
        return self._status['usecaseParameters'].get('dropList')

    @property
    def fe_selected_list(self):
        """ Get the list of selected feature engineering modules in the usecase.

        Returns:
            list(str): Names of the feature engineering modules selected for the usecase
        """
        return self._status['usecaseParameters'].get('featuresEngineeringSelectedList')

    @property
    def normal_models_list(self):
        """ Get the list of selected normal models in the usecase.

        Returns:
            list(str): Names of the normal models selected for the usecase
        """
        return self._status['usecaseParameters'].get('normalModels')

    @property
    def lite_models_list(self):
        """ Get the list of selected lite models in the usecase.

        Returns:
            list(str): Names of the lite models selected for the usecase
        """
        return self._status['usecaseParameters'].get('liteModels')

    @property
    def simple_models_list(self):
        """ Get the list of selected simple models in the usecase.

        Returns:
            list(str): Names of the simple models selected for the usecase
        """
        return self._status['usecaseParameters'].get('simpleModels')

    @classmethod
    def _start_usecase(cls, name, dataset_id, data_type, type_problem, **kwargs):
        """ Start a usecase of the given data type and problem type with a specific
        training configuration (on the platform).

        Args:
            name (str): Registration name for the usecase to create
            dataset_id (str): Unique id of the training dataset resource
            data_type (str): Type of data used in the usecase (among "tabular", "images"
                and "timeseries")
            type_problem: Type of problem to compute with the usecase (among "regression",
                "classification", "multiclassification" and "object-detection")
            **kwargs:

        Returns:
            :class:`.BaseUsecase`: Newly created usecase object
        """
        logger.info('[Usecase] Starting usecase')

        if data_type == 'tabular' or data_type == 'timeseries':
            data = dict(name=name, datasetId=dataset_id, **kwargs)
        elif data_type == 'images':
            csv_id, folder_id = dataset_id
            data = dict(name=name, datasetId=csv_id, datasetFolderId=folder_id, **kwargs)
        else:
            raise PrevisionException('invalid data type: {}'.format(data_type))

        endpoint = '/usecases/{}/{}'.format(data_type, type_problem)
        start = client.request(endpoint, requests.post, data=data)

        if start.status_code != 202:
            logger.error(data)
            logger.error('response:')
            logger.error(start.text)
            raise PrevisionException('usecase failed to start')

        start_response = parse_json(start)
        usecase = cls.from_id(start_response['_id'], version=start_response['version'])
        events_url = '/usecases/{}/versions/{}'.format(start_response['_id'], start_response['version'])
        pio.client.event_manager.wait_for_event(usecase._id,
                                                cls.resource,
                                                EventTuple('USECASE_UPDATE', 'status', 'running'),
                                                specific_url=events_url)
        return usecase

    def predict_single(self,
                       use_best_single=False,
                       confidence=False,
                       explain=False,
                       **predict_data):
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
        if use_best_single:
            best = self.best_single
        else:
            best = self.best_model
        return best.predict_single(confidence=confidence,
                                   explain=explain,
                                   **predict_data)

    def predict_from_dataset(self,
                             dataset,
                             use_best_single=False,
                             confidence=False,
                             dataset_folder=None) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the usecase.

        Arguments:
            dataset (:class:`.Dataset`): Reference to the dataset object to make
                predictions for
            use_best_single (bool, optional): Whether to use the best single model
                instead of the best model overall (default: ``False``)
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            dataset_folder (:class:`.Dataset`): Matching folder dataset for the
                predictions, if necessary

        Returns:
            ``pd.DataFrame``: Predictions as a ``pandas`` dataframe
        """
        if use_best_single:
            best = self.best_single
        else:
            best = self.best_model

        return best.predict_from_dataset(dataset, confidence=confidence, dataset_folder=dataset_folder)

    def predict(self, df, confidence=False,
                use_best_single=False) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the usecase with a Scikit-learn style blocking prediction mode.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours).
            Prefer using this for simple models and small dataframes, or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): ``pandas`` DataFrame containing the test data
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            use_best_single (bool, optional): Whether to use the best single model
                instead of the best model overall (default: ``False``)

        Returns:
            tuple(pd.DataFrame, str): Prediction data (as ``pandas`` dataframe) and prediction job ID.
        """
        if use_best_single:
            best = self.best_single
        else:
            best = self.best_model

        return best.predict(df=df, confidence=confidence)

    def predict_proba(self, df, confidence=False, use_best_single=False) -> pd.DataFrame:
        """ Get the predictions for a dataset stored in the current active [client]
        workspace using the best model of the usecase with a Scikit-learn style blocking prediction mode,
        and returns the probabilities.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours).
            Prefer using this for simple models and small dataframes, or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): ``pandas`` DataFrame containing the test data
            confidence (bool, optional): Whether to predict with confidence values
                (default: ``False``)
            use_best_single (bool, optional): Whether to use the best single model
                instead of the best model overall (default: ``False``)

        Returns:
            tuple(pd.DataFrame, str): Prediction probabilities data (as ``pandas`` dataframe)
            and prediction job ID.
        """
        preds = self.predict(df)
        if preds.shape[1] == 2:
            return preds[self.column_config.target_column].values
        else:
            return preds.values

    def wait_until(self, condition, raise_on_error=True, timeout=config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseUsecase`) -> bool.): Function to use to check the
                break condition
            raise_on_error (bool, optional): If true then the function will stop on error,
                otherwise it will continue waiting (default: ``True``)
            timeout (float, optional): Maximal amount of time to wait before forcing exit

        .. example::

            usecase.wait_until(lambda usecase: len(usecase) > 3)

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
                elif self._status['status'] == 'failed':
                    raise PrevisionException('Resource failed while waiting')
            except PrevisionException as e:
                logger.warning(e.__repr__())
                if raise_on_error:
                    raise

            time.sleep(config.scheduler_refresh_rate)

    def stop(self):
        """ Stop a usecase (stopping all nodes currently in progress). """
        logger.info('[Usecase] stopping usecase')
        response = client.request('/usecases/{}/versions/{}/stop'.format(self.id, self.version),
                                  requests.put)
        events_url = '/usecases/{}/versions/{}'.format(self.id, self.version)
        pio.client.event_manager.wait_for_event(self._id,
                                                self.resource,
                                                EventTuple('USECASE_UPDATE', 'status', 'done'),
                                                specific_url=events_url)
        logger.info('[Usecase] stopping:' + '  '.join(str(k) + ': ' + str(v)
                                                      for k, v in parse_json(response).items()))

    def delete(self):
        """ Delete a usecase from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}'.format(self._id),
                                  method=requests.delete)
        return (json.loads(response.content.decode('utf-8')))

    def get_predictions(self, full=False):
        """
        Retrieves the list of predictions for the current usecase from client workspace
        (with the full predictions object if necessary)
        Args:
            full (boolean): If true, return full prediction objects (else only metadata)
        """
        response = client.request(endpoint='/usecases/{}/predictions'.format(self._id),
                                  method=requests.get)
        preds_list = (json.loads(response.content.decode('utf-8')))['items']
        preds_dict = {}
        for pred in preds_list:
            _id = pred.pop('_id')
            if full:
                response = client.request(endpoint='/usecases/{}/predictions/{}/download'.format(self._id, _id),
                                          method=requests.get)
                preds_dict[_id] = zip_to_pandas(response)
            else:
                preds_dict[_id] = pred
        return preds_dict

    def delete_prediction(self, prediction_id):
        """ Delete a prediction in the list for the current usecase from the actual [client] workspace.

        Args:
            prediction_id (str): Unique id of the prediction to delete

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}/versions/{}/predictions/{}'.format(self._id, self.version,
                                                                                            prediction_id),
                                  method=requests.delete)
        return (json.loads(response.content.decode('utf-8')))

    def delete_predictions(self):
        """ Delete all predictions in the list for the current usecase from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}/versions/{}/predictions'.format(self._id, self.version),
                                  method=requests.delete)
        return (json.loads(response.content.decode('utf-8')))

    def share(self, with_users):
        """ Share a usecase in the actual [client] workspace with other users (specified
        via their emails).

        Args:
            with_users (list(str)): List of emails of the users to share the usecase with
        """
        endpoint = '/usecases/{}/sharing'.format(self._id)
        for user in with_users:
            response = client.request(endpoint=endpoint, method=requests.post, data={'email': user})
            sharing_result = json.loads(response.content.decode('utf-8'))
            if sharing_result.get('status', 200) != 200:
                logger.error(sharing_result)
                raise PrevisionException(sharing_result['message'])
            self.shared_users.append(sharing_result['email'])

    def unshare(self, from_users):
        """ Unshare a usecase in the actual [client] workspace from other users (specified
        via their emails).

        Args:
            from_users (list(str)): List of emails of the users to unshare the usecase from
        """
        for user in from_users:
            endpoint = '/usecases/{}/sharing/{}'.format(self._id, user)
            response = client.request(endpoint=endpoint, method=requests.delete)
            sharing_result = json.loads(response.content.decode('utf-8'))
            if sharing_result.get('status', 200) != 200:
                if sharing_result['status'] == 400:
                    logger.warning(sharing_result)
                else:
                    logger.error(sharing_result)
                    raise PrevisionException(sharing_result['message'])
            self.shared_users.remove(user)

    @property
    def score(self):
        """ Get the current score of the usecase (i.e. the score of the model that is
        currently considered the best performance-wise for this usecase).

        Returns:
            float: Usecase score (or infinity if not available).
        """
        try:
            return self._status['score']
        except KeyError:
            return float('inf')

    def get_cv(self, use_best_single=False) -> pd.DataFrame:
        """ Get the cross validation dataset from the best model of the usecase.

        Args:
            use_best_single (bool, optional): Whether to use the best single model
                instead of the best model overall (default: ``False``)

        Returns:
            ``pd.DataFrame``: Cross validation dataset
        """
        if use_best_single:
            best = self.best_single
        else:
            best = self.best_model

        return best.cross_validation

    def print_info(self):
        """ Print all info on the usecase. """
        for k, v in self._usecase_info.items():
            print(str(k) + ': ' + str(v))

    def _save_json(self):
        raise NotImplementedError

    def save(self, directory='.'):
        version_dict = self._save_json()
        with open(os.path.join(directory, self.name) + '.pio', 'w') as f:
            json.dump(version_dict, f)

    @classmethod
    def load(cls, pio_file):
        with open(pio_file, 'r') as f:
            mdl = json.load(f)
        uc = cls.from_id(mdl['_id'])
        if mdl['usecaseParameters'].get('datasetsHoldout'):
            uc.holdout_dataset = mdl['usecaseParameters'].get('datasetsHoldout')[0]

        return uc
