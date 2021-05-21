# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Dict, Union

from pandas.core.frame import DataFrame
from previsionio.usecase_config import TypeProblem
import time
import json
import uuid
import requests
import pandas as pd
import previsionio as pio
from functools import lru_cache

from .logger import logger
from .dataset import Dataset
from .deployed_model import DeployedModel
from .prevision_client import client
from .api_resource import ApiResource
from .utils import handle_error_response, parse_json, EventTuple, \
    PrevisionException, zip_to_pandas


class Model(ApiResource):
    """ A Model object is generated by Prevision AutoML plateform when you launch a use case.
    All models generated by Prevision.io are deployable in our Store

    With this Model class, you can select the model with the optimal hyperparameters
    that responds to your buisiness requirements, then you can deploy it
    as a real-time/batch endpoint that can be used for a web Service.

    Args:
        _id (str): Unique id of the model
        usecase_version_id (str): Unique id of the usecase version of the model
        name (str, optional): Name of the model (default: ``None``)
    """

    def __init__(self, _id: str, usecase_version_id: str, project_id: str, model_name: str = None,
                 deployable: bool = False, **other_params):
        """ Instantiate a new :class:`.Model` object to manipulate a model resource on the platform. """
        super().__init__(_id=_id)
        self._id = _id
        self.usecase_version_id = usecase_version_id
        self.project_id = project_id
        self.name = model_name
        self.tags = {}
        self.deployable = deployable

        for k, v in other_params.items():
            self.__setattr__(k, v)

    def __repr__(self):
        return str(self._id)

    def __str__(self):
        """ Show up the Model object attributes.

        Returns:
            str: JSON-formatted info
        """
        args_to_show = {k: self.__dict__[k]
                        for k in self.__dict__
                        if all(map(lambda x: x not in k.lower(),
                                   ["event", "compositiondetails"])
                               )
                        }

        return json.dumps(args_to_show, sort_keys=True, indent=4, separators=(',', ': '))

    @classmethod
    def from_id(cls, _id: str) -> Union[
            'RegressionModel',
            'ClassificationModel',
            'MultiClassificationModel',
            'TextSimilarityModel']:
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
        end_point = '/models/{}'.format(_id)
        response = client.request(endpoint=end_point,
                                  method=requests.get)
        handle_error_response(response, end_point)
        model = json.loads(response.content.decode('utf-8'))
        training_type = model['training_type']
        if training_type == TypeProblem.Regression:
            return RegressionModel(**model)
        elif training_type == TypeProblem.Classification:
            return ClassificationModel(**model)
        elif training_type == TypeProblem.MultiClassification:
            return MultiClassificationModel(**model)
        elif training_type == TypeProblem.TextSimilarity:
            return TextSimilarityModel(**model)
        else:
            raise PrevisionException('Training type {} not supported'.format(model['training_type']))

    @property
    @lru_cache()
    def hyperparameters(self):
        """ Return the hyperparameters of a model.

        Returns:
            dict: Hyperparameters of the model
        """
        url = '/models/{}/hyperparameters/download'.format(self._id)
        response = client.request(
            endpoint=url,
            method=requests.get)
        handle_error_response(response, url)
        return (json.loads(response.content.decode('utf-8')))

    def wait_for_prediction(self, prediction_id: str):
        """ Wait for a specific prediction to finish.

        Args:
            predict_id (str): Unique id of the prediction to wait for
        """
        specific_url = '/predictions/{}'.format(prediction_id)
        pio.client.event_manager.wait_for_event(prediction_id,
                                                specific_url,
                                                EventTuple('PREDICTION_UPDATE', 'state', 'done', [('state', 'failed')]),
                                                specific_url=specific_url)

    def _predict_bulk(self,
                      dataset_id: str,
                      confidence: bool = False,
                      dataset_folder_id: str = None):
        """ (Util method) Private method used to handle bulk predict.

        .. note::

            This function should not be used directly. Use predict_from_* methods instead.

        Args:
            dataset_id (str): Unique id of the dataset to predict with
            confidence (bool, optional): Whether to predict with confidence estimator (default: ``False``)
            dataset_folder_id (str, optional): Unique id of the associated folder dataset to predict with,
                if need be (default: ``None``)

        Returns:
            str: A prediction job ID

        Raises:
            PrevisionException: Any error while starting the prediction on the platform or parsing the result
        """
        data = {
            'dataset_id': dataset_id,
            'model_id': self._id,
            'confidence': str(confidence).lower(),
        }

        if dataset_folder_id is not None:
            data['folder_dataset_id'] = dataset_folder_id

        predict_start = client.request('/usecase-versions/{}/predictions'.format(self.usecase_version_id),
                                       requests.post, data=data)
        predict_start_parsed = parse_json(predict_start)

        if '_id' not in predict_start_parsed:
            err = 'Error starting prediction: {}'.format(predict_start_parsed)
            logger.error(err)
            raise PrevisionException(err)

        return predict_start_parsed['_id']

    def _get_predictions(self, predict_id: str, separator=',') -> pd.DataFrame:
        """ Get the result prediction dataframe from a given predict id.

        Args:
            predict_id (str): Prediction job ID

        Returns:
            ``pd.DataFrame``: Prediction dataframe.
        """
        url = '/predictions/{}/download'.format(predict_id)
        pred_response = pio.client.request(url,
                                           requests.get)
        logger.debug('[Predict {0}] Downloading prediction file'.format(predict_id))
        handle_error_response(pred_response, url)

        return zip_to_pandas(pred_response, separator=separator)

    def predict_from_dataset(self, dataset: Dataset,
                             confidence: bool = False,
                             dataset_folder: Dataset = None) -> Union[pd.DataFrame, None]:
        """ Make a prediction for a dataset stored in the current active [client]
        workspace (using the current SDK dataset object).

        Args:
            dataset (:class:`.Dataset`): Dataset resource to make a prediction for
            confidence (bool, optional): Whether to predict with confidence values (default: ``False``)
            dataset_folder (:class:`.Dataset`, None): Matching folder dataset resource for the prediction,
                if necessary

        Returns:
            ``pd.DataFrame``: Prediction results dataframe
        """

        predict_id = self._predict_bulk(dataset.id,
                                        confidence=confidence,
                                        dataset_folder_id=dataset_folder.id if dataset_folder else None)

        self.wait_for_prediction(predict_id)

        # FIXME : wait_for_prediction() seems to be broken...
        retry_count = 60
        retry = 0
        while retry < retry_count:
            retry += 1
            try:
                preds = self._get_predictions(predict_id, separator=dataset.separator)
                return preds
            except Exception:
                # FIXME:
                # sometimes I observed error 500, with prediction on image usecase
                logger.warning('wait_for_prediction has prolly exited {} seconds too early'
                               .format(retry))
                time.sleep(1)
        return None

    def predict(self, df: DataFrame, confidence: bool = False, prediction_dataset_name: str = None) -> pd.DataFrame:
        """ Make a prediction in a Scikit-learn blocking style.

        .. warning::

            For large dataframes and complex (blend) models, this can be slow (up to 1-2 hours). Prefer using
            this for simple models and small dataframes or use option ``use_best_single = True``.

        Args:
            df (``pd.DataFrame``): A ``pandas`` dataframe containing the testing data
            confidence (bool, optional): Whether to predict with confidence estimator (default: ``False``)

        Returns:
            ``pd.DataFrame``: Prediction results dataframe
        """
        if prediction_dataset_name is None:
            prediction_dataset_name = 'test_{}_{}'.format(self.name, str(uuid.uuid4())[-6:])

        dataset = Dataset._new(self.project_id, prediction_dataset_name, dataframe=df)

        predict_id = self._predict_bulk(dataset.id,
                                        confidence=confidence)
        self.wait_for_prediction(predict_id)

        return self._get_predictions(predict_id, separator=dataset.separator)

    def enable_deploy(self):
        data = {"deploy": True}
        url = '/models/{}'.format(self._id)
        response = client.request(url, requests.put, data=data)
        handle_error_response(response, url, data, message_prefix="Error cannot enable deploy")
        self.deployable = True
        res = parse_json(response)
        return res

    def disable_deploy(self):
        data = {"deploy": False}
        url = '/models/{}'.format(self._id)
        response = client.request(url, requests.put, data=data)
        handle_error_response(response, url, data, message_prefix="Error cannot disable deploy")
        self.deployable = False
        res = parse_json(response)
        return res

    def deploy(self) -> DeployedModel:
        """ (Not Implemented yet) Deploy the model as a REST API app.

        Keyword Arguments:
            app_type {enum} -- it can be 'model', 'notebook', 'shiny', 'dash' or 'node' application

        Returns:
            str: Path of the deployed application
        """
        raise NotImplementedError


class ClassicModel(Model):

    @property
    @lru_cache()
    def feature_importance(self) -> pd.DataFrame:
        """ Return a dataframe of feature importances for the given model features, with their corresponding
        scores (sorted by descending feature importance scores).

        Returns:
            ``pd.DataFrame``: Dataframe of feature importances

        Raises:
            PrevisionException: Any error while fetching data from the platform or parsing the result
        """
        endpoint = '/models/{}/features-importances/download'.format(self._id)
        response = client.request(
            endpoint=endpoint,
            method=requests.get)
        handle_error_response(response, endpoint, message_prefix="Error cannot fetch feature importance")

        df_feat_importance = zip_to_pandas(response)

        return df_feat_importance.sort_values(by="importance", ascending=False)

    @property
    def cross_validation(self) -> pd.DataFrame:
        """ Get model's cross validation dataframe.

        Returns:
            ``pd.Dataframe``: Cross-validation dataframe
        """
        logger.debug('getting cv, model_id: {}'.format(self.id))
        url = '/models/{}/cross-validation/download'.format(self._id)
        cv_response = client.request(url,
                                     requests.get)
        handle_error_response(cv_response, url, message_prefix="Error cannot fetch cross validation")
        df_cv = zip_to_pandas(cv_response)

        return df_cv

    def chart(self):
        """ Return chart analysis information for a model.

        Returns:
            dict: Chart analysis results

        Raises:
            PrevisionException: Any error while fetching data from the platform or parsing the result
        """
        endpoint = '/models/{}/analysis'.format(self._id)
        response = client.request(
            endpoint=endpoint,
            method=requests.get)

        handle_error_response(response, endpoint)

        result = (json.loads(response.content.decode('utf-8')))

        # drop chart-related information
        return result

    def predict_single(self, data: Dict, confidence: bool = False, explain: bool = False):
        """ Make a prediction for a single instance. Use :py:func:`predict_from_dataset_name` or predict methods
        to predict multiple instances at the same time (it's faster).

        Args:
            data (dict): Features names and values (without target feature) - missing feature keys
                will be replaced by nans
            confidence (bool, optional): Whether to predict with confidence values (default: ``False``)
            explain (bool, optional): Whether to explain prediction (default: ``False``)


        .. note::

            You can set both ``confidence`` and ``explain`` to true.

        Returns:
            dict: Dictionary containing the prediction result

            .. note::

                The prediction format depends on the problem type (regression, classification, etc...)
        """
        payload = {
            'features': {
                str(k): v for k, v in data.items() if str(v) != 'nan'
            },
            'explain': explain,
            'confidence': confidence,
            'best': False,
            'model_id': self._id
        }

        logger.debug('[Predict Unit] sending payload ' + str(payload))
        url = '/usecase-versions/{}/unit-prediction'.format(self.usecase_version_id)
        response = client.request(url,
                                  requests.post,
                                  data=payload,
                                  content_type='application/json')
        handle_error_response(response, url, payload, message_prefix="Error while doing a predict")
        response_json = parse_json(response)
        return response_json


class ClassificationModel(ClassicModel):
    """ A model object for a (binary) classification usecase, i.e. a usecase where the target
    is categorical with exactly 2 modalities.

    Args:
        _id (str): Unique id of the model
        uc_id (str): Unique id of the usecase of the model
        uc_version (str, int): Version of the usecase of the model (either an integer for a specific
            version, or "last")
        name (str, optional): Name of the model (default: ``None``)
    """

    def __init__(self, _id, usecase_version_id, **other_params):
        """ Instantiate a new :class:`.ClassificationModel` object to manipulate a classification model
        resource on the platform. """
        super().__init__(_id, usecase_version_id, **other_params)
        self._predict_threshold = 0.5

    @property
    @lru_cache()
    def optimal_threshold(self):
        """ Get the value of threshold probability that optimizes the F1 Score.

        Returns:
            float: Optimal value of the threshold (if it not a classification problem it returns ``None``)

        Raises:
            PrevisionException: Any error while fetching data from the platform or parsing the result
        """
        endpoint = '/models/{}/analysis'.format(self._id)
        response = client.request(endpoint=endpoint,
                                  method=requests.get)
        handle_error_response(response, endpoint)
        resp = json.loads(response.content.decode('utf-8'))
        return resp["optimal_proba"]

    def get_dynamic_performances(self, threshold: float = 0.5):
        """ Get model performance for the given threshold.

        Args:
            threshold (float, optional): Threshold to check the model's performance for (default: 0.5)

        Returns:
            dict: Model classification performance dict with the following keys:

                - ``confusion_matrix``
                - ``accuracy``
                - ``precision``
                - ``recall``
                - ``f1_score``

        Raises:
            PrevisionException: Any error while fetching data from the platform or parsing the result
        """
        threshold = float(threshold)
        if any((threshold > 1, threshold < 0)):
            err = 'threshold value has to be between 0 and 1'
            logger.error(err)
            raise ValueError(err)

        result = dict()
        query = '?threshold={}'.format(str(threshold))
        endpoint = '/models/{}/dynamic-analysis{}'.format(self._id, query)

        response = client.request(endpoint=endpoint,
                                  method=requests.get)
        handle_error_response(response, endpoint)
        resp = json.loads(response.content.decode('utf-8'))

        result['confusion_matrix'] = resp["confusion_matrix"]
        for metric in ['accuracy', 'precision', 'recall', 'f1Score']:
            result[metric] = resp["score"][metric]

        return result


class RegressionModel(ClassicModel):
    """ A model object for a regression usecase, i.e. a usecase where the target is numerical.

    Args:
        _id (str): Unique id of the model
        uc_id (str): Unique id of the usecase of the model
        uc_version (str, int): Version of the usecase of the model (either an integer for a specific
            version, or "last")
        name (str, optional): Name of the model (default: ``None``)
    """


class MultiClassificationModel(ClassicModel):
    """ A model object for a multi-classification usecase, i.e. a usecase where the target
    is categorical with strictly more than 2 modalities.

    Args:
        _id (str): Unique id of the model
        uc_id (str): Unique id of the usecase of the model
        uc_version (str, int): Version of the usecase of the model (either an integer for a specific
            version, or "last")
        name (str, optional): Name of the model (default: ``None``)
    """


class TextSimilarityModel(Model):

    def _predict_bulk(self,
                      queries_dataset_id: str,
                      queries_dataset_content_column: str,
                      top_k: int,
                      matching_id_description_column: str = None):
        """ (Util method) Private method used to handle bulk predict.

        .. note::

            This function should not be used directly. Use predict_from_* methods instead.

        Args:
            queries_dataset_id (str): Unique id of the quries dataset to predict with
            queries_dataset_content_column (str): Content queries column name
            queries_dataset_matching_id_description_column (str): Matching id description column name
            top_k (integer): Number of the nearest description to predict
        Returns:
            str: A prediction job ID

        Raises:
            PrevisionException: Any error while starting the prediction on the platform or parsing the result
        """
        data = {
            'model_id': self._id,
            'queries_dataset_id': queries_dataset_id,
            'queries_dataset_content_column': queries_dataset_content_column,
            'top_k': top_k
        }
        if matching_id_description_column:
            data['queries_dataset_matching_id_description_column'] = matching_id_description_column
        endpoint = '/usecase-versions/{}/predictions'.format(self.usecase_version_id)
        predict_start = client.request(endpoint,
                                       requests.post, data=data)
        handle_error_response(predict_start, endpoint, data)
        predict_start_parsed = parse_json(predict_start)

        if '_id' not in predict_start_parsed:
            err = 'Error starting prediction: {}'.format(predict_start_parsed)
            logger.error(err)
            raise PrevisionException(err)

        return predict_start_parsed['_id']

    def predict_from_dataset(self, queries_dataset: Dataset, queries_dataset_content_column: str, top_k: int = 10,
                             queries_dataset_matching_id_description_column: str = None) -> Union[pd.DataFrame, None]:
        """ Make a prediction for a dataset stored in the current active [client]
        workspace (using the current SDK dataset object).

        Args:
            dataset (:class:`.Dataset`): Dataset resource to make a prediction for
            queries_dataset_content_column (str): Content queries column name
            top_k (integer): Number of the nearest description to predict
            queries_dataset_matching_id_description_column (str): Matching id description column name

        Returns:
            ``pd.DataFrame``: Prediction results dataframe
        """
        predict_id = self._predict_bulk(queries_dataset.id,
                                        queries_dataset_content_column,
                                        top_k=top_k,
                                        matching_id_description_column=queries_dataset_matching_id_description_column)
        self.wait_for_prediction(predict_id)
        # FIXME : wait_for_prediction() seems to be broken...
        retry_count = 60
        retry = 0
        while retry < retry_count:
            retry += 1
            try:
                preds = self._get_predictions(predict_id, separator=queries_dataset.separator)
                return preds
            except Exception:
                # FIXME:
                # sometimes I observed error 500, with prediction on image usecase
                logger.warning('wait_for_prediction has prolly exited {} seconds too early'
                               .format(retry))
                time.sleep(1)
        return None
