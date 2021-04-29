import requests
from functools import lru_cache
import time
import json
from .logger import logger
from .api_resource import ApiResource
from .usecase_config import UsecaseConfig
from .prevision_client import client
from .utils import PrevisionException, parse_json, EventTuple
from . import config
from .model import TextSimilarityModel
from .dataset import Dataset
from .usecase import Usecase
import previsionio as pio


class ModelEmbedding(object):
    TFIDF = 'tf_idf'
    Transformer = 'transformer'
    TransformerFineTuned = 'transformer_fine_tuned'


class TextSimilarityModels(object):
    BruteForce = 'brute_force'
    ClusterPruning = 'cluster_pruning'
    IVFOPQ = 'ivfopq'
    HKM = 'hkm'
    LSH = 'lsh'


class Preprocessing(object):
    config = {}

    def __init__(self, word_stemming='yes', ignore_stop_word='auto', ignore_punctuation='no'):
        self.word_stemming = word_stemming
        self.ignore_stop_word = ignore_stop_word
        self.ignore_punctuation = ignore_punctuation


class ListModelsParameters(UsecaseConfig):

    config = {
        'models_parameters': 'models_params'
    }

    def __init__(self, models_parameters=None):

        if models_parameters is None:

            models_parameters_1 = ModelsParameters(ModelEmbedding.TFIDF,
                                                   Preprocessing(),
                                                   [TextSimilarityModels.BruteForce,
                                                    TextSimilarityModels.ClusterPruning])
            models_parameters_2 = ModelsParameters(ModelEmbedding.Transformer,
                                                   {},
                                                   [TextSimilarityModels.BruteForce])
            models_parameters_3 = ModelsParameters(ModelEmbedding.TransformerFineTuned,
                                                   {},
                                                   [TextSimilarityModels.BruteForce])
            models_parameters = [models_parameters_1, models_parameters_2, models_parameters_3]
        self.models_parameters = []
        for element in models_parameters:
            if isinstance(element, ModelsParameters):
                self.models_parameters.append(element)
            else:
                self.models_parameters.append(ModelsParameters(**element))


class ModelsParameters(UsecaseConfig):
    """ Training configuration that holds the relevant data for a usecase description:
    the wanted feature engineering, the selected models, the training speed...

    Args:

    """

    config = {
        'model_embedding': 'model_embedding',
        'preprocessing': 'preprocessing',
        'models': 'models'
    }

    def __init__(self,
                 model_embedding='tf_idf',
                 preprocessing=Preprocessing(),
                 models=['brute_force']):
        self.model_embedding = model_embedding
        if isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        elif preprocessing == {}:
            self.preprocessing = preprocessing
        else:
            self.preprocessing = Preprocessing(**preprocessing)
        self.models = models


class TextSimilarity(ApiResource):

    default_metric = 'accuracy_at_k'
    default_top_k = 10
    data_type = 'tabular'
    type_problem = 'text-similarity'
    resource = 'usecase-versions'

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        self.name: str = usecase_info.get('name')
        self.metric = usecase_info.get('metric')
        self.top_k = usecase_info.get('top_K')
        self.lang = usecase_info.get('lang')
        self.dataset = usecase_info.get('dataset_id')
        usecase_version_params = usecase_info['usecase_version_params']
        self.description_column_config = DescriptionsColumnConfig(
            content_column=usecase_version_params.get('content_column'),
            id_column=usecase_version_params.get('id_column'))
        if usecase_info.get('queries_dataset_id'):
            self.queries_dataset = usecase_info.get('queries_dataset_id')
            content_column = usecase_version_params.get('queries_dataset_content_column')
            matching_id = usecase_version_params.get('queries_dataset_matching_id_description_column')
            queries_dataset_id_column = usecase_version_params.get('queries_dataset_id_column', None)
            self.queries_column_config = QueriesColumnConfig(queries_dataset_content_column=content_column,
                                                             queries_dataset_matching_id_description_column=matching_id,
                                                             queries_dataset_id_column=queries_dataset_id_column)
        else:
            self.queries_dataset = None
            self.queries_column_config = None
        models_parameters = usecase_version_params.get('models_params')
        self.models_parameters = ListModelsParameters(models_parameters=models_parameters)

        self._id = usecase_info.get('_id')
        self.usecase_id = usecase_info.get('usecase_id')
        self.project_id = usecase_info.get('project_id')
        self.version = usecase_info.get('version', 1)
        self._usecase_info = usecase_info
        self.dataset_id = usecase_info.get('dataset_id')
        self.predictions = {}
        self.predict_token = None

        self.model_class = TextSimilarityModel

        self._models = {}

    @classmethod
    def fit(self, project_id, name, dataset, description_column_config, metric=None, top_k=None, lang='auto',
            queries_dataset=None, queries_column_config=None,
            models_parameters=ListModelsParameters(), **kwargs):
        """ Start a supervised usecase training with a specific training configuration
        (on the platform).

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset
                object to use for as training dataset
            description_column_config (:class:`.DescriptionsColumnConfig`): Description column configuration
                (see the documentation of the :class:`.DescriptionsColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the usecase (default: ``None``)
            queries_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a queries dataset (default: ``None``)

        Returns:
            :class:`.TextSimilarity`: Newly created supervised usecase object
        """

        description_column_config = description_column_config.to_kwargs()
        if queries_column_config:
            queries_column_config = queries_column_config.to_kwargs()
            training_args = dict(description_column_config + queries_column_config)
        else:
            training_args = dict(description_column_config)
        training_args.update(to_json(models_parameters))
        training_args.update()

        if queries_dataset:
            if isinstance(queries_dataset, str):
                training_args['queries_dataset_id'] = queries_dataset
            else:
                training_args['queries_dataset_id'] = queries_dataset.id

        if not metric:
            metric = self.default_metric
        if not top_k:
            top_k = self.default_top_k
        training_args['metric'] = metric if isinstance(metric, str) else metric.value
        training_args['top_k'] = top_k
        training_args['lang'] = lang if isinstance(lang, str) else self.lang
        if isinstance(dataset, str):
            dataset_id = dataset
        elif isinstance(dataset, tuple):
            dataset_id = [d.id for d in dataset]
        else:
            dataset_id = dataset.id

        data = dict(name=name, dataset_id=dataset_id, **training_args)

        endpoint = '/projects/{}/{}/{}/{}'.format(project_id, 'usecases', self.data_type, self.type_problem)
        start = client.request(endpoint, requests.post, data=data, content_type='application/json')

        if start.status_code != 200:
            logger.error(data)
            logger.error('response:')
            logger.error(start.text)
            logger.error(start.__dict__)
            raise PrevisionException('usecase failed to start')

        start_response = parse_json(start)
        usecase = self.from_id(start_response['_id'])
        events_url = '/{}/{}'.format(self.resource, start_response['_id'])
        pio.client.event_manager.wait_for_event(usecase._id,
                                                self.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', 'state', 'running'),
                                                specific_url=events_url)
        return usecase

    def update_status(self):
        return super().update_status(specific_url='/{}/{}'.format(self.resource,
                                                                  self._id))

    @classmethod
    def from_id(cls, _id):
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
        return super().from_id(specific_url='/{}/{}'.format(cls.resource, _id))

    @property
    def usecase(self):
        """Get a usecase of current usecase version.

        Returns:
            :class:`.Usecase`: Fetched usecase

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return Usecase.from_id(self.usecase_id)

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

    @property
    def models(self):
        """Get the list of models generated for the current use case. Only the models that
        are done training are retrieved.

        Returns:
            list(:class:`.Model`): List of models found by the platform for the usecase
        """
        end_point = '/{}/{}/models'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get)
        models = json.loads(response.content.decode('utf-8'))['items']

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

    def best_model(self):
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
    def running(self):
        """ Get a flag indicating whether or not the usecase is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return status['state'] == 'running'

    @property
    def status(self):
        """ Get a flag indicating whether or not the usecase is currently running.

        Returns:
            bool: Running status
        """
        status = self._status
        return status['state']

    def print_info(self):
        """ Print all info on the usecase. """
        for k, v in self._usecase_info.items():
            print(str(k) + ': ' + str(v))

    def wait_until(self, condition, raise_on_error=True, timeout=config.default_timeout):
        """ Wait until condition is fulfilled, then break.

        Args:
            condition (func: (:class:`.BaseUsecaseVersion`) -> bool.): Function to use to check the
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
                elif self._status['state'] == 'failed':
                    raise PrevisionException('Resource failed while waiting')
            except PrevisionException as e:
                logger.warning(e.__repr__())
                if raise_on_error:
                    raise

            time.sleep(config.scheduler_refresh_rate)

    def stop(self):
        """ Stop a usecase (stopping all nodes currently in progress). """
        logger.info('[Usecase] stopping usecase')
        response = client.request('/{}/{}/stop'.format(self.resource, self.id),
                                  requests.put)
        events_url = '/{}/{}'.format(self.resource, self.id)
        pio.client.event_manager.wait_for_event(self.resource_id,
                                                self.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', 'state', 'done'),
                                                specific_url=events_url)
        logger.info('[Usecase] stopping:' + '  '.join(str(k) + ': ' + str(v)
                                                      for k, v in parse_json(response).items()))


class DescriptionsColumnConfig(UsecaseConfig):
    """ Description Column configuration for starting a usecase: this object defines
    the role of specific columns in the dataset.

    Args:
        content_column (str, required): Name of the content column in the description dataset
        id_column (str, optional): Name of the id column in the description dataset
    """

    config = {
    }

    def __init__(self,
                 content_column,
                 id_column):
        self.content_column = content_column
        self.id_column = id_column


class QueriesColumnConfig(UsecaseConfig):
    """ Description Column configuration for starting a usecase: this object defines
    the role of specific columns in the dataset.

    Args:
        content_column (str, required): Name of the content column in the description dataset
        id_column (str, optional): Name of the id column in the description dataset
    """

    # config = {
    #     'queries_dataset_content_column': 'queriesDatasetContentColumn',
    #     'queries_dataset_matching_id_description_column': 'queriesDatasetMatchingIdDescriptionColumn',
    #     'queries_dataset_id_column': 'queriesDatasetIdColumn',
    # }

    def __init__(self,
                 queries_dataset_content_column,
                 queries_dataset_matching_id_description_column,
                 queries_dataset_id_column=None):
        self.queries_dataset_content_column = queries_dataset_content_column
        self.queries_dataset_matching_id_description_column = queries_dataset_matching_id_description_column
        self.queries_dataset_id_column = queries_dataset_id_column


def to_json(obj):
    if isinstance(obj, bool):
        return obj
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        obj_list = []
        for e in obj:
            obj_list.append(to_json(e))
        return obj_list
    elif isinstance(obj, dict):
        obj_d = {}
        for key, value in obj.items():
            obj_d[key] = to_json(value)
        return obj_d
    elif hasattr(obj, '__dict__'):
        obj_dict = {}
        for key, value in obj.__dict__.items():
            if key in obj.config:
                key = obj.config[key]
            obj_dict[key] = to_json(value)
        return obj_dict
