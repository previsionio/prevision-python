import requests
from functools import lru_cache
import time
import json
from .logger import logger
from .usecase_config import DataType, UsecaseConfig, TypeProblem
from .prevision_client import client
from .utils import PrevisionException, parse_json, EventTuple
from . import config
from .model import TextSimilarityModel
from .usecase import ClassicUsecaseVersion
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


class TextSimilarity(ClassicUsecaseVersion):

    default_metric = 'accuracy_at_k'
    default_top_k = 10
    data_type = DataType.Tabular
    type_problem = TypeProblem.TextSimilarity
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
    def fit(cls, project_id, name, dataset, description_column_config, metric=None, top_k=None, lang: str='auto',
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
            metric = cls.default_metric
        if not top_k:
            top_k = cls.default_top_k
        training_args['metric'] = metric if isinstance(metric, str) else metric.value
        training_args['top_k'] = top_k
        training_args['lang'] = lang
        if isinstance(dataset, str):
            dataset_id = dataset
        elif isinstance(dataset, tuple):
            dataset_id = [d.id for d in dataset]
        else:
            dataset_id = dataset.id

        data = dict(name=name, dataset_id=dataset_id, **training_args)

        endpoint = '/projects/{}/{}/{}/{}'.format(project_id, 'usecases', cls.data_type, cls.type_problem)
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
