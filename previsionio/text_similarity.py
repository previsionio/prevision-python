from enum import Enum
from typing import Dict, List, Union
from previsionio.dataset import Dataset
import requests
from .usecase_config import DataType, UsecaseConfig, TypeProblem, YesOrNo, YesOrNoOrAuto
from .prevision_client import client
from .utils import parse_json, EventTuple, to_json
from .model import TextSimilarityModel
from .usecase_version import BaseUsecaseVersion
import previsionio as pio


class ModelEmbedding(Enum):
    """Embedding models for Text Similarity"""
    TFIDF = 'tf_idf'
    """Term Frequency - Inverse Document Frequency"""
    Transformer = 'transformer'
    """Transformer"""
    TransformerFineTuned = 'transformer_fine_tuned'
    """fine tuned Transformer"""


class TextSimilarityModels(Enum):
    """Similarity search models for Text Similarity"""
    BruteForce = 'brute_force'
    """Brute force search"""
    ClusterPruning = 'cluster_pruning'
    """Cluster Pruning"""
    IVFOPQ = 'ivfopq'
    """InVerted File system and Optimized Product Quantization"""
    HKM = 'hkm'
    """Hierarchical K-Means"""
    LSH = 'lsh'
    """Locality Sensitive Hashing"""


class TextSimilarityLang(Enum):
    Auto = 'auto'
    French = 'fr'
    English = 'en'


class Preprocessing(object):
    config = {}

    def __init__(self,
                 word_stemming: YesOrNo = YesOrNo.Yes,
                 ignore_stop_word: YesOrNoOrAuto = YesOrNoOrAuto.Auto,
                 ignore_punctuation: YesOrNo = YesOrNo.Yes):
        self.word_stemming = word_stemming
        self.ignore_stop_word = ignore_stop_word
        self.ignore_punctuation = ignore_punctuation


class ModelsParameters(UsecaseConfig):
    """ Training configuration that holds the relevant data for a usecase description:
    the wanted feature engineering, the selected models, the training speed...

    Args:
        preprocessing (Preprocessing, optional): Dictionary of the text preprocessings to be applied
            (only for "tf_idf" embedding model),

            - *word_stemming*: default to "yes"
            - *ignore_stop_word*: default to "auto", choice will be made depending on if the
              text descriptions contain full sentences or not
            - *ignore_punctuation*: default to "no".
        model_embedding (ModelEmbedding, optional): Name of the embedding model to be used
            (among: "tf_idf", "transformer", "transformer_fine_tuned").
        models (list(TextSimilarityModels), optional): Names of the searching models to be used (among:
            "brute_force", "cluster_pruning", "ivfopq", "hkm", "lsh").
    """

    config = {
        'model_embedding': 'model_embedding',
        'preprocessing': 'preprocessing',
        'models': 'models'
    }

    def __init__(self,
                 model_embedding: ModelEmbedding = ModelEmbedding.TFIDF,
                 preprocessing: Preprocessing = Preprocessing(),
                 models: List[TextSimilarityModels] = [TextSimilarityModels.BruteForce]):
        self.model_embedding = model_embedding
        if isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        elif preprocessing == {}:
            self.preprocessing = preprocessing
        else:
            self.preprocessing = Preprocessing(**preprocessing)
        self.models = models


class ListModelsParameters(UsecaseConfig):

    config = {
        'models_parameters': 'models_params'
    }

    def __init__(self, models_parameters: Union[List[ModelsParameters], List[Dict], None] = None):

        if models_parameters is None:

            models_parameters_1 = ModelsParameters(ModelEmbedding.TFIDF,
                                                   Preprocessing(),
                                                   [TextSimilarityModels.BruteForce,
                                                    TextSimilarityModels.ClusterPruning])
            models_parameters_2 = ModelsParameters(ModelEmbedding.Transformer,
                                                   Preprocessing(),
                                                   [TextSimilarityModels.BruteForce, TextSimilarityModels.IVFOPQ])
            models_parameters_3 = ModelsParameters(ModelEmbedding.TransformerFineTuned,
                                                   Preprocessing(),
                                                   [TextSimilarityModels.BruteForce, TextSimilarityModels.IVFOPQ])
            models_parameters = [models_parameters_1, models_parameters_2, models_parameters_3]
        self.models_parameters = []
        for element in models_parameters:
            if isinstance(element, ModelsParameters):
                self.models_parameters.append(element)
            else:
                self.models_parameters.append(ModelsParameters(**element))


class DescriptionsColumnConfig(UsecaseConfig):
    """ Description Column configuration for starting a usecase: this object defines
    the role of specific columns in the dataset.

    Args:
        content_column (str, required): Name of the column containing the text descriptions in the
            description dataset.
        id_column (str, optional): Name of the id column in the description dataset.
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
        content_column (str, required): Name of the column containing the text queries in the
            description dataset.
        id_column (str, optional): Name of the id column in the description dataset.
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


class TextSimilarity(BaseUsecaseVersion):
    """ A text similarity usecase version """

    default_metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity.accuracy_at_k
    default_top_k: int = 10
    data_type: DataType = DataType.Tabular
    training_type: TypeProblem = TypeProblem.TextSimilarity
    resource: str = 'usecase-versions'
    model_class = TextSimilarityModel

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        usecase_version_params = usecase_info['usecase_version_params']
        self.metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity(
            usecase_version_params.get('metric', self.default_metric))
        self.top_k: int = usecase_version_params.get('top_K', self.default_top_k)
        self.lang: TextSimilarityLang = TextSimilarityLang(usecase_version_params.get('lang', TextSimilarityLang.Auto))

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

        self._id: str = usecase_info['_id']
        self.usecase_id: str = usecase_info['usecase_id']
        self.project_id: str = usecase_info['project_id']
        self.version: int = usecase_info.get('version', 1)
        self._usecase_info = usecase_info
        self.dataset_id: str = usecase_info['dataset_id']
        self.predictions = {}
        self.predict_token = None

        self._models = {}

    @classmethod
    def from_id(cls, _id: str) -> 'TextSimilarity':
        return cls(**super()._from_id(_id))

    @classmethod
    def load(cls, pio_file: str) -> 'TextSimilarity':
        return cls(**super()._load(pio_file))

    @classmethod
    def _fit(
        cls,
        project_id: str,
        name: str,
        dataset: Dataset,
        description_column_config: DescriptionsColumnConfig,
        metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity.accuracy_at_k,
        top_k: int = 10,
        lang: TextSimilarityLang = TextSimilarityLang.Auto,
        queries_dataset: Dataset = None,
        queries_column_config: QueriesColumnConfig = None,
        models_parameters: ListModelsParameters = ListModelsParameters(),
        **kwargs
    ) -> 'TextSimilarity':
        """ Start a supervised usecase training with a specific training configuration
        (on the platform).

        Args:
            name (str): Name of the usecase to create
            dataset (:class:`.Dataset`): Reference to the dataset
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

        training_args = to_json(description_column_config)
        assert isinstance(training_args, Dict)
        if queries_column_config:
            training_args.update(to_json(queries_column_config))
        training_args.update(to_json(models_parameters))

        if queries_dataset:
            if isinstance(queries_dataset, str):
                training_args['queries_dataset_id'] = queries_dataset
            else:
                training_args['queries_dataset_id'] = queries_dataset.id

        if not metric:
            metric = cls.default_metric
        if not top_k:
            top_k = cls.default_top_k
        training_args['metric'] = metric.value
        training_args['top_k'] = top_k
        training_args['lang'] = lang.value
        if isinstance(dataset, str):
            dataset_id = dataset
        else:
            dataset_id = dataset.id

        data = dict(name=name, dataset_id=dataset_id, **training_args)

        endpoint = '/projects/{}/{}/{}/{}'.format(project_id, 'usecases', cls.data_type.value, cls.training_type.value)
        start = client.request(endpoint,
                               method=requests.post,
                               data=data,
                               content_type='application/json',
                               message_prefix='Text similarity usecase start')

        start_response = parse_json(start)
        usecase = cls.from_id(start_response['_id'])
        events_url = '/{}/{}'.format(cls.resource, start_response['_id'])
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(usecase._id,
                                                cls.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', ('state', 'running')),
                                                specific_url=events_url)
        return usecase

    def new_version(
        self,
        description: str = None,
        dataset: Dataset = None,
        description_column_config: DescriptionsColumnConfig = None,
        metric: pio.metrics.TextSimilarity = None,
        top_k: int = None,
        lang: TextSimilarityLang = TextSimilarityLang.Auto,
        queries_dataset: Dataset = None,
        queries_column_config: Union[QueriesColumnConfig, None] = None,
        models_parameters: ListModelsParameters = None,
        **kwargs
    ) -> 'TextSimilarity':
        """ Start a text similarity usecase training to create a new version of the usecase (on the
        platform): the training configs are copied from the current version and then overridden
        for the given parameters.

        Args:
            description (str, optional): additional description of the version
            dataset (:class:`.Dataset`, :class:`.DatasetImages`, optional): Reference to the dataset
                object to use for as training dataset
            description_column_config (:class:`.DescriptionsColumnConfig`, optional): Column configuration for the
                usecase (see the documentation of the :class:`.ColumnConfig` resource for more details on each possible
                column types)
            metric (metrics.TextSimilarity, optional): Specific metric to use for the usecase (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
        Returns:
            :class:`.TextSimilarity`: Newly created text similarity usecase version object (new version)
        """

        if not dataset:
            dataset_id = self.dataset_id
        else:
            dataset_id = dataset.id

        if not description_column_config:
            description_column_config = self.description_column_config

        if not metric:
            metric = self.metric
        if not top_k:
            top_k = self.top_k
        if not lang:
            lang = self.lang
        if not queries_dataset:
            queries_dataset_id = self.queries_dataset
        else:
            queries_dataset_id = queries_dataset.id

        if not queries_column_config:
            queries_column_config = self.queries_column_config

        if not models_parameters:
            models_parameters = self.models_parameters

        training_args = to_json(description_column_config)
        assert isinstance(training_args, Dict)
        if queries_column_config:
            training_args.update(to_json(queries_column_config))
        training_args.update(to_json(models_parameters))

        if queries_dataset_id:
            training_args['queries_dataset_id'] = queries_dataset_id

        training_args['metric'] = metric.value
        training_args['top_k'] = top_k
        training_args['lang'] = lang.value

        data = dict(dataset_id=dataset_id, **training_args)

        if description:
            data["description"] = description

        endpoint = "/usecases/{}/versions".format(self.usecase_id)
        resp = client.request(endpoint=endpoint,
                              data=data,
                              method=requests.post,
                              content_type='application/json',
                              message_prefix='Text similarity usecase start')

        start_response = parse_json(resp)
        usecase = self.from_id(start_response['_id'])
        events_url = '/{}/{}'.format(self.resource, start_response['_id'])
        assert pio.client.event_manager is not None
        pio.client.event_manager.wait_for_event(usecase._id,
                                                self.resource,
                                                EventTuple('USECASE_VERSION_UPDATE', ('state', 'running')),
                                                specific_url=events_url)
        return usecase
