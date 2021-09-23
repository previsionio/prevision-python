from enum import Enum
from typing import Dict, List, Union
from previsionio.dataset import Dataset
from .experiment_config import DataType, ExperimentConfig, TypeProblem, YesOrNo, YesOrNoOrAuto
from .utils import to_json
from .model import TextSimilarityModel
from .experiment_version import BaseExperimentVersion
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
    IVFOPQ = 'ivf_opq'
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


class ModelsParameters(ExperimentConfig):
    """ Training configuration that holds the relevant data for an experiment description:
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
            "brute_force", "cluster_pruning", "ivf_opq", "hkm", "lsh").
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


class ListModelsParameters(ExperimentConfig):

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
                                                   # [TextSimilarityModels.BruteForce, TextSimilarityModels.IVFOPQ])
                                                   [TextSimilarityModels.BruteForce])
            models_parameters_3 = ModelsParameters(ModelEmbedding.TransformerFineTuned,
                                                   Preprocessing(),
                                                   #" [TextSimilarityModels.BruteForce, TextSimilarityModels.IVFOPQ])
                                                   [TextSimilarityModels.BruteForce])
            models_parameters = [models_parameters_1, models_parameters_2, models_parameters_3]
        self.models_parameters = []
        for element in models_parameters:
            if isinstance(element, ModelsParameters):
                self.models_parameters.append(element)
            else:
                self.models_parameters.append(ModelsParameters(**element))


class DescriptionsColumnConfig(ExperimentConfig):
    """ Description Column configuration for starting an experiment: this object defines
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


class QueriesColumnConfig(ExperimentConfig):
    """ Description Column configuration for starting an experiment: this object defines
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


class TextSimilarity(BaseExperimentVersion):
    """ A text similarity experiment version """

    default_metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity.accuracy_at_k
    default_top_k: int = 10
    data_type: DataType = DataType.Tabular
    training_type: TypeProblem = TypeProblem.TextSimilarity
    resource: str = 'experiment-versions'
    model_class = TextSimilarityModel

    def __init__(self, **experiment_version_info):
        super().__init__(**experiment_version_info)

        self.predictions = {}
        self.predict_token = None

    def _update_from_dict(self, **experiment_version_info):
        super()._update_from_dict(**experiment_version_info)

        experiment_version_params = experiment_version_info['experiment_version_params']

        dataset_id: str = experiment_version_info['dataset_id']
        self.dataset: Dataset = Dataset.from_id(dataset_id)
        self.description_column_config = DescriptionsColumnConfig(
            content_column=experiment_version_params.get('content_column'),
            id_column=experiment_version_params.get('id_column'))
        self.metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity(
            experiment_version_params.get('metric', self.default_metric))
        self.top_k: int = experiment_version_params.get('top_K', self.default_top_k)
        self.lang: TextSimilarityLang = TextSimilarityLang(experiment_version_params.get('lang', TextSimilarityLang.Auto))

        if experiment_version_info.get('queries_dataset_id'):
            queries_dataset_id = experiment_version_info['queries_dataset_id']
            self.queries_dataset: Dataset = Dataset.from_id(queries_dataset_id)
            content_column = experiment_version_params.get('queries_dataset_content_column')
            matching_id = experiment_version_params.get('queries_dataset_matching_id_description_column')
            queries_dataset_id_column = experiment_version_params.get('queries_dataset_id_column', None)
            self.queries_column_config = QueriesColumnConfig(queries_dataset_content_column=content_column,
                                                             queries_dataset_matching_id_description_column=matching_id,
                                                             queries_dataset_id_column=queries_dataset_id_column)
        else:
            self.queries_dataset = None
            self.queries_column_config = None

        models_parameters = experiment_version_params.get('models_params')
        self.models_parameters = ListModelsParameters(models_parameters=models_parameters)

    @classmethod
    def from_id(cls, _id: str) -> 'TextSimilarity':
        return cls(**super()._from_id(_id))

    @classmethod
    def load(cls, pio_file: str) -> 'TextSimilarity':
        return cls(**super()._load(pio_file))

    @staticmethod
    def _build_experiment_version_creation_data(description, dataset, description_column_config, metric,
                                             top_k, lang, queries_dataset, queries_column_config,
                                             models_parameters,
                                             parent_version=None) -> Dict:
        data = super(TextSimilarity, TextSimilarity)._build_experiment_version_creation_data(
            description,
            parent_version=parent_version,
        )

        data['dataset_id'] = dataset.id
        data.update(to_json(description_column_config))
        data['metric'] = metric if isinstance(metric, str) else metric.value
        data['top_k'] = top_k
        data['lang'] = lang if isinstance(lang, str) else lang.value
        if queries_dataset is not None:
            if queries_column_config is None:
                raise ValueError('arg queries_column_config must be set if queries_dataset is set')
            data['queries_dataset_id'] = queries_dataset.id
            data.update(to_json(queries_column_config))
        data.update(to_json(models_parameters))

        return data

    @classmethod
    def _fit(
        cls,
        experiment_id: str,
        dataset: Dataset,
        description_column_config: DescriptionsColumnConfig,
        metric: pio.metrics.TextSimilarity = pio.metrics.TextSimilarity.accuracy_at_k,
        top_k: int = 10,
        lang: TextSimilarityLang = TextSimilarityLang.Auto,
        queries_dataset: Dataset = None,
        queries_column_config: Union[QueriesColumnConfig, None] = None,
        models_parameters: ListModelsParameters = ListModelsParameters(),
        description: str = None,
        parent_version: str = None,
    ) -> 'TextSimilarity':
        """ Start a supervised experiment training with a specific training configuration
        (on the platform).

        Args:
            name (str): Name of the experiment to create
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            description_column_config (:class:`.DescriptionsColumnConfig`): Description column configuration
                (see the documentation of the :class:`.DescriptionsColumnConfig` resource for more details
                on each possible column types)
            metric (str, optional): Specific metric to use for the experiment (default: ``None``)
            queries_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a queries dataset (default: ``None``)

        Returns:
            :class:`.TextSimilarity`: Newly created supervised experiment object
        """

        return super()._fit(
            experiment_id,
            description=description,
            parent_version=parent_version,
            dataset=dataset,
            description_column_config=description_column_config,
            metric=metric,
            top_k=top_k,
            lang=lang,
            queries_dataset=queries_dataset,
            queries_column_config=queries_column_config,
            models_parameters=models_parameters,
        )

    def new_version(
        self,
        dataset: Dataset = None,
        description_column_config: DescriptionsColumnConfig = None,
        metric: pio.metrics.TextSimilarity = None,
        top_k: int = None,
        lang: TextSimilarityLang = None,
        queries_dataset: Dataset = None,
        queries_column_config: Union[QueriesColumnConfig, None] = None,
        models_parameters: ListModelsParameters = None,
        description: str = None,
    ) -> 'TextSimilarity':
        """ Start a text similarity experiment training to create a new version of the experiment (on the
        platform): the training configs are copied from the current version and then overridden
        for the given parameters.

        Args:
            description (str, optional): additional description of the version
            dataset (:class:`.Dataset`, :class:`.DatasetImages`, optional): Reference to the dataset
                object to use for as training dataset
            description_column_config (:class:`.DescriptionsColumnConfig`, optional): Column configuration for the
                experiment (see the documentation of the :class:`.ColumnConfig` resource for more details on each possible
                column types)
            metric (metrics.TextSimilarity, optional): Specific metric to use for the experiment (default: ``None``)
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
        Returns:
            :class:`.TextSimilarity`: Newly created text similarity experiment version object (new version)
        """
        return TextSimilarity._fit(
            self.experiment_id,
            dataset if dataset is not None else self.dataset,
            description_column_config if description_column_config is not None else self.description_column_config,
            metric=metric if metric is not None else self.metric,
            top_k=top_k if top_k is not None else self.top_k,
            lang=lang if lang is not None else self.lang,
            queries_dataset=queries_dataset if queries_dataset is not None else self.queries_dataset,
            queries_column_config=queries_column_config if queries_column_config is not None else self.queries_column_config,
            models_parameters=models_parameters if models_parameters is not None else self.models_parameters,
            description=description,
            parent_version=self.version,
        )
