import copy
from enum import Enum
from typing import List, Union


def _drop_in_list(feature_list: List, param_to_drop: List):
    new = copy.copy(feature_list)
    for arg in param_to_drop:
        new.remove(arg.value)
    return new


class TypeProblem(Enum):
    """ Type of supervised problems available with Prevision.io. """
    Classification = 'classification'
    """Prediction using classification approach, for when the output variable is a category"""
    Regression = 'regression'
    """Prediction using regression problem, for when the output variable is a real or continuous value"""
    MultiClassification = 'multiclassification'
    """Prediction using classification approach, for when the output variable many categories"""
    TextSimilarity = 'text-similarity'
    """Ranking of texts by keywords"""
    ObjectDetection = "object-detection"
    """Detection of pattern in images"""


class DataType(Enum):
    """ Type of data available with Prevision.io. """
    Tabular = 'tabular'
    """Data arranged in a table"""
    TimeSeries = 'timeseries'
    """Data points indexed in time order"""
    Images = 'images'
    """Catalogue of images"""


class ExperimentState(Enum):
    """ Possible state of an Experiment in Prevision.io """
    Done = 'done'
    """The experiment finished properly"""
    Running = 'running'
    """The experiment is still running"""
    Failed = 'failed'
    """The experiment finished with an error"""
    Pending = 'pending'
    """The experiment is waiting for hardware ressources"""


class AdvancedModel(Enum):
    """ Types of normal models that can be trained with Prevision.io.
    The ``Full`` member is a shortcut to get all available models at once.
    To just drop a single model from a list of models, use:

    .. code-block:: python

        LiteModel.drop(LiteModel.xxx)
    """
    LightGBM = 'LGB'
    """LightGBM"""
    XGBoost = 'XGB'
    """XGBoost"""
    NeuralNet = 'NN'
    """NeuralNet"""
    ExtraTrees = 'ET'
    """ExtraTrees"""
    LinReg = 'LR'
    """Linear Regression"""
    RandomForest = 'RF'
    """Random Forest"""
    CatBoost = 'CB'
    """CatBoost"""
    Full = [LightGBM, XGBoost, NeuralNet, ExtraTrees, LinReg, RandomForest, CatBoost]
    """Evaluate all models"""


class NormalModel(Enum):
    """ Types of lite models that can be trained with Prevision.io.
    The ``Full`` member is a shortcut to get all available models at once.
    To just drop a single model from a list of models, use:

    .. code-block:: python

        Model.drop(Model.xxx)
    """
    LightGBM = 'LGB'
    """LightGBM"""
    XGBoost = 'XGB'
    """XGBoost"""
    NeuralNet = 'NN'
    """NeuralNet"""
    ExtraTrees = 'ET'
    """ExtraTrees"""
    LinReg = 'LR'
    """Linear Regression"""
    RandomForest = 'RF'
    """Random Forest"""
    NaiveBayesClassifier = 'NBC'
    """Random Forest"""
    CatBoost = 'CB'
    """CatBoost"""
    Full = [LightGBM, XGBoost, NeuralNet, ExtraTrees, LinReg, RandomForest, NaiveBayesClassifier, CatBoost]
    """Evaluate all models"""


class SimpleModel(Enum):
    """ Types of simple models that can be trained with Prevision.io.
    The ``Full`` member is a shortcut to get all available simple models at once.
    To just drop a single model from a list of simple models, use:

    .. code-block:: python

        SimpleModel.drop(SimpleModel.xxx)
    """
    DecisionTree = 'DT'
    """DecisionTree"""
    LinReg = 'LR'
    """Linear Regression"""
    Full = [DecisionTree, LinReg]
    """Evaluate all simple models"""


class Feature(Enum):
    """ Types of feature engineering that can be applied to a dataset with Prevision.io.
    The ``Full`` member is a shortcut to get all available feature engineering modules at once.
    To just drop a feature engineering module from a list of modules, use:

    .. code-block:: python

        Feature.drop(Feature.xxx)
    """
    Counts = 'Counter'
    """Value type counting"""
    DateTime = 'Date'
    """Date transformation"""
    Frequency = 'freq'
    """Frequency encoding"""
    TextTfidf = 'text_tfidf'
    """Statistical analysis"""
    TextWord2vect = 'text_word2vec'
    """Word embedding"""
    TextEmbedding = 'text_embedding'
    """Sentence embedding"""
    TargetEncoding = 'tenc'
    """Target encoding"""
    PolynomialFeatures = 'poly'
    """Polynomial feature"""
    PCA = 'pca'
    """Principal component analysis"""
    KMeans = 'kmean'
    """K-Means clustering"""
    Full = [Counts, DateTime, Frequency, TextTfidf,
            TextWord2vect, TextEmbedding, TargetEncoding,
            PolynomialFeatures, PCA, KMeans]
    """Full feature engineering"""


class Profile(Enum):
    """ Training profile type. """
    Quick = 'quick'
    """Quickest profile, lowest predictive performance"""
    Normal = 'normal'
    """Normal profile, best balance"""
    Advanced = 'advanced'
    """Slowest profile, for maximal optimization"""


class ExperimentConfig(object):

    config = {}

    @classmethod
    def from_dict(cls, kwargs_dict):
        reverse_dict = dict([(v, k) for k, v in cls.config.items()])

        class_args = {}

        for k, v in kwargs_dict.items():
            if k not in reverse_dict:
                continue

            if v == '' or not v:
                continue

            class_args[reverse_dict[k]] = v

        return cls(**class_args)


class TrainingConfig(ExperimentConfig):
    """ Training configuration that holds the relevant data for an experiment description:
    the wanted feature engineering, the selected models, the training speed...

    Args:
        profile (Profile): Type of training profile to use:

            - "quick": this profile runs very fast but has a lower performance
              (it is recommended for early trials)
            - "advanced": this profile runs slower but has increased performance
              (it is usually for optimization steps at the end of your project)
            - the "normal" profile is something in-between to help you investigate
              an interesting result

        advanced_models (list(AdvancedModel), optional): Names of the advanced models to use in the experiment
            (among: "LR", "RF", "ET", "XGB", "LGB", "CB" and "NN"). The advanced models will be
            hyperparametrized, resulting in a more accurate modelization at the cost of a longer
            training time.
        normal_models (list(NormalModel), optional): Names of the (normal) models to use in the experiment
            (among: "LR", "RF", "ET", "XGB", "LGB", "CB", 'NB' and "NN"). The normal models only
            use default parameters.
        simple_models (list(SimpleModel), optional): Names of the (simple) models to use in the experiment
            (among: "LR" and "DT"). These models are easy to ineterpret and fast to train but only
            offer a limited modelization complexity.
        features (list(Feature), optional): Names of the feature engineering modules to use (among:
            "Counter", "Date", "freq", "text_tfidf", "text_word2vec", "text_embedding", "tenc",
            "ee", "poly", "pca" and "kmean")
        with_blend (bool, optional): If true, Prevision.io's pipeline will add "blend" models
            at the end of the training by cherry-picking already trained models and fine-tuning
            hyperparameters (usually gives even better performance)
        feature_time_seconds (int, optional): feature selection take at most fsel_time in seconds
        feature_number_kept (int, optional): a feature selection algorithm is launched to keep at most
            `feature_number_kept` features
    """

    config = {
        'features': 'features_engineering_selected_list',
        'feature_time_seconds': 'features_selection_time',
        'feature_number_kept': 'features_selection_count',
        'advanced_models': 'normal_models',
        'normal_models': 'lite_models'
    }

    def __init__(self,
                 profile: Profile = Profile.Quick,
                 advanced_models: List[AdvancedModel] = [AdvancedModel.XGBoost, AdvancedModel.LinReg],
                 normal_models: List[NormalModel] = [NormalModel.XGBoost, NormalModel.LinReg],
                 simple_models: List[SimpleModel] = [],
                 features: List[Feature] = [Feature.Frequency, Feature.TargetEncoding, Feature.Counts],
                 with_blend: bool = False,
                 feature_time_seconds: int = 3600,
                 feature_number_kept: Union[int, None] = None):
        """

        Args:
            profile:
            normal_models:
            lite_models:
            simple_models:
            features (Feature, optional): Names of the feature engineering modules to use
            with_blend (bool, optional): models selectioned are also launched as blend
            feature_time_seconds (int, optional): feature selection take at most fsel_time in seconds
            feature_number_kept (int, optional): a feature selection algorithm is launched to keep at most
                `feature_number_kept` features
        """

        self.features = features

        self.advanced_models = advanced_models
        self.normal_models = normal_models
        self.simple_models = simple_models

        self.profile = profile
        self.with_blend = with_blend
        self.feature_time_seconds = feature_time_seconds

        if feature_number_kept:
            self.feature_number_kept = feature_number_kept


class YesOrNo(Enum):
    Yes = "yes"
    No = "no"


class YesOrNoOrAuto(Enum):
    Yes = "yes"
    No = "no"
    Auto = "auto"


class ColumnConfig(ExperimentConfig):
    """ Column configuration for starting an experiment: this object defines
    the role of specific columns in the dataset (and optionally the list of columns
    to drop).

    Args:
        target_column (str, optional): Name of the target column in the dataset
        id_column (str, optional): Name of the id column in the dataset that does
            not have any signal and will be ignored for computation
        fold_column (str, optional): Name of the fold column used that should be used to
            compute the various folds in the dataset
        weight_column (str, optional): Name of the weight column used to assign non-equal
            importance weights to the various rows in the dataset
        filename_column (str, optional): Name of the filename column in the dataset for
            an image-based experiment
        time_column (str, optional): Name of the time column in the dataset for a
            timeseries experiment
        group_columns (list(str), optional): Names of the columns in the dataset that define a
            unique time serie for a timeseries experiment
        apriori_columns (list(str), optional): Names of the columns that are known *a priori* in
            the dataset for a timeseries experiment
        drop_list (list(str), optional): Names of all the columns that should be dropped
            from the dataset while training the experiment
    """

    config = {
        'group_columns': 'group_list',
        'apriori_columns': 'apriori_list',
        'drop_list': 'drop_list',
    }

    def __init__(self,
                 target_column: Union[str, None] = None,
                 filename_column: Union[str, None] = None,
                 id_column: Union[str, None] = None,
                 fold_column: Union[str, None] = None,
                 weight_column: Union[str, None] = None,
                 time_column: Union[str, None] = None,
                 group_columns: Union[List[str], None] = None,
                 apriori_columns: Union[List[str], None] = None,
                 drop_list: Union[List[str], None] = None):
        self.target_column: Union[str, None] = target_column
        self.filename_column: Union[str, None] = filename_column
        self.id_column: Union[str, None] = id_column
        self.fold_column: Union[str, None] = fold_column
        self.weight_column: Union[str, None] = weight_column
        self.time_column: Union[str, None] = time_column
        self.group_columns: Union[List[str], None] = group_columns
        self.apriori_columns: Union[List[str], None] = apriori_columns
        self.drop_list: Union[List[str], None] = drop_list


base_config = TrainingConfig(profile=Profile.Normal,
                             advanced_models=AdvancedModel.Full.value,
                             normal_models=NormalModel.Full.value,
                             simple_models=SimpleModel.Full.value,
                             features=_drop_in_list(Feature.Full.value, [Feature.PCA, Feature.KMeans]),
                             with_blend=True)

quick_config = TrainingConfig(profile=Profile.Quick,
                              advanced_models=_drop_in_list(AdvancedModel.Full.value, [AdvancedModel.NeuralNet]),
                              normal_models=_drop_in_list(NormalModel.Full.value, [NormalModel.NeuralNet]),
                              simple_models=_drop_in_list(SimpleModel.Full.value, [SimpleModel.LinReg]),
                              features=_drop_in_list(Feature.Full.value, [Feature.PCA, Feature.KMeans]),
                              with_blend=False)

ultra_config = TrainingConfig(profile=Profile.Quick,
                              features=_drop_in_list(Feature.Full.value, [Feature.PCA, Feature.KMeans]),
                              advanced_models=[AdvancedModel.XGBoost],
                              normal_models=[NormalModel.XGBoost],
                              simple_models=[SimpleModel.LinReg],
                              with_blend=False)

nano_config = TrainingConfig(profile=Profile.Quick,
                             advanced_models=[AdvancedModel.LinReg],
                             normal_models=[NormalModel.LinReg],
                             simple_models=[],
                             features=[],
                             with_blend=False)
