import copy
from typing import List, Union


class ParamList(list):

    """ A list of params to be passed to a usecase. """

    def drop(self, *args):
        new = copy.copy(self)
        for arg in args:
            new.remove(arg)
        return new


class TypeProblem(object):
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


class DataType(object):
    """ Type of data available with Prevision.io. """
    Tabular = 'tabular'
    """Data arranged in a table"""
    TimeSeries = 'timeseries'
    """Data points indexed in time order"""
    Images = 'images'
    """Catalogue of images"""


class AdvancedModel(object):
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
    Full = ParamList(['LGB', 'XGB', 'NN', 'ET', 'LR', 'RF', 'CB'])
    """Evaluate all models"""


class NormalModel(object):
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
    Full = ParamList(['LGB', 'XGB', 'NN', 'ET', 'LR', 'RF', 'NBC', 'CB'])
    """Evaluate all models"""


class SimpleModel(object):
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
    Full = ParamList(['DT', 'LR'])
    """Evaluate all simple models"""


class Feature(object):
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
    Full = ParamList(['Counter', 'Date', 'freq', 'text_tfidf',
                      'text_word2vec', 'text_embedding', 'tenc',
                      'poly', 'pca', 'kmean'])
    """Full feature engineering"""


class Profile(object):
    """ Training profile type. """
    Quick = 'quick'
    """Quickest profile, lowest predictive performance"""
    Normal = 'normal'
    """Normal profile, best balance"""
    Advanced = 'advanced'
    """Slowest profile, for maximal optimization"""


class UsecaseConfig(object):

    list_args = {'fe_selected_list', 'drop_list', 'normal_models'}

    config = {}

    def to_kwargs(self):
        kwargs = []
        for key, value in self.__dict__.items():
            if not value:
                continue
            if key not in self.config:
                kwargs.append((key, value))
            elif isinstance(value, list):
                kwargs.append((self.config[key], value))
            elif isinstance(value, bool):
                kwargs.append((self.config[key], str(value).lower()))
            elif isinstance(value, int):
                kwargs.append((self.config[key], value))
            else:
                kwargs.append((self.config[key], str(value)))
        return kwargs

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


class TrainingConfig(UsecaseConfig):
    """ Training configuration that holds the relevant data for a usecase description:
    the wanted feature engineering, the selected models, the training speed...

    Args:
        profile (str): Type of training profile to use:

            - "quick": this profile runs very fast but has a lower performance
              (it is recommended for early trials)
            - "advanced": this profile runs slower but has increased performance
              (it is usually for optimization steps at the end of your project)
            - the "normal" profile is something in-between to help you investigate
              an interesting result

        advanced_models (list(str), optional): Names of the advanced models to use in the usecase
            (among: "LR", "RF", "ET", "XGB", "LGB", "CB" and "NN"). The advanced models will be
            hyperparametrized, resulting in a more accurate modelization at the cost of a longer
            training time.
        normal_models (list(str), optional): Names of the (normal) models to use in the usecase
            (among: "LR", "RF", "ET", "XGB", "LGB", "CB", 'NB' and "NN"). The normal models only
            use default parameters.
        simple_models (list(str), optional): Names of the (simple) models to use in the usecase
            (among: "LR" and "DT"). These models are easy to ineterpret and fast to train but only
            offer a limited modelization complexity.
        features (list(str), optional): Names of the feature engineering modules to use (among:
            "Counter", "Date", "freq", "text_tfidf", "text_word2vec", "text_embedding", "tenc",
            "ee", "poly", "pca" and "kmean")
        with_blend (bool, optional): If true, Prevision.io's pipeline will add "blend" models
            at the end of the training by cherry-picking already trained models and fine-tuning
            hyperparameters (usually gives even better performance)
        fe_selected_list (list(str), optional): Override for the features list, to restrict it
            only this list
    """

    config = {
        'fe_selected_list': 'features_engineering_selected_list'
    }

    def __init__(self,
                 profile=Profile.Quick,
                 advanced_models=[AdvancedModel.XGBoost, AdvancedModel.LinReg],
                 normal_models=[NormalModel.XGBoost, NormalModel.LinReg],
                 simple_models=[],
                 features=[Feature.Frequency, Feature.TargetEncoding, Feature.Counts],
                 with_blend=False,
                 fe_selected_list=[]):
        """

        Args:
            profile:
            normal_models:
            lite_models:
            simple_models:
            features:
            with_blend:
            fe_selected_list:
        """

        if fe_selected_list:
            self.fe_selected_list = fe_selected_list
        else:
            self.fe_selected_list = [f for f in Feature.Full if f in features]

        self.normal_models = advanced_models
        self.lite_models = normal_models
        self.simple_models = simple_models

        self.profile = profile
        self.with_blend = with_blend


class ColumnConfig(UsecaseConfig):
    """ Column configuration for starting a usecase: this object defines
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
            an image-based usecase
        time_column (str, optional): Name of the time column in the dataset for a
            timeseries usecase
        group_columns (list(str), optional): Names of the columns in the dataset that define a
            unique time serie for a timeseries usecase
        apriori_columns (list(str), optional): Names of the columns that are known *a priori* in
            the dataset for a timeseries usecase
        drop_list (list(str), optional): Names of all the columns that should be dropped
            from the dataset while training the usecase
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
                             advanced_models=AdvancedModel.Full,
                             normal_models=NormalModel.Full,
                             simple_models=SimpleModel.Full,
                             features=Feature.Full.drop(Feature.PCA, Feature.KMeans),
                             with_blend=True)

quick_config = TrainingConfig(profile=Profile.Quick,
                              advanced_models=AdvancedModel.Full.drop(AdvancedModel.NeuralNet),
                              normal_models=NormalModel.Full.drop(NormalModel.NeuralNet),
                              simple_models=SimpleModel.Full.drop(SimpleModel.LinReg),
                              features=Feature.Full.drop(Feature.PCA, Feature.KMeans),
                              with_blend=False)

ultra_config = TrainingConfig(profile=Profile.Quick,
                              features=Feature.Full.drop(Feature.PCA, Feature.KMeans),
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
