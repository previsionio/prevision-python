from __future__ import print_function
import logging
from previsionio.logger import logger, event_logger


__version__ = '11.1.0'


def verbose(v, debug: bool = False, event_log: bool = False):
    """ Set the SDK level of verbosity.

    Args:
        v (bool): Whether to activate info logging
        debug (bool, optional): Whether to activate detailed
            debug logging (default: ``False``)
        event_log (bool, optional): Whether to activate detailed
            event managers debug logging (default: ``False``)
    """
    if event_log:
        event_logger.setLevel(logging.DEBUG)

    if v:
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


verbose(False)


class Config:
    def __init__(self):
        self.auto_update: bool = True
        self.zip_files: bool = True
        self.success_codes: list = list(range(200, 211)) + [226]
        self.retry_codes: list = [500, 502, 503, 504]
        self.request_retries: int = 6
        self.request_retry_time: int = 10
        self.scheduler_refresh_rate: int = 10
        self.default_timeout: float = 3600.


config = Config()

from previsionio.prevision_client import client
from previsionio.usecase_config import (
    AdvancedModel,
    NormalModel,
    SimpleModel,
    TypeProblem,
    Feature,
    TrainingConfig,
    Profile,
    base_config,
    quick_config,
    ultra_config,
    ColumnConfig,
    YesOrNo,
    YesOrNoOrAuto,
)

import previsionio.metrics as metrics

from previsionio.connector import Connector
from previsionio.datasource import DataSource
from previsionio.project import Project
from previsionio.usecase import Usecase
from previsionio.supervised import Supervised
from previsionio.timeseries import (
    TimeSeries,
    TimeWindow,
    TimeWindowException,
)
from previsionio.model import (
    Model,
    ClassificationModel,
    RegressionModel,
    MultiClassificationModel,
    TextSimilarityModel,
)
from previsionio.text_similarity import (
    TextSimilarity,
    DescriptionsColumnConfig,
    QueriesColumnConfig,
    ListModelsParameters,
    ModelsParameters,
    TextSimilarityModels,
    ModelEmbedding,
    Preprocessing,
)
from previsionio.dataset import Dataset, DatasetImages
from previsionio.analyzer import cv_classif_analysis
from previsionio.deployed_model import DeployedModel
from previsionio.usecase_deployment import UsecaseDeployment
from previsionio.prediction import ValidationPrediction, DeploymentPrediction

__all__ = [
    'client',
    'AdvancedModel',
    'NormalModel',
    'SimpleModel',
    'TypeProblem',
    'Feature',
    'TrainingConfig',
    'YesOrNo',
    'YesOrNoOrAuto',
    'Profile',
    'base_config',
    'quick_config',
    'ultra_config',
    'ColumnConfig',
    'metrics',
    'Connector',
    'DataSource',
    'Supervised',
    'TimeSeries',
    'TimeWindow',
    'TimeWindowException',
    'Dataset',
    'DatasetImages',
    'Model',
    'RegressionModel',
    'ClassificationModel',
    'MultiClassificationModel',
    'TextSimilarityModel',
    'cv_classif_analysis',
    'DeployedModel',
    'TextSimilarity',
    'DescriptionsColumnConfig',
    'QueriesColumnConfig',
    'ListModelsParameters',
    'ModelsParameters',
    'TextSimilarityModels',
    'ModelEmbedding',
    'Preprocessing',
    'Project',
    'Usecase',
    'UsecaseDeployment',
    'ValidationPrediction',
    'DeploymentPrediction'
]
