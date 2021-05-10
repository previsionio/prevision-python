from __future__ import print_function
import logging
from .logger import logger, event_logger


__version__ = '10.21.1'


def verbose(v, debug=False, event_log=False):
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
        self.auto_update = True
        self.zip_files = True
        self.request_retries = 3
        self.request_retry_time = 10
        self.scheduler_refresh_rate = 10
        self.default_timeout = 10800


config = Config()

from previsionio.prevision_client import client
from previsionio.usecase_config import \
    Model, \
    SimpleModel, \
    TypeProblem, \
    Feature, \
    TrainingConfig, \
    Profile, \
    base_config, \
    quick_config, \
    ultra_config, \
    ClusterMagnitude, \
    ClusteringTrainingConfig, \
    clustering_base_config, \
    ColumnConfig

import previsionio.metrics as metrics

from previsionio.connector import Connector
from previsionio.datasource import DataSource
from previsionio.supervised import Supervised, SupervisedImages, \
    Regression, Classification, MultiClassification, \
    RegressionImages, ClassificationImages, MultiClassificationImages
from previsionio.timeseries import TimeSeries, TimeWindow, TimeWindowException

from previsionio.text_similarity import TextSimilarity, DescriptionsColumnConfig, \
    QueriesColumnConfig, ListModelsParameters, ModelsParameters, TextSimilarityModels, ModelEmbedding, Preprocessing
from previsionio.dataset import Dataset, DatasetImages
# from previsionio.experiment import Experiment
from previsionio.plotter import PrevisionioPlotter, PlotlyPlotter, MatplotlibPlotter
from previsionio.analyzer import cv_classif_analysis
from previsionio.deployed_model import DeployedModel

__all__ = ['client',
           'Model',
           'SimpleModel',
           'TypeProblem',
           'Feature',
           'TrainingConfig',
           'Profile',
           'base_config',
           'quick_config',
           'ultra_config',
           'ClusterMagnitude',
           'ClusteringTrainingConfig',
           'clustering_base_config',
           'ColumnConfig',
           'metrics',
           'Connector',
           'DataSource',
           'Supervised',
           'SupervisedImages',
           'Regression',
           'RegressionImages',
           'Classification',
           'ClassificationImages',
           'MultiClassification',
           'MultiClassificationImages',
           'TimeSeries',
           'TimeWindow',
           'TimeWindowException',
           'Dataset',
           'DatasetImages',
           'PrevisionioPlotter',
           'PlotlyPlotter',
           'MatplotlibPlotter',
           'cv_classif_analysis',
           'DeployedModel',
           'TextSimilarity',
           'DescriptionsColumnConfig',
           'QueriesColumnConfig',
           'ListModelsParameters',
           'ModelsParameters',
           'TextSimilarityModels',
           'ModelEmbedding',
           'Preprocessing'
           ]
