import logging
from previsionio.logger import logger, event_logger


__version__ = '11.2.0'


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
from previsionio.experiment_config import (
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

from previsionio.connector import Connector
from previsionio.datasource import DataSource
from previsionio.exporter import Exporter, ExporterWriteMode
from previsionio.export import Export
from previsionio.project import Project
from previsionio.experiment import Experiment
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
from previsionio.deployed_model import DeployedModel
from previsionio.experiment_deployment import ExperimentDeployment
from previsionio.prediction import ValidationPrediction, DeploymentPrediction

from previsionio.pipeline import PipelineScheduledRun

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
    'Connector',
    'DataSource',
    'Exporter',
    'ExporterWriteMode',
    'Export',
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
    'Experiment',
    'ExperimentDeployment',
    'ValidationPrediction',
    'DeploymentPrediction',
    'PipelineScheduledRun'
]

from . import _version
__version__ = _version.get_versions()['version']
