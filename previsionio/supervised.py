# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Union

from .dataset import Dataset, DatasetImages
from .experiment_config import DataType, TypeProblem, TrainingConfig, ColumnConfig
from . import metrics
from .experiment_version import ClassicExperimentVersion
from .model import RegressionModel, ClassificationModel, MultiClassificationModel
from .utils import to_json

MODEL_CLASS_DICT = {
    TypeProblem.Regression: RegressionModel,
    TypeProblem.Classification: ClassificationModel,
    TypeProblem.MultiClassification: MultiClassificationModel
}


class Supervised(ClassicExperimentVersion):

    """ A supervised experiment version, for tabular data """

    data_type = DataType.Tabular

    def __init__(self, **experiment_version_info):
        super().__init__(**experiment_version_info)

    def _update_from_dict(self, **experiment_version_info):
        super()._update_from_dict(**experiment_version_info)
        if 'folder_dataset_id' in experiment_version_info:
            self.dataset_images: DatasetImages = DatasetImages.from_id(experiment_version_info['folder_dataset_id'])
        else:
            self.dataset_images = None
        self.model_class = MODEL_CLASS_DICT.get(self.training_type, RegressionModel)

    @classmethod
    def from_id(cls, _id: str) -> 'Supervised':
        """Get a supervised experiment version from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment version to retrieve

        Returns:
            :class:`.Supervised`: Fetched experiment version

        Raises:
            PrevisionException: Any error while fetching data from the platform or parsing result
        """
        return cls(**super()._from_id(_id))

    @staticmethod
    def _build_experiment_version_creation_data(description, dataset, column_config, metric,
                                             holdout_dataset, training_config,
                                             parent_version=None) -> Dict:
        data = super(Supervised, Supervised)._build_experiment_version_creation_data(
            description,
            parent_version=parent_version,
        )

        # because if Image there is the dataset and the images zip
        if isinstance(dataset, tuple):
            data['dataset_id'], data['folder_dataset_id'] = dataset[0].id, dataset[1].id
        else:
            data['dataset_id'] = dataset.id

        data.update(to_json(column_config))
        data['metric'] = metric if isinstance(metric, str) else metric.value
        data['holdout_dataset_id'] = holdout_dataset.id if holdout_dataset is not None else None
        data.update(to_json(training_config))

        return data

    @classmethod
    def _fit(
        cls,
        experiment_id: str,
        dataset: Union[Dataset, Tuple[Dataset, DatasetImages]],
        column_config: ColumnConfig,
        metric: metrics.Enum,
        holdout_dataset: Dataset = None,
        training_config: TrainingConfig = TrainingConfig(),
        description: str = None,
        parent_version: str = None,
    ) -> 'Supervised':
        """ Start a supervised experiment version training with a specific configuration (on the platform).

        Args:
            experiment_id (str): The id of the experiment from which this version is created
            dataset (:class:`.Dataset`, :class:`.DatasetImages`): Reference to the dataset(s)
                object(s) to use for as training dataset(s)
            column_config (:class:`.ColumnConfig`): Column configuration for the experiment
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (metrics.Enum): Specific metric to use for the experiment version
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
            description (str, optional): The description of this experiment version (default: ``None``)
            parent_version (str, optional): The parent version of this experiment_version (default: ``None``)


        Returns:
            :class:`.Supervised`: Newly created supervised experiment version object
        """
        return super()._fit(
            experiment_id,
            description=description,
            parent_version=parent_version,
            dataset=dataset,
            column_config=column_config,
            metric=metric,
            holdout_dataset=holdout_dataset,
            training_config=training_config,
        )

    def new_version(
        self,
        dataset: Dataset = None,
        dataset_images: DatasetImages = None,
        column_config: ColumnConfig = None,
        metric: metrics.Enum = None,
        holdout_dataset: Dataset = None,
        training_config: TrainingConfig = None,
        description: str = None,
    ) -> 'Supervised':
        """ Start a supervised experiment version training from this version to create a new version of the experiment
        (on the platform). The training parameters are copied from the current version and then overridden
        for the given parameters.

        Args:
            dataset (:class:`.Dataset`): Reference to the dataset
                object to use for as training dataset
            dataset_images (:class:`.DatasetImages`): Reference to the images dataset
                object to use for as training dataset
            column_config (:class:`.ColumnConfig`, optional): Column configuration for the experiment
                (see the documentation of the :class:`.ColumnConfig` resource for more details
                on each possible column types)
            metric (metrics.Enum, optional): Specific metric to use for the experiment version
            holdout_dataset (:class:`.Dataset`, optional): Reference to a dataset object to
                use as a holdout dataset (default: ``None``)
            training_config (:class:`.TrainingConfig`, optional): Specific training configuration
                (see the documentation of the :class:`.TrainingConfig` resource for more details
                on all the parameters)
            description (str, optional): The description of this experiment version (default: ``None``)
        Returns:
            :class:`.Supervised`: Newly created supervised experiment object (new version)
        """
        dataset = dataset if dataset is not None else self.dataset
        dataset_images = dataset_images if dataset_images is not None else self.dataset_images
        dataset = (dataset, dataset_images)
        return Supervised._fit(
            self.experiment_id,
            dataset,
            column_config if column_config is not None else self.column_config,
            metric if metric is not None else self.metric,
            holdout_dataset=holdout_dataset if holdout_dataset is not None else self.holdout_dataset,
            training_config=training_config if training_config is not None else self.training_config,
            description=description,
            parent_version=self.version,
        )

    def _save_json(self):
        json_dict = {
            '_id': self.id,
            'experiment_version_params': self._status.get('experiment_version_params', {}),
            'dataset_id': self._status['dataset_id'],
            'training_type': self.training_type.value,
            'data_type': self.data_type.value,
        }
        if self.holdout_dataset_id:
            json_dict['holdout_id'] = self.holdout_dataset_id
        return json_dict
