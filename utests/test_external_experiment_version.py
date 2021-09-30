import os
import uuid
import pandas as pd
import pytest

import previsionio as pio
from previsionio.project import Project
from previsionio.experiment import Experiment
from previsionio.external_experiment_version import ExternalExperimentVersion
from previsionio.model import (Model,
                               ExternalRegressionModel, ExternalClassificationModel, ExternalMultiClassificationModel)


# NOTE: use Unittest classes instead of assert everywhere...


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_external_models')
TESTING_ID = str(uuid.uuid4())

pio.config.zip_files = False
pio.config.default_timeout = 1000

type_problems = [
    'regression',
    #'classification',
    #"'multiclassification',
]
type_problem_2_projet_experiment_version_creation_method_name = {
    type_problem: f'create_external_{type_problem}' for type_problem in type_problems
}

TEST_DATASETS_PATH = {
    'regression': os.path.join(DATA_PATH, 'regression_holdout_dataset.csv'),
    #'classification': os.path.join(DATA_PATH, 'classification_holdout_dataset.csv'),
    #'multiclassification': os.path.join(DATA_PATH, 'multiclassification_holdout_dataset.csv'),
}

TEST_EXTERNAL_MODELS = {
    'regression': (
        'test_sdk_regression_external_model',
        os.path.join(DATA_PATH, 'regression_model.onnx'),
        os.path.join(DATA_PATH, 'regression_model.yaml'),
    ),
    'classification': (
        'test_sdk_classification_external_model',
        os.path.join(DATA_PATH, 'classification_model.onnx'),
        os.path.join(DATA_PATH, 'classification_model.yaml'),
    ),
    'multiclassification': (
        'test_sdk_multiclassification_external_model',
        os.path.join(DATA_PATH, 'multiclassification_model.onnx'),
        os.path.join(DATA_PATH, 'multiclassification_model.yaml'),
    ),
}


TEST_PIO_DATASETS = {}


def make_pio_datasets():
    for problem_type, p in TEST_DATASETS_PATH.items():
        project = Project.from_id(PROJECT_ID)
        dataset = project.create_dataset(p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                         dataframe=pd.read_csv(p))
        TEST_PIO_DATASETS[problem_type] = dataset


def setup_module(module):
    project_name = f'project_sdk_TEST_EXTERNAL_MODELS_{TESTING_ID}'
    project = Project.new(name=project_name,
                          description="description_sdk_TEST_EXTERNAL_MODELS")
    global PROJECT_ID
    PROJECT_ID = project.id
    make_pio_datasets()


def teardown_module(module):
    project = Project.from_id(PROJECT_ID)
    project.delete()


def create_external_experiment_version(
    project_id,
    type_problem,
    experiment_name,
    external_models,
    target_column='TARGET',  # because all our utests datasets has a target_column named 'TARGET'
    dataset=None,
    experiment_version_description=None,
) -> ExternalExperimentVersion:

    project = Project.from_id(project_id)
    holdout_dataset = TEST_PIO_DATASETS[type_problem]
    experiment_version_creation_method_name =\
        type_problem_2_projet_experiment_version_creation_method_name[type_problem]
    experiment_version_creation_method = getattr(project, experiment_version_creation_method_name)

    # NOTE: should allow to change the default metric
    return experiment_version_creation_method(
        experiment_name,
        holdout_dataset,
        target_column,
        external_models,
        dataset=dataset,
        experiment_version_description=experiment_version_description,
    )


# create an external experiment version only from type_problem, all other args are default
def create_external_experiment_version_from_type_problem(type_problem, experiment_name=None):
    if experiment_name is None:
        experiment_name = f'test_sdk_external_models_{type_problem}_{TESTING_ID}'

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    experiment_version_description = f'description_version_{experiment_name}'
    experiment_version: ExternalExperimentVersion = create_external_experiment_version(
        PROJECT_ID,
        type_problem,
        experiment_name,
        external_models,
        experiment_version_description=experiment_version_description,
    )
    return experiment_version


def test_experiment_version():
    experiment_name = f'test_sdk_external_models_test_experiment_version_{TESTING_ID}'
    type_problem = 'regression'
    experiment_version = create_external_experiment_version_from_type_problem(type_problem,
                                                                              experiment_name=experiment_name)

    experiment_id = experiment_version.experiment_id
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id in [experiment.id for experiment in experiments]

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    experiment_version_new = experiment_version.new_version(external_models)

    experiment_versions = Experiment.from_id(experiment_id).versions
    assert experiment_version_new.id in [experiment_version.id for experiment_version in experiment_versions]

    Experiment.from_id(experiment_id).delete()
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id not in [experiment.id for experiment in experiments]


def test_experiment_latest_versions():
    experiment_name = f'test_sdk_external_models_test_experiment_latest_versions_{TESTING_ID}'
    type_problem = 'classification'
    experiment_version = create_external_experiment_version_from_type_problem(type_problem,
                                                                              experiment_name=experiment_name)

    experiment_id = experiment_version.experiment_id
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id in [experiment.id for experiment in experiments]

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    experiment_version_new = experiment_version.new_version(external_models)
    assert experiment_version.id != experiment_version_new.id
    assert experiment_version.experiment_id == experiment_version_new.experiment_id
    assert experiment_version.project_id == experiment_version_new.project_id

    latest_version = Experiment.from_id(experiment_version_new.experiment_id).latest_version
    assert experiment_version_new._id == latest_version._id
    latest_version.new_version(external_models)

    Experiment.from_id(experiment_version_new.experiment_id).delete()
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id not in [experiment.id for experiment in experiments]


def test_stop_running_experiment_version():
    experiment_name = f'test_sdk_external_models_test_stop_running_experiment_version_{TESTING_ID}'
    type_problem = 'multiclassification'
    experiment_version = create_external_experiment_version_from_type_problem(type_problem,
                                                                              experiment_name=experiment_name)

    experiment_id = experiment_version.experiment_id
    assert experiment_version.running
    experiment_version.stop()
    assert not experiment_version.running
    Experiment.from_id(experiment_version.experiment_id).delete()
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id not in [experiment.id for experiment in experiments]


@pytest.fixture(scope='module', params=type_problems)
def setup_experiment_class(request):
    type_problem = request.param
    experiment_name = f'test_sdk_external_models_{type_problem}_{TESTING_ID}'
    experiment_version = create_external_experiment_version_from_type_problem(type_problem,
                                                                              experiment_name=experiment_name)
    assert experiment_version.running
    experiment_version.wait_until(lambda experiment: experiment.done or experiment._status['state'] == 'failed')
    assert not experiment_version.running
    assert experiment_version.done
    yield type_problem, experiment_version
    Experiment.from_id(experiment_version.experiment_id).delete()


# NOTE: copy paste from test_supervised.py, atm test nothing else than experiment version launching and
#       stop, which is a part of TestPredict and other tests
"""
class TestExperimentVersionGeneric:

    def test_check_config(self, setup_experiment_class):
        training_type, experiment_version = setup_experiment_class
        # assert all([c in experiment_version.drop_list for c in DROP_COLS])
        # experiment_version.update_status()
        # assert set(experiment_version.feature_list) == set(experiment_version_config.features)
        # assert set(experiment_version.advanced_models_list) == set(experiment_version_config.advanced_models)
        # assert set(experiment_version.simple_models_list) == set(experiment_version_config.simple_models)
        # assert set(experiment_version.normal_models_list) == set(experiment_version_config.normal_models)
"""


class TestPredict:

    def test_predict(self, setup_experiment_class):
        type_problem, experiment_version = setup_experiment_class
        dataset_path = TEST_DATASETS_PATH[type_problem]
        data = pd.read_csv(dataset_path)
        preds = experiment_version.predict(data)
        assert len(preds) == len(data)


class TestInfos:

    def test_info(self, setup_experiment_class):
        type_problem, experiment_version = setup_experiment_class
        # test models
        assert len(experiment_version.models) > 0
        # test Score
        assert experiment_version.score is not None
        """
        # test cv
        df_cv = experiment_version.get_cv()
        assert isinstance(df_cv, pd.DataFrame)
        """
        """
        # test hyper parameters
        hyper_params = experiment_version.best_model.hyperparameters
        assert hyper_params is not None
        """
        """
        # test feature importance
        feat_importance = experiment_version.best_model.feature_importance
        assert list(feat_importance.columns) == ['feature', 'importance']
        """
        # test correlation matrix
        matrix = experiment_version.correlation_matrix
        assert isinstance(matrix, pd.DataFrame)
        # test schema
        schema = experiment_version.schema
        assert schema is not None
        # test features stats
        stats = experiment_version.features_stats
        assert stats is not None
        # test fastest model
        model = experiment_version.fastest_model
        # test experiment version id
        assert model.experiment_version_id == experiment_version._id
        # test print info
        experiment_version.print_info()
        assert isinstance(experiment_version.models, list)
        model = experiment_version.models[0]
        model_copy = Model.from_id(model._id)
        """
        assert isinstance(model.hyperparameters, dict)
        assert model_copy.hyperparameters == model.hyperparameters
        """

        # assert isinstance(model.cross_validation, pd.DataFrame)
        assert isinstance(model.chart(), dict)

        # assert isinstance(model_copy.cross_validation, pd.DataFrame)
        assert isinstance(model_copy.chart(), dict)
        if type_problem == 'classification':
            assert isinstance(model_copy, ExternalClassificationModel)
            assert model_copy.optimal_threshold == model.optimal_threshold
            assert isinstance(model.get_dynamic_performances(), dict)
        elif type_problem == 'multiclassification':
            assert isinstance(model_copy, ExternalMultiClassificationModel)
        elif type_problem == 'regression':
            assert isinstance(model_copy, ExternalRegressionModel)
