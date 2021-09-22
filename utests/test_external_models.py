import os
import uuid
from typing import Tuple
import pandas as pd
import pytest
import previsionio as pio
from previsionio import logger
from .datasets import make_supervised_datasets, remove_datasets


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_external_models')
TESTING_ID = str(uuid.uuid4())

pio.config.zip_files = False
pio.config.default_timeout = 1000

type_problem_2_projet_usecase_version_creation_method_name = {
    'regression': "create_external_regression",
    #'classification': "create_external_classification",
    #'multiclassification': "create_external_multiclassification",
}
type_problems = type_problem_2_projet_usecase_version_creation_method_name.keys()

TEST_DATASETS_PATH = {
    'regression': os.path.join(DATA_PATH, 'regression_holdout_dataset.csv'),
    'classification': os.path.join(DATA_PATH, 'classification_holdout_dataset.csv'),
    'multiclassification': os.path.join(DATA_PATH, 'multiclassification_holdout_dataset.csv'),
}

# just for fast debug with onmy one type_problem in type_problems
TEST_DATASETS_PATH = {type_problem: TEST_DATASETS_PATH[type_problem] for type_problem in type_problems}


TEST_PIO_DATASETS = {}
def make_pio_datasets():
    for problem_type, p in TEST_DATASETS_PATH.items():
        project = pio.Project.from_id(PROJECT_ID)
        dataset = project.create_dataset(p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                         dataframe=pd.read_csv(p))
        TEST_PIO_DATASETS[problem_type] = dataset


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


def setup_module(module):
    project_name = f'project_sdk_TEST_EXTERNAL_MODELS_{TESTING_ID}'
    project = pio.Project.new(name=project_name,
                              description="description_sdk_TEST_EXTERNAL_MODELS")
    global PROJECT_ID
    PROJECT_ID = project.id
    make_pio_datasets()


def teardown_module(module):
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


def create_external_usecase_version(
    project_id,
    type_problem,
    usecase_name,
    external_models,
    target_column='TARGET',  # because all our utests datasets has a target_column named 'TARGET'
    dataset=None,
    usecase_version_description=None,
) -> pio.external_models.ExternalUsecaseVersion:

    project = pio.Project.from_id(project_id)
    holdout_dataset = TEST_PIO_DATASETS[type_problem]
    usecase_version_creation_method_name = type_problem_2_projet_usecase_version_creation_method_name[type_problem]
    usecase_version_creation_method = getattr(project, usecase_version_creation_method_name)

    # NOTE: should allow to change the default metric
    return usecase_version_creation_method(
        usecase_name,
        holdout_dataset,
        target_column,
        external_models,
        dataset=dataset,
        usecase_version_description=usecase_version_description,
    )


# create an external usecase version only from type_problem, all other args are default
def create_external_usecase_version_from_type_problem(type_problem, usecase_name=None):
    if usecase_name is None:
        usecase_name = f'test_sdk_external_models_{type_problem}_{TESTING_ID}'

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    usecase_version_description = f'description_version_{usecase_name}'
    usecase_version: pio.ExternalUsecaseVersion = create_external_usecase_version(
        PROJECT_ID,
        type_problem,
        usecase_name,
        external_models,
        usecase_version_description=usecase_version_description,
    )
    return usecase_version


def test_usecase_version():
    usecase_name = f'test_sdk_external_models_test_usecase_version_{TESTING_ID}'
    type_problem = 'regression'
    usecase_version = create_external_usecase_version_from_type_problem(type_problem, usecase_name=usecase_name)

    usecase_id = usecase_version.usecase_id
    usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id in [usecase.id for usecase in usecases]

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    usecase_version_new = usecase_version.new_version(external_models)

    usecase_versions = pio.Usecase.from_id(usecase_id).versions
    assert usecase_version_new.id in [usecase_version.id for usecase_version in usecase_versions]

    pio.Usecase.from_id(usecase_id).delete()
    usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id not in [usecase.id for usecase in usecases]


def test_usecase_latest_versions():
    usecase_name = f'test_sdk_external_models_test_usecase_latest_versions_{TESTING_ID}'
    # type_problem = 'classification'
    # NOTE: set classification at the end to have a different type_problem than in test_usecase_version()
    type_problem = 'regression'
    usecase_version = create_external_usecase_version_from_type_problem(type_problem, usecase_name=usecase_name)

    usecase_id = usecase_version.usecase_id
    usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id in [usecase.id for usecase in usecases]

    external_model = TEST_EXTERNAL_MODELS[type_problem]
    external_models = [external_model]
    usecase_version_new = usecase_version.new_version(external_models)
    assert usecase_version.id != usecase_version_new.id
    assert usecase_version.usecase_id == usecase_version_new.usecase_id
    assert usecase_version.project_id == usecase_version_new.project_id

    latest_version = pio.Usecase.from_id(usecase_version_new.usecase_id).latest_version
    assert usecase_version_new._id == latest_version._id
    latest_version.new_version(external_models)

    pio.Usecase.from_id(usecase_version_new.usecase_id).delete()
    usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id not in [usecase.id for usecase in usecases]


def test_stop_running_usecase_version():
    usecase_name = f'test_sdk_external_models_test_stop_running_usecase_version_{TESTING_ID}'
    # type_problem = 'classification'
    # NOTE: set multiclassification at the end to have a different type_problem than in test_usecase_version()
    type_problem = 'regression'
    usecase_version = create_external_usecase_version_from_type_problem(type_problem, usecase_name=usecase_name)

    usecase_id = usecase_version.usecase_id
    assert usecase_version.running
    usecase_version.stop()
    assert not usecase_version.running
    pio.Usecase.from_id(usecase_version.usecase_id).delete()
    usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id not in [usecase.id for usecase in usecases]


@pytest.fixture(scope='module', params=type_problems)
def setup_usecase_class(request):
    type_problem = request.param
    logger.info(f'type_problem: {type_problem}')
    usecase_name = f'test_sdk_external_models_{type_problem}_{TESTING_ID}'
    usecase_version = create_external_usecase_version_from_type_problem(type_problem, usecase_name=usecase_name)
    assert usecase_version.running
    usecase_version.wait_until(
         lambda usecase: (len(usecase.models) > 0) or (usecase._status['state'] == 'failed'))
    usecase_version.stop()
    assert not usecase_version.running
    usecase_version.wait_until(lambda usecase_version: usecase_version._status['state'] == 'done', timeout=60)
    assert usecase_version._status['state'] == 'done'
    yield type_problem, usecase_version
    pio.Usecase.from_id(usecase_version.usecase_id).delete()


class TestUsecaseVersionGeneric:

    def test_check_config(self, setup_usecase_class):
        training_type, uc = setup_usecase_class
        # NOTE: atm test nothing else than usecase version launching and stop
        """
        assert all([c in uc.drop_list for c in DROP_COLS])
        uc.update_status()
        assert set(uc.feature_list) == set(uc_config.features)
        assert set(uc.advanced_models_list) == set(uc_config.advanced_models)
        assert set(uc.simple_models_list) == set(uc_config.simple_models)
        assert set(uc.normal_models_list) == set(uc_config.normal_models)
        """


class TestPredict:

    def test_predict(self, setup_usecase_class):
        type_problem, usecase_version = setup_usecase_class
        dataset_path = TEST_DATASETS_PATH[type_problem]
        data = pd.read_csv(dataset_path)
        logger.info('start predict')
        preds = usecase_version.predict(data)
        assert len(preds) == len(data)
        # test_predict_unit
        pred = usecase_version.predict_single(data.iloc[0].to_dict())
        assert pred is not None
