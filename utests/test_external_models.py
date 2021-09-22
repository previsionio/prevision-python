import os
import uuid
from typing import Tuple
import pandas as pd
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from .utils import train_model, get_testing_id, DROP_COLS


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_external_models')
TESTING_ID = str(uuid.uuid4())

pio.config.zip_files = False
pio.config.default_timeout = 1000

type_problem_2_projet_usecase_version_creation_method_name = {
    'regression': "create_external_regression", # only regression for fast debug
}
type_problems = type_problem_2_projet_usecase_version_creation_method_name.keys()


test_datasets = {}
def make_pio_datasets(paths):
    for problem_type, p in paths.items():
        project = pio.Project.from_id(PROJECT_ID)
        dataset = project.create_dataset(p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                         dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset


test_external_models = {
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
    project_name = f'project_sdk_test_external_models_{TESTING_ID}'
    project = pio.Project.new(name=project_name,
                              description="description_sdk_test_external_models")
    global PROJECT_ID
    PROJECT_ID = project._id
    paths = {
        'regression': os.path.join(DATA_PATH, 'regression_holdout_dataset.csv'),
    }
    make_pio_datasets(paths)


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
    holdout_dataset = test_datasets[type_problem]
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


def test_usecase_version():
    usecase_name = f'test_sdk_external_models_test_usecase_version_{TESTING_ID}'
    type_problem = 'regression'
    external_model = test_external_models[type_problem]
    external_models = [external_model]
    usecase_version_description = f'description_version_{usecase_name}'
    usecase_version: pio.ExternalUsecaseVersion = create_external_usecase_version(
        PROJECT_ID,
        type_problem,
        usecase_name,
        external_models,
        usecase_version_description=usecase_version_description,
    )
    usecase_id = usecase_version.usecase_id
    project_usecases = pio.Usecase.list(PROJECT_ID)
    assert usecase_id in [usecase.id for usecase in project_usecases]

    usecase_version_new = usecase_version.new_version(external_models)

    usecase_versions = pio.Usecase.from_id(usecase_id).versions
    assert usecase_version_new.id in [usecase_version.id for usecase_version in usecase_versions]

    pio.Usecase.from_id(usecase_id).delete()
    project_usecases = pio.Usecase.list(PROJECT_ID)
    project_usecases_ids = [usecase.id for usecase in project_usecases]
    assert usecase_id not in project_usecases_ids
