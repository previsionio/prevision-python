import os
from typing import Tuple
import pandas as pd
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from .utils import train_model, get_testing_id, DROP_COLS

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_usecase_external_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_external_models')

test_datasets = {}

training_type_2_pio_projet_method_name = {
    'regression': "create_external_regression", # only regression for fast debug
}
training_types = training_type_2_pio_projet_method_name.keys()


def make_pio_datasets(paths):
    for problem_type, p in paths.items():
        project = pio.Project.from_id(PROJECT_ID)
        dataset = project.create_dataset(p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                         dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id
    paths = {
        'regression': os.path.join(DATA_PATH, 'regression_holdout_dataset.csv'),
    }
    make_pio_datasets(paths)


def teardown_module(module):
    pass # do nothing during debug
    # project = pio.Project.from_id(PROJECT_ID)
    # project.delete()


def create_external_usecase_version(project_id, uc_name, training_type, usecase_version_description=None):
    project = pio.Project.from_id(project_id)
    holdout_dataset = test_datasets[training_type]
    training_type_class = training_type_2_pio_projet_method_name[training_type]
    training_type_func = getattr(project, training_type_class)

    holdout_dataset = test_datasets[training_type]

    target_column = 'TARGET'
    external_model = (
        'my_external_model',
        os.path.join(DATA_PATH, 'regression_model.onnx'),
        os.path.join(DATA_PATH, 'regression_model.yaml'),
    )
    external_models = [external_model]

    return training_type_func(
        uc_name,
        holdout_dataset,
        target_column,
        external_models,
        usecase_version_description=usecase_version_description,
    )


def test_usecase_version():
    uc_name = TESTING_ID + '_file_del'
    usecase_version: pio.ExternalUsecaseVersion = create_external_usecase_version(
        PROJECT_ID,
        uc_name,
        'regression',
        usecase_version_description='This is an external regression usecase_version'
    )
    """
    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name in [u.name for u in usecases]

    usecase_new_version = usecase_version.new_version()
    usecase_versions = pio.Usecase.from_id(usecase_version.usecase_id).versions
    assert usecase_new_version._id in [u._id for u in usecase_versions]

    pio.Usecase.from_id(usecase_new_version.usecase_id).delete()

    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name not in [u.name for u in usecases]
    """
