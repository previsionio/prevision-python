import os
from previsionio.usecase_config import UsecaseState
import time
import pandas as pd
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id
from tempfile import TemporaryDirectory

TESTING_ID = get_testing_id()
N_DATASETS = 0
PROJECT_NAME = "sdk_test_dataset_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

test_datasets = {}
paths = {}


def setup_module(module):
    remove_datasets(DATA_PATH)
    paths.update(make_supervised_datasets(DATA_PATH))
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id


def teardown_module(module):
    remove_datasets(DATA_PATH)
    project = pio.Project.from_id(PROJECT_ID)
    for ds in project.list_datasets(all=True):
        if TESTING_ID in ds.name:
            ds.delete()
    project.delete()


def test_upload_dataset_from_dataframe():
    project = pio.Project.from_id(PROJECT_ID)
    paths_df = {k: paths[k] for k in paths if k != 'zip_regression'}
    for problem_type, p in paths_df.items():
        dataset = project.create_dataset(p.split('/')[-1][:-4] + str(TESTING_ID),
                                         dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset

    datasets = [ds for ds in project.list_datasets(all=True) if TESTING_ID in ds.name]
    ds_names = [k + str(TESTING_ID) for k in paths_df]

    assert len(datasets) == len(paths_df)
    global N_DATASETS
    N_DATASETS += len(datasets)

    for ds in datasets:
        assert ds.name in ds_names


def test_upload_dataset_from_filename():
    project = pio.Project.from_id(PROJECT_ID)
    paths_files = {k: paths[k] for k in ('regression', 'zip_regression')}
    for p in paths_files.values():
        project.create_dataset(p.split('/')[-1][:-4] + str(TESTING_ID),
                               file_name=p)

    datasets = [ds for ds in project.list_datasets(all=True) if TESTING_ID in ds.name]
    assert len(datasets) == len(paths_files) + N_DATASETS


def test_from_id_new():
    ds = test_datasets['classification']
    new = pio.Dataset.from_id(ds._id)
    assert new._id == ds._id


def test_download():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    with TemporaryDirectory() as dir:
        path = ds.download(dir)
    assert os.path.isfile(path)
    os.remove(path)


def test_embedding():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    ds.start_embedding()
    t0 = time.time()
    states = [UsecaseState.Pending, UsecaseState.Running]
    while ds.get_embedding_status() in states and time.time() < t0 + pio.config.default_timeout:
        ds.update_status()

    embedding = ds.get_embedding()
    assert isinstance(embedding, dict)
    assert 'labels' in embedding
    assert 'tensors' in embedding


def test_delete_dataset():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    ds.delete()
