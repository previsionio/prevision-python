import os
import time
import pandas as pd
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id

TESTING_ID = get_testing_id()

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
    for ds in pio.Dataset.list(PROJECT_ID, all=True):
        if TESTING_ID in ds.name:
            ds.delete()
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


def test_upload_datasets():
    for problem_type, p in paths.items():
        dataset = pio.Dataset._new(PROJECT_ID, p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                   dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset

    datasets = [ds for ds in pio.Dataset.list(PROJECT_ID, all=True) if TESTING_ID in ds.name]
    ds_names = [k + str(TESTING_ID) + '.csv' for k in paths]
    assert len(datasets) == len(paths)
    for ds in datasets:
        assert ds.name in ds_names


def test_from_id_new():
    ds = test_datasets['classification']
    new = pio.Dataset.from_id(ds._id)
    assert new._id == ds._id


def test_download():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    path = ds.download()
    assert os.path.isfile(path)
    os.remove(path)


def test_embedding():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    ds.start_embedding()
    t0 = time.time()
    while ds.get_embedding_status() in ['pending', 'running'] and time.time() < t0 + pio.config.default_timeout:
        ds.update_status()

    embedding = ds.get_embedding()
    assert isinstance(embedding, dict)
    assert 'labels' in embedding
    assert 'tensors' in embedding


def test_delete_dataset():
    ds = test_datasets['regression']
    ds = pio.Dataset.from_id(ds._id)
    ds.delete()
