import os
import time
import pandas as pd
import pytest
import previsionio as pio
from previsionio.utils import PrevisionException
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_dataset_" + str(TESTING_ID)
#PROJECT_ID = "6082fb73b153a8001c3052df"
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
        dataset = pio.Dataset.new(PROJECT_ID, p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
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


def test_get_by_name():
    # test with fake name
    for (fake_name, v) in [('foobar', 'last'), ('foobar', -5)]:
        with pytest.raises(PrevisionException) as e:
            pio.dataset.Dataset.get_by_name(PROJECT_ID, fake_name, version=v)
        assert (e.match(r"DatasetNotFoundError"))

    # delete the first uploaded dataset
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    ds = pio.dataset.Dataset.get_by_name(PROJECT_ID, ds_name)
    assert ds is not None
    assert ds.name == ds_name


def test_download():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    ds = pio.dataset.Dataset.get_by_name(PROJECT_ID, ds_name)
    path = ds.download()
    assert os.path.isfile(path)
    os.remove(path)


def test_embedding():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    ds = pio.dataset.Dataset.get_by_name(PROJECT_ID, ds_name)
    ds.start_embedding()
    t0 = time.time()
    while ds.get_embedding_status() in ['pending', 'running'] and time.time() < t0 + pio.config.default_timeout:
        ds.update_status()

    embedding = ds.get_embedding()
    assert isinstance(embedding, dict)
    assert 'labels' in embedding
    assert 'tensors' in embedding


def test_delete_dataset():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    pio.dataset.Dataset.get_by_name(PROJECT_ID, ds_name).delete()
