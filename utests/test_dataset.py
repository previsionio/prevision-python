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

pio.config.zip_files = False
pio.config.default_timeout = 80

test_datasets = {}
paths = {}


def setup_module(module):
    remove_datasets(DATA_PATH)
    paths.update(make_supervised_datasets(DATA_PATH))


def teardown_module(module):
    remove_datasets(DATA_PATH)
    for ds in pio.Dataset.list():
        if TESTING_ID in ds.name:
            ds.delete()


def test_upload_datasets():
    for problem_type, p in paths.items():
        dataset = pio.Dataset.new(name=p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'),
                                  dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset

    datasets = [ds for ds in pio.Dataset.list() if TESTING_ID in ds.name]
    ds_names = [k + str(TESTING_ID) + '.csv' for k in paths]
    assert len(datasets) == len(paths)
    for ds in datasets:
        assert ds.name in ds_names


def test_from_id_new():
    ds = test_datasets['classification']
    new = pio.Dataset.from_id(ds._id)
    assert new._id == ds._id


def test_get_by_name():
    # test with none dataset_name
    with pytest.raises(AttributeError):
        pio.dataset.Dataset.get_by_name()

    # test with fake name
    for (fake_name, v) in [('foobar', 'last'), ('foobar', -5)]:
        with pytest.raises(PrevisionException) as e:
            pio.dataset.Dataset.get_by_name(name=fake_name, version=v)
        assert (e.match(r"DatasetNotFoundError"))

    # delete the first uploaded dataset
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    ds = pio.dataset.Dataset.get_by_name(ds_name)
    assert ds is not None
    assert ds.name == ds_name


def test_download():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    csv_path = pio.dataset.Dataset.download(dataset_name=ds_name)
    assert os.path.isfile(csv_path)
    os.remove(csv_path)


def test_embedding():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    ds = pio.dataset.Dataset.from_name(ds_name)
    ds.start_embedding()
    time.sleep(20)
    embedding = ds.get_embedding()
    assert isinstance(embedding, dict)
    assert 'labels' in embedding
    assert 'tensors' in embedding


def test_delete_dataset():
    ds_name = 'regression' + str(TESTING_ID) + '.csv'
    pio.dataset.Dataset.get_by_name(ds_name).delete()
