from logging import DEBUG
import logging
import os
# from previsionio.logger import event_logger
from previsionio.experiment_config import ExperimentState
import time
import pandas as pd
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id
from tempfile import TemporaryDirectory

TESTING_ID = get_testing_id()
PROJECT_NAME = "sdk_test_dataset_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

paths = {}

# event_logger.setLevel(DEBUG)
logging.getLogger().setLevel(DEBUG)


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
    paths_df = {k: paths[k] for k in paths if k == 'classification'}
    for _, p in paths_df.items():
        project.create_dataset(
            p.split('/')[-1][:-4] + str(TESTING_ID),
            dataframe=pd.read_csv(p),
            origin="pipeline_intermediate_file",
        )

    datasets = [ds for ds in project.list_datasets(all=True) if TESTING_ID in ds.name]
    ds_names = [k + str(TESTING_ID) for k in paths_df]

    assert len(datasets) == len(paths_df)

    for ds in datasets:
        assert ds.name in ds_names

    for ds in datasets:
        ds.delete()


def test_upload_dataset_from_filename():
    project = pio.Project.from_id(PROJECT_ID)
    paths_files = {k: paths[k] for k in paths if 'regression' in k}
    # paths_files = {k: paths[k] for k in ('regression', 'zip_regression')}
    for p in paths_files.values():
        project.create_dataset(p.split('/')[-1][:-4] + str(TESTING_ID),
                               file_name=p)

    datasets = [ds for ds in project.list_datasets(all=True) if TESTING_ID in ds.name]
    assert len(datasets) == len(paths_files)
    for ds in datasets:
        ds.delete()


def test_from_id_new():
    project = pio.Project.from_id(PROJECT_ID)
    dataset = project.create_dataset(paths["classification"].split('/')[-1][:-4] + str(TESTING_ID),
                                     dataframe=pd.read_csv(paths["classification"]),
                                     origin="pipeline_intermediate_file")

    new = pio.Dataset.from_id(dataset._id)
    assert new._id == dataset._id
    dataset.delete()


def test_download():
    project = pio.Project.from_id(PROJECT_ID)
    dataset = project.create_dataset(paths["regression"].split('/')[-1][:-4] + str(TESTING_ID),
                                     dataframe=pd.read_csv(paths["regression"]), origin="pipeline_intermediate_file")
    ds = pio.Dataset.from_id(dataset._id)
    with TemporaryDirectory() as dir:
        path_zip = ds.download(directoy_path=dir)
        assert os.path.isfile(path_zip)
        path_parquet = ds.download(directoy_path=dir, extension="parquet")
        assert os.path.isfile(path_parquet)
    dataset.delete()


def test_embedding():
    project = pio.Project.from_id(PROJECT_ID)
    ds = project.create_dataset(paths["regression"].split('/')[-1][:-4] + str(TESTING_ID),
                                dataframe=pd.read_csv(paths["regression"]), origin="pipeline_intermediate_file")

    ds.start_embedding()
    t0 = time.time()
    states = [ExperimentState.Pending, ExperimentState.Running]
    while ds.get_embedding_status() in states and time.time() < t0 + pio.config.default_timeout:
        ds.update_status()
        time.sleep(1)

    embedding = ds.get_embedding()
    assert isinstance(embedding, dict)
    assert 'labels' in embedding
    assert 'tensors' in embedding
    ds.delete()


def test_delete_dataset():
    project = pio.Project.from_id(PROJECT_ID)
    ds = project.create_dataset(paths["regression"].split('/')[-1][:-4] + str(TESTING_ID),
                                dataframe=pd.read_csv(paths["regression"]), origin="pipeline_intermediate_file")
    ds.delete()
