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
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

test_datasets = {}
dataset_name = 'cats_and_dogs_train'
dataset_test_name = TESTING_ID + '-' + dataset_name

def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id


def test_upload_dataset_image():
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_img/{}'.format(dataset_name))

    # upload ZIP images folder
    dataset_zip = pio.DatasetImages.new(
        PROJECT_ID,
        dataset_test_name,
        file_name=os.path.join(datapath, '{}.zip'.format(dataset_name))
    )
    test_datasets['zip'] = dataset_zip
    # bug web metaData without rowsPerPage
    assert len(pio.DatasetImages.list(PROJECT_ID)) == 1
    path = dataset_zip.download()
    assert os.path.isfile(path)
    os.remove(path)
    dataset_zip.delete()
