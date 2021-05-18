import os
import previsionio as pio
from .utils import get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_dataset_image_" + str(TESTING_ID)
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


def teardown_module(module):
    project = pio.Project.from_id(PROJECT_ID)
    for image_folder in project.list_image_folders(all=True):
        if TESTING_ID in image_folder.name:
            image_folder.delete()
    project.delete()


def test_upload_dataset_image():
    project = pio.Project.from_id(PROJECT_ID)
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_img/{}'.format(dataset_name))
    # upload ZIP images folder
    dataset_zip = project.create_image_folder(
        dataset_test_name,
        file_name=os.path.join(datapath, '{}.zip'.format(dataset_name))
    )
    test_datasets['zip'] = dataset_zip
    # bug web metaData without rowsPerPage
    assert len(project.list_image_folders()) == 1
    path = dataset_zip.download()
    assert os.path.isfile(path)
    os.remove(path)
    dataset_zip.delete()
