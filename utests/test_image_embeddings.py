import os
import pandas as pd
import previsionio as pio
from .utils import get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_image_embeddings_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

col_config = pio.ColumnConfig(target_column='class', filename_column='filename')
experiment_version_config = pio.TrainingConfig(
    advanced_models=[pio.AdvancedModel.LinReg],
    normal_models=[],
    simple_models=[],
    features=[pio.Feature.Counts],
    profile=pio.Profile.Quick,
)

test_datasets = {}
dataset_name = 'cats_and_dogs_train'
dataset_test_name = TESTING_ID + '-' + dataset_name


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id
    upload_datasets()


def upload_datasets():
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_img/{}'.format(dataset_name))
    # upload CSV reference file
    project = pio.Project.from_id(PROJECT_ID)
    dataset_csv = project.create_dataset(
        name=dataset_test_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(dataset_name)))
    )
    # upload ZIP images folder
    dataset_zip = project.create_image_folder(
        name=dataset_test_name,
        file_name=os.path.join(datapath, '{}.zip'.format(dataset_name))
    )
    test_datasets['csv'] = dataset_csv
    test_datasets['zip'] = dataset_zip


def teardown_module(module):
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


def test_run_image_embeddings():
    experiment_name = TESTING_ID + '_img_embeds'
    dataset = test_datasets['csv']
    dataset_images = test_datasets['zip']
    project = pio.Project.from_id(PROJECT_ID)
    experiment_version = project.fit_image_classification(
        experiment_name,
        dataset,
        dataset_images,
        column_config=col_config,
        metric=pio.metrics.Classification.AUC,
        training_config=experiment_version_config,
    )
    experiment_version.wait_until(lambda experimentv: len(experimentv.models) > 0)
    pio.Experiment.from_id(experiment_version.experiment_id).delete()
