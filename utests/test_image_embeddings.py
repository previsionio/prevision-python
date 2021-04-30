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
uc_config = pio.TrainingConfig(normal_models=[pio.Model.LinReg],
                               lite_models=[],
                               simple_models=[],
                               features=[pio.Feature.Counts],
                               profile=pio.Profile.Quick)

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
    dataset_csv = pio.Dataset._new(
        PROJECT_ID,
        name=dataset_test_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(dataset_name)))
    )
    # upload ZIP images folder
    dataset_zip = pio.DatasetImages._new(
        PROJECT_ID,
        name=dataset_test_name,
        file_name=os.path.join(datapath, '{}.zip'.format(dataset_name))
    )
    test_datasets['csv'] = dataset_csv
    test_datasets['zip'] = dataset_zip


def teardown_module(module):
    pio.Dataset.get_by_name(project_id=PROJECT_ID, name=dataset_test_name).delete()
    pio.DatasetImages.get_by_name(project_id=PROJECT_ID, name=dataset_test_name).delete()
    for uc_dict in pio.Supervised.list():
        uc = pio.Supervised.from_id(uc_dict['usecase_id'])
        if TESTING_ID in uc.name:
            uc.delete()


def test_run_image_embeddings():
    uc_name = TESTING_ID + '_img_embeds'
    datasets = (test_datasets['csv'], test_datasets['zip'])
    uc = pio.MultiClassificationImages.fit(uc_name, dataset=datasets, column_config=col_config,
                                           metric=pio.metrics.MultiClassification.error_rate,
                                           training_config=uc_config)
    uc.wait_until(lambda usecase: len(usecase) > 0)
    uc.delete()
