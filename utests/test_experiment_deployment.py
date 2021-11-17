import pandas as pd
import previsionio as pio

from . import DATA_PATH
from .datasets import make_supervised_datasets, remove_datasets
from .utils import train_model, get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_experiment_deployment" + str(TESTING_ID)
PROJECT_ID = ""

experiment_version_config = pio.TrainingConfig(
    advanced_models=[pio.AdvancedModel.LinReg],
    normal_models=[pio.NormalModel.LinReg],
    simple_models=[pio.SimpleModel.DecisionTree],
    features=[pio.Feature.Counts],
    profile=pio.Profile.Quick,
)

training_type_2_pio_class = {
    'regression': "fit_regression",
    'classification': "fit_classification",
    'multiclassification': "fit_multiclassification",
}
training_types = training_type_2_pio_class.keys()

test_datasets = {}


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
    remove_datasets(DATA_PATH)
    paths = make_supervised_datasets(DATA_PATH)
    make_pio_datasets(paths)


def teardown_module(module):
    remove_datasets(DATA_PATH)
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


def supervised_from_filename(training_type, experiment_name):
    dataset = test_datasets[training_type]
    training_type_class = training_type_2_pio_class[training_type]
    return train_model(PROJECT_ID, experiment_name, dataset, training_type,
                       training_type_class, experiment_version_config)


def test_experiment_version():
    experiment_name = TESTING_ID + '_file_del'
    experiment_version: pio.Supervised = supervised_from_filename('regression', experiment_name)
    experiment_version.wait_until(
        lambda experiment: (len(experiment.models) > 0) or (experiment._status['state'] == 'failed'))
    assert experiment_version.running
    experiment_version.stop()
    experiment_version_best_model = experiment_version.best_model

    project = pio.Project.from_id(PROJECT_ID)
    experiment_deployment = project.create_experiment_deployment('test_sdk_' + TESTING_ID,
                                                                 experiment_version_best_model)

    experiment_version_dataset_id = experiment_version.dataset._id
    prediction_dataset = pio.Dataset.from_id(experiment_version_dataset_id)

    experiment_deployment.wait_until(lambda experiment_deployment: experiment_deployment.run_state == 'done')

    deployment_prediction = experiment_deployment.predict_from_dataset(prediction_dataset)
    prediction_df = deployment_prediction.get_result()
    assert isinstance(prediction_df, pd.DataFrame)

    # Test deployed model
    # import os
    # experiment_deployment.create_api_key()
    # creds = experiment_deployment.get_api_keys()[-1]
    # model = pio.DeployedModel(
    #     prevision_app_url=experiment_deployment.url,
    #     client_id=creds['client_id'],
    #     client_secret=creds['client_secret'],
    #     prevision_url=os.getenv('PREVISION_URL', ''),
    # )
    # prediction, confidence, explain = model.predict(
    #     predict_data={'feat_0': 0, 'feat_1': 0},
    #     use_confidence=True,
    #     explain=True,
    # )
    # assert prediction is not None
    # assert confidence is not None
    # assert explain is not None
