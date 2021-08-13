import pandas as pd
import previsionio as pio

from . import DATA_PATH
from .datasets import make_supervised_datasets, remove_datasets
from .utils import train_model, get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_usecase_deployment" + str(TESTING_ID)
PROJECT_ID = ""

uc_config = pio.TrainingConfig(advanced_models=[pio.AdvancedModel.LinReg],
                               normal_models=[pio.NormalModel.LinReg],
                               simple_models=[pio.SimpleModel.DecisionTree],
                               features=[pio.Feature.Counts],
                               profile=pio.Profile.Quick)
test_datasets = {}

training_type_2_pio_class = {
    'regression': "fit_regression",
    'classification': "fit_classification",
    'multiclassification': "fit_multiclassification",
}
training_types = training_type_2_pio_class.keys()


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


def supervised_from_filename(training_type, uc_name):
    dataset = test_datasets[training_type]
    training_type_class = training_type_2_pio_class[training_type]
    return train_model(PROJECT_ID, uc_name, dataset, training_type, training_type_class, uc_config)


def test_usecase_version():
    uc_name = TESTING_ID + '_file_del'
    usecase_version: pio.Supervised = supervised_from_filename('regression', uc_name)
    usecase_version.wait_until(
        lambda usecase: (len(usecase.models) > 0) or (usecase._status['state'] == 'failed'))
    assert usecase_version.running
    usecase_version.stop()
    uc_best_model = usecase_version.best_model

    project = pio.Project.from_id(PROJECT_ID)
    usecase_deployment = project.create_usecase_deployment('test_sdk_' + TESTING_ID, uc_best_model)

    uc_dataset_id = usecase_version.dataset_id
    prediction_dataset = pio.Dataset.from_id(uc_dataset_id)

    usecase_deployment.wait_until(lambda usecase_deployment: usecase_deployment.run_state == 'done')

    deployement_prediction = usecase_deployment.predict_from_dataset(prediction_dataset)
    prediction_df = deployement_prediction.get_result()
    assert isinstance(prediction_df, pd.DataFrame)
