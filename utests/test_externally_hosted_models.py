import pytest
import previsionio as pio
from previsionio.experiment_version import ExternallyHostedExperimentVersion
from previsionio.experiment_deployment import ExternallyHostedModelDeployment
from .utils import get_testing_id


TESTING_ID = get_testing_id()
PROJECT_NAME = "sdk_test_externally_hosted_models" + str(TESTING_ID)
PROJECT_ID = ""
configs = [
    {
        "holdout": "data_externally_hosted_models/regression_holdout_dataset.parquet",
        "pred": "data_externally_hosted_models/regression_pred_dataset.parquet",
        "yaml": "data_externally_hosted_models/regression_model.yaml",
        "type_problem": pio.TypeProblem.Regression,
        "input": {
            'feat_0': 0,
            'feat_1': 1,
            'feat_2': 2,
            'feat_3': 3,
        },
        "output": {
            'pred_Target': 42,
        },
    },
    {
        "holdout": "data_externally_hosted_models/multiclassification_holdout_dataset.parquet",
        "pred": "data_externally_hosted_models/multiclassification_pred_dataset.parquet",
        "yaml": "data_externally_hosted_models/multiclassification_model.yaml",
        "type_problem": pio.TypeProblem.MultiClassification,
        "input": {
            'sepal_length': 0,
            'sepal_width': 1,
            'petal_length': 2,
            'petal_width': 3,
        },
        "output": {
            'pred_Iris-setosa': .1,
            'pred_Iris-versicolor': .2,
            'pred_Iris-virginica': .7,
        },
    },
]


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME, description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id


def teardown_module(module):
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


@pytest.mark.parametrize('config', configs)
def test_all(config):
    print(config)

    project = pio.Project.from_id(PROJECT_ID)

    holdout_dataset = project.create_dataset(
        config['holdout'].split('/')[-1],
        file_name=config['holdout'],
    )
    pred_dataset = project.create_dataset(
        config['pred'].split('/')[-1],
        file_name=config['pred'],
    )

    # test create externally hosted model
    externally_hosted_model = project.create_externally_hosted_model(
        f"test_externally_hosted_{config['type_problem'].value}",
        holdout_dataset,
        'TARGET',
        [('my_externally_hosted_model', config['yaml'])],
        config['type_problem'],
        pred_dataset=pred_dataset,
    )
    externally_hosted_model.wait_until(
        lambda x: (len(x.models) > 0) or (x._status['state'] == 'failed')
    )

    # test from_id
    check_model = ExternallyHostedExperimentVersion.from_id(
        externally_hosted_model._id
    )
    assert isinstance(check_model, ExternallyHostedExperimentVersion)

    # test experiments listing
    experiment_id = externally_hosted_model.experiment_id
    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_id in [experiment.id for experiment in experiments]

    # test new version
    new_externally_hosted_model = externally_hosted_model.new_version(
        [('my_externally_hosted_model', config['yaml'])]
    )
    assert isinstance(new_externally_hosted_model, ExternallyHostedExperimentVersion)

    # test deploy model
    model = externally_hosted_model.models[0]
    externally_hosted_model_deployment = project.create_externally_hosted_model_deployment(
        f"test_externally_hosted_{config['type_problem'].value}",
        main_model=model,
    )
    externally_hosted_model_deployment.wait_until(
        lambda x: x.deploy_state in ['done', 'failed']
    )

    # test from_id
    check_model = ExternallyHostedModelDeployment.from_id(
        externally_hosted_model_deployment._id
    )
    assert isinstance(check_model, ExternallyHostedModelDeployment)

    # test send unit
    res_unit = externally_hosted_model_deployment.log_unit_prediction(
        config['input'],
        config['output'],
    )
    assert res_unit['message'] == 'OK'

    # test send bulk
    res_bulk = externally_hosted_model_deployment.log_bulk_prediction(
        config['holdout'],
        config['pred'],
    )
    assert isinstance(res_bulk, dict)

    # test list bulk
    res_list = externally_hosted_model_deployment.list_log_bulk_predictions()
    assert len(res_list) >= 1

    # Experiment.from_id(experiment_id).delete()
    # experiments = Experiment.list(PROJECT_ID)
    # assert experiment_id not in [experiment.id for experiment in experiments]
