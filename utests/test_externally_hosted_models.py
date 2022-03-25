import previsionio as pio
from previsionio.experiment_version import ExternallyHostedExperimentVersion
from previsionio.experiment_deployment import ExternallyHostedModelDeployment
from .utils import get_testing_id


TESTING_ID = get_testing_id()
PROJECT_NAME = "sdk_test_externally_hosted_models" + str(TESTING_ID)
PROJECT_ID = ""
PATHS = {
    "input": "data_externally_hosted_models/regression_holdout_dataset_input.parquet",
    "output": "data_externally_hosted_models/regression_holdout_dataset_output.parquet",
    "yaml": "data_externally_hosted_models/regression_model.yaml",
}


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME, description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id
    global holdout_dataset
    holdout_dataset = project.create_dataset('input', file_name=PATHS['input'])
    global pred_dataset
    pred_dataset = project.create_dataset('output', file_name=PATHS['output'])


def teardown_module(module):
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


def test_all():
    project = pio.Project.from_id(PROJECT_ID)

    # test create
    externally_hosted_model = project.create_externally_hosted_model(
        'test_externally_hosted',
        holdout_dataset,
        'TARGET',
        [('my_externally_hosted_model', 'regression_model.yaml')],
        pio.TypeProblem.Regression,
        pred_dataset=pred_dataset,
    )
    externally_hosted_model.wait_until(
        lambda x: (len(x.models) > 0) or (x._status['state'] == 'failed')
    )

    # test from from_id
    check_model = ExternallyHostedExperimentVersion.from_id(
        externally_hosted_model._id
    )
    assert isinstance(check_model, ExternallyHostedExperimentVersion)

    # test deploy model
    model = externally_hosted_model.models[0]
    externally_hosted_model_deployment = project.create_externally_hosted_model_deployment(
        'test_externally_hosted',
        main_model=model,
    )
    externally_hosted_model_deployment.wait_until(
        lambda x: x.deploy_state in ['done', 'failed']
    )

    # test from from_id
    check_model = ExternallyHostedModelDeployment.from_id(
        externally_hosted_model_deployment._id
    )
    assert isinstance(check_model, ExternallyHostedModelDeployment)

    experiment_id = experiment_version.experiment_id
    experiments = Experiment.list(PROJECT_ID)
    assert experiment_id in [experiment.id for experiment in experiments]

    experiment_version_new = experiment_version.new_version([('my_externally_hosted_model', 'regression_model.yaml')])
    # test send unit
    _input = {
        'feat_0': 0,
        'feat_1': 1,
        'feat_2': 2,
        'feat_3': 3,
    }
    output = {'pred_Target': 42}
    res_unit = externally_hosted_model_deployment.log_unit_prediction(_input, output)
    assert res_unit['message'] == 'OK'

    # test send bulk
    res_bulk = externally_hosted_model_deployment.log_bulk_prediction(
        'regression_holdout_dataset_input.parquet',
        'regression_holdout_dataset_output.parquet',
    )
    assert isinstance(res_bulk, dict)

    # test list bulk
    res_list = externally_hosted_model_deployment.list_log_bulk_predictions()
    assert len(res_list) >= 1

    # Experiment.from_id(experiment_id).delete()
    # experiments = Experiment.list(PROJECT_ID)
    # assert experiment_id not in [experiment.id for experiment in experiments]
