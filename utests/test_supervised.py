import os
from previsionio.experiment_version import ClassicExperimentVersion
from typing import Tuple
from previsionio.model import ClassificationModel
import pandas as pd
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import train_model, get_testing_id, DROP_COLS

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_experiment_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

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


def test_delete_experiment():
    experiment_name = TESTING_ID + '_test_delete_experiment'
    experiment_version = supervised_from_filename('regression', experiment_name)
    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_name in [u.name for u in experiments]
    pio.Experiment.from_id(experiment_version.experiment_id).delete()
    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_name not in [u.name for u in experiments]


def test_experiment_version():
    experiment_name_desired = TESTING_ID + '_test_experiment_version'
    experiment_version: pio.Supervised = supervised_from_filename('regression', experiment_name_desired)
    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_name_desired in [e.name for e in experiments]

    experiment_new_version = experiment_version.new_version()
    experiment_versions = pio.Experiment.from_id(experiment_version.experiment_id).versions
    assert experiment_new_version.id in [ev.id for ev in experiment_versions]

    experiment = pio.Experiment.from_id(experiment_new_version.experiment_id)
    experiment.delete()

    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment.id not in [exp.id for exp in experiments]


def test_experiment_latest_versions():
    experiment_name = TESTING_ID + '_test_experiment_latest_versions'
    experiment_version: pio.Supervised = supervised_from_filename('regression', experiment_name)
    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_name in [u.name for u in experiments]

    experiment_new_version = experiment_version.new_version()
    assert experiment_version._id != experiment_new_version._id
    assert experiment_version.experiment_id == experiment_new_version.experiment_id
    assert experiment_version.project_id == experiment_new_version.project_id

    # experiments = pio.Experiment.list(PROJECT_ID)
    latest_version = pio.Experiment.from_id(experiment_new_version.experiment_id).latest_version
    assert experiment_new_version._id == latest_version._id
    latest_version.new_version()

    pio.Experiment.from_id(experiment_new_version.experiment_id).delete()

    experiments = pio.Experiment.list(PROJECT_ID)
    assert experiment_name not in [u.name for u in experiments]


def test_stop_running_experiment():
    experiment_name = TESTING_ID + '_test_stop_running_experiment'
    experiment_version = supervised_from_filename('regression', experiment_name)
    experiment_version.wait_until(
        lambda experiment: (len(experiment.models) > 0) or (experiment._status['state'] == 'failed'))
    assert experiment_version.running
    experiment_version.stop()
    experiment_version.update_status()
    assert not experiment_version.running
    pio.Experiment.from_id(experiment_version.experiment_id).delete()


@pytest.fixture(scope='module', params=training_types)
def setup_experiment_class(request):
    experiment_name = '{}_{}'.format(request.param[0:5], TESTING_ID)
    experiment_version = supervised_from_filename(request.param, experiment_name)
    experiment_version.wait_until(
        lambda experiment: (len(experiment.models) > 0) or (experiment._status['state'] == 'failed'))
    assert experiment_version.running
    experiment_version.stop()
    experiment_version.wait_until(lambda experiment: experiment._status['state'] == 'done', timeout=60)
    assert experiment_version._status['state'] == 'done'
    yield request.param, experiment_version
    _ = pio.Experiment.from_id(experiment_version.experiment_id).delete()


options_parameters = ('options',
                      [{'confidence': False},
                       {'confidence': True}])

predict_u_options_parameters = ('options',
                                [{'confidence': False, 'explain': True},
                                 {'confidence': True}])

predict_test_ids = [('confidence-' if opt['confidence'] else 'normal-')
                    for opt in predict_u_options_parameters[1]]


class TestUCGeneric:
    def test_check_config(self, setup_experiment_class):
        training_type, experiment_version = setup_experiment_class
        assert all([c in experiment_version.drop_list for c in DROP_COLS])
        experiment_version.update_status()
        assert set(experiment_version.feature_list) == set(experiment_version_config.features)
        assert set(experiment_version.advanced_models_list) == set(experiment_version_config.advanced_models)
        assert set(experiment_version.simple_models_list) == set(experiment_version_config.simple_models)
        assert set(experiment_version.normal_models_list) == set(experiment_version_config.normal_models)


class TestPredict:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_predict(self, setup_experiment_class, options):
        training_type, experiment_version = setup_experiment_class
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(training_type)))
        preds = experiment_version.predict(data, **options)
        assert len(preds) == len(data)
        if options['confidence']:
            if training_type == 'regression':
                conf_cols = ['_quantile={}'.format(q) for q in [1, 5, 10, 25, 50, 75, 95, 99]]
                for q in conf_cols:
                    assert any([q in col for col in preds])
            elif training_type == 'classification':
                assert 'confidence' in preds
                assert 'credibility' in preds
        # test_predict_unit
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(training_type)))
        pred = experiment_version.predict_single(data.iloc[0].to_dict(), **options)
        assert pred is not None


class TestInfos:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_info(self, setup_experiment_class: Tuple[str, ClassicExperimentVersion], options):
        training_type, experiment_version = setup_experiment_class
        # test models
        assert len(experiment_version.models) > 0
        # test Score
        assert experiment_version.score is not None
        # test cv
        df_cv = experiment_version.get_cv()
        assert isinstance(df_cv, pd.DataFrame)
        # test hyper parameters
        hyper_params = experiment_version.best_model.hyperparameters
        assert hyper_params is not None
        # test feature importance
        feat_importance = experiment_version.best_model.feature_importance
        assert list(feat_importance.columns) == ['feature', 'importance']
        # test correlation matrix
        matrix = experiment_version.correlation_matrix
        assert isinstance(matrix, pd.DataFrame)
        # test schema
        schema = experiment_version.schema
        assert schema is not None
        # test features stats
        stats = experiment_version.features_stats
        assert stats is not None
        # test fastest model
        model = experiment_version.fastest_model
        # test experiment version id
        assert model.experiment_version_id == experiment_version._id
        # test print info
        experiment_version.print_info()
        assert isinstance(experiment_version.models, list)
        model = experiment_version.models[0]
        model_copy = pio.Model.from_id(model._id)
        assert isinstance(model.hyperparameters, dict)
        assert model_copy.hyperparameters == model.hyperparameters

        print("experiment_version.status", experiment_version.status)

        assert isinstance(model.cross_validation, pd.DataFrame)
        assert isinstance(model.chart(), dict)
        if training_type == 'classification':
            assert isinstance(model_copy, ClassificationModel)
            assert model_copy.optimal_threshold == model.optimal_threshold
            assert isinstance(model.get_dynamic_performances(), dict)
