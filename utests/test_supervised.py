import os
from previsionio.usecase_version import ClassicUsecaseVersion
from typing import Tuple
from previsionio.model import ClassificationModel
import pandas as pd
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import train_model, get_testing_id, DROP_COLS

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_usecase_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000

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


def test_delete_usecase():
    uc_name = TESTING_ID + '_file_del'
    usecase_version = supervised_from_filename('regression', uc_name)
    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name in [u.name for u in usecases]
    usecase_version.usecase.delete()
    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name not in [u.name for u in usecases]


def test_usecase_version():
    uc_name = TESTING_ID + '_file_del'
    usecase_version: pio.Supervised = supervised_from_filename('regression', uc_name)
    usecases = pio.Usecase.list(PROJECT_ID)
    print("usecase_version.usecase_id", usecase_version.usecase_id)
    print("usecase_version.project_id", usecase_version.project_id)
    assert uc_name in [u.name for u in usecases]

    usecase_new_version = usecase_version.new_version()
    print("usecase_new_version.usecase_id", usecase_new_version.usecase_id)
    print("usecase_new_version.project_id", usecase_new_version.project_id)
    # usecases = pio.Usecase.list(PROJECT_ID)
    usecase_versions = usecase_version.usecase.versions
    assert usecase_new_version._id in [u['_id'] for u in usecase_versions]

    usecase_new_version.usecase.delete()

    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name not in [u.name for u in usecases]


def test_stop_running_usecase():
    uc_name = TESTING_ID + '_file_run'
    usecase_version = supervised_from_filename('regression', uc_name)
    usecase_version.wait_until(
        lambda usecase: (len(usecase.models) > 0) or (usecase._status['state'] == 'failed'))
    assert usecase_version.running
    usecase_version.stop()
    usecase_version.update_status()
    assert not usecase_version.running
    usecase_version.usecase.delete()


@pytest.fixture(scope='module', params=training_types)
def setup_usecase_class(request):
    usecase_name = '{}_{}'.format(request.param[0:5], TESTING_ID)
    uc = supervised_from_filename(request.param, usecase_name)
    uc.wait_until(
        lambda usecase: (len(usecase.models) > 0) or (usecase._status['state'] == 'failed'))
    assert uc.running
    uc.stop()
    uc.wait_until(lambda usecase: usecase._status['state'] == 'done', timeout=60)
    assert uc._status['state'] == 'done'
    yield request.param, uc
    _ = uc.usecase.delete()


options_parameters = ('options',
                      [{'confidence': False},
                       {'confidence': True}])

predict_u_options_parameters = ('options',
                                [{'confidence': False, 'explain': True},
                                 {'confidence': True}])

predict_test_ids = [('confidence-' if opt['confidence'] else 'normal-')
                    for opt in predict_u_options_parameters[1]]


class TestUCGeneric:
    def test_check_config(self, setup_usecase_class):
        training_type, uc = setup_usecase_class
        assert all([c in uc.drop_list for c in DROP_COLS])
        uc.update_status()
        assert set(uc.feature_list) == set(uc_config.features)
        assert set(uc.advanced_models_list) == set(uc_config.advanced_models)
        assert set(uc.simple_models_list) == set(uc_config.simple_models)
        assert set(uc.normal_models_list) == set(uc_config.normal_models)


class TestPredict:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_predict(self, setup_usecase_class, options):
        training_type, uc = setup_usecase_class
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(training_type)))
        preds = uc.predict(data, **options)
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
        pred = uc.predict_single(data.iloc[0].to_dict(), **options)
        assert pred is not None


class TestInfos:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_info(self, setup_usecase_class: Tuple[str, ClassicUsecaseVersion], options):
        training_type, uc = setup_usecase_class
        # test models
        assert len(uc.models) > 0
        # test Score
        assert uc.score is not None
        # test cv
        df_cv = uc.get_cv()
        assert isinstance(df_cv, pd.DataFrame)
        # test hyper parameters
        hyper_params = uc.best_model.hyperparameters
        assert hyper_params is not None
        # test feature importance
        feat_importance = uc.best_model.feature_importance
        assert list(feat_importance.columns) == ['feature', 'importance']
        # test correlation matrix
        matrix = uc.correlation_matrix
        assert isinstance(matrix, pd.DataFrame)
        # test schema
        schema = uc.schema
        assert schema is not None
        # test features stats
        stats = uc.features_stats
        assert stats is not None
        # test fastest model
        model = uc.fastest_model
        # test usecase version id
        assert model.usecase_version_id == uc._id
        # test print info
        uc.print_info()
        assert isinstance(uc.models, list)
        model = uc.models[0]
        model_copy = pio.Model.from_id(model._id)
        assert isinstance(model.hyperparameters, dict)
        assert model_copy.hyperparameters == model.hyperparameters

        print("uc.status", uc.status)

        assert isinstance(model.cross_validation, pd.DataFrame)
        assert isinstance(model.chart(), dict)
        if training_type == 'classification':
            assert isinstance(model_copy, ClassificationModel)
            assert model_copy.optimal_threshold == model.optimal_threshold
            assert isinstance(model.get_dynamic_performances(), dict)
