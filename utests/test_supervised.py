import os
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

type_problem_2_pio_class = {
    'regression': "fit_regression",
    'classification': "fit_classification",
    'multiclassification': "fit_multiclassification",
}
type_problems = type_problem_2_pio_class.keys()


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


def supervised_from_filename(type_problem, uc_name):
    dataset = test_datasets[type_problem]
    type_problem_class = type_problem_2_pio_class[type_problem]
    return train_model(PROJECT_ID, uc_name, dataset, type_problem, type_problem_class, uc_config)


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

    new_version_uc_name = uc_name + '_new'
    usecase_new_version = usecase_version.new_version(new_version_uc_name)
    print("usecase_new_version.usecase_id", usecase_new_version.usecase_id)
    print("usecase_new_version.project_id", usecase_new_version.project_id)
    #usecases = pio.Usecase.list(PROJECT_ID)
    usecase_versions = usecase_version.usecase.versions
    assert usecase_new_version._id in [u['_id'] for u in usecase_versions]

    usecase_new_version.usecase.delete()

    usecases = pio.Usecase.list(PROJECT_ID)
    assert uc_name not in [u.name for u in usecases]


def test_stop_running_usecase():
    uc_name = TESTING_ID + '_file_run'
    usecase_version = supervised_from_filename('regression', uc_name)
    usecase_version.wait_until(lambda usecase: len(usecase.models) > 0)
    assert usecase_version.running
    usecase_version.stop()
    usecase_version.update_status()
    assert not usecase_version.running
    usecase_version.usecase.delete()


@pytest.fixture(scope='module', params=type_problems)
def setup_usecase_class(request):
    usecase_name = '{}_{}'.format(request.param[0:5], TESTING_ID)
    uc = supervised_from_filename(request.param, usecase_name)
    uc.wait_until(lambda usecase: len(usecase.models) > 0)
    uc.stop()
    uc.wait_until(lambda usecase: usecase._status['state'] == 'done')
    yield request.param, uc
    usecase = uc.usecase.delete()


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
        type_problem, uc = setup_usecase_class
        assert all([c in uc.drop_list for c in DROP_COLS])
        uc.update_status()
        assert sorted(uc.fe_selected_list) == sorted(uc_config.fe_selected_list)
        assert sorted(uc.normal_models_list) == sorted(uc_config.normal_models)
        assert sorted(uc.simple_models_list) == sorted(uc_config.simple_models)


class TestPredict:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_predict(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(type_problem)))
        preds = uc.predict(data, **options)
        assert len(preds) == len(data)
        if options['confidence']:
            if type_problem == 'regression':
                conf_cols = ['_quantile={}'.format(q) for q in [1, 5, 10, 25, 50, 75, 95, 99]]
                for q in conf_cols:
                    assert any([q in col for col in preds])
            elif type_problem == 'classification':
                assert 'confidence' in preds
                assert 'credibility' in preds
        # test_predict_unit
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(type_problem)))
        pred = uc.predict_single(data.iloc[0].to_dict(), **options)
        assert pred is not None


class TestInfos:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_info(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
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

        assert isinstance(model.cross_validation, pd.DataFrame)
        assert isinstance(model.chart(), dict)
        if type_problem == 'classification':
            assert model_copy.optimal_threshold == model.optimal_threshold
            assert isinstance(model.get_dynamic_performances(), dict)
