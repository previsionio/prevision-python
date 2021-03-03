import os
import pandas as pd
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import train_model, get_testing_id, DROP_COLS

TESTING_ID = get_testing_id()

pio.config.zip_files = False
pio.config.default_timeout = 80

uc_config = pio.TrainingConfig(normal_models=[pio.Model.LinReg],
                               lite_models=[pio.Model.LinReg],
                               simple_models=[pio.SimpleModel.DecisionTree],
                               features=[pio.Feature.Counts],
                               profile=pio.Profile.Quick)
test_datasets = {}

type_problem_2_pio_class = {
    'regression': pio.Regression,
    'classification': pio.Classification,
    'multiclassification': pio.MultiClassification,
}
type_problems = type_problem_2_pio_class.keys()


def make_pio_datasets(paths):
    for problem_type, p in paths.items():
        dataset = pio.Dataset.new(p.split('/')[-1].replace('.csv', str(TESTING_ID) + '.csv'), dataframe=pd.read_csv(p))
        test_datasets[problem_type] = dataset


def setup_module(module):
    remove_datasets(DATA_PATH)
    paths = make_supervised_datasets(DATA_PATH)
    make_pio_datasets(paths)


def teardown_module(module):
    remove_datasets(DATA_PATH)
    for ds in pio.Dataset.list():
        if TESTING_ID in ds.name:
            ds.delete()
    for uc_dict in pio.Supervised.list():
        uc = pio.Supervised.from_id(uc_dict['usecase_id'])
        if TESTING_ID in uc.name:
            uc.delete()


def supervised_from_filename(type_problem, uc_name):
    dataset = test_datasets[type_problem]
    type_problem_class = type_problem_2_pio_class[type_problem]
    return train_model(uc_name, dataset, type_problem, type_problem_class, uc_config)


def test_delete_usecase():
    uc_name = TESTING_ID + '_file_del'
    uc = supervised_from_filename('regression', uc_name)
    usecases = pio.Supervised.list()
    assert uc_name in [u['name'] for u in usecases]
    uc.delete()
    usecases = pio.Supervised.list()
    assert uc_name not in [u['name'] for u in usecases]


def test_stop_running_usecase():
    uc_name = TESTING_ID + '_file_run'
    uc = supervised_from_filename('regression', uc_name)
    uc.wait_until(lambda usecase: len(usecase) > 0)
    assert uc.running

    uc.stop()
    uc.update_status()
    assert not uc.running
    uc.delete()


# def test_uc_from_name():
#     uc_name_asked = TESTING_ID + '_df_name'
#     uc = supervised_from_filename('regression', uc_name_asked)
#     uc_name_returned = uc.name
#     uc2 = pio.Regression.from_name(uc_name_returned)
#     uc.wait_until(lambda usecase: usecase.get_nb_models() > 0)
#     assert uc2.name == uc_name_returned
#     uc.delete()


@pytest.fixture(scope='module', params=type_problems)
def setup_usecase_class(request):
    usecase_name = '{}_{}'.format(request.param[0:5], TESTING_ID)
    uc = supervised_from_filename(request.param, usecase_name)
    # uc.wait_until(lambda usecase: len(usecase) > 0)
    # time.sleep(30)
    # uc.stop()
    uc.wait_until(lambda usecase: usecase._status['state'] == 'done')
    yield request.param, uc
    uc.delete()


options_parameters = ('options',
                      [{'confidence': False, 'use_best_single': False},
                       {'confidence': False, 'use_best_single': True},
                       pytest.param({'confidence': True, 'use_best_single': False}),
                       {'confidence': True, 'use_best_single': True}])

predict_u_options_parameters = ('options',
                                [{'confidence': False, 'use_best_single': False, 'explain': True},
                                 {'confidence': False, 'use_best_single': True},
                                 {'confidence': True, 'use_best_single': False},
                                 {'confidence': True, 'use_best_single': True}])

predict_test_ids = [('confidence-' if opt['confidence'] else 'normal-') +
                    ('best_single' if opt['use_best_single'] else 'best_model')
                    for opt in predict_u_options_parameters[1]]


class TestUCGeneric:
    def test_drop_col(self, setup_usecase_class):
        type_problem, uc = setup_usecase_class
        assert all([c in uc.drop_list for c in DROP_COLS])

    def test_drop_fe(self, setup_usecase_class):
        type_problem, uc = setup_usecase_class
        uc.update_status()
        assert sorted(uc.fe_selected_list) == sorted(uc_config.fe_selected_list)

    def test_drop_models(self, setup_usecase_class):
        type_problem, uc = setup_usecase_class
        uc.update_status()
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

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_predict_unit(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class

        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(type_problem)))
        pred = uc.predict_single(**options, **data.iloc[0].to_dict())
        assert pred is not None

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_sk_predict(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class

        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(type_problem)))
        preds = uc.predict(data, **options)
        assert len(preds) == len(data)
        if options['confidence']:
            if type_problem == 'regression':
                conf_cols = ['target_quantile={}'.format(q) for q in [1, 5, 10, 25, 50, 75, 95, 99]]
                for q in conf_cols:
                    assert any(q in col for col in preds)
            elif type_problem == 'classification':
                assert 'confidence' in preds
                assert 'credibility' in preds

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_proba_predict(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        if type_problem == 'classification':
            data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(type_problem)))
            preds = uc.predict_proba(data, **options)
            assert len(preds) == len(data)
            if options['confidence']:
                assert 'confidence' in preds
                assert 'credibility' in preds
        else:
            print('\nInvalid usecase type for predict_proba: "{}"'.format(type_problem))

    # @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    # @pytest.mark.xfail(reason='usecase_info keys change during training')
    # def test_get_usecase_params(self, setup_usecase_class, options):
    #     uc_info_keys = {'GROUP_column', 'algorithms', 'allClusters', 'avg', 'bar', 'bestModelID', 'clusterStats',
    #                     'clusterfe', 'cvFileID', 'datasetStats', 'dataset_stats_id', 'datasets_holdout', 'email',
    #                     'fe_length', 'featureImportance', 'global_end_dw', 'global_end_fw', 'global_start_dw',
    #                     'global_start_fw', 'hopts_length', 'hyperParameters', 'image_training', 'labels', 'losses',
    #                     'metric', 'models', 'nbColumns', 'nbFeatures', 'nbRows', 'nmodels', 'owner', 'pause',
    #                     'predsTargets', 'requestValues', 'runningExplain', 'runningPrediction', 'state', 'stdev',
    #                     'task', 'topicModeling', 'totalPreds', 'train_length', 'tsne', 'types', 'uploadDate',
    #                     'use_case', 'use_case_description'}
    #
    #     type_problem, uc = setup_usecase_class
    #     uc.update_uc_info()
    #
    #     for k in uc.usecase_info.keys():
    #         assert k in uc_info_keys


class TestInfos:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_nb_models(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        nb_models = len(uc)
        assert isinstance(nb_models, int)
        assert nb_models > 0

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_score(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        assert uc.score is not None

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_cv(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        df_cv = uc.get_cv()
        assert isinstance(df_cv, pd.DataFrame)

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_get_hyperparameters(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        hyper_params = uc.best_model.hyperparameters
        assert hyper_params is not None

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_get_feature_importance(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        feat_importance = uc.best_model.feature_importance
        assert list(feat_importance.columns) == ['feature', 'importance']

    # @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    # def test_get_correlation_matrix(self, setup_usecase_class, options):
    #     type_problem, uc = setup_usecase_class
    #     matrix = uc.correlation_matrix
    #     assert isinstance(matrix, pd.DataFrame)

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_get_schema(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        schema = uc.schema
        assert isinstance(schema, pd.DataFrame)

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_get_feature_stats(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        stats = uc.feature_stats
        assert stats is not None

    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_get_fastest_model(self, setup_usecase_class, options):
        type_problem, uc = setup_usecase_class
        model = uc.fastest_model
        assert model.uc_id == uc._id
        assert model.uc_version == uc.version
        assert model.tags['fastest']

    # @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    # def test_print_infos(self, setup_usecase_class, options):
    #     type_problem, uc = setup_usecase_class
    #     uc.print_info()
