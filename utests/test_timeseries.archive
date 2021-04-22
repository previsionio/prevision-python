import os
import pandas as pd
import numpy as np
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id

pio.config.zip_files = True

TESTING_ID = get_testing_id()


def setup_module():
    remove_datasets(DATA_PATH)
    make_supervised_datasets(DATA_PATH)


def teardown_module(module):
    remove_datasets(DATA_PATH)
    for ds in pio.Dataset.list():
        if TESTING_ID in ds.name:
            ds.delete()
    for uc_dict in pio.Supervised.list():
        uc = pio.Supervised.from_id(uc_dict['usecaseId'])
        if TESTING_ID in uc.name:
            uc.delete()


def get_data(path, groups):
    if isinstance(groups, tuple):
        a, b = groups
        data = pd.concat([pd.read_csv(path).assign(group1=i, group2=j) for i in range(a) for j in range(b)])
        group_list = ['group1', 'group2']
    elif groups == 1:
        data = pd.read_csv(path)
        group_list = []
    else:
        data = pd.concat([pd.read_csv(path).assign(group=i) for i in range(groups)])
        group_list = ['group']

    return data, group_list


def train_model(uc_name, groups=1, time_window=pio.TimeWindow(-90, -30, 1, 15)):
    path = os.path.join(DATA_PATH, 'ts.csv')
    data, group_list = get_data(path, groups)
    fname = '{}_{}.csv'.format(uc_name, '-'.join(group_list))
    data.to_csv(fname, index=False)
    dataset = pio.Dataset.new(name=uc_name,
                              dataframe=data)

    uc_config = pio.TrainingConfig(normal_models=[pio.Model.LinReg],
                                   lite_models=[pio.Model.LinReg],
                                   features=[pio.Feature.Counts],
                                   profile=pio.Profile.Quick)

    col_config = pio.ColumnConfig(target_column='target',
                                  time_column='time',
                                  # group_columns=group_list
                                  )

    uc = pio.TimeSeries.fit(uc_name,
                            dataset,
                            time_window=time_window,
                            training_config=uc_config,
                            column_config=col_config)
    return uc


windows = [
    # dws, dwe, fws, fwe
    (-10, -5, 3, 4),
    (-90, -30, 1, 15),
    (-17, -15, 1, 3),
]

wrong_windows = [
    (-10, -90, 1, 15),
    (90, -15, 1, 15),
    (-90, -10, -1, 15)
]


@pytest.mark.parametrize(
    'groups', [1, 3, (2, 2)],
    ids=['no groups', '3 groups', '4 groups - 2 columns (2, 2)']
)
def test_ts_groups(groups):
    group_name = '{}_{}'.format(str(groups[0]), str(groups[1])) if isinstance(groups, tuple) else groups
    uc_name_asked = 'ts_{}grp_{}'.format(group_name, TESTING_ID)
    uc = train_model(uc_name_asked, groups)
    uc.wait_until(lambda usecase: len(usecase) > 0)
    uc.stop()
    usecases = pio.Supervised.list()
    assert uc.id in [u['usecaseId'] for u in usecases]


def time_window_test(dws, dwe, fws, fwe):
    ts_label = '_'.join(str(s).replace('-', 'm') for s in (dws, dwe, fws, fwe))
    uc_name_asked = 'ts_time{}_{}'.format(ts_label, TESTING_ID)

    uc = train_model(uc_name_asked,
                     time_window=pio.TimeWindow(dws, dwe, fws, fwe))
    uc_name_returned = uc.name

    uc.wait_until(lambda usecase: len(usecase) > 0)
    uc.stop()
    return uc_name_returned


@pytest.mark.parametrize('dws, dwe, fws, fwe', windows,
                         ids=['-'.join(str(s) for s in w) for w in windows])
def test_time_window(dws, dwe, fws, fwe):
    uc_name_returned = time_window_test(dws, dwe, fws, fwe)
    usecases = [uc['name'] for uc in pio.Supervised.list()]
    assert uc_name_returned in usecases


@pytest.mark.parametrize('dws, dwe, fws, fwe', wrong_windows,
                         ids=['-'.join(str(s) for s in w) for w in wrong_windows])
def test_wrong_time_window(dws, dwe, fws, fwe):
    with pytest.raises(pio.TimeWindowException):
        time_window_test(dws, dwe, fws, fwe)


ts_params = [(1, False), (3, False)]
ts_ids = ['no groups-legacy', '3 groups-legacy']


@pytest.fixture(scope='class', params=ts_params, ids=ts_ids)
def setup_ts_class(request):
    groups = request.param
    group_name = '{}_{}'.format(str(groups[0]), str(groups[1])) if isinstance(groups, tuple) else groups
    uc_name = 'ts_{}grp_{}'.format(group_name, TESTING_ID)
    uc = train_model(uc_name, groups)

    uc.wait_until(lambda usecase: len(usecase) > 0)
    uc.stop()
    yield groups, uc
    uc.delete()


options_parameters = ('options', [
    {'confidence': True, 'use_best_single': True},
    {'confidence': True, 'use_best_single': False},
    {'confidence': False, 'use_best_single': True},
    {'confidence': False, 'use_best_single': False}
])

predict_test_ids = [('confidence-' if opt['confidence'] else 'normal-') +
                    ('best_single' if opt['use_best_single'] else 'best_model')
                    for opt in options_parameters[1]]


class TestPredict:
    @pytest.mark.parametrize(*options_parameters, ids=predict_test_ids)
    def test_predict(self, setup_ts_class, options):
        groups, uc = setup_ts_class

        path = os.path.join(DATA_PATH, 'ts.csv')
        test_data, group_list = get_data(path, groups)
        test_data.loc[test_data['time'] > '2018-01-01', 'target'] = np.nan
        preds = uc.predict(test_data, **options)
        assert groups * len(preds) == (test_data['time'] >= '2018-01-01').sum()
