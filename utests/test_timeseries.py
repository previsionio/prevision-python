import os

import pandas as pd
import numpy as np
import pytest
import previsionio as pio
from .datasets import make_supervised_datasets, remove_datasets
from . import DATA_PATH
from .utils import get_testing_id

from typing import Union
from pandas import DataFrame, Series
FrameOrSeriesUnion = Union["DataFrame", "Series"]


pio.config.zip_files = True

TESTING_ID = get_testing_id()
PROJECT_NAME = "sdk_test_experiment_timeseries_" + str(TESTING_ID)
PROJECT_ID = ""
pio.config.zip_files = False
pio.config.default_timeout = 1000


def setup_module():
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id
    remove_datasets(DATA_PATH)
    make_supervised_datasets(DATA_PATH)


def teardown_module(module):
    remove_datasets(DATA_PATH)
    project = pio.Project.from_id(PROJECT_ID)
    project.delete()


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
    project = pio.Project.from_id(PROJECT_ID)
    dataset = project.create_dataset(name=uc_name,
                                     dataframe=data)

    uc_config = pio.TrainingConfig(advanced_models=[pio.AdvancedModel.LinReg],
                                   normal_models=[pio.NormalModel.LinReg],
                                   features=[pio.Feature.Counts],
                                   profile=pio.Profile.Quick)

    col_config = pio.ColumnConfig(target_column='target',
                                  time_column='time',
                                  # group_columns=group_list
                                  )

    uc = project.fit_timeseries_regression(uc_name,
                                           dataset,
                                           time_window=time_window,
                                           training_config=uc_config,
                                           column_config=col_config)
    return uc


windows = [
    (-10, -5, 3, 4),
    (-90, -30, 1, 15)
]

wrong_windows = [
    (-10, -90, 1, 15),
    (90, -15, 1, 15)
]


@pytest.mark.parametrize(
    'groups', [1, 3, (2, 2)],
    ids=['no groups', '3 groups', '4 groups - 2 columns (2, 2)']
)
def test_ts_groups(groups):
    group_name = '{}_{}'.format(str(groups[0]), str(groups[1])) if isinstance(groups, tuple) else groups
    uc_name_asked = 'ts_{}grp_{}'.format(group_name, TESTING_ID)
    experiment_version = train_model(uc_name_asked, groups)
    experiment_version.wait_until(lambda experiment: len(experiment) > 0)
    experiment_version.stop()
    experiment = pio.Experiment.from_id(experiment_version.experiment_id)
    project = pio.Project.from_id(PROJECT_ID)
    experiments = project.list_experiments()
    assert experiment.id in [uc.id for uc in experiments]

    path = os.path.join(DATA_PATH, 'ts.csv')
    test_data, group_list = get_data(path, groups)
    test_data.loc[test_data['time'] > '2018-01-01', 'target'] = np.nan
    # preds = experiment_version.predict(test_data, confidence=False)
    # preds = experiment_version.predict(test_data, confidence=True)


def time_window_test(dws, dwe, fws, fwe):
    ts_label = '_'.join(str(s).replace('-', 'm') for s in (dws, dwe, fws, fwe))
    uc_name_asked = 'ts_time{}_{}'.format(ts_label, TESTING_ID)
    uc = train_model(uc_name_asked,
                     time_window=pio.TimeWindow(dws, dwe, fws, fwe))
    uc.wait_until(lambda experimentv: len(experimentv.models) > 0)
    uc.stop()
    return uc


@pytest.mark.parametrize('dws, dwe, fws, fwe', windows,
                         ids=['-'.join(str(s) for s in w) for w in windows])
def test_time_window(dws, dwe, fws, fwe):
    experiment_version = time_window_test(dws, dwe, fws, fwe)
    experiment = pio.Experiment.from_id(experiment_version.experiment_id)
    project = pio.Project.from_id(PROJECT_ID)
    experiments = project.list_experiments()
    assert experiment.id in [experiment.id for experiment in experiments]
    experiments_versions = experiment.versions
    assert experiment_version.id in [experiment_version.id for experiment_version in experiments_versions]


def test_version():
    dws, dwe, fws, fwe = (-10, -5, 3, 4)
    ts_label = '_'.join(str(s).replace('-', 'm') for s in (dws, dwe, fws, fwe))
    uc_name_asked = 'ts_time{}_{}'.format(ts_label, TESTING_ID)

    uc = train_model(uc_name_asked,
                     time_window=pio.TimeWindow(dws, dwe, fws, fwe))

    uc.wait_until(lambda experimentv: len(experimentv.models) > 0)
    uc.stop()
    new_uc = uc.new_version()
    new_uc.wait_until(lambda experimentv: len(experimentv.models) > 1)
    uc.stop()


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
    experiment_version = train_model(uc_name, groups)

    experiment_version.wait_until(lambda experimentv: len(experimentv.models) > 0)
    experiment_version.stop()
    yield groups, experiment_version
    pio.Experiment.from_id(experiment_version.experiment_id).delete()
