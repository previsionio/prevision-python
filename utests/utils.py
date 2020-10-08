import previsionio as pio

DROP_COLS = ['feat_0']


def train_model(uc_name, dataset, type_problem, type_problem_class, training_config):
    return type_problem_class.fit(uc_name,
                                  dataset,
                                  pio.ColumnConfig(target_column='target',
                                                   drop_list=DROP_COLS
                                                   ),
                                  training_config=training_config)


def get_testing_id(id_type='usecase'):
    if id_type == 'usecase':
        existing_usecase_names = list(pio.Supervised.list())
        i = 0
        testing_id = 'test_{}'.format(i)
        while any(testing_id in usecase_name for usecase_name in [u['name'] for u in existing_usecase_names]):
            i += 1
            testing_id = 'test_{}'.format(i)
    elif id_type == 'connector':
        existing_connector_names = list(pio.Connector.list())
        i = 0
        testing_id = 'test_{}'.format(i)
        while any(testing_id in connector_name for connector_name in [u.name for u in existing_connector_names]):
            i += 1
            testing_id = 'test_{}'.format(i)
    elif id_type == 'datasource':
        existing_datasource_names = list(pio.DataSource.list())
        i = 0
        testing_id = 'test_{}'.format(i)
        while any(testing_id in datasource_name for datasource_name in [u.name for u in existing_datasource_names]):
            i += 1
            testing_id = 'test_{}'.format(i)
    return testing_id
