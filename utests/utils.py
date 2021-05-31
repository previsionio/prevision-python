import uuid

import previsionio as pio

DROP_COLS = ['feat_0']


def train_model(project_id, uc_name, dataset, training_type, training_type_func, training_config):
    project = pio.Project.from_id(project_id)
    training_type_func = getattr(project, training_type_func)
    return training_type_func(uc_name,
                             dataset,
                             pio.ColumnConfig(target_column='target',
                                              drop_list=DROP_COLS
                                              ),
                             training_config=training_config)


def get_testing_id():
    return str(uuid.uuid4())
