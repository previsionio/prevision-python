import uuid

import previsionio as pio

DROP_COLS = ['feat_0']


def train_model(project_id, uc_name, dataset, type_problem, type_problem_class, training_config):
    return type_problem_class.fit(project_id,
                                  uc_name,
                                  dataset,
                                  pio.ColumnConfig(target_column='target',
                                                   drop_list=DROP_COLS
                                                   ),
                                  training_config=training_config)


def get_testing_id():
    return str(uuid.uuid4())
