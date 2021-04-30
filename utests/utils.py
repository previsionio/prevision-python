import uuid

import previsionio as pio

DROP_COLS = ['feat_0']


def train_model(project_id, uc_name, dataset, type_problem, type_problem_func, training_config):
    project = pio.Project.from_id(project_id)
    type_problem_func = getattr(project, type_problem_func)
    return type_problem_func(uc_name,
                                  dataset,
                                  pio.ColumnConfig(target_column='target',
                                                   drop_list=DROP_COLS
                                                   ),
                                  training_config=training_config)


def get_testing_id():
    return str(uuid.uuid4())
