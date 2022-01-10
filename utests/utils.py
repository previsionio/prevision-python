import os
import json
import uuid
import previsionio as pio

DROP_COLS = ['feat_0']


def train_model(project_id, experiment_name, dataset, training_type, training_type_func, training_config):
    project = pio.Project.from_id(project_id)
    training_type_func = getattr(project, training_type_func)
    return training_type_func(
        experiment_name,
        dataset,
        pio.ColumnConfig(target_column='target', drop_list=DROP_COLS),
        training_config=training_config,
    )


def get_testing_id():
    return str(uuid.uuid4())


def get_connectors_config():
    # Try local config
    if os.path.exists("connectors_config"):
        connectors_config_path = "connectors_config"
        print("\nUsing local connectors_config")
    # Else use config defined in CI/CD
    else:
        connectors_config_path = os.getenv("CONNECTORS_CONFIG_FILE")
        if connectors_config_path is None:
            raise ValueError("connectors tests unavailable, missing config file env var")
        print("\nUsing CI/CD connectors_config")
    return json.load(open(connectors_config_path))
