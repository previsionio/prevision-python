import time
import previsionio as pio
from .utils import get_testing_id


TESTING_ID = get_testing_id()
PROJECT_NAME = "sdk_test_exporter_" + str(TESTING_ID)

pio.config.default_timeout = 600


def setup_module(module):
    # Create project
    global project
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")

    # Create dataset
    global dataset
    dataset = project.create_dataset('test_exporter',
                                     file_name='utests/data_exporer/titanic.csv')

    # Train one model
    training_config = pio.TrainingConfig(advanced_models=[],
                                         normal_models=[],
                                         simple_models=[pio.SimpleModel.DecisionTree],
                                         features=[],
                                         profile=pio.Profile.Quick)
    column_config = pio.ColumnConfig(target_column='Survived', id_column='PassengerId')
    experiment_version = project.fit_classification(
        name='test_exporter_classif',
        dataset=dataset,
        column_config=column_config,
        metric=pio.metrics.Classification.AUC,
        training_config=training_config,
        holdout_dataset=None,
    )

    # Create validation_prediction
    experiment_version.wait_until(
        lambda experimentv: (len(experimentv.models) > 0) or (experimentv._status['state'] == 'failed'))
    if experiment_version._status['state'] == 'failed':
        raise 'Could not train experiment'
    global validation_prediction
    validation_prediction = experiment_version.predict_from_dataset(dataset)

    # Create experiment deployment
    uc_best_model = experiment_version.best_model
    experiment_deployment = project.create_experiment_deployment('test_sdk_' + TESTING_ID, uc_best_model)

    # Create deployment_prediction
    experiment_deployment.wait_until(lambda experimentd: experimentd.run_state == 'done')
    global deployment_prediction
    deployment_prediction = experiment_deployment.predict_from_dataset(dataset)


def teardown_module(module):
    project.delete()


def check_exporter_and_exports(exporter):
    exporters = project.list_exporter(True)
    exporters_id = [exprtr._id for exprtr in exporters]
    assert exporter._id in exporters_id

    export = exporter.export_dataset(dataset, wait_for_export=True)
    check_export(exporter, export)

    time.sleep(1)
    export = exporter.export_file('utests/data/titanic.csv', wait_for_export=True, origin="pipeline-file",
                                  pipeline_scheduled_run_id="611a303307edd8001cb046d5")
    check_export(exporter, export)

    time.sleep(1)
    export = exporter.export_prediction(validation_prediction, wait_for_export=True)
    check_export(exporter, export)

    time.sleep(1)
    export = exporter.export_prediction(deployment_prediction, wait_for_export=True)
    check_export(exporter, export)

    exporter2 = pio.Exporter.from_id(exporter._id)
    assert exporter2._id == exporter._id

    exporter.delete()
    exporters = project.list_exporter(True)
    exporters_id = [exprtr._id for exprtr in exporters]
    assert exporter._id not in exporters_id


def check_export(exporter, export):
    exports = exporter.list_exports()
    exports_ids = [exprt._id for exprt in exports]
    assert export._id in exports_ids


def test_exporter_FTP():
    from .connectors_config import ftp_config
    connector = project.create_ftp_connector("test_ftp_connector", ftp_config['host'],
                                             ftp_config['port'], ftp_config['username'],
                                             ftp_config['password'])
    exporter = project.create_exporter(connector, 'test_ftp_exporter',
                                       description="test_ftp_exporter description",
                                       path='titanic_765765.csv',
                                       write_mode=pio.ExporterWriteMode.replace)
    check_exporter_and_exports(exporter)


def test_exporter_SFTP():
    from .connectors_config import sftp_config
    connector = project.create_sftp_connector("test_ftp_connector", sftp_config['host'],
                                              sftp_config['port'], sftp_config['username'],
                                              sftp_config['password'])
    exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                       description="test_sftp_exporter description",
                                       path='/upload/test_sqk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.replace)
    check_exporter_and_exports(exporter)


def test_exporter_MySQL():
    from .connectors_config import mysql_config
    connector = project.create_sql_connector("test_sftp_connector", mysql_config['host'],
                                             mysql_config['port'], mysql_config['username'],
                                             mysql_config['password'])
    exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                       description="test_sftp_exporter description",
                                       database=mysql_config['database'],
                                       table=mysql_config['table'],
                                       write_mode=pio.ExporterWriteMode.replace)
    check_exporter_and_exports(exporter)


def test_exporter_S3():
    from .connectors_config import S3_config
    connector = project.create_s3_connector("test_s3_connector", username=S3_config['username'],
                                            password=S3_config['password'])
    exporter = project.create_exporter(connector, 'test_s3_exporter',
                                       description="test_s3_exporter description",
                                       bucket=S3_config['bucket'],
                                       path='/test_sdk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.replace)
    check_exporter_and_exports(exporter)


def test_exporter_GCP_bucket():
    from .connectors_config import gcp_config
    connector = project.create_gcp_connector("test_gcp_connector",
                                             googleCredentials=gcp_config['googleCredentials'])
    exporter = project.create_exporter(connector, 'test_gcp_exporter',
                                       description="test_gcp_exporter description",
                                       bucket=gcp_config['bucket'],
                                       path='/test_sdk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.replace)
    check_exporter_and_exports(exporter)
