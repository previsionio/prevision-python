# import pytest
import previsionio as pio
from .utils import get_testing_id


try:
    from .connectors_config import (ftp_config, sftp_config, mysql_config,
                                    hive_config, S3_config, gcp_config)

    TESTING_ID = get_testing_id()
    TESTING_ID_CONNECTOR = get_testing_id()
    TESTING_ID_EXPORTER = get_testing_id()
    PROJECT_NAME = "sdk_test_exporter_" + str(TESTING_ID)
    PROJECT_ID = ""

    pio.config.default_timeout = 120

    connectors = {
        'FTP': ftp_config,
        'SFTP': sftp_config,
        'SQL': mysql_config,
        'HIVE': hive_config,
        'S3': S3_config,
        'GCP': gcp_config,
    }

    def setup_module(module):
        global project
        project = pio.Project.new(name=PROJECT_NAME,
                                  description="description test sdk")
        global dataset
        dataset = project.create_dataset('test_exporter',
                                         file_name='utests/data/titanic.csv')

    def teardown_module(module):
        project.delete()

    def check_exporter_and_exports(exporter):
        exporters = project.list_exporter(True)
        exporters_id = [exprtr._id for exprtr in exporters]
        assert exporter._id in exporters_id

        export = exporter.apply_file('utests/data/titanic.csv',
                                     wait_for_export=True)
        check_export(export)

        export = exporter.apply_dataset(dataset, wait_for_export=True)
        check_export(export)

        exporter.delete()
        exporters = project.list_exporter(True)
        exporters_id = [exprtr._id for exprtr in exporters]
        assert exporter._id not in exporters_id

    def check_export(exporter, export):
        exports = exporter.list_exports()
        exports_ids = [exprt._id for exprt in exports]
        assert export._id in exports_ids

    def test_exporter_FTP():
        connector = project.create_ftp_connector("test_ftp_connector", ftp_config['host'],
                                                 ftp_config['port'], ftp_config['username'],
                                                 ftp_config['password'])
        exporter = project.create_exporter(connector, 'test_ftp_exporter',
                                           description="test_ftp_exporter description",
                                           path='titanic_765765.csv',
                                           write_mode=pio.ExporterWriteMode.timestamp)
        check_exporter_and_exports(exporter)

    def test_exporter_SFTP():
        connector = project.create_sftp_connector("test_ftp_connector", sftp_config['host'],
                                                  sftp_config['port'], sftp_config['username'],
                                                  sftp_config['password'])
        exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                           description="test_sftp_exporter description",
                                           path='/upload/test_sqk/titanic.csv',
                                           write_mode=pio.ExporterWriteMode.timestamp)
        check_exporter_and_exports(exporter)

    def test_exporter_MySQL():
        connector = project.create_sql_connector("test_sftp_connector", mysql_config['host'],
                                                 mysql_config['port'], mysql_config['username'],
                                                 mysql_config['password'])
        exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                           description="test_sftp_exporter description",
                                           database=mysql_config['database'],
                                           table=mysql_config['table'],
                                           write_mode=pio.ExporterWriteMode.append)
        check_exporter_and_exports(exporter)

    def test_exporter_S3():
        connector = project.create_s3_connector("test_s3_connector", username=S3_config['username'],
                                                password=S3_config['password'])
        exporter = project.create_exporter(connector, 'test_s3_exporter',
                                           description="test_s3_exporter description",
                                           bucket=S3_config['bucket'],
                                           path='/test_sdk/titanic.csv',
                                           write_mode=pio.ExporterWriteMode.timestamp)
        check_exporter_and_exports(exporter)

    def test_exporter_GCP_bucket():
        connector = project.create_gcp_connector("test_gcp_connector",
                                                 googleCredentials=gcp_config['googleCredentials'])
        exporter = project.create_exporter(connector, 'test_gcp_exporter',
                                           description="test_gcp_exporter description",
                                           bucket=gcp_config['bucket'],
                                           path='/test_sdk/titanic.csv',
                                           write_mode=pio.ExporterWriteMode.timestamp)
        check_exporter_and_exports(exporter)

except:
    print("exporter tests unavailable")
