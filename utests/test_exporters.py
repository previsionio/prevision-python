import copy
import pytest
import previsionio as pio
from previsionio.project import connectors_names
from .utils import get_testing_id

from .connectors_config import ftp_config, sftp_config, mysql_config, \
        hive_config, S3_config, gcp_config
TESTING_ID = get_testing_id()
TESTING_ID_CONNECTOR = get_testing_id()
TESTING_ID_EXPORTER = get_testing_id()
PROJECT_NAME = "sdk_test_exporter_" + str(TESTING_ID)
PROJECT_ID = ""

pio.config.default_timeout = 120


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id


#def teardown_module(module):
#    project = pio.Project.from_id(PROJECT_ID)
#    project.delete()


connectors = {
        'FTP': ftp_config,
        'SFTP': sftp_config,
        'SQL': mysql_config,
        'HIVE': hive_config,
        'S3': S3_config,
        'GCP': gcp_config
    }


def test_exporter_FTP():
    project = pio.Project.from_id(PROJECT_ID)
    connector = project.create_ftp_connector("test_ftp_connector", ftp_config['host'],
                                             ftp_config['port'], ftp_config['username'],
                                             ftp_config['password'])
    exporter = project.create_exporter(connector, 'test_ftp_exporter',
                                       description="test_ftp_exporter description",
                                       path='titanic_765765.csv',
                                       write_mode=pio.ExporterWriteMode.timestamp)
    assert exporter is not None
    #exporter.apply_file('utests/data/titanic.csv', 'titanic')


def test_exporter_SFTP():
    project = pio.Project.from_id(PROJECT_ID)
    connector = project.create_sftp_connector("test_ftp_connector", sftp_config['host'],
                                              sftp_config['port'], sftp_config['username'],
                                              sftp_config['password'])
    exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                       description="test_sftp_exporter description",
                                       path='/upload/test_sqk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.timestamp)
    assert exporter is not None
    print("exporter._id", exporter._id)
    #dataset = project.create_dataset('test_exporter',
    #                                file_name='utests/data/titanic.csv')
    #exporter.apply_dataset(dataset)
    export = exporter.apply_file('utests/data/titanic.csv', 'titanic')



def test_exporter_MySQL():
    project = pio.Project.from_id(PROJECT_ID)
    connector = project.create_sql_connector("test_sftp_connector", mysql_config['host'],
                                              mysql_config['port'], mysql_config['username'],
                                              mysql_config['password'])
    exporter = project.create_exporter(connector, 'test_sftp_exporter',
                                       description="test_sftp_exporter description",
                                       database=mysql_config['database'],
                                       table=mysql_config['table'],
                                       write_mode=pio.ExporterWriteMode.append)
    assert exporter is not None


def test_exporter_S3():
    project = pio.Project.from_id(PROJECT_ID)
    connector = project.create_s3_connector("test_s3_connector", username=S3_config['username'],
                                            password=S3_config['password'])
    exporter = project.create_exporter(connector, 'test_s3_exporter',
                                       description="test_s3_exporter description",
                                       bucket=S3_config['bucket'],
                                       path='/test_sdk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.timestamp)
    assert exporter is not None


def test_exporter_GCP_bucket():
    project = pio.Project.from_id(PROJECT_ID)
    connector = project.create_gcp_connector("test_gcp_connector",
                                             googleCredentials=gcp_config['googleCredentials'])
    exporter = project.create_exporter(connector, 'test_gcp_exporter',
                                       description="test_gcp_exporter description",
                                       bucket=gcp_config['bucket'],
                                       path='/test_sdk/titanic.csv',
                                       write_mode=pio.ExporterWriteMode.timestamp)
    assert exporter is not None