import copy
import pytest
import previsionio as pio
from previsionio.project import connectors_names
from .utils import get_testing_id


try: 
    from .connectors_config import ftp_config, sftp_config, mysql_config, \
    hive_config, S3_config, gcp_config
     
    TESTING_ID = get_testing_id()
    TESTING_ID_CONNECTOR = get_testing_id()
    TESTING_ID_DATASOURCE = get_testing_id()
    SQL_CONNECTORS = ['SQL', 'HIVE']
    OBJECT_CONNECTORS = ['FTP', 'SFTP']
    PROJECT_NAME = "sdk_test_datasource_" + str(TESTING_ID)
    PROJECT_ID = ""

    pio.config.default_timeout = 120


    def setup_module(module):
        project = pio.Project.new(name=PROJECT_NAME,
                                description="description test sdk")
        global PROJECT_ID
        PROJECT_ID = project._id


    def teardown_module(module):
        project = pio.Project.from_id(PROJECT_ID)
        for ds in project.list_datasource(all=True):
            if TESTING_ID_DATASOURCE in ds.name:
                ds.delete()
        for conn in project.list_connectors(all=True):
            if TESTING_ID_CONNECTOR in conn.name:
                conn.delete()
        project.delete()


    connectors = {
        'FTP': ftp_config,
        'SFTP': sftp_config,
        'SQL': mysql_config,
        'HIVE': hive_config,
        'S3': S3_config,
        'GCP': gcp_config
    }

    connectors_options = ('options', [
        {'type': 'FTP', **connectors['FTP']},
        {'type': 'SFTP', **connectors['SFTP']},
        {'type': 'SQL', **connectors['SQL']},
        {'type': 'HIVE', **connectors['HIVE']},
        {'type': 'S3', **connectors['S3']},
        {'type': 'GCP', **connectors['GCP']},
    ])
    connectors_sql_options = ('options', [c for c in connectors_options[1] if c['type'] in SQL_CONNECTORS])
    connector_test_ids = ['connector-' + opt['type'] for opt in connectors_options[1]]
    connector_sql_test_ids = ['connector-' + opt['type'] for opt in connectors_sql_options[1]]

    connectors_object_options = ('options', [c for c in connectors_options[1] if c['type'] in OBJECT_CONNECTORS])

    connector_object_test_ids = ['connector-' + opt['type'] for opt in connectors_object_options[1]]

    datasources_options = ('options', [
        {'connector': 'FTP', 'path': ftp_config['file']},
        {'connector': 'SFTP', 'path': sftp_config['file']},
        {'connector': 'SQL', 'database': mysql_config['database'], 'table': mysql_config['table']},
        {'connector': 'HIVE', 'database': hive_config['database'], 'table': hive_config['table']},
        {'connector': 'S3', 'bucket': S3_config['bucket'], 'path': S3_config['file']},
        {'connector': 'GCP', 'gCloud': 'BigQuery', 'database': gcp_config['database'], 'table': gcp_config['table']},
        {'connector': 'GCP', 'gCloud': 'Storage', 'bucket': gcp_config['bucket'], 'path': gcp_config['file']},
    ])

    datasource_test_ids = ['datasource-' + opt['connector'] for opt in datasources_options[1]]


    def prepare_connector_options(base_options):
        options = copy.deepcopy(base_options)
        connector_type = options.pop('type')
        options['name'] = connector_type + TESTING_ID_CONNECTOR
        for key in ['database', 'table', 'file', 'bucket']:
            if key in options:
                del options[key]

        return options, connector_type


    @pytest.fixture(scope='module')
    def setup_connector_class():

        def _wrapped_setter(connector_options):
            options, connector_type = prepare_connector_options(connector_options)
            project = pio.Project.from_id(PROJECT_ID)
            connector_func = getattr(project, connectors_names[connector_type])
            connector = connector_func(**options)
            return connector

        return _wrapped_setter


    example_datasource_id = None


    @pytest.mark.parametrize(*connectors_options, ids=connector_test_ids)
    def test_connector_new(setup_connector_class, options):
        conn = setup_connector_class(options)
        assert conn is not None


    @pytest.mark.parametrize(*connectors_options, ids=connector_test_ids)
    def test_connector_new_test(setup_connector_class, options):
        conn = setup_connector_class(options)
        assert conn.test()


    @pytest.mark.parametrize(*connectors_sql_options, ids=connector_sql_test_ids)
    def test_sql_connector_list_databases(setup_connector_class, options):
        conn = setup_connector_class(options)
        databases = conn.list_databases()
        assert databases is not None
        assert len(databases) > 0
        assert options['database'] in databases


    @pytest.mark.parametrize(*connectors_sql_options, ids=connector_sql_test_ids)
    def test_sql_connector_list_tables(setup_connector_class, options):
        conn = setup_connector_class(options)
        tables = conn.list_tables(options['database'])
        assert tables is not None
        assert len(tables) > 0
        assert options['table'] in tables

    # @pytest.mark.parametrize(*connectors_object_options, ids=connector_object_test_ids)
    # def test_object_connector_list_files(setup_connector_class, options):
    #     conn = setup_connector_class(options)
    #     files = conn.list_files()
    #     # web error  {‘items': False}
    #     assert files is not None
    #     assert len(files) > 0
    #     assert options['file'] in files


    @pytest.mark.parametrize(*datasources_options, ids=datasource_test_ids)
    def test_datasource_new(setup_connector_class, options):
        datasource_options = copy.deepcopy(options)
        connector_type = datasource_options.pop('connector')
        connector_options = connectors[connector_type]
        connector_options['type'] = connector_type
        conn = setup_connector_class(connector_options)
        project = pio.Project.from_id(PROJECT_ID)
        datasource_options['connector'] = conn
        datasource_options['name'] = 'datasource_{}_{}'.format(TESTING_ID_DATASOURCE, connector_type)
        datasource = project.create_datasource(**datasource_options)

        global example_datasource_id
        if example_datasource_id is None:
            example_datasource_id = datasource._id
        assert datasource is not None


    def test_datasource_from_id():
        global example_datasource_id
        assert example_datasource_id is not None
        datasource = pio.DataSource.from_id(example_datasource_id)
        assert datasource is not None

except:
    print("datasource tests unavailable")
