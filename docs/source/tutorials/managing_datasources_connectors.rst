.. _managing_datasources_connectors:

Managing datasources & connectors
=================================

Datasources and connectors are Prevision.io's way of keeping a link to a source of data and
taking snapshots when needed. The distant data source can be an FTP server, a database, an
Amazon bucket...

Connectors hold the credentials to connect to the distant data source and datasources specify
the exact resource to resource from it (be it the path to the file to load, the name of the
database table to parse...).

For more info on all the options of connectors and datasources, check out the :ref:`api_reference`.

Listing available connectors and datasources
--------------------------------------------

Like all SDK API resources, connectors and datasources already registered on the platform can be listed
using the ``list()`` method:

.. code-block:: py

    connectors = pio.Connector.list()
    for connector in connectors:
        print(connector.name)

    datasources = pio.Datasource.list()
    for datasource in datasources:
        print(datasource.name)

Creating a connector
--------------------

To create a connector, use the ``new()`` method of the connector class you want to use. For example,
to create a connector to an SQL database, use the :class:`.SQLConnector` and pass in your credentials:

.. code-block:: py

    connector = pio.SQLConnector.new('my_sql_connector',
                                     'https://myserver.com',
                                     port=3306,
                                     username='username',
                                     password='password')

Creating a datasource
---------------------

After you've created a connector, you need to use a datasource to actually refer to and fetch a resource
in the distant data source. To create a datasource, you need to link the matching connector and to supply
the relevant info, depending on the connector type.

.. code-block:: py

    datasource = pio.Datasource.new(connector,
                                    'my_sql_datasource',
                                    database='my_db',
                                    table='table1')

You can then create datasets from this datasource as explained in the guide on :ref:`using_datasets`.
