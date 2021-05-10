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

Connectors and datasources already registered on the platform can be listed
using the ``list_connectors()`` and ``list_datasource()`` method from project class:

.. code-block:: py

    connectors = project.list_connectors()
    for connector in connectors:
        print(connector.name)

    datasources = project.list_datasource()
    for datasource in datasources:
        print(datasource.name)

Creating a connector
--------------------

To create a connector, use the appropriate method of project class. For example,
to create a connector to an SQL database, use the ``create_sql_connector()`` and pass in your credentials:

.. code-block:: py

    connector = project.create_sql_connector('my_sql_connector',
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

    datasource = project.create_datasource(connector,
                                           'my_sql_datasource',
                                           database='my_db',
                                           table='table1')

You can then create datasets from this datasource as explained in the guide on :ref:`using_datasets`.
