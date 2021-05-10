.. _using_datasets:

Loading & fetching datasets
===========================

Loading up data
---------------

To use a dataset in a Prevision.io's usecase, you need to upload it on the platform. This can be done
on the online platform, in the "Data" page, or through the SDK.

When using the SDK, you can reference a file path directly, or use a pre-read ``pandas`` dataframe, to
easily create a new Prevision.io dataset on the platform:

.. code-block:: py

    # create a project
    project = pio.Project.new(name="project_name",
                              description="project description")
    # load some data from a CSV file
    data_path = 'helloworld.csv'
    dataset = project.create_dataset(name='helloworld', file_name=data_path)

    # or use a pandas DataFrame
    dataframe = pd.read_csv(data_path)
    dataset = project.create_dataset(name='helloworld', dataframe=dataframe)

If you have a datasource you to take a snapshot of to create a dataset (see :ref:`managing_datasources_connectors`),
then use the SDK resource object in your arguments:

.. code-block:: py

    datasource = pio.DataSource.from_id('my_datasource_id')
    dataset = project.create_dataset(name='helloworld', datasource=datasource)

Listing available datasets
--------------------------

To get a list of all the datasets currently available on the platform (in your workspace), use the ``list_datasets()``
method:

.. code-block:: py

    datasets = project.list_datasets()
    for dataset in datasets:
        print(dataset.name)


Fetching data from the platform
--------------------------------

If you already uploaded a dataset on the platform and want to grab it locally to perform some preprocessing,
or a train/test split, simply use the ``from_id()`` SDK methods:

.. code-block:: py

    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')
