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

    # load some data from a CSV file
    data_path = 'helloworld.csv'
    dataset = pio.Dataset.new(name='helloworld', file_name=data_path)

    # or use a pandas DataFrame
    dataframe = pd.read_csv(data_path)
    dataset = pio.Dataset.new(name='helloworld', dataframe=dataframe)

If you have a datasource you to take a snapshot of to create a dataset (see :ref:`managing_datasources_connectors`),
then use the SDK resource object in your arguments:

.. code-block:: py

    datasource = pio.Dataset.from_name('my_datasource')
    dataset = pio.Dataset.new(name='helloworld', datasource=datasource)

Listing available datasets
--------------------------

To get a list of all the datasets currently available on the platform (in your workspace), use the ``list()``
method:

.. code-block:: py

    datasets = pio.Dataset.list()
    for dataset in datasets:
        print(dataset.name)


Fetching data from the platform
--------------------------------

If you already uploaded a dataset on the platform and want to grab it locally to perform some preprocessing,
or a train/test split, simply use the ``from_name()`` or ``from_id()`` SDK methods:

.. code-block:: py

    # load a dataset by name
    dataset = pio.Dataset.from_name('helloworld')

    # or by unique id
    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')
