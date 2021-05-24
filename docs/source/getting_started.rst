.. _getting_started:

***************
Getting started
***************

The following document is a step by step usage example of the `Prevision.io <https://prevision.io/>`_ Python SDK. The full documentation of the software is available `here <https://previsionio.readthedocs.io/fr/latest/>`_.

Pre-requisites
==============

You need to have an account at cloud.prevision.io or on an on-premise version installed in your company. Contact us or your IT manager for more information.

You will be working on a specific "instance". This instance corresponds to the subdomain at the beginning of the url in your prevision.io address: ``https://<your instance>.prevision.io``.

Get the package
===============

1. clone the git repo:

.. code-block:: bash

    git clone https://github.com/previsionio/prevision-python.git

2. install as a Python package:

.. code-block:: bash

    cd prevision-python
    python setup.py install


Set up your client
==================

Prevision.io's SDK client uses a specific master token to authenticate with the instance's server and allow you to perform various requests. To get your master token, log in the online interface on your instance, navigate to the admin page and copy the token.

You can either set the token and the instance name as environment variables, by specifying
``PREVISION_URL`` and ``PREVISION_MASTER_TOKEN``, or at the beginning of your script:

.. code-block:: python

    import previsionio as pio

    # The client is initialized with your master token and the url of the prevision.io server
    # (or local installation, if applicable)
    url = "https://<your instance>.prevision.io"
    token = "<your token>"
    pio.client.init_client(url, token)

    # You can manage the verbosity (only output warnings and errors by default)
    pio.verbose(
        False,           # whether to activate info logging
        debug=False,     # whether to activate detailed debug logging
        event_log=False, # whether to activate detailed event managers debug logging
    )

Create a project
================

First things first, to upload data or train a usecase, you need to create a project.

.. code-block:: python

    # create project
    project = pio.Project.new(name="project_name",
                              description="project description")

Data
====

To train a usecase, you need to gather some training data. This data must be uploaded to your instance using either a data source, a file path or a :class:`.pandas.DataFrame`.

Managing datasources & connectors
---------------------------------

Datasources and connectors are Prevision.io's way of keeping a link to a source of data and taking snapshots when needed. The avaible data sources are:

- SQL
- HIVE
- FTP
- SFTP
- S3
- GCP

Connectors hold the credentials to connect to the distant data sources. Then you can specify the exact resource to extract from a data source (be it the path to the file to load, the name of the database table to parse...).

For more info on all the options of connectors and datasources, check out the :ref:`api_reference`.

Creating a connector
~~~~~~~~~~~~~~~~~~~~

To create a connector, use the appropriate method of project class. For example,
to create a connector to an SQL database, use the ``create_sql_connector()`` and pass in your credentials:

.. code-block:: py

    connector = project.create_sql_connector('my_sql_connector',
                                             'https://myserver.com',
                                             port=3306,
                                             username='username',
                                             password='password')

Creating a data source
~~~~~~~~~~~~~~~~~~~~~~

After you've created a connector, you need to use a datasource to actually refer to and fetch a resource
in the distant data source. To create a datasource, you need to link the matching connector and to supply
the relevant info, depending on the connector type.

.. code-block:: py

    datasource = project.create_datasource(connector,
                                           'my_sql_datasource',
                                           database='my_db',
                                           table='table1')

You can then create datasets from this datasource as explained in the guide on :ref:`using_datasets`.

Listing available connectors and data sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connectors and datasources already registered on the platform can be listed
using the ``list_connectors()`` and ``list_datasource()`` method from project class:

.. code-block:: py

    connectors = project.list_connectors()
    for connector in connectors:
        print(connector.name)

    datasources = project.list_datasource()
    for datasource in datasources:
        print(datasource.name)

Uploading Data
--------------

.. code-block:: python

    # Upload tabular data from a CSV file
    data_path = 'path/to/your/data.csv'
    dataset = project.create_dataset(name='helloworld', file_name=data_path)

    # or use a pandas DataFrame
    dataframe = pd.read_csv(data_path)
    dataset = project.create_dataset(name='helloworld', dataframe=dataframe)

    # or use a created data source
    datasource = pio.DataSource.from_id('my_datasource_id')
    dataset = project.create_dataset(name='helloworld', datasource=datasource)

    # Upload an image folder
    image_folder_path = 'path/to/your/image_data.zip'
    image_folder = project.create_image_folder(name='helloworld', file_name=image_folder_path)


This will automatically upload the data as a new dataset on your Prevision.io's instance. If you go to the online interface, you will see this new dataset in the list of datasets (in the "Data" tab).

Listing available datasets
--------------------------

To get a list of all the datasets currently available on the platform (in your workspace), use the ``list_datasets()``
method:

.. code-block:: py

    # List tabular datasets
    datasets = project.list_datasets()
    for dataset in datasets:
        print(dataset.name)

    # List image folders
    image_folders = project.list_image_folders()
    for folder in image_folders:
        print(folder.name)

Downloading data from the platform
----------------------------------

If you already uploaded a dataset on the platform and want to grab it locally, simply use the ``Dataset.from_id()`` SDK methods:

.. code-block:: py

    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')

Starting Regression/Classification/Multi-classification
=======================================================

Configuring the dataset
-----------------------

To start a usecase you need to specify the dataset to be used and its configuration (target column, weight column, id column, ...). To get a full documentation check the api documentation of the :class:`.ColumnConfig` in :ref:`config_reference`.

.. code-block:: python

    column_config = pio.ColumnConfig(target_column='TARGET', id_column='ID')

Configuring the training parameters
-----------------------------------

If you want, you can also specify some training parameters, such as which models are used, which transformations are applied, and how the models are optimized. To get a full documentation check the api documentation of the :class:`.TrainingConfig` in :ref:`config_reference`.

.. code-block:: python

    training_config = pio.TrainingConfig(
        advanced_models=[pio.AdvancedModel.LinReg],
        normal_models=[pio.NormalModel.LinReg],
        simple_models=[pio.SimpleModel.DecisionTree],
        features=[pio.Feature.Counts],
        profile=pio.Profile.Quick
    )

Starting training
-----------------

You can now create a new usecase based on :

 - a usecase name
 - a dataset
 - a column config
 - (optional) a metric type
 - (optional) a training config
 - (optional) a holdout dataset (dataset only used for evaluation)

.. code-block:: python

    usecase_version = project.fit_classification(
        name='helloworld_classif',
        dataset=dataset,
        column_config=column_config,
        metric=pio.metrics.Classification.AUC,
        training_config=uc_config,
        holdout_dataset=None,
    )

If you want to use image data for your usecase, you need to provide the API with both the tabular dataset and the image folder:

.. code-block:: python

    usecase_version = project.fit_image_classification(
        name='helloworld_images_classif',
        dataset=(dataset, image_folder),
        column_config=column_config,
        metric=pio.metrics.Classification.AUC,
        training_config=uc_config,
        holdout_dataset=None,
    )

Making predictions
------------------

To make prediction from a dataset and a usecase, you need to wait until at least one model is trained. This can be achieved in the following way:

.. code-block:: python

    # (block until there is at least 1 model trained)
    usecase_version.wait_until(lambda usecasev: len(usecasev.models) > 0)

    # check out the usecase status and other info
    usecase_version.print_info()
    print('Current (best model) score:', usecase_version.score)

    # predict from uploaded dataset on the plateform
    preds = usecase_version.predict_from_dataset(test_dataset)

    # or predict from a `pandas.DataFrame`
    preds = usecase_version.predict(test_dataframe)

Starting Text Similarity
========================

If you want, you can also specify some training parameters, such as which models are used,
which embedding and preprocessing are applied.

.. code-block:: python

    models_parameters_1 = pio.ModelsParameters(pio.ModelEmbedding.TFIDF,
                                               pio.Preprocessing(),
                                               [pio.TextSimilarityModels.BruteForce, pio.TextSimilarityModels.ClusterPruning])
    models_parameters_2 = pio.ModelsParameters(pio.ModelEmbedding.Transformer,
                                               {},
                                               [pio.TextSimilarityModels.BruteForce])
    models_parameters_3 = pio.ModelsParameters(pio.ModelEmbedding.TransformerFineTuned,
                                               {},
                                               [pio.TextSimilarityModels.BruteForce])
    models_parameters = [models_parameters_1, models_parameters_2, models_parameters_3]
    models_parameters = pio.ListModelsParameters(models_parameters=models_parameters)


.. note::

    If you want the default configuration of text similarity models, simply use:

    .. code-block:: python

        models_parameters = pio.ListModelsParameters()


Starting text similarity training
---------------------------------

You can then create a new text similarity usecase based on :

 - a usecase name
 - a dataset
 - a description column config
 - (optional) a queries dataset
 - (optional) a qeries column config
 - (optional) a metric type
 - (optional) a top k
 - (optional) a language
 - (optional) a models parameters list

.. code-block:: python

    usecase_verion = project.fit_text_similarity('helloworld_text_similarity',
                                                 dataset,
                                                 description_column_config,
                                                 metric=pio.metrics.TextSimilarity.accuracy_at_k,
                                                 top_k=10,
                                                 models_parameters=models_parameters)

Monitoring training
-------------------

You can retrieve at any moment the number of models trained so far and the current error score,
as well as some additional info.

.. code-block:: python

    >>> usecase_verion.score
    0.0585

    >>> usecase_verion.print_info()
    scores_cv: 0.0585



You can also wait until a certain condition is reached, such as a number of models or a certain score:

.. code-block:: python

    # will block until there are more than 3 models
    uc.wait_until(lambda usecasev: len(usecasev.models) > 0)

    # will block until error is lower than 0.3 (warning, it may never reach it and wait forever)
    uc.wait_until(lambda usecasev: usecasev.score < .3)


The ``wait_until`` method takes a function that takes the usecase as an argument, and can therefore access any info
relative to the usecase.

Making predictions
------------------

Once we have at least a model, we can start making predictions. We don't need to wait until the complete training
process is done, and we'll always have access to the best model trained so far.

.. code-block:: python

    # we have some test data here:
    data_path = 'data/titanic_test.csv'
    test_dataset = project.create_dataset(name='helloworld_test', file_name=data_path)

    preds = usecase_verion.predict_from_dataset(test_dataset)

    # scikit-learn style:
    df = pd.read_csv(data_path)
    preds = uc.predict(df)

For text similarity, you can create a new prediction based on :
  - a dataset queries
  - a query colmun name
  - (optional) topK
  - (optional) description id column name

.. code-block:: python

    # we have some test data here:
    data_path = 'data/queries_test.csv'
    test_dataset = project.create_dataset(name='helloworld_test', file_name=data_path)

    preds = usecase_verion.predict_from_dataset(test_dataset,
                                                'query',
                                                top_k=10,
                                                queries_dataset_matching_id_description_column='true_item_id')

Additional util methods
=======================

Retrieving a use case
---------------------

Since a use case can be somewhat long to train, it can be useful to separate the training, monitoring and prediction phases.

To do that, we need to be able to recreate a usecase object in python from its name:

.. code-block:: python

    usecase_version = pio.Supervised.from_id('<a usecase id>')
    # usecase_version now has all the same methods as a usecase_version created directly from a file or a dataframe
    usecase_version.print_info()

Stopping and deleting
---------------------

Once you're satisfied with model performance, don't want to wait for the complete training process to be over, or need to free up some resources to start a new training, you can stop the usecase_version simply:

.. code-block:: python

    usecase_version.stop()

You'll still be able to make predictions and get info, but the performance won't improve anymore. Note: there's no difference in state between a stopped usecase and a usecase that has completed its training completely.

You can decide to completely delete the usecase:

.. code-block:: python

    uc = usecase_version.usecase
    uc.delete()

However be careful, in that case any detail about the usecase will be removed, and you won't be able to make predictions from it anymore.
