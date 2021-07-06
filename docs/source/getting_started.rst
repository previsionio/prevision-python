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

.. code-block:: bash

    pip install previsionio


Set up your client
==================

Prevision.io's SDK client uses a specific master token to authenticate with the instance's server and allows you to perform various requests. To get your master token, log in the online interface on your instance, navigate to the admin page and copy the token.

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

Connectors hold the credentials to connect to the distant data sources. Then you can specify the exact resource to extract from a data source (be it the path to the file to load, the name of the database table to parse, ...).

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

For more information on all the available connectors, check out the :ref:`project_reference` full documentation.

Creating a data source
~~~~~~~~~~~~~~~~~~~~~~

After you've created a connector, you need to use a datasource to actually refer to and fetch a resource
in the distant data source. To create a datasource, you need to link the matching connector and to supply
the relevant info, depending on the connector type:

.. code-block:: py

    datasource = project.create_datasource(connector,
                                           'my_sql_datasource',
                                           database='my_db',
                                           table='table1')

For more details on the creation of a datasource, check out the :ref:`project_reference` full documentation of the method ``create_datasource``.

You can then create datasets from this datasource as explained in :ref:`Uploading Data`.

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

.. _Uploading Data:

Uploading Data
--------------

You can upload data from three different sources: a path to a local (``csv``, ``zip``) file, a :class:`.pandas.DataFrame` or a created data source

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

If you already uploaded a dataset on the platform and want to grab it locally, simply use the ``Dataset.from_id()`` SDK method:

.. code-block:: py

    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')

Regression/Classification/Multi-classification usecase
======================================================

Configuring the dataset
-----------------------

To start a usecase you need to specify the dataset to be used and its configuration (target column, weight column, id column, ...). To get a full documentation check the api reference of the :class:`.ColumnConfig` in :ref:`config_reference`.

.. code-block:: python

    column_config = pio.ColumnConfig(target_column='TARGET', id_column='ID')

.. _configuring train:

Configuring the training parameters
-----------------------------------

If you want, you can also specify some training parameters, such as which models are used, which transformations are applied, and how the models are optimized. To get a full documentation check the api reference of the :class:`.TrainingConfig` in :ref:`config_reference`.

.. code-block:: python

    training_config = pio.TrainingConfig(
        advanced_models=[pio.AdvancedModel.LinReg],
        normal_models=[pio.NormalModel.LinReg],
        simple_models=[pio.SimpleModel.DecisionTree],
        features=[pio.Feature.Counts],
        profile=pio.Profile.Quick,
    )

Starting training
-----------------

You can now create a new usecase based on:

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
        training_config=training_config,
        holdout_dataset=None,
    )

If you want to use image data for your usecase, you need to provide the API with both the tabular dataset and the image folder:

.. code-block:: python

    usecase_version = project.fit_image_classification(
        name='helloworld_images_classif',
        dataset=(dataset, image_folder),
        column_config=column_config,
        metric=pio.metrics.Classification.AUC,
        training_config=training_config,
        holdout_dataset=None,
    )

To get an exhaustive list of the available metrics go to the api reference :ref:`metrics_reference`.

.. _making prediction:

Making predictions
------------------

To make predictions from a dataset and a usecase, you need to wait until at least one model is trained. This can be achieved in the following way:

.. code-block:: python

    # block until there is at least 1 model trained
    usecase_version.wait_until(lambda usecasev: len(usecasev.models) > 0)

    # check out the usecase status and other info
    usecase_version.print_info()
    print('Current (best model) score:', usecase_version.score)

    # predict from uploaded dataset on the plateform
    preds = usecase_version.predict_from_dataset(test_dataset)

    # or predict from a `pandas.DataFrame`
    preds = usecase_version.predict(test_dataframe)

.. note::

    The ``wait_until`` method takes a function that takes the usecase as an argument, and can therefore access any info relative to the usecase.

Time Series usecase
===================

A time series usecase is very similar to a regression usecase. The main differences rely in the dataset configuration, and the specification of a time window.

Configuring the dataset
-----------------------

Here you need to specify which column in the dataset defines the time steps. Also you can specify the ``group_columns`` (columns defining a unique time serie) as well as the ``apriori_columns`` (columns containing information known in advanced):

.. code-block:: python

    column_config = pio.ColumnConfig(
        target_column='Sales',
        id_column='ID',
        time_column='Date',
        group_columns=['Store', 'Product'],
        apriori_columns=['is_holiday'],
    )

Configuring the training parameters
-----------------------------------

The training config is the same as for a regression usecase (detailed in :ref:`configuring train`).

Starting training
-----------------

You can now create a new usecase based on:

 - a usecase name
 - a dataset
 - a column config
 - a time window
 - (optional) a metric type
 - (optional) a training config

In particular the ``time_window`` parameter defines the period in the past that you have for each prediction, and the period in the future that you want to predict:

.. code-block:: python

    # Define your time window:
    # example here using 2 weeks in the past to predict the next week
    time_window = pio.TimeWindow(
        derivation_start=-28,
        derivation_end=-14,
        forecast_start=1,
        forecast_end=7,
    )

    usecase_version = project.fit_timeseries_regression(
        name='helloworld_time_series',
        dataset=dataset,
        time_window=time_window,
        column_config=column_config,
        metric=pio.metrics.Regression.RMSE,
        training_config=training_config,
        holdout_dataset=None,
    )

To get a full documentation check the api reference :ref:`time_series_reference`.

Making predictions
------------------

The predictions workflow is the same as for a regression usecase (detailed in :ref:`making prediction`).

Text Similarity usecase
=======================

A Text Similarity usecase matches the most similar texts between a dataset containing descriptions (can be seen as a catalog) and a dataset containing queries. It first converts texts to numerical vectors (text embeddings) and then performs a similarity search to retrieve the most similar documents to a query.

Configuring the datasets
------------------------

To start a usecase you need to specify the datasets to be used and their configuration. Note that a *DescriptionsDataset* is required while a *QueriesDataset* is optional during training (used for scoring). To get a full documentation check the api reference of the :class:`.DescriptionsColumnConfig` and the :class:`.QueriesColumnConfig` in :ref:`text_similarity_reference`.

.. code-block:: python

    # Required: configuration of the DescriptionsDataset
    description_column_config = pio.TextSimilarity.DescriptionsColumnConfig(
        content_column='text_descriptions',
        id_column='ID',
    )

    # Optional: configuration of the QueriesDataset
    queries_column_config = pio.TextSimilarity.QueriesColumnConfig(
        content_column='text_queries',
        id_column='ID',
    )

Configuring the training parameters
-----------------------------------

If you want, you can also specify some training parameters, such as which embedding models, searching models and preprocessing are used. To get a full documentation check the api reference of the :class:`.ModelsParameters` in :ref:`text_similarity_reference`. Here you need to specify one configuration per embedding model you want to use:

.. code-block:: python

    # Using TF-IDF as embedding model
    models_parameters_1 = pio.ModelsParameters(
        model_embedding=pio.ModelEmbedding.TFIDF,
        preprocessing=pio.Preprocessing(),
        models=[pio.TextSimilarityModels.BruteForce, pio.TextSimilarityModels.ClusterPruning],
    )

    # Using Transformer as embedding model
    models_parameters_2 = pio.ModelsParameters(
        model_embedding=pio.ModelEmbedding.Transformer,
        preprocessing=pio.Preprocessing(),
        models=[pio.TextSimilarityModels.BruteForce, pio.TextSimilarityModels.IVFOPQ],
    )

    # Using fine-tuned Transformer as embedding model
    models_parameters_3 = pio.ModelsParameters(
        model_embedding=pio.ModelEmbedding.TransformerFineTuned,
        preprocessing=pio.Preprocessing(),
        models=[pio.TextSimilarityModels.BruteForce, pio.TextSimilarityModels.IVFOPQ],
    )

    # Gather everything
    models_parameters = [models_parameters_1, models_parameters_2, models_parameters_3]
    models_parameters = pio.ListModelsParameters(models_parameters=models_parameters)


.. note::

    If you want the default configuration of text similarity models, simply use:

    .. code-block:: python

        models_parameters = pio.ListModelsParameters()


Starting the training
---------------------

You can then create a new text similarity usecase based on:

 - a usecase name
 - a dataset
 - a description column config
 - (optional) a queries dataset
 - (optional) a queries column config
 - (optional) a metric type
 - (optional) the number of *top k* results tou want per query
 - (optional) a language
 - (optional) a models parameters list

.. code-block:: python

    usecase_verion = project.fit_text_similarity(
        name='helloworld_text_similarity',
        dataset=dataset,
        description_column_config=description_column_config,
        metric=pio.metrics.TextSimilarity.accuracy_at_k,
        top_k=10,
        queries_dataset=queries_dataset,
        queries_column_config=queries_column_config,
        models_parameters=models_parameters,
    )

To get an exhaustive list of the available metrics go to the class :class:`.previsionio.metrics.TextSimilarity` in the api reference :ref:`metrics_reference`.

Making predictions
------------------

To make predictions from a dataset and a usecase, you need to wait until at least one model is trained. This can be achieved in the following way:

.. code-block:: python

    # block until there is at least 1 model trained
    usecase_version.wait_until(lambda usecasev: len(usecasev.models) > 0)

    # check out the usecase status and other info
    usecase_version.print_info()
    print('Current (best model) score:', usecase_version.score)

    # predict from uploaded dataset on the plateform
    preds = usecase_version.predict_from_dataset(
        queries_dataset=queries_dataset,
        queries_dataset_content_column='queries',
        top_k=10,
        queries_dataset_matching_id_description_column=None, # Optional
    )

.. note::

    The ``wait_until`` method takes a function that takes the usecase as an argument, and can therefore access any info relative to the usecase.

Additional util methods
=======================

Retrieving a use case
---------------------

Since a use case can be somewhat long to train, it can be useful to separate the training, monitoring and prediction phases.

To do that, we need to be able to recreate a usecase object in python from its name:

.. code-block:: python

    usecase_version = pio.Supervised.from_id('<a usecase id>')
    # Usecase_version now has all the same methods as a usecase_version
    # created directly from a file or a dataframe
    usecase_version.print_info()

Stopping and deleting
---------------------

Once you're satisfied with model performance, don't want to wait for the complete training process to be over, or need to free up some resources to start a new training, you can stop the usecase_version simply:

.. code-block:: python

    usecase_version.stop()

You'll still be able to make predictions and get info, but the performance won't improve anymore. Note: there's no difference in state between a stopped usecase and a usecase that has completed its training completely.

You can decide to completely delete the usecase:

.. code-block:: python

    uc = pio.Usecase.from_id(usecase_version.usecase_id)
    uc.delete()

However be careful, in that case any detail about the usecase will be removed, and you won't be able to make predictions from it anymore.

Using deployed model
--------------------

Prevision.io's SDK allows to make a prediction from a model deployed with the Prevision.io's platform.

To deploy a model you need to log to the web interface of your instance, select a project and the usecase you want to work with. Then go to the ``Models`` tab and toggle on the models you want to use in production. Then click on the ``Deployments`` tab at the left of the screen, go to ``Deployments usecases`` at the top left of the screen and click on the ``Deploy a new usecase`` button at the top right of the screen. Fill in the different fields and click on ``Deploy``.

Once the model is deployed and your on its main page, go to the bottom of the page and click on ``generate new key`` wich will create a ``Client Id`` and a ``Client secret``. You will need the ``url`` (displayed at the top of the page in the interface) the ``Client Id`` and the ``Client secret`` to call it via the python SDK:

.. code-block:: python

    import previsionio as pio

    # Initialize the deployed model object from the url of the model, your client id and client secret for this model, and your credentials
    model = pio.DeployedModel(prevision_app_url, client_id, client_secret)

    # Make a prediction
    prediction, confidance, explain = model.predict(
        predict_data={'feature1': 1, 'feature2': 2},
        use_confidence=True,
        explain=True,
    )

To get a full documentation check the api reference :ref:`deployed_model_reference`.
