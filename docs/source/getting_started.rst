.. _getting_started:

Getting started
===============

Pre-requisites
--------------

You need to have an account at cloud.prevision.io or on an on-premise version installed in your company. Contact
us or your IT manager for more info.

You will be working on a specific "instance". This instance corresponds to the subdomain at the beginning of the
url in your prevision.io address: ``https://<your instance>.prevision.io``.

Get the package
---------------

1. clone the git repo:

.. code-block:: bash

    git clone https://github.com/previsionio/prevision-python.git

2. install as a Python package:

.. code-block:: bash

    cd prevision-python
    python setup.py install


Setup your client
-----------------

Prevision.io's SDK client uses a specific master token to authenticate with the instance's server
and allow you to perform various requests. To get your master token, log in the online interface on
your instance, navigate to the admin page and copy the token.

You can either set the token and the instance name as environment variables, by specifying
``PREVISION_URL`` and ``PREVISION_MASTER_TOKEN``, or at the beginning of your script:

.. code-block:: python

    import previsionio as pio

    # We initialize the client with our master token and the url of the prevision.io server
    # (or local installation, if applicable)
    url = """https://<your instance>.prevision.io"""
    token = """<your token>"""
    pio.client.init_client(url, token)

A small example
---------------

create a project
~~~~~~~~~~~~~~~~

First things first, to upload data or train a usecase, you need to create a project.

.. code-block:: python

    # create project
    project = pio.Project.new(name="project_name",
                              description="project description")

Getting some data
~~~~~~~~~~~~~~~~~

To train a usecase, you need to gather some training data. This data
can be passed a ``pandas`` ``DataFrame`` or a string representing a path to a file.

.. code-block:: python


    # load some data from a CSV file
    data_path = 'data/titanic.csv'
    dataset = project.create_dataset(name='helloworld', file_name=data_path)

    # or use a pandas DataFrame
    dataframe = pd.read_csv(data_path)
    dataset = project.create_dataset(name='helloworld', dataframe=dataframe)

This will automatically read the given data and upload it as a new dataset on your Prevision.io's
instance. If you go to the online interface, you will see this new dataset in the list of datasets
(in the "Data" tab).

You can also load in your script a dataset that has already been uploaded on the platform:

.. code-block:: python

    # by unique id
    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')

.. note::

    If you want to list all of the available datasets on your instance, simply use:

    .. code-block:: python

        datasets = project.list_datasets()


Configuring a usecase
~~~~~~~~~~~~~~~~~~~~~

If you want, you can also specify some training parameters, such as which models are used,
which transformations are applied, and how the models are optimized.

.. code-block:: python

    uc_config = pio.TrainingConfig(advanced_models=[pio.AdvancedModel.LinReg],
                                   normal_models=[pio.NormalModel.LinReg],
                                   simple_models=[pio.SimpleModel.DecisionTree],
                                   features=[pio.Feature.Counts],
                                   profile=pio.Profile.Quick)

For a full details on training config and training parameters, see the training config documentation.


Starting training
~~~~~~~~~~~~~~~~~

You can then create a new usecase based on :

 - a usecase name
 - a dataset
 - a column config
 - (optional) a metric type
 - (optional) a training config

.. code-block:: python

    usecase_version = project.fit_classification('helloworld_classif',
                                                 dataset,
                                                 metric=pio.metrics.Classification.AUC,
                                                 training_config=uc_config)

.. note::

    For more complex usecase setups (for example with an image dataset), refer to the :ref:`starting_usecase`
    guide.


Configuring a text similarity usecase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

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
-----------------------

Retrieving a use case
~~~~~~~~~~~~~~~~~~~~~

Since a use case can be somewhat long to train, it can be useful to separate the training, monitoring and prediction
phases.

To do that, we need to be able to recreate a usecase object in python from its name:

.. code-block:: python

    usecase_version = pio.Supervised.from_id('<a usecase id>')
    # usecase_version now has all the same methods as a usecase_version created directly from a file or a dataframe
    >>> usecase_version.print_info()
    scores_cv: 0.0585
    state: running

Stopping and deleting
~~~~~~~~~~~~~~~~~~~~~

Once you're satisfied with model performance, don't want to wait for the complete training process to be over, or need
to free up some resources to start a new training, you can stop the usecase_version simply:

.. code-block:: python

    usecase_version.stop()

You'll still be able to make predictions and get info, but the performance won't improve anymore.
Note: there's no difference in state between a stopped usecase and a usecase that has completed its training completely.

You can decide to completely delete the usecase:

.. code-block:: python

    uc = usecase_version.usecase
    uc.delete()

However, be careful because, in that case, any detail about the usecase will be removed, and you won't be able to
make predictions anymore.
