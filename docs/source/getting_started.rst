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

Getting some data
~~~~~~~~~~~~~~~~~

First things first, to train a usecase, you need to gather some training data. This data
can be passed a ``pandas`` ``DataFrame`` or a string representing a path to a file.

.. code-block:: python

    # load some data from a CSV file
    data_path = 'data/titanic.csv'
    dataset = pio.Dataset.new(name='helloworld', file_name=data_path)

    # or use a pandas DataFrame
    dataframe = pd.read_csv(data_path)
    dataset = pio.Dataset.new(name='helloworld', dataframe=dataframe)

This will automatically read the given data and upload it as a new dataset on your Prevision.io's
instance. If you go to the online interface, you will see this new dataset in the list of datasets
(in the "Data" tab).

You can also load in your script a dataset that has already been uploaded on the platform; to do so,
either use its name or its unique id:

.. code-block:: python

    # load a dataset by name
    dataset = pio.Dataset.from_name('helloworld')

    # or by unique id
    dataset = pio.Dataset.from_id('5ebaad70a7271000e7b28ea0')

.. note::

    If you want to list all of the available datasets on your instance, simply use:

    .. code-block:: python

        datasets = pio.Dataset.list()


Configuring a usecase
~~~~~~~~~~~~~~~~~~~~~

If you want, you can also specify some training parameters, such as which models are used,
which transformations are applied, and how the models are optimized.

.. code-block:: python

    uc_config = pio.TrainingConfig(models=[pio.Model.XGBoost],
                                   features=pio.Feature.Full,
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

    uc = pio.Classification.fit('helloworld_classif',
                                 dataset,
                                 metric=pio.metrics.Classification.AUC,
                                 training_config=uc_config)

.. note::

    For more complex usecase setups (for example with an image dataset), refer to the :ref:`starting_usecase`
    guide.

Monitoring training
~~~~~~~~~~~~~~~~~~~

You can retrieve at any moment the number of models trained so far and the current error score,
as well as some additional info.

.. code-block:: python

    >>> uc.score
    0.0585

    >>> len(uc)
    4

    >>> uc.print_info()
    scores_cv: 0.0585
    nbPreds: None
    datasetId: 5b4f039f997e790001081657
    nbModels: 4
    state: RUNNING


You can also wait until a certain condition is reached, such as a number of models or a certain score:

.. code-block:: python

    # will block until there are more than 3 models
    uc.wait_until(lambda usecase: len(usecase) > 3)

    # will block until error is lower than 0.3 (warning, it may never reach it and wait forever)
    uc.wait_until(lambda usecase: usecase.score < .3)


The ``wait_until`` method takes a function that takes the usecase as an argument, and can therefore access any info
relative to the usecase. For now, ``len(usecase)`` and ``usecase.score`` are the most useful ones,
but other will come with time.

Making predictions
~~~~~~~~~~~~~~~~~~

Once we have at least a model, we can start making predictions. We don't need to wait until the complete training
process is done, and we'll always have access to the best model trained so far.

.. code-block:: python

    # we have some test data here:
    data_path = 'data/titanic_test.csv'
    test_dataset = pio.Dataset.new(name='helloworld_test', file_name=data_path)

    # we can predict asynchronously (just like training, either from a file path or a dataframe):
    predict_id = uc.predict_from_dataset(test_dataset)

    # We wait until they're ready
    uc.wait_for_prediction(predict_id)

    # and retrieve them once they're done
    preds = uc.download_prediction(predict_id)

    # If the data is not too large, we can predict synchronously, scikit-learn style:
    df = pd.read_csv(data_path)
    preds = uc.predict(df)


Additional util methods
-----------------------

Retrieving a use case
~~~~~~~~~~~~~~~~~~~~~

Since a use case can be somewhat long to train, it can be useful to separate the training, monitoring and prediction
phases.

To do that, we need to be able to recreate a usecase object in python from its name:

.. code-block:: python

    uc = pio.Supervised.from_name('titanic')
    # or
    uc = pio.Supervised.from_id('<a usecase id>')
    # uc now has all the same methods as a usecase created directly from a file or a dataframe
    >>> uc.print_info()
    scores_cv: 0.0585
    nbPreds: None
    datasetId: 5b4f039f997e790001081657
    nbModels: 4
    state: RUNNING

Stopping and deleting
~~~~~~~~~~~~~~~~~~~~~

Once you're satisfied with model performance, don't want to wait for the complete training process to be over, or need
to free up some resources to start a new training, you can stop the model simply:

.. code-block:: python

    uc.stop()

You'll still be able to make predictions and get info, but the performance won't improve anymore.
Note: there's no difference in state between a stopped usecase and a usecase that has completed its training completely.

You can also decide to completely delete the usecase:

.. code-block:: python

    uc.delete()

However, be careful because, in that case, any detail about the usecase will be removed, and you won't be able to
make predictions anymore.
