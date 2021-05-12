Setting logging
===============

Prevision.io's SDK can provide with more detailed information if you change the
logging level. By default, it will only output warnings and errors.

To change the logging level, use ``pio.verbose()`` method:

.. automodule:: previsionio.__init__
    :members: verbose

For example:

.. code-block:: py
    :linenos:

    import previsionio as pio

    # CHANGE LOGGING LEVEL ------------------------------------------
    pio.verbose(True, debug=True) # (add event_log=True
                                  # for events logging)

    # CLIENT INITIALIZATION -----------------------------------------
    url = """https://<your instance>.prevision.io"""
    token = """<your token>"""
    pio.client.init_client(url, token)

    # TESTING LOGS --------------------------------------------------
    # fetching a dataset from the platform
    dataset = pio.Dataset.from_id('dataset_id')

    # fetching a usecase  from the platform
    usecase = pio.Usecase.from_id('usecase_id')

    # fetching a usecase version from the platform
    usecase_version = pio.Supervised.from_id('usecase_version_id')
    usecase_version = pio.Classification.from_id('usecase_version_id')

    # fetching a model from the platform
    model = pio.Model.from_id('helloworld classif')
