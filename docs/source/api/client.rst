Client
======

Prevision.io's SDK client uses a specific master token to authenticate with the instance's server
and allow you to perform various requests. To get your master token, log in the online interface,
navigate to the admin page and copy the token.

You can either set the token and the instance name as environment variables, by specifying
``PREVISION_URL`` and ``PREVISION_MASTER_TOKEN``, or at the beginning of your script:

.. code-block:: python

    import previsionio as pio

    # We initialize the client with our master token and the url of the prevision.io server
    # (or local installation, if applicable)
    url = """https://<your instance>.prevision.io"""
    token = """<your token>"""
    pio.client.init_client(url, token)

.. automodule:: previsionio.prevision_client
    :members:
