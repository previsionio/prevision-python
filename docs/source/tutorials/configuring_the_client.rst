Configuring the client
======================

To connect with Prevision.io's Python SDK onto your instance, you need to have your master token.
This token can be found either:

- by going to the online web interface, in the "Administration & API key" page
- by calling:

  .. code-block:: python
    
    client.init_client_with_login(prevision_url, email, password)

Then, to use your client credentials, you have 2 options:

- set the credentials as environment variables so that they are automatically reloaded when you run your scripts: you need to set
  ``PREVISION_URL`` to your instance url (i.e. something of the form: ``https://<instance_name>.prevision.io``) and
  ``PREVISION_MASTER_TOKEN`` to the master token you just retrieved

- set the credentials at the beginning of your script, using the ``init_client()`` method:

  .. code-block:: python

    import previsionio as pio

    url = """https://<your instance>.prevision.io"""
    token = """<your token>"""
    pio.client.init_client(url, token)

For a full description of Prevision.io's client API, check out the :ref:`api_reference`.
