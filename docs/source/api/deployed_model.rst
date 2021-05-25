.. _deployed_model_reference:

Deployed model
==============

Prevision.io's SDK allows to make a prediction from a model deployed with the Prevision.io's platform.

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

.. automodule:: previsionio.deployed_model
    :members:
