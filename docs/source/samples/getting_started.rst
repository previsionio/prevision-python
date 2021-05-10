Getting started
===============

This piece of code shows how to:

- initialize a connection to your instance and authentify with your token
- load some data
- start a usecase
- get info about the usecase and its model
- make some predictions

.. code-block:: py
    :linenos:

    import previsionio as pio
    import pandas as pd

    # CLIENT INITIALIZATION -----------------------------------------
    url = """https://<your instance>.prevision.io"""
    token = """<your token>"""
    pio.client.init_client(url, token)

    # CREATE PROJECT --------------------------------------------------
    project = pio.Project.new(name="project_name",
                              description="project description")
    # DATA LOADING --------------------------------------------------
    # load data from a CSV
    dataframe = pd.read_csv('helloworld_train.csv')
    # upload it to the platform
    dataset = project.create_dataset(name='helloworld_train', dataframe=dataframe)

    # USECASE TRAINING ----------------------------------------------
    # setup usecase
    uc_config = pio.TrainingConfig(advanced_models=[pio.AdvancedModel.LinReg],
                                   normal_models=[pio.NormalModel.LinReg],
                                   simple_models=[pio.SimpleModel.DecisionTree],
                                   features=[pio.Feature.Counts],
                                   profile=pio.Profile.Quick)

    # run training
    usecase_version = project.fit_classification('helloworld_classif',
                                                 dataset,
                                                 metric=pio.metrics.Classification.AUC,
                                                 training_config=uc_config)

    # (block until there is at least 1 model trained)
    usecase_version.wait_until(lambda usecase: len(usecase.models) > 0)

    # check out the usecase status and other info
    usecase_version.print_info()
    print('Current (best model) score:', usecase_version.score)

    # PREDICTIONS ---------------------------------------------------
    # load up test data
    test_datapath = 'helloworld_test.csv'
    test_dataset = project.create_dataset(name='helloworld_test', file_name=test_datapath)

    preds = usecase_version.predict_from_dataset(test_dataset)

    df = pd.read_csv(test_datapath)
    preds = usecase_version.predict(df)
