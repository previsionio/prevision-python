import previsionio as pio

if __name__ == '__main__':
    pio.client.init_client('url',
                           'token')

    # create train & test dataset
    train = pio.Dataset.from_id('train_id')
    test = pio.Dataset.from_id('test_id')

    train_df = train.to_pandas()
    test_df = test.to_pandas()

    train_df['new_feature'] = train_df['Length'] * train_df['YearConstruction']
    test_df['new_feature'] = test_df['Length'] * test_df['YearConstruction']

    PROJECT_ID = "example_id"
    project = pio.Project.from_id(PROJECT_ID)

    train_fe = project.create_dataset('train_fe', dataframe=train_df)
    test_fe = project.create_dataset('test_fe', dataframe=test_df)

    uc_config = pio.TrainingConfig(advanced_models=[pio.AdvancedModel.LinReg],
                                   normal_models=[pio.NormalModel.LinReg],
                                   simple_models=[pio.SimpleModel.DecisionTree],
                                   features=[pio.Feature.Counts],
                                   profile=pio.Profile.Quick,
                                   with_blend=False)

    col_config = pio.ColumnConfig(target_column='TARGET')

    usecase_version = project.fit_regression('example_basic',
                                             dataset=train_fe,
                                             column_config=col_config,
                                             training_config=uc_config)

    usecase_version.wait_until(lambda usecasev: len(usecasev.models) > 1)

    preds = usecase_version.predict_from_dataset(test_fe)

    # Get some Use case derived informations:
    # correlation matrix
    print('*************************************')
    print('***         GET CORR MATRIX       ***')
    CM = usecase_version.correlation_matrix
    print(CM)

    # basic feauture stats
    print('*************************************')
    print('***    GET FEATURE STATISTICS    ***')
    FS = usecase_version.features_stats
    print(FS)

    # one specefic feature infos
    print('*************************************')
    print('***    GET FEATURE STATISTICS    ***')
    FI = usecase_version.get_feature_info(feature_name='bathrooms')
    print(FI)

    # list the created models:
    print('*************************************')
    print('***    LIST MODELS    ***')
    print(usecase_version.models)
    print('-------')
    #print(usecase_version.list_models['xgb'])

    # Get some typical models
    # a- the model with the best performances (having the minimal value of the chosen erro metric)
    print('*************************************')
    print('***         GET BEST MODEL        ***')
    best_model = usecase_version.best_model
    print(best_model.__dict__)

    # b- the fastest model (having the minimal prediction time)
    print('************************************')
    print('***       GET FASTEST MODEL      ***')
    fastest_model = usecase_version.fastest_model
    print(fastest_model.__dict__)

    # c- a specefic model (if a specefic model is desired, the SDK gives utilies to deal with it)
    print('************************************')
    print('***     GET A SPECEFIC MODEL     ***')
    specefic_model = pio.Model.from_id('xxxxx')
    print(specefic_model.__dict__)

    # prediction for a new dataframe sample
    df = test_fe._data.sample(frac=0.2, replace=True, random_state=42)
    print('***************************************')
    print('*** GET PREDICTIONS of an input dataframe ***')
    df_preds = best_model.predict(df)
    print(df_preds.head())

    # predict from a registred dataset name
    print('*******************************************')
    print('*** GET PREDICTIONS of registred dataset ***')
    dataset_preds = best_model.predict_from_dataset(test_fe)
    print(dataset_preds.head())

    # Get the Cross validation predictions
    print('*******************************************')
    print('***        GET CROSS VALIDATION        ***')
    cv = fastest_model.cross_validation
    print(cv.head())

    # unit prediction
    print('*******************************************')
    print('***        ONE UNIT PREDICTION         ***')
    unit = df.iloc[7]
    print("single unit :", unit)
    print("prediction : ", best_model.predict_single(unit))
