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

    train_fe = pio.Dataset.new('train_fe', dataframe=train_df)
    test_fe = pio.Dataset.new('test_fe', dataframe=test_df)

    uc_config = pio.TrainingConfig(models=[pio.Model.XGBoost],
                                   features=pio.Feature.Full,
                                   profile=pio.Profile.Quick,
                                   with_blend=False)

    col_config = pio.ColumnConfig(target_column='TARGET')

    uc = pio.Regression.fit('example_basic',
                            dataset=train_fe,
                            column_config=col_config,
                            training_config=uc_config)

    uc.wait_until(lambda u: len(u) > 1)

    predict_id = uc.predict_from_dataset(test_fe)
    uc.wait_for_prediction(predict_id)
    preds = uc.download_prediction(predict_id)

    print(preds.head())
