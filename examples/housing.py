import os
import sys
path_elements = os.getcwd().split(os.sep)
sys.path.append(os.getcwd())  # os.sep.join(path_elements[:-1]))

from previsionio.utils import PrevisionException
import previsionio as pio


TOKEN = """"""
URL = 'https://dev.prevision.io'

# if __name__ == '__main__':
pio.prevision_client.client.init_client(URL, TOKEN)

# get train & test dataset stocked on the datastore
train = pio.dataset.Dataset.get_by_name(dataset_name='regression_house_80')
test = pio.dataset.Dataset.get_by_name(dataset_name='regression_house_20')
# transform a var
train._data['bathrooms'] = train._data['bathrooms'].astype('int').apply(lambda x: round(x))
test._data['bathrooms'] = test._data['bathrooms'].astype('int').apply(lambda x: round(x))
# register new datasets
train_fe = pio.Dataset.new('regression_house_80_fe', dataframe=train._data)
test_fe = pio.Dataset.new('regression_house_20_fe', dataframe=test._data)
# auto ml use case starting
uc_config = pio.TrainingConfig(models=[pio.Model.XGBoost, pio.Model.RandomForest],
                               features=pio.Feature.Full,
                               profile=pio.Profile.Quick,
                               with_blend=False)

col_config = pio.ColumnConfig(target_column='TARGET', id_column='ID')

uc = pio.Regression.fit('housing_from_sdk',
                        dataset=train_fe,
                        holdout_dataset=test_fe,
                        column_config=col_config,
                        training_config=uc_config)

uc.wait_until(lambda u: len(u) > 1)

# Get some Use case derived informations:
# correlation matrix
print('*************************************')
print('***         GET CORR MATRIX       ***')
CM = uc.get_correlation_matrix()
print(CM)

# basic feauture stats
print('*************************************')
print('***    GET FEATURE STATISTICS    ***')
FS = uc.get_feature_stats()
print(FS)

# one specefic feature infos
print('*************************************')
print('***    GET FEATURE STATISTICS    ***')
FI = uc.get_feature_info(feature_name='bathrooms')
print(FI)

# list the created models:
print('*************************************')
print('***    LIST MODELS    ***')
print(uc.list_models())
print('-------')
print(uc.list_models['xgb'])

# Get some typical models
# a- the model with the best performances (having the minimal value of the chosen erro metric)
print('*************************************')
print('***         GET BEST MODEL        ***')
best_model = uc.get_best_model()
print(best_model.__dict__)

# b- the fastest model (having the minimal prediction time)
print('************************************')
print('***       GET FASTEST MODEL      ***')
fastest_model = uc.get_fastest_model()
print(fastest_model.__dict__)

# c- a specefic model (if a specefic model is desired, the SDK gives utilies to deal with it)
print('************************************')
print('***     GET A SPECEFIC MODEL     ***')
specefic_model = uc.get_model_from_name('XGB-1')
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
dataset_preds = best_model.predict_from_dataset_name('test_fe')
print(dataset_preds.head())

# Get the Cross validation predictions
print('*******************************************')
print('***        GET CROSS VALIDATION        ***')
cv = fastest_model.get_cv()
print(cv.head())

# unit prediction
print('*******************************************')
print('***        ONE UNIT PREDICTION         ***')
unit = df.iloc[7]
print("single unit :", unit)
print("prediction : ", best_model.predict_single(unit))
