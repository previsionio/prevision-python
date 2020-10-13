# [Prevision.io - Python SDK]
# Sample: Getting started
#
# ---------------------------------------------------------------
# Author: Mina Pêcheux
# Date: July 2020
# ===============================================================

import previsionio as pio
import pandas as pd

# CLIENT INITIALIZATION -----------------------------------------
url = """https://<your instance>.prevision.io"""
token = """<your token>"""
pio.client.init_client(url, token)

# DATA LOADING --------------------------------------------------
# load data from a CSV
dataframe = pd.read_csv('helloworld_train.csv')
# upload it to the platform
dataset = pio.Dataset.new(name='helloworld_train', dataframe=dataframe)

# USECASE TRAINING ----------------------------------------------
# setup usecase
uc_config = pio.TrainingConfig(models=[pio.Model.XGBoost],
                                features=pio.Feature.Full,
                                profile=pio.Profile.Quick)

# run training
uc = pio.Classification.fit('helloworld_classif',
                                dataset,
                                metric=pio.metrics.Classification.AUC,
                                training_config=uc_config)

# (block until there is at least 1 model trained)
uc.wait_until(lambda usecase: len(usecase) > 0)

# check out the usecase status and other info
uc.print_info()
print('Current number of models:', len(uc))
print('Current (best model) score:', uc.score)

# PREDICTIONS ---------------------------------------------------
# load up test data
test_datapath = 'helloworld_test.csv'
test_dataset = pio.Dataset.new(name='helloworld_test', file_name=test_datapath)

# 1. use an ASYNC prediction
predict_id = uc.predict_from_dataset(test_dataset)
uc.wait_for_prediction(predict_id)
preds = uc.download_prediction(predict_id)
print(preds)

# 2. or use a SYNC prediction (Scikit-learn blocking style)
# WARNING: should only be used for small datasets
df = pd.read_csv(test_datapath)
preds = uc.predict(df)
