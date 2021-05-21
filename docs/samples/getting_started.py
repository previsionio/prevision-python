# [Prevision.io - Python SDK]
# Sample: Getting started
#
# ---------------------------------------------------------------
# Author: Mina Pêcheux
# Date: July 2020
# ===============================================================

from previsionio.usecase_config import ColumnConfig
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
dataset = project.create_dataset(name='dataset_name', dataframe=dataframe)

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
                                             column_config=ColumnConfig(target_column="target"),
                                             training_config=uc_config)

# (block until there is at least 1 model trained)
usecase_version.wait_until(lambda usecasev: len(usecasev.models) > 0)

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
