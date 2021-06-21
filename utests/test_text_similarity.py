import os
from previsionio.text_similarity import ModelEmbedding, TextSimilarityLang, TextSimilarityModels
from previsionio.usecase_config import YesOrNo, YesOrNoOrAuto
import time
import pandas as pd
import unittest
import previsionio as pio
from .utils import get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_usecase_" + str(TESTING_ID)
PROJECT_ID = ""

test_datasets = {}
describe_dataset_file_name = 'manutan_items_100'
description_dataset_name = TESTING_ID + '-' + describe_dataset_file_name
queries_dataset_file_name = 'manutan_queries_100'
queries_dataset_name = TESTING_ID + '-' + queries_dataset_file_name


def upload_datasets():
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_text_similarity')
    # upload CSV reference file
    description_dataset_csv = pio.Dataset._new(
        PROJECT_ID,
        name=description_dataset_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(describe_dataset_file_name)))
    )

    queries_dataset_csv = pio.Dataset._new(
        PROJECT_ID,
        name=queries_dataset_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(queries_dataset_file_name)))
    )
    test_datasets['description'] = description_dataset_csv
    test_datasets['queries'] = queries_dataset_csv


def setup_module(module):
    project = pio.Project.new(name=PROJECT_NAME,
                              description="description test sdk")
    global PROJECT_ID
    PROJECT_ID = project._id
    upload_datasets()


class BaseTrainSearchDelete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upload_datasets()

    @classmethod
    def tearDownClass(cls):
        for ds in test_datasets.values():
            ds.delete()

    def test_train_stop_delete_text_similarity(self):

        description_column_config = pio.DescriptionsColumnConfig('item_desc', 'item_id')
        uc = pio.TextSimilarity._fit(PROJECT_ID,
                                     'test_sdk_1_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k)

        uc.wait_until(lambda usecase: usecase._status['state'] == 'done')
        time.sleep(40)
        uc.stop()
        uc.update_status()
        assert not uc.running
        uc.usecase.delete()

    def test_train_new_stop_delete_text_similarity(self):

        description_column_config = pio.DescriptionsColumnConfig('item_desc', 'item_id')
        uc = pio.TextSimilarity._fit(PROJECT_ID,
                                     'test_sdk_1_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k)

        uc.wait_until(lambda usecase: usecase._status['state'] == 'done')

        new_version = uc.new_version()
        new_version.wait_until(lambda usecase: usecase._status['state'] == 'done')

        time.sleep(40)
        new_version.stop()
        new_version.update_status()
        assert not new_version.running

        uc.stop()
        uc.update_status()
        assert not uc.running
        uc.usecase.delete()

    def test_train_search_delete_text_similarity_with_queries_dataset(self):

        description_column_config = pio.DescriptionsColumnConfig(content_column='item_desc', id_column='item_id')
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query',
                                                        queries_dataset_matching_id_description_column='true_item_id')
        uc = pio.TextSimilarity._fit(PROJECT_ID,
                                     'test_sdk_2_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k,
                                     top_k=10,
                                     lang=TextSimilarityLang.Auto,
                                     queries_dataset=test_datasets['queries'],
                                     queries_column_config=queries_column_config)

        uc.wait_until(lambda usecase: usecase._status['state'] == 'done')
        assert not uc.running
        assert uc.score is not None
        nb_model = len(uc.models)
        nb_prediction = 0
        for model in uc.models:
            model.predict_from_dataset(test_datasets['queries'],
                                       'query',
                                       top_k=10,
                                       queries_dataset_matching_id_description_column='true_item_id')
            nb_prediction += 1
        assert nb_prediction == nb_model
        uc.usecase.delete()

    def test_train_delete_text_similarity_with_queries_dataset_all_models(self):
        description_column_config = pio.DescriptionsColumnConfig(content_column='item_desc', id_column='item_id')
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query',
                                                        queries_dataset_matching_id_description_column='true_item_id')
        usecase_config = [{'model_embedding': ModelEmbedding.TFIDF,
                           'preprocessing': {'word_stemming': YesOrNo.Yes,
                                             'ignore_stop_word': YesOrNoOrAuto.Auto,
                                             'ignore_punctuation': YesOrNo.No},
                           'models': [TextSimilarityModels.BruteForce, TextSimilarityModels.ClusterPruning]},
                          {'model_embedding': ModelEmbedding.Transformer,
                           'preprocessing': {},
                           'models': [TextSimilarityModels.BruteForce, TextSimilarityModels.LSH,
                                      TextSimilarityModels.HKM]},
                          {'model_embedding': ModelEmbedding.TransformerFineTuned,
                           'preprocessing': {},
                           'models': [TextSimilarityModels.BruteForce, TextSimilarityModels.LSH,
                                      TextSimilarityModels.HKM]}]
        models_parameters = pio.ListModelsParameters(usecase_config)
        uc = pio.TextSimilarity._fit(PROJECT_ID,
                                     'test_sdk_3_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k,
                                     top_k=10,
                                     lang=TextSimilarityLang.Auto,
                                     queries_dataset=test_datasets['queries'],
                                     queries_column_config=queries_column_config,
                                     models_parameters=models_parameters)

        uc.wait_until(lambda usecase: usecase._status['state'] == 'done')
        assert not uc.running
        assert uc.score is not None
        uc.stop()
        uc.update_status()
        assert not uc.running
        uc.usecase.delete()
