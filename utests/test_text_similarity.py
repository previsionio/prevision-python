import os
import pandas as pd
import pytest
import unittest
import previsionio as pio
from .utils import get_testing_id

TESTING_ID = get_testing_id()

test_datasets = {}
describe_dataset_file_name = 'manutan_items_100'
description_dataset_name = TESTING_ID + '-' + describe_dataset_file_name
queries_dataset_file_name = 'manutan_queries_100'
queries_dataset_name = TESTING_ID + '-' + queries_dataset_file_name

def upload_datasets():
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_text_similarity')
    # upload CSV reference file
    description_dataset_csv = pio.Dataset.new(
        name=description_dataset_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(describe_dataset_file_name)))
    )

    queries_dataset_csv = pio.Dataset.new(
        name=queries_dataset_name,
        dataframe=pd.read_csv(os.path.join(datapath, '{}.csv'.format(queries_dataset_file_name)))
    )
    test_datasets['description'] = description_dataset_csv
    test_datasets['queries'] = queries_dataset_csv

def setup_module(module):
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
        uc = pio.TextSimilarity.fit('test_sdk_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k)

        uc.wait_until(lambda usecase: usecase._status['status'] == 'done')
        len_models = 0
        while len_model==0:
            len_models = len(uc.models)
        uc.stop()
        uc.update_status()
        assert not uc.running
        uc.delete()

    def test_train_search_delete_text_similarity_with_queries_dataset(self):

        description_column_config = pio.DescriptionsColumnConfig(content_column='item_desc', id_column='item_id')
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query', queries_dataset_matching_id_description_column='true_item_id')
        uc = pio.TextSimilarity.fit('test_sdk__2__text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k,
                                     top_k=10,
                                     lang='auto',
                                     queries_dataset=test_datasets['queries'],
                                     queries_column_config=queries_column_config)

        uc.wait_until(lambda usecase: usecase._status['status'] == 'done')
        assert not uc.running
        assert uc.score is not None
        nb_model = len(uc.models)
        nb_prediction = 0
        for model in uc.models:
            preds = model.predict_from_dataset(test_datasets['queries'], 'query', 'true_item_id', 10)
            nb_prediction += 1
        assert nb_prediction == nb_model
        uc.delete()


    def test_train_delete_text_similarity_with_queries_dataset_all_models(self):

        description_column_config = pio.DescriptionsColumnConfig(content_column='item_desc', id_column='item_id')
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query', queries_dataset_matching_id_description_column='true_item_id')
        usecase_config = [{'modelEmbedding': 'tf_idf', 'preprocessing': {'word_stemming': 'yes', 'ignore_stop_word': 'auto', 'ignore_punctuation': 'no'}, 'models': ['brute_force', 'cluster_pruning']},
                          {'modelEmbedding': 'transformer', 'preprocessing': {}, 'models': ['brute_force', 'lsh', 'hkm']},
                          {'modelEmbedding': 'transformer_fine_tuned', 'preprocessing': {}, 'models': ['brute_force', 'lsh', 'hkm']}]
        models_parameters = pio.ListModelsParameters(usecase_config)
        uc = pio.TextSimilarity.fit('test_sdk_text_similarity_{}'.format(TESTING_ID),
                                     test_datasets['description'],
                                     description_column_config,
                                     metric=pio.metrics.TextSimilarity.accuracy_at_k,
                                     top_k=10,
                                     lang='auto',
                                     queries_dataset=test_datasets['queries'],
                                     queries_column_config=queries_column_config)

        uc.wait_until(lambda usecase: usecase._status['status'] == 'done')
        assert not uc.running
        assert uc.score is not None
        uc.stop()
        uc.update_status()
        assert not uc.running
        uc.delete()
