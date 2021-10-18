import os
import pandas as pd
import unittest

import previsionio as pio
from previsionio.experiment import Experiment
from previsionio.text_similarity import ModelEmbedding, TextSimilarityLang, TextSimilarityModels
from previsionio.experiment_config import DataType, TypeProblem, YesOrNo, YesOrNoOrAuto

from .utils import get_testing_id

TESTING_ID = get_testing_id()

PROJECT_NAME = "sdk_test_text_sim_experiment_" + str(TESTING_ID)
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


class BaseTrainSearchDelete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = pio.Project.new(name=PROJECT_NAME,
                                  description="description test sdk")
        global PROJECT_ID
        PROJECT_ID = project._id
        upload_datasets()

    @classmethod
    def tearDownClass(cls):
        project = pio.Project.from_id(PROJECT_ID)
        project.delete()

    def test_train_stop_delete_text_similarity(self):
        experiment_name = 'test_sdk_1_text_similarity_{}'.format(TESTING_ID)
        experiment = Experiment.new(PROJECT_ID, 'prevision-auto-ml', experiment_name,
                                    DataType.Tabular, TypeProblem.TextSimilarity)
        experiment_id = experiment.id
        description_dataset = test_datasets['description']
        description_column_config = pio.DescriptionsColumnConfig('item_desc', 'item_id')
        experiment_version = pio.TextSimilarity._fit(
            experiment_id,
            description_dataset,
            description_column_config,
            metric=pio.metrics.TextSimilarity.accuracy_at_k,
        )

        experiment_version.wait_until(lambda experiment: experiment._status['state'] == 'done')
        experiment_version.stop()
        experiment_version.update_status()
        assert not experiment_version.running
        pio.Experiment.from_id(experiment_version.experiment_id).delete()
        project_experiments = pio.Experiment.list(PROJECT_ID)
        project_experiments_ids = [experiment.id for experiment in project_experiments]
        assert experiment_id not in project_experiments_ids

    def test_train_new_stop_delete_text_similarity(self):
        experiment_name = 'test_sdk_1_text_similarity_{}'.format(TESTING_ID)
        experiment = Experiment.new(PROJECT_ID, 'prevision-auto-ml', experiment_name,
                                    DataType.Tabular, TypeProblem.TextSimilarity)
        experiment_id = experiment.id
        description_dataset = test_datasets['description']
        description_column_config = pio.DescriptionsColumnConfig('item_desc', 'item_id')
        experiment_version = pio.TextSimilarity._fit(
            experiment_id,
            description_dataset,
            description_column_config,
            metric=pio.metrics.TextSimilarity.accuracy_at_k,
        )

        experiment_version.wait_until(lambda experiment: experiment._status['state'] == 'done')

        new_version = experiment_version.new_version()
        new_version.wait_until(lambda experiment: experiment._status['state'] == 'done')

        new_version.stop()
        new_version.update_status()
        assert not new_version.running

        experiment_version.stop()
        experiment_version.update_status()
        assert not experiment_version.running
        pio.Experiment.from_id(experiment_version.experiment_id).delete()
        project_experiments = pio.Experiment.list(PROJECT_ID)
        project_experiments_ids = [experiment.id for experiment in project_experiments]
        assert experiment_id not in project_experiments_ids

    def test_train_search_delete_text_similarity_with_queries_dataset(self):
        experiment_name = 'test_sdk_2_text_similarity_{}'.format(TESTING_ID)
        experiment = Experiment.new(PROJECT_ID, 'prevision-auto-ml', experiment_name,
                                    DataType.Tabular, TypeProblem.TextSimilarity)
        experiment_id = experiment.id
        description_dataset = test_datasets['description']
        description_column_config = pio.DescriptionsColumnConfig('item_desc', 'item_id')
        queries_dataset = test_datasets['queries']
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query',
                                                        queries_dataset_matching_id_description_column='true_item_id')

        experiment_version = pio.TextSimilarity._fit(
            experiment_id,
            description_dataset,
            description_column_config,
            metric=pio.metrics.TextSimilarity.accuracy_at_k,
            top_k=10,
            lang=TextSimilarityLang.Auto,
            queries_dataset=queries_dataset,
            queries_column_config=queries_column_config,
        )

        experiment_version.wait_until(lambda experiment: experiment._status['state'] == 'done')
        assert not experiment_version.running
        assert experiment_version.score is not None
        nb_model = len(experiment_version.models)
        nb_prediction = 0
        for model in experiment_version.models:
            model.predict_from_dataset(test_datasets['queries'],
                                       'query',
                                       top_k=10,
                                       queries_dataset_matching_id_description_column='true_item_id')
            nb_prediction += 1
        assert nb_prediction == nb_model
        pio.Experiment.from_id(experiment_version.experiment_id).delete()
        project_experiments = pio.Experiment.list(PROJECT_ID)
        project_experiments_ids = [experiment.id for experiment in project_experiments]
        assert experiment_id not in project_experiments_ids

    def test_train_delete_text_similarity_with_queries_dataset_all_models(self):
        experiment_name = 'test_sdk_3_text_similarity_{}'.format(TESTING_ID)
        experiment = Experiment.new(PROJECT_ID, 'prevision-auto-ml', experiment_name,
                                    DataType.Tabular, TypeProblem.TextSimilarity)
        experiment_id = experiment.id
        experiment_config = [{'model_embedding': ModelEmbedding.TFIDF,
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
        models_parameters = pio.ListModelsParameters(experiment_config)
        description_dataset = test_datasets['description']
        description_column_config = pio.DescriptionsColumnConfig(content_column='item_desc', id_column='item_id')
        queries_dataset = test_datasets['queries']
        queries_column_config = pio.QueriesColumnConfig(queries_dataset_content_column='query',
                                                        queries_dataset_matching_id_description_column='true_item_id')

        experiment_version = pio.TextSimilarity._fit(
            experiment_id,
            description_dataset,
            description_column_config,
            metric=pio.metrics.TextSimilarity.accuracy_at_k,
            top_k=10,
            lang=TextSimilarityLang.Auto,
            queries_dataset=queries_dataset,
            queries_column_config=queries_column_config,
            models_parameters=models_parameters,
        )

        experiment_version.wait_until(lambda experiment: experiment._status['state'] == 'done')
        assert not experiment_version.running
        assert experiment_version.score is not None
        experiment_version.stop()
        experiment_version.update_status()
        assert not experiment_version.running
        pio.Experiment.from_id(experiment_version.experiment_id).delete()
        project_experiments = pio.Experiment.list(PROJECT_ID)
        project_experiments_ids = [experiment.id for experiment in project_experiments]
        assert experiment_id not in project_experiments_ids
