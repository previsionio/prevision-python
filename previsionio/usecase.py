# -*- coding: utf-8 -*-
from __future__ import print_function
import json
from typing import Dict, List, Union
import pandas as pd
import requests
import time
import previsionio as pio
import os
from functools import lru_cache

from . import config
from .usecase_config import DataType, TrainingConfig, ColumnConfig, TypeProblem
from .logger import logger
from .prevision_client import client
from .utils import parse_json, EventTuple, PrevisionException, zip_to_pandas, get_all_results
from .api_resource import ApiResource
from .dataset import Dataset

from enum import Enum


class Usecase(ApiResource):
    """ A Usecase

    Args:
        _id (str): Unique id of the usecase
        name (str): Name of the usecase

    """

    resource = 'usecases'

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        self._id = usecase_info.get('_id')
        self.name: str = usecase_info.get('name')
        self.project_id: str = usecase_info.get('project_id')
        self.training_type: str = usecase_info.get('training_type')
        self.data_type: str = usecase_info.get('data_type')
        self.version_ids: list = usecase_info.get('version_ids')

    @classmethod
    def from_id(cls, _id):
        """Get a usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase version to retrieve

        Returns:
            :class:`.BaseUsecaseVersion`: Fetched usecase

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return super().from_id(specific_url='/{}/{}'.format(cls.resource, _id))

    @classmethod
    def list(cls, project_id: str, all: bool = True) -> List['Usecase']:
        """ List all the available usecase in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Usecase` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Usecase`): Fetched dataset objects
        """
        resources = super().list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    @property
    def versions(self):
        """Get the list of all versions for the current use case.

        Returns:
            list(dict): List of the usecase versions (as JSON metadata)
        """
        end_point = '/{}/{}/versions'.format(self.resource, self._id)
        response = client.request(endpoint=end_point, method=requests.get)
        res = parse_json(response)
        # TODO create usecase version object
        return res['items']

    def delete(self):
        """ Delete a usecase from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}'.format(self._id),
                                  method=requests.delete)
        return response
