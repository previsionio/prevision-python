# -*- coding: utf-8 -*-
from __future__ import print_function
from previsionio.usecase_config import DataType, TypeProblem
from typing import List
import requests

from .prevision_client import client
from .utils import parse_json
from .api_resource import ApiResource


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
        self.training_type: TypeProblem = TypeProblem(usecase_info.get('training_type'))
        self.data_type: DataType = DataType(usecase_info.get('data_type'))
        self.version_ids: list = usecase_info.get('version_ids')

    @classmethod
    def from_id(cls, _id: str) -> 'Usecase':
        """Get a usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase version to retrieve

        Returns:
            :class:`.BaseUsecaseVersion`: Fetched usecase

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(specific_url='/{}/{}'.format(cls.resource, _id)))

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
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    @property
    def versions(self):
        """Get the list of all versions for the current use case.

        Returns:
            list(dict): List of the usecase versions (as JSON metadata)
        """
        end_point = '/{}/{}/versions'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Usecase versions listing')
        res = parse_json(response)
        # TODO create usecase version object
        return res['items']

    def delete(self):
        """ Delete a usecase from the actual [client] workspace.

        Returns:
            dict: Deletion process results
        """
        response = client.request(endpoint='/usecases/{}'.format(self._id),
                                  method=requests.delete,
                                  message_prefix='Usecase deletion')
        return response
