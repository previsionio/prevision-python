# -*- coding: utf-8 -*-
from __future__ import print_function

from previsionio.text_similarity import TextSimilarity
from previsionio.supervised import Supervised
from previsionio.timeseries import TimeSeries
from previsionio.usecase_config import DataType, TypeProblem
from typing import Dict, List, Type, Union
import requests

from .prevision_client import client
from .utils import parse_json
from .api_resource import ApiResource


def get_usecase_version_class(training_type: TypeProblem, data_type: DataType) -> Union[Type[TextSimilarity], Type[Supervised], Type[TimeSeries]]:
    """ Get the type of UsecaseVersion class used by this Usecase

    Returns:
        (:type:`.TextSimilarity` | :type:`.Supervised` | :type:`.TimeSeries`): Type of UsecaseVersion
    """
    default: Dict[DataType, Union[Type[Supervised], Type[TimeSeries]]] = {
        DataType.Tabular: Supervised,
        DataType.TimeSeries: TimeSeries
    }
    class_dict = {
        TypeProblem.TextSimilarity: {DataType.Tabular: TextSimilarity},
    }
    class_type = class_dict.get(training_type, default).get(data_type)
    assert class_type is not None
    return class_type


class Usecase(ApiResource):
    """ A Usecase

    Args:
        _id (str): Unique id of the usecase
        name (str): Name of the usecase

    """

    resource = 'usecases'

    def __init__(self, **usecase_info):
        super().__init__(**usecase_info)
        self._id: str = usecase_info['_id']
        self.name: str = usecase_info['name']
        self.project_id: str = usecase_info['project_id']
        self.training_type: TypeProblem = TypeProblem(usecase_info['training_type'])
        self.data_type: DataType = DataType(usecase_info['data_type'])

    @classmethod
    def from_id(cls, _id: str) -> 'Usecase':
        """Get a usecase from the platform by its unique id.

        Args:
            _id (str): Unique id of the usecase version to retrieve

        Returns:
            :class:`.Usecase`: Fetched usecase

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
    def usecase_version_class(self) -> Union[Type[TextSimilarity], Type[Supervised], Type[TimeSeries]]:
        """ Get the type of UsecaseVersion class used by this Usecase

        Returns:
            (:type:`.TextSimilarity` | :type:`.Supervised` | :type:`.TimeSeries`): Type of UsecaseVersion
        """
        return get_usecase_version_class(self.training_type, self.data_type)

    @property
    def latest_version(self) -> Union[TextSimilarity, Supervised, TimeSeries]:
        """Get the latest version of this use case.

        Returns:
            (:class:`.TextSimilarity` | :class:`.Supervised` | :class:`.TimeSeries`):
            latest UsecaseVersion in this Usecase
        """
        end_point = '/{}/{}/versions'.format(self.resource, self._id)
        format = {
            "rowsPerPage": 1,
            "sortBy": "created_at",
            "descending": True
        }

        response = client.request(endpoint=end_point,
                                  format=format,
                                  method=requests.get,
                                  message_prefix='Latest usecase version')
        res = parse_json(response)
        assert len(res['items']) == 1
        return self.usecase_version_class(**res['items'][0])

    @property
    def versions(self) -> List[Union[TextSimilarity, Supervised, TimeSeries]]:
        """Get the list of all versions for the current use case.

        Returns:
            list(:class:`.TextSimilarity` | :class:`.Supervised` | :class:`.TimeSeries`):
            List of the usecase versions (as JSON metadata)
        """
        end_point = '/{}/{}/versions'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Usecase versions listing')
        res = parse_json(response)
        return [self.usecase_version_class(**val) for val in res['items']]

    def delete(self):
        """ Delete a usecase from the actual [client] workspace."""
        _ = client.request(endpoint='/usecases/{}'.format(self._id),
                           method=requests.delete,
                           message_prefix='Usecase deletion')
