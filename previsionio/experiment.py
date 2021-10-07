# -*- coding: utf-8 -*-
from previsionio.text_similarity import TextSimilarity
from previsionio.supervised import Supervised
from previsionio.timeseries import TimeSeries
from previsionio.external_experiment_version import ExternalExperimentVersion
from previsionio.experiment_config import DataType, TypeProblem
from typing import Dict, List, Union
import requests

from .prevision_client import client
from .utils import parse_json
from .api_resource import ApiResource


def get_experiment_version_class(
    training_type: TypeProblem,
    data_type: DataType,
    provider: str,
) -> Union[TextSimilarity, Supervised, TimeSeries, ExternalExperimentVersion]:
    """ Get the type of ExperimentVersion class used by this Experiment

    Returns:
        (:class:`.previsionio.text_similarity.TextSimilarity` | :class:`.Supervised` |
        :class:`.TimeSeries` | :class:`.ExternalExperimentVersion`):
        Type of ExperimentVersion
    """
    if provider == 'external':
        experiment_version_class = ExternalExperimentVersion
    else:
        if training_type == TypeProblem.TextSimilarity:
            experiment_version_class = TextSimilarity
        else:
            if data_type == DataType.TimeSeries:
                experiment_version_class = TimeSeries
            elif data_type in [DataType.Tabular, DataType.Images]:
                experiment_version_class = Supervised
            else:
                raise ValueError('There is no experiment_version_class with the following values: '
                                 f'training_type: {training_type}'
                                 f'data_type: {data_type}'
                                 f'provider: {provider}')
    return experiment_version_class


class Experiment(ApiResource):
    """ An Experiment

    Args:
        _id (str): Unique id of the experiment
        name (str): Name of the experiment

    """

    resource = 'experiments'

    def __init__(self,
                 _id: str,
                 project_id: str,
                 provider: str,
                 name: str,
                 training_type: TypeProblem,
                 data_type: DataType) -> 'Experiment':
        super().__init__(_id=_id)
        self.project_id = project_id
        self.provider = provider
        self.name = name
        self.training_type: TypeProblem = TypeProblem(training_type)
        self.data_type: DataType = DataType(data_type)

    # TODO: build a class enum for possible providers
    @classmethod
    def new(cls,
            project_id: str,
            provider: str,
            name: str,
            data_type: DataType,
            training_type: TypeProblem) -> 'Experiment':
        url = f'/projects/{project_id}/experiments'
        data = {
            'provider': provider,
            'name': name,
            'data_type': data_type.value,
            'training_type': training_type.value,
        }
        response = client.request(url,
                                  method=requests.post,
                                  data=data,
                                  message_prefix='Experiment creation')
        experiment_info = parse_json(response)
        experiment = cls.from_dict(experiment_info)
        return experiment

    @classmethod
    def from_dict(cls, experiment_info: Dict) -> 'Experiment':
        experiment = cls(
            experiment_info['_id'],
            experiment_info['project_id'],
            experiment_info['provider'],
            experiment_info['name'],
            experiment_info['training_type'],
            experiment_info['data_type'],
        )
        return experiment

    @classmethod
    def from_id(cls, _id: str) -> 'Experiment':
        """Get an experiment from the platform by its unique id.

        Args:
            _id (str): Unique id of the experiment version to retrieve

        Returns:
            :class:`.Experiment`: Fetched experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        experiment_info = super()._from_id(_id)
        experiment = cls.from_dict(experiment_info)
        return experiment

    @classmethod
    def list(cls, project_id: str, all: bool = True) -> List['Experiment']:
        """ List all the available experiment in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Experiment` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            project_id (str): project id
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.Experiment`): Fetched dataset objects
        """
        experiment_infos = super()._list(all=all, project_id=project_id)
        return [cls.from_dict(experiment_info) for experiment_info in experiment_infos]

    @property
    def experiment_version_class(self) -> Union[TextSimilarity, Supervised, TimeSeries, ExternalExperimentVersion]:
        """ Get the type of ExperimentVersion class used by this Experiment

        Returns:
            (:class:`previsionio.text_similarity.TextSimilarity` | :class:`.Supervised` |
            :class:`.TimeSeries` | :class:`.ExternalExperimentVersion`):
            Type of ExperimentVersion
        """
        return get_experiment_version_class(self.training_type, self.data_type, self.provider)

    @property
    def latest_version(self) -> Union[TextSimilarity, Supervised, TimeSeries]:
        """Get the latest version of this experiment version.

        Returns:
            (:class:`previsionio.text_similarity.TextSimilarity` | :class:`.Supervised` | :class:`.TimeSeries`):
            latest ExperimentVersion in this Experiment
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
                                  message_prefix='Latest experiment version')
        res = parse_json(response)
        assert len(res['items']) == 1
        return self.experiment_version_class(**res['items'][0])

    @property
    def versions(self) -> List[Union[TextSimilarity, Supervised, TimeSeries]]:
        """Get the list of all versions for the current experiment version.

        Returns:
            list(:class:`previsionio.text_similarity.TextSimilarity` | :class:`.Supervised` | :class:`.TimeSeries`):
            List of the experiment versions (as JSON metadata)
        """
        end_point = '/{}/{}/versions'.format(self.resource, self._id)
        response = client.request(endpoint=end_point,
                                  method=requests.get,
                                  message_prefix='Experiment versions listing')
        res = parse_json(response)
        return [self.experiment_version_class(**val) for val in res['items']]

    def delete(self):
        """Delete an experiment from the actual [client] workspace.

        Raises:
            PrevisionException: If the experiment does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()
