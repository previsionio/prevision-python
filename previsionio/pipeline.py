import requests
from typing import Dict, List
from .api_resource import ApiResource
from . import client
from .logger import logger
from .utils import parse_json


class PipelineScheduledRun(ApiResource):
    resource = 'pipeline-scheduled-runs'

    def __init__(
        self,
        _id: str,
        name: str,
        project_id: str,
        pipeline_template_id: Dict,
        nodes_properties: List,
        exec_type: str,
        enabled: bool,
        description: str = None,
        draft: bool = True,
        created_at: str = None,
        **kwargs
    ):
        """ A Pipeline Scheduled Run """
        super().__init__(_id=_id)
        self._id = _id
        self.name = name
        self.project_id = project_id
        self.pipeline_template_id = pipeline_template_id
        self.nodes_properties = nodes_properties
        self.exec_type = exec_type
        self.enabled = enabled
        self.description = description
        self.draft = draft
        self.created_at = created_at

    @classmethod
    def from_id(cls, _id: str) -> 'PipelineScheduledRun':
        """ Get a pipeline scheduled run from the platform by its unique id.

        Args:
            _id (str): Unique id of the pipeline scheduled run to retrieve

        Returns:
            :class:`.PipelineScheduledRun`: Fetched experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(_id=_id))

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available pipeline scheduled runs in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Dataset` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            project_id (str): Unique reference of the project id on the platform
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.PipelineScheduledRun`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    def trigger(self):
        """ Trigger an execution of a pipeline scheduled run. """
        url = '/{}/{}/trigger'.format(self.resource, self._id)
        client.request(url, method=requests.post, message_prefix='PipelineScheduledRun trigger')
        logger.info('pipeline scheduled run with id {} triggered'.format(self._id))

    def get_executions(self, limit: int = 15):
        """Get executions of a pipeline scheduled run.

        Args:
            limit (int): Number of executions to retrieve.

        Returns:
            list(dict)
        """
        url = '/{}/{}?limit={}'.format(self.resource, self._id, limit)
        run_resp = client.request(url, method=requests.get, message_prefix='PipelineScheduledRun get element')
        run_json = parse_json(run_resp)
        return run_json['executions']
