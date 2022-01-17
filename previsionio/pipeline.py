import requests
from .api_resource import ApiResource
from . import client
from .utils import parse_json


class PipelineScheduledRun(ApiResource):
    resource = 'pipeline-scheduled-runs'

    def __init__(self, _id: str, name, project_id, pipeline_template_id, nodes_properties, exec_type,
                 enabled, description=None, draft=True, created_at=None, **kwargs):
        """ A Pipeline Scheduled Run
        """
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
        return cls(**super()._from_id(_id=_id))

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available pipeline scheduled run in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.Dataset` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.PipelineScheduledRun`): Fetched dataset objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    def trigger(self):
        """Trigger an execution of a pipeline scheduled run.

        """
        url = '/{}/{}/trigger'.format(self.resource, self._id)
        _ = client.request(url, method=requests.post, message_prefix='PipelineScheduledRun trigger')

    def get_executions(self, limit=15):
        """Get executions of a pipeline scheduled run.

        """
        url = '/{}/{}'.format(self.resource, self._id) + "?limit={}".format(limit)
        run_resp = client.request(url, method=requests.get, message_prefix='PipelineScheduledRun get element')
        run_json = parse_json(run_resp)
        return run_json['executions']
