import requests
from typing import Dict, List
from .api_resource import ApiResource
from . import client
from .logger import logger
from .utils import parse_json, PrevisionException


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

    def delete(self):
        """Delete a PipelineScheduledRun.

        Raises:
            PrevisionException: If the project does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    @classmethod
    def new(cls, project_id, pipeline_template_id, name, description=None, nodes_params=[],
            exec_type="manual", exec_cron=None, exec_period_start=None, exec_period_end=None):
        """ Create a pipeline Scheduled Run.

        Args:
            project_id(str): project id
            pipeline_template_id(str): pipeline template id
            name (str): pipeline scheduled run name
            description (str, optional): Pipeline scheduled run description
            nodes_params (list(dict)): Pipeline Nodes parameters.
                E.g [{'_id': 'xxx', 'properties':{'property_name': 'property_value'}}
            exec_type (str, optional): Run mode, possible values: manual or recurring
                (default: ``manual``)
            exec_cron (str, optional): Cron for recurring pipeline scheduled run
            exec_period_start (str, optional): Start period of recurring pipeline scheduled run
            exec_period_end (str, optional): End period of recurring pipeline scheduled run

        Returns:
            :class:`.PipelineScheduledRun`: Newly created PipelineScheduledRun

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        run_url = '/projects/{}/pipeline-scheduled-runs'.format(project_id)
        data = {'pipeline_template_id': pipeline_template_id,
                'name': name}
        if description:
            data['description'] = description

        if exec_type == 'manual':
            unvalid_args = []
            if exec_cron:
                unvalid_args.append('exec_cron')
            if exec_period_start:
                unvalid_args.append('exec_period_start')
            if exec_period_end:
                unvalid_args.append('exec_period_end')
            if unvalid_args:
                msg = 'Arguments {} are available only for exec type manual'.format(', '.join(unvalid_args))
                raise PrevisionException(msg)
        run_response = client.request(run_url,
                                      method=requests.post,
                                      data=data,
                                      message_prefix='Create Scheduled Run')
        run_response_json = parse_json(run_response)
        pipeline_scheduled_run_id = run_response_json['_id']
        for node in nodes_params:
            node_id = node['_id']
            node_data = [{'name': name, 'value': value} for name, value in node['properties'].items()]
            node_put_url = '/pipeline-scheduled-runs/{}/node/{}'.format(pipeline_scheduled_run_id, node_id)
            _ = client.request(node_put_url,
                               method=requests.put,
                               data=node_data,
                               message_prefix='Configure Scheduled Run Node')

        # strange behavior of backend mecanisme, without updating pipeline scheduled run, confirm failed
        update_run_url = '/pipeline-scheduled-runs/{}'.format(pipeline_scheduled_run_id)

        data = {'name': name, 'exec_type': exec_type}
        if exec_cron:
            data['exec_cron'] = exec_cron
        if exec_period_start:
            data['exec_period_start'] = exec_period_start
        if exec_period_end:
            data['exec_period_end'] = exec_period_end
        _ = client.request(update_run_url,
                           data=data,
                           method=requests.put,
                           message_prefix='Updating Scheduled Run')
        confirm_url = '/pipeline-scheduled-runs/{}/confirm'.format(pipeline_scheduled_run_id)
        confirm_response = client.request(confirm_url,
                                          method=requests.post,
                                          message_prefix='Confirm Scheduled Run')
        run_json = parse_json(confirm_response)
        return cls(**run_json)

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available pipeline scheduled runs in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.PipelineScheduledRun` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            project_id (str): Unique reference of the project id on the platform
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.PipelineScheduledRun`): Fetched PipelineScheduledRun objects
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


class PipelineTemplate(ApiResource):
    resource = 'pipeline-templates'

    def __init__(self, _id, name, project_id, nodes, edges, description=None, draft=True, used_in_run=False, **kwargs):

        self._id = _id
        self.name = name
        self.project_id = project_id
        self.nodes = nodes
        self.edges = edges
        self.description = description
        self.draft = draft
        self.used_in_run = used_in_run

    @classmethod
    def from_id(cls, _id: str) -> 'PipelineTemplate':
        """ Get a pipeline template run from the platform by its unique id.

        Args:
            _id (str): Unique id of the pipeline template run to retrieve

        Returns:
            :class:`.PipelineTemplate`: Fetched experiment

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return cls(**super()._from_id(_id=_id))

    @classmethod
    def list(cls, project_id: str, all: bool = True):
        """ List all the available pipeline templates in the current active [client] workspace.

        .. warning::

            Contrary to the parent ``list()`` function, this method
            returns actual :class:`.PipelineTemplate` objects rather than
            plain dictionaries with the corresponding data.

        Args:
            project_id (str): Unique reference of the project id on the platform
            all (boolean, optional): Whether to force the SDK to load all items of
                the given type (by calling the paginated API several times). Else,
                the query will only return the first page of result.

        Returns:
            list(:class:`.PipelineTemplate`): Fetched PipelineTemplate objects
        """
        resources = super()._list(all=all, project_id=project_id)
        return [cls(**conn_data) for conn_data in resources]

    def delete(self):
        """Delete a PipelineTemplate.

        Raises:
            PrevisionException: If the project does not exist
            requests.exceptions.ConnectionError: Error processing the request
        """
        super().delete()

    def get_nodes_properties(self):
        """ Get nodes properties of pipeline template run.

        Returns:
            list(dict): Information about nodes properties:

                - ``_id``: str
                - ``name``: str
                - ``properties``: dict<property_name:property_type>

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        nodes_properties = []
        for node in self.nodes:
            node_id = node['_id']
            node_name = node['pipeline_component_id']['metadata']['name']
            node_properties = {}
            for property in node['pipeline_component_id']['interfaces']['properties']:
                if 'hidden' in property and property['hidden']:
                    continue
                node_properties[property['name']] = property['type']
            nodes_properties.append({'_id': node_id, 'name': node_name, 'properties': node_properties})
        return nodes_properties

    def create_scheduled_run(self, name, description=None, nodes_params=[], exec_type="manual",
                             exec_cron=None, exec_period_start=None, exec_period_end=None) -> 'PipelineScheduledRun':
        """ Create a pipeline Scheduled Run.

        Args:
            name (str): pipeline scheduled run name
            description (str, optional): Pipeline scheduled run description
            nodes_params (list(dict)): Pipeline Nodes parameters.
                E.g [{'_id': 'xxx', 'properties':{'property_name': 'property_value'}}
            exec_type (str, optional): Run mode, possible values: manual or recurring
                (default: ``manual``)
            exec_cron (str, optional): Cron for recurring pipeline scheduled run
            exec_period_start (str, optional): Start period of recurring pipeline scheduled run
            exec_period_end (str, optional): End period of recurring pipeline scheduled run

        Returns:
            :class:`.PipelineScheduledRun`: Newly created PipelineScheduledRun

        Raises:
            PrevisionException: Any error while fetching data from the platform
                or parsing result
        """
        return PipelineScheduledRun.new(self.project_id, self._id, name, description=description,
                                        nodes_params=nodes_params, exec_type=exec_type,
                                        exec_cron=exec_cron, exec_period_start=exec_period_start,
                                        exec_period_end=exec_period_end)
