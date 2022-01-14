from .api_resource import ApiResource


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

    def trigger():
        """Trigger an execution of a pipeline scheduled run.

        """
        url = '/{}/{}/trigger'.format(self.resource, self._id)
        run_resp = client.request(url, method=requests.get, message_prefix='PipelineScheduledRun trigger')
        run_json = parse_json(run_resp)

    def get_executions():
        """Get executions of a pipeline scheduled run.

        """
        url = '/{}/{}'.format(self.resource, self._id)
        run_resp = client.request(url, method=requests.get, message_prefix='PipelineScheduledRun get element')
        run_json = parse_json(run_resp)
        return run_json['executions']
