from previsionio.pipeline import PipelineScheduledRun

PipelineScheduledRun_ID = "61e5627cdf5fef0028e68215"
PROJECT_ID = "61e53eef2d67a00028511129"


def test_pipeline_schedule_run_trigger():

    scheduled_run = PipelineScheduledRun.from_id(PipelineScheduledRun_ID)
    executions = scheduled_run.get_executions(limit=1000)
    executions_count = len(executions)
    scheduled_run.trigger()
    executions = scheduled_run.get_executions(limit=1000)
    assert executions_count + 1 == len(executions)

    l = PipelineScheduledRun.list(project_id=PROJECT_ID)
    assert len(l) == 1
