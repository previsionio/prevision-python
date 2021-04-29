import previsionio as pio
from .utils import get_testing_id

TESTING_ID = get_testing_id()

pio.config.zip_files = False
pio.config.default_timeout = 1000

test_datasets = {}
paths = {}


def test_create_delete_project():
    project = pio.Project.new(name="sdk_test_project_" + str(TESTING_ID),
                              description="description test sdk")
    projects_names = [proj.name for proj in pio.Project.list(all=True) if TESTING_ID in proj.name]
    assert project.name in projects_names
    project_info = project.info()
    assert project_info['name'] == "sdk_test_project_" + str(TESTING_ID)

    assert type(project.users()) == list
    project.add_user('david.fradel@prevision.io', 'admin')
    project_copy = pio.Project.from_id(project._id)
    assert project_copy.info()['_id'] == project_info['_id']

    project.delete()
    projects_names = [proj.name for proj in pio.Project.list(all=True) if TESTING_ID in proj.name]
    assert project.name not in projects_names
