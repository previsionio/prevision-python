# Prevision-python tests
Requires pytest, pytest-cov

Setup two environment variables (e.g. local setup):

- `PREVISION_URL=http://localhost:3001`
- `PREVISION_MASTER_TOKEN=***`

Run :

```bash
py.test --cov=../previsionio --verbose --cov-config .coveragerc --cov-report html  --junit-xml=prevision_python.xml
```

To only run `test_supervised.py`:

```bash
py.test "test_supervised.py" --cov=../previsionio --verbose --cov-config .coveragerc --cov-report html  --junit-xml=prevision_python.xml
```
