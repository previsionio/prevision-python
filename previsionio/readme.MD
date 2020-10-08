# Prevision.io

This package allows you to control the Prevision.io Automated Machine Learning Platform from a python script or notebook.

## Configure client
### Using master token
```python
import previsionio as pio
token = """your_master_token"""
previsionio_url = 'https://cloud.prevision.io'
pio.client.init_client(previsionio_url, token)
```


### Using login & password
```python
import previsionio as pio
previsionio_url = 'https://cloud.prevision.io'
pio.client.init_client_with_login(previsionio_url, 'email', 'password')
```

### Using environment variables
Client will initialize automatically if variables `PREVISION_URL` and `PREVISION_MASTER_TOKEN` are set.



## Start a usecase
### From a file (.zip or .csv)

```python
import previsionio as pio

# Assuming we have some data at 'data/titanic.csv / .zip
data_path = 'data/titanic.csv'
zip_path = 'data/titanic.zip'

# We can create a usecase by first setting some parameters 
# (Usecase.print_params() to display required & optional parameters)
usecase_params = {
    'target_column': 'Survived',
    'type_problem': 'classification',
    'use_case': 'usecase_name',
}

# We can start it directly from the file (.csv or .zip)
uc = pio.Usecase.from_filename(zip_path, **usecase_params)
```

### From a pandas DataFrame

```python
import pandas as pd
import previsionio as pio
data_path = 'data/titanic.csv'

usecase_params = {
    'target_column': 'Survived',
    'type_problem': 'classification',
    'use_case': 'usecase_name',
}

df = pd.read_csv('data/titanic.csv')
uc = pio.Usecase.from_dataframe(df, **usecase_params)
```


### From an existing usecase
If a usecase has already been created (through the web app or the API), you can load it here to make predictions.

```python
uc_old = pio.Usecase.from_usecase_name('some_old_usecase')
```


## Waiting for training
The usecase will refresh in the background, but you might need to block until a certain condition (more than N models, less than X AUC, etc.)

(Waiting for training completion is not implemented for the moment)

```python
# We can wait for example until we have more than 3 models
uc.wait_until(lambda usecase: usecase.get_nb_models() > 3)

# Or until AUC < 0.2
uc.wait_until(lambda usecase: usecase.get_score() < 0.2)
```

You can check the usecase advancement using `uc.print_info()`

## Predicting
Once at least one model is available, you can make predictions using the usecase

```python
test = pd.read_csv('data/titanic.csv')
predict_job_id = uc.predict_from_dataframe(df)
```

You might need to wait a bit until predictions are ready and available.

```python
uc.wait_for_prediction(predict_job_id)
```
And then you can retrieve them as a pandas DataFrame object:
```python
preds = uc.get_prediction(predict_job_id)
```

## Managing usecases
You can stop or delete usecases:
```python
uc.stop()
uc.delete()
```

And you can get a list of projects on the platform:
```python
import previsionio as pio
usecases = pio.client.get_usecases()
```
