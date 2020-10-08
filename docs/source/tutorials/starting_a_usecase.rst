.. _starting_usecase:

Starting a usecase
==================

To run a use case and train models on your training subset, you need to define some configuration parameters,
and then simply use the SDK's :class:`.BaseUsecase`-derived methods to have the platform automatically take care
of starting everything for you.

Configuring the use case columns
--------------------------------

In order for the platform to know what your training target is, or whether you have some specific id columns that
should not be taken into account during computation, you need to specify some "column configuration" for your use case.

These columns are bundled together into a :class:`.ColumnConfig` instance; there are 5 interesting parameters:

- ``target_column``: the name of the target column in the dataset
- ``id_column`` (optional): the name of an id column that has no value for the model (it doesn’t have any true signal) but
  is just a handy list of references for example; this column should thus be ignored during training (but it will eventually
  be rematched to the prediction sample to give you back the full data)
- ``fold_column`` (optional): if you want to perform a custom stratification to improve the quality of your predictions (which
  is sometimes better than regular cross-validation), you can pass a specific column name to use as reference; if none is provided,
  a random stratification will be used and will try to force the same distribution of the target between folds
- ``weight_column`` (optional): sometimes, a numerical does not contain an actual feature but rather an indication of how important
  each row is — if that is the case, you can pass the name of this column as `weight_column` (the higher the weight, the more important
  the row — by default, all rows are considered to be of equal importance); note that if this is provided, the optimised metric will
  become weighted
- ``drop_list`` (optional): you can pass a list of column names that you wish to exclude from the training (they will simply be ignored)

There are additional columns required in case of a timeseries or image-based usecase: take a look at the :class:`.ColumnConfig` API reference
for more details.

Here is an example of a very basic column configuration instance:

.. code-block:: python

    column_config = pio.ColumnConfig(target_column='TARGET', id_column='ID')

Configuring the training profile
--------------------------------

You can also fine-tune your use case options by configuring a training profile. This ensemble of variables will decide several things for
your use case: what models are tested out, what metric is used, the desired types of feature engineering...

The function offers you a range of options to choose from, among which some that are used quite often:

- ``models``: the list of "full" models you want to add to your training pipeline chosen among "LR", "RF", "ET", "XGB", "LGB" and "NN"
- ``simple_models``: the list of "simple" models you want to add to your training pipeline chosen among "LR" and "DT"
- ``fe_selected_list``: the list of feature engineering blocks to integrate in the pipeline (these will be applied on your dataset during training to extract relevant
  information and transform it in the best possible way for the models fit step)
- ``profile``: this Prevision.io specific is a way of setting a global run mode that determines both training time and performance. You may choose between 3 profiles:

    1. the "quick" profile runs very fast but has a lower performance (it is recommended for early trials)
    2. the "advanced" profile runs slower but has increased performance (it is usually for optimization steps at the end of your project)
    3. the "normal" profile is something in-between to help you investigate an interesting result

- ``with_blend``: if you turn this setting on, you will allow Prevision.io’s training pipeline to append additional "blend" models at the end that are based on some cherry-picked already-trained models and proceed to further optimization to usually
  get even better results

A common "quick-test" training config could be:

.. code-block:: python

    training_config = pio.TrainingConfig(models=[pio.Model.XGBoost, pio.Model.RandomForest],
                                         features=pio.Feature.Full,
                                         profile=pio.Profile.Quick,
                                         with_blend=False)

Starting the use case!
----------------------

To create the usecase and start your training session, you need to call the ``fit()`` function of one of the SDK's usecase classes. The class you pick
depends on the type of problem your usecase uses: regression, (binary) classification, multiclassification or timeseries; and on whether it uses a simple
tabular dataset or images.

For a full list of the usecase objects and their API, check out the :ref:`api_reference`.

You also need to provide the API with the dataset you want to use (for a tabular usecase) or the CSV reference dataset and ZIP image dataset (for an image
usecase).

The following example shows how to start a regression on a simple tabular dataset:

.. code-block:: python

    uc = pio.Regression.fit('helloworld reg',
                            dataset,
                            column_config=column_config,
                            training_config=training_config)

If you are running an image usecase, then you need to pass the two datasets as a tuple:

The following example shows how to start a regression on a simple tabular dataset (where the CSV reference dataset is a :class:`.Dataset` instance and
the ZIP image dataset is a :class:`.DatasetImages` instance):

.. code-block:: python

    uc = pio.RegressionImages.fit('helloworld images reg',
                                  (dataset_csv, dataset_zip),
                                  column_config=column_config,
                                  training_config=training_config)

When you start your usecase, you can either let the SDK pick a default metric according to your usecase type, or you can choose one yourself from the
list of available :ref:`metrics`.
