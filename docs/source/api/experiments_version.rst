Experiments Version
================

Prevision.io's Python SDK enables you to very easily run experiments of different types:
regression, (binary) classification, multiclassification or timeseries.

All these classes inherit from the base :class:`previsionio.experiment.BaseExperimentVersion` class,
and then from the :class:`previsionio.supervised.Supervised` class.

When starting an experiment, you also need to specify a training configuration.

Take a look at the specific documentation pages for a more in-depth explanation of each layer
and of the experiment configuration options:

.. toctree::
   :maxdepth: 3

   experiments/base_experiment.rst
   experiments/supervised.rst
   experiments/time_series.rst
   experiments/text_similarity.rst
   experiments/config.rst
