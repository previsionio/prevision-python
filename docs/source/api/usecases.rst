Usecases
========

Prevision.io's Python SDK enables you to very easily run usecases of different types:
regression, (binary) classification, multiclassification or timeseries.

All these classes inherit from the base :class:`previsionio.usecase.BaseUsecase` class,
and then from the :class:`previsionio.supervised.Supervised` class.

When starting a usecase, you also need to specify a training configuration.

Take a look at the specific documentation pages for a more in-depth explanation of each layer
and of the usecase configuration options:

.. toctree::
   :maxdepth: 3

   usecases/base_usecase.rst
   usecases/supervised.rst
   usecases/config.rst
