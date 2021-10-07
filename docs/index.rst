.. prevision-python documentation master file, created by
   sphinx-quickstart on Tue Jul 17 11:11:51 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to prevision-python's documentation!
============================================

Prevision.io is an automated SaaS machine learning platform that enables you to create deploy and monitor powerful predictive models and business applications.

This documentation focuses on how to use Prevision.io's Python SDK for a direct usage in your data science scripts.

To take a quick peek at the available features, look at the :ref:`getting_started` guide.

If you'd rather examine the Python API directly, here is the direct :ref:`api_reference`.

The compatibility version between Prevision.io’s Python SDK and Prevision Platform works as follows:


.. list-table:: Compatibility matrix
   :widths: auto
   :header-rows: 1

   * -
     - Prevision 10.10
     - Prevision 10.11
     - Prevision 10.12
     - Prevision 10.13
     - Prevision 10.14
     - Prevision 10.15
     - Prevision 10.16
     - Prevision 10.17
     - Prevision 10.18
     - Prevision 10.19
     - Prevision 10.20
     - Prevision 10.21
     - Prevision 10.22
     - Prevision 10.23
     - Prevision 10.24
     - Prevision 11.0
   * - Prevision Python SDK 10.10
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.11
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.12
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.13
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.14
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.15
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.16
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.17
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.18
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
   * - Prevision Python SDK 10.19
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 10.20
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 10.21
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 10.22
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 10.23
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 10.24
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✘
   * - Prevision Python SDK 11.0
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✘
     - ✓


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/getting_started.rst
   source/api.rst
