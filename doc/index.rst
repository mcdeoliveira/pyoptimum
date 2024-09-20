.. PyOptimum documentation master file, created by
   sphinx-quickstart on Sat Jun 17 11:00:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyOptimum's documentation!
=====================================

This package contains some utility functions to facilitate the connection to the
`Optimize API <https://optimize.vicbee.net/optimize/api/ui>`_ and
`Models API <https://optimize.vicbee.net/models/api/ui>`_ in python. See
`demo <https://vicbee.net/optimize.html>`_ for a more interesting jupyter
notebook demo.

This library is a simple wrapper of the
`requests <https://requests.readthedocs.io>`_ and
`aiohttp <https://aiohttp.readthedocs.io>`_
python libraries that helps making token management transparent to the user.

Documentation is limited to the functionality provided by this library.
See `Optimize API <https://optimize.vicbee.net/optimize/api/ui>`_ for a complete
API documentation.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   src/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   src/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
