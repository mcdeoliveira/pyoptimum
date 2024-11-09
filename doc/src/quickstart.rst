Quick start
===========

Installation
------------

Install using pip

.. code-block:: python

    pip install pyoptimum


Hello World
-----------

Try the following code

.. code-block:: python

    from pyoptimum import Client
    username = 'demo@optimize.vicbee.net'
    password = 'optimize'
    client = Client(username, password)

to create a client object then call the :meth:`pyoptimum.Client.call` to solve a small
portfolio problem

.. code-block:: python

    Q = [[0.01, 0.001],[0.001, 0.02]]
    r = [0.1, 0.3]
    mu = 0.3
    data = {'Q': Q, 'r': r, 'mu': mu}
    response = client.call('portfolio', data)

in which `Q` is the model's covariance matrix, `r` is the vector of expected returns,
and `mu` the portfolio desired return. The response object ``response`` contains the
optimized portfolio::

    response = {
        'obj': 0.019999999998211167,
        'status': 'optimal',
        'x': [-5.532633539658773e-11, 0.9999999999580456]
    }

in which `obj` is the portfolio variance, and `x` the optimal portfolio weights.
The returned portfolio is "optimal," as shown in `status`.

Basic error handling
--------------------

Running the code below

.. code-block:: python

    Q = [[0.01, 0.001],[0.001, 0.02]]
    r = [0.1, 0.3, 0.4]
    mu = 0.3
    data = {'Q': Q, 'r': r, 'mu': mu}
    response = client.call('portfolio', data)

will raise the exception::

    Traceback (most recent call last):
      ...
      File ".../pyoptimum/__init__.py", line 115, in call
        raise PyOptimumException(self.detail)
    pyoptimum.PyOptimumException: 'Q' must be an array of size 3 x 3
        Dimensions: Q[2, 2]!=[3, 3], r[3]==[3]

The details of the exception can be retrieved from the ``client``::

    print(client.detail)

to produce::

    'Q' must be an array of size 3 x 3
    Dimensions: Q[2, 2]!=[3, 3], r[3]==[3]

which indicates a mismatch between the dimension of the covariance matrix `Q` and the
vector of expected returns `r`.

Asynchronous Client
-------------------

:class:`pyoptimum.AsyncClient` is an an asynchronous version of :class:`pyoptimum.Client` with basically the same interface and some additional features.

.. code-block:: python

    from pyoptimum import AsyncClient
    client = AsyncClient(username, password)

one can then handle calls asynchronously as in

.. code-block:: python

    response = await client.call('portfolio', data)

Besides the automatic token renewal feature, the :meth:`pyoptimum.AsyncClient:call` will
also poll the APIs for asynchronous resources when computations are deferred. See
`this case study <https://vicbee.net/case.html>`_ for more details on the architecture.

Models and Portfolio
--------------------

The classes
:class:`pyoptimum.portfolio.Portfolio` and :class:`pyoptimum.model.Model`
implement additional functionality that further simplifies working with portfolios and mean variance models.
Details are provided in the :ref:`Reference` section.

Learn more
----------

- Checkout the detailed documentation in :ref:`Reference`.
- See `Optimize API <https://optimize.vicbee.net/optimize/api/ui>`_ and `Models API <https://optimize.vicbee.net/models/api/ui>`_ for a complete API documentation.
- Checkout `vicbee.net <https://vicbee.net>`_ and run the following jupiter notebook

    .. image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/mcdeoliveira/pyoptimum-examples/master?filepath=examples%2Fportfolio.ipynb

  for more details on the features of the API.
