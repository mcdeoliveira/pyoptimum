import warnings
from copy import deepcopy
from typing import Union, Optional, Iterable, Tuple, Dict

import numpy as np
from numpy import typing as npt


class Model:
    r"""
    Helper class representing a mean and variance portfolio model

    Model parameters are the arrays:

    * :math:`r`: a vector with the expected returns of each asset
    * :math:`Q`: a vector representing the diagonal component of the portfolio covariance matrix
    * :math:`F`: an array representing the systematic part of the return
    * :math:`D`: the covariance matrix of the model factors

    If :math:`x` are portfolio weights, that is :math:`x` sums to one, then the expected return :math:`\bar{r}` and variance :math:`\sigma^2` calculated by ``Model`` are of the form:

    .. math::
        :nowrap:

        \begin{align}
            \bar{r} &= r^T x, & \sigma^2 &= x^T (\operatorname{diag}(Q) + F D F^T) x
        \end{align}

    :param data: a dictionary with model parameters ``r``, ``Q``, ``F``, and  ``D``, or a ``Model`` instance; ``F`` and ``D`` can be omitted, in which case the covarance is diagonal.
    """

    def __init__(self, data: Union[dict, "Model"]):
        if isinstance(data, Model):
            # copy constructor
            for k, v in data.__dict__.items():
                setattr(self, k, deepcopy(v))

        else:
            # from dict
            self.r = np.array(data['r'])
            self.Q = np.array(data['Q'])
            assert self.Q.ndim == 1, "Q must be a one-dimensional array"

            self.F = np.array(data['F']) if 'F' in data else None
            if self.F is None:
                assert 'D' not in data and 'Di' not in data, "If F is absent then D and Di should also not be given"

            self._std = None
            self._Di = None
            self._D = None
            if 'Di' in data:
                assert 'D' not in data, "Di and D cannot be both in model data"
                self.Di = np.array(data['Di'])
            elif self.F is not None:
                self.D = np.array(data['D'])

    @property
    def has_factors(self) -> bool:
        """
        :return: True if ``F`` and ``D`` are present
        """
        return self.F is not None

    @property
    def std(self) -> npt.NDArray:
        """
        :return: the standard deviation of the individual assets in the model
        """
        if self._std is None:
            self._std = np.sqrt(self.Q + np.diag(self.F @ self.D @ self.F.transpose())) if self.F is not None else np.sqrt(self.Q)
        return self._std

    @property
    def D(self) -> Optional[npt.NDArray]:
        """
        :return: the covariance of the factors; may be ``None``
        """
        if self._D is None and self.F is not None:
            # calculate inverse first
            D = np.linalg.inv(self.Di)
            D = (D + D.T)/2
            self._D = D
        return self._D

    @D.setter
    def D(self, value: npt.NDArray) -> None:
        """
        :param value: the covariance of the factors
        """
        assert self.F is not None, "Cannot set D if F is None"
        self._D = value
        self._Di = None
        self._std = None

    @property
    def Di(self) -> Optional[npt.NDArray]:
        """
        :return: the inverse of the covariance of the factors
        """
        if self._Di is None and self.F is not None:
            # calculate inverse first
            Di = np.linalg.inv(self.D)
            Di = (Di + Di.T)/2
            self._Di = Di
        return self._Di

    @Di.setter
    def Di(self, value: npt.NDArray) -> None:
        """
        :param value: the inverse of the covariance of the factors
        """
        assert self.F is not None, "Cannot set Di if F is None"
        self._Di = value
        self._D = None
        self._std = None

    def to_dict(self, fields: Optional[Iterable]=None,
                as_list: bool=False,
                normalize_variance=False) -> Dict[str, npt.NDArray]:
        """
        Returns a dictionary with the model parameters

        :param fields: list of fields to be present (default: [``r``, ``Q``, ``F``, ``D``] or [``r``, ``Q``] if ``F`` and ``D`` are ``None``)
        :param as_list: return parameters are lists if ``True``
        :param normalize_variance: normalize the parameters so that the covariance matrix has one as its largest diagonal entry
        :return: a dictionary with the parameters
        """
        # normalize
        alpha = np.max(self.std) ** 2 if normalize_variance else 1.0
        if fields:
            d = {f: getattr(self, f) / alpha if f in ['Q', 'D'] else getattr(self, f) for f in fields}
        elif self.F is None:
            d = { 'r': self.r, 'Q': self.Q / alpha }
        else:
            d = { 'r': self.r, 'D': self.D / alpha, 'F': self.F, 'Q': self.Q / alpha }
        return {k: v.tolist() for k, v in d.items()} if as_list else d

    def return_and_variance(self, x: npt.NDArray) -> Tuple[float, float]:
        """
        Calculate the expected return and standard deviation of the portoflio holdings ``x``

        :param x: the portfolio holdings
        :return: a tuple with the return and the standard deviation
        """
        # normalize for calculating return and standard deviation
        value = np.sum(x)
        if value < 0:
            value = -value
            warnings.warn("Total portfolio is negative")
        elif value == 0.:
            value = 1
            warnings.warn("Total portfolio is zero")
        mu = np.dot(x, self.r) / value
        if self.F is None:
            std = np.sqrt(np.dot(self.Q * x, x)) / value
        else:
            v = self.F.transpose() @ x
            std = np.sqrt(np.dot(self.Q * x, x) + np.dot(self.D @ v, v)) / value
        return mu, std

    def unconstrained_frontier(self, x_bar: float=1.) -> Tuple[float, float, float]:
        r"""
        Calculates the parameters of the unconstrained optimal frontier

        The unconstrained optimal frontier is the solution to the quadratic optimization problem

        .. math::

            \sigma^2(\bar{\mu}) = \min_{x} \left \{ \sigma^2(x) : \quad r^T x = \bar{\mu}, \quad \sum_{i} x_i = \bar{x} \right \}

        which has as solution

        .. math::

            \sigma^2(\bar{\mu}) = a (\bar{\mu} - \mu^*)^2 + \sigma^2_0

        This function calculates and returns the parameters :math:`a`, :math:`\mu^*`, and :math:`\sigma_0`.

        :param x_bar: the portfolio sum (defaul: ``1``)
        :return: tuple with ``a``, ``mu_star``, and ``sigma_0``
        """
        try:
            if self.F is None:
                b: npt.NDArray = np.vstack((self.r, np.ones((len(self.r))))).transpose()
                bsb: npt.NDArray = b.transpose() @ ((1/self.Q[:,np.newaxis]) * b)
            else:
                q: npt.NDArray = np.diag(self.Q) + self.F @ self.D @ self.F.transpose()
                b: npt.NDArray = np.vstack((self.r, np.ones((len(self.r))))).transpose()
                bsb: npt.NDArray = b.transpose() @ np.linalg.solve(q, b)
            bsb_inv = np.linalg.inv(bsb)
            a: float = bsb_inv[0, 0]
            b: float = -bsb_inv[0, 1]
            c: float = bsb_inv[1, 1]
            mu_star = b * x_bar / a
            sigma_0 = np.sqrt(c - b ** 2 / a) * x_bar
        except np.linalg.LinAlgError:
            zero_q = self.Q == 0.0
            zero_r = self.r == 0.0
            zero_f = np.sum(self.F, axis=1) == 0 if self.F is not None else True
            if np.all(zero_f == zero_q) and np.all(zero_q == zero_r):
                # zero return and zero variance
                if self.F is None:
                    reduced_model = Model({'r': self.r[~zero_q], 'Q': self.Q[~zero_q]})
                else:
                    reduced_model = Model({'r': self.r[~zero_q], 'Q': self.Q[~zero_q],
                                           'F': self.F[~zero_q, :], 'D': self.D})
                return reduced_model.unconstrained_frontier(x_bar)
            else:
                raise ValueError("Risk-free asset is not supported in this version")
        return a, mu_star, sigma_0

