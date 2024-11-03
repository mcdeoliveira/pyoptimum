import warnings
from copy import deepcopy
from typing import Union, Optional, Iterable, Tuple

import numpy as np
from numpy import typing as npt


class Model:

    def __init__(self, data: Union[dict, "Model"]):
        if isinstance(data, Model):
            # copy constructor
            for k, v in data.__dict__.items():
                setattr(self, k, deepcopy(v))

        else:
            # from dict
            self.r = np.array(data['r'])
            self.F = np.array(data['F'])
            self.Q = np.array(data['Q'])
            assert self.Q.ndim == 1, "Q must be a one-dimensional array"

            self._std = None
            self._Di = None
            self._D = None
            if 'Di' in data:
                assert 'D' not in data, "Di and D cannot be both in model data"
                self.Di = np.array(data['Di'])
            else:
                self.D = np.array(data['D'])

    @property
    def std(self):
        if self._std is None:
            self._std = np.sqrt(self.Q + np.diag(self.F @ self.D @ self.F.transpose()))
        return self._std

    @property
    def D(self):
        if self._D is None:
            # calculate inverse first
            D = np.linalg.inv(self.Di)
            D = (D + D.T)/2
            self._D = D
        return self._D

    @D.setter
    def D(self, value: npt.NDArray):
        self._D = value
        self._Di = None
        self._std = None

    @property
    def Di(self):
        if self._Di is None:
            # calculate inverse first
            Di = np.linalg.inv(self.D)
            Di = (Di + Di.T)/2
            self._Di = Di
        return self._Di

    @Di.setter
    def Di(self, value: npt.NDArray):
        self._Di = value
        self._D = None
        self._std = None

    def to_dict(self, fields: Optional[Iterable]=None,
                as_list: bool=False,
                normalize=False) -> dict:
        # normalize
        alpha = np.max(self.std) ** 2 if normalize else 1.0
        if fields:
            d = {f: getattr(self, f) for f in fields}
            if normalize:
                for f in ['Q', 'D']:
                    if f in fields:
                        d[f] /= alpha
        else:
            d = { 'r': self.r, 'D': self.D / alpha, 'F': self.F, 'Q': self.Q / alpha, 'std': self.std }
        return {k: v.tolist() for k, v in d.items()} if as_list else d

    def unconstrained_frontier(self, x_bar: float=1.):
        q = np.diag(self.Q) + self.F @ self.D @ self.F.transpose()
        b = np.vstack((self.r, np.ones((len(self.r))))).transpose()
        try:
            bsb = b.transpose() @ np.linalg.solve(q, b)
            bsb_inv = np.linalg.inv(bsb)
            a = bsb_inv[0, 0]
            b = -bsb_inv[0, 1]
            c = bsb_inv[1, 1]
            mu_star = b * x_bar / a
            sigma_0 = np.sqrt(c - b ** 2 / a) * x_bar
        except np.linalg.LinAlgError:
            zero_q = self.Q == 0.0
            zero_f = np.sum(self.F, axis=1) == 0
            if np.all(zero_f == zero_q):
                # zero return and zero variance
                reduced_model = Model({'r': self.r[~zero_q], 'Q': self.Q[~zero_q],
                                       'F': self.F[~zero_q,:], 'D': self.D })
                return reduced_model.unconstrained_frontier(x_bar)
            else:
                raise ValueError("Risk-free asset is not supported in this version")
        return a, mu_star, sigma_0

    def return_and_variance(self, x: npt.NDArray) -> Tuple[float, float]:
        # normalize for calculating return and standard deviation
        value = sum(x)
        if value < 0:
            value = -value
            warnings.warn("Total portfolio is negative")
        elif value == 0.:
            value = 1
            warnings.warn("Total portfolio is zero")
        mu = np.dot(x, self.r) / value
        v = self.F.transpose() @ x
        std = np.sqrt(np.dot(self.Q * x, x) + np.dot(self.D @ v, v)) / value
        return mu, std
