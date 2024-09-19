import io
from typing import Optional, Iterable, Literal, Any

import numpy as np
import pandas as pd

from numpy import typing as npt


class Model:

    def __init__(self, data: dict):
        self.r = np.array(data['r'])
        self.D = np.array(data['D'])
        self.F = np.array(data['F'])
        self.Q = np.array(data['Q'])
        self.std = np.sqrt(self.Q + np.diag(self.F @ self.D @ self.F.transpose()))

    def to_dict(self, fields: Optional[Iterable]=None, as_list: bool=False) -> dict:
        if fields:
            d = {f: getattr(self, f) for f in fields}
        else:
            d = { 'r': self.r, 'D': self.D, 'F': self.F, 'Q': self.Q, 'std': self.std }
        return {k: v.tolist() for k, v in d.items()} if as_list else d


LESS_THAN_OR_EQUAL = "\u2264"
GREATER_THAN_OR_EQUAL = "\u2265"

class Portfolio:

    MethodLiteral = Literal['approximate', 'optimal', 'none']
    ConstraintFunctionLiteral = Literal['purchases', 'sales', 'holdings', 'short sales']
    ConstraintSignLiteral = Literal[LESS_THAN_OR_EQUAL, GREATER_THAN_OR_EQUAL]
    ConstraintUnitLiteral = Literal['shares', 'value', 'percent value']

    def __init__(self, portfolio_client, model_client):
        self.portfolio_client = portfolio_client
        self.model_client = model_client
        self.models: dict = {}
        self.model_weights: dict[str, float] = {}
        self.portfolio = None
        self.frontier = None
        self.frontier_query_params = {}
        self.frontier_method: Portfolio.MethodLiteral = 'none'

    @staticmethod
    def unconstrained_frontier(model: Model, x_bar: float=1.):
        # TODO: improve calculation by using the lemma of the inverse instead of
        #       assembling the entire covariance matrix
        q = np.diag(model.Q) + model.F @ model.D @ model.F.transpose()
        b = np.vstack((model.r, np.ones((len(model.r))))).transpose()
        bsb = b.transpose() @ np.linalg.solve(q, b)
        bsb_inv = np.linalg.inv(bsb)
        a = bsb_inv[0, 0]
        b = -bsb_inv[0, 1]
        c = bsb_inv[1, 1]
        mu_star = b * x_bar / a
        sigma_0 = np.sqrt(c - b ** 2 / a) * x_bar
        return a, mu_star, sigma_0

    @staticmethod
    def return_and_variance(x: npt.NDArray, model: Model) -> tuple[float, float]:
        # normalize for calculating return and standard deviation
        value = sum(x)
        mu = np.dot(x, model.r) / value
        v = model.F.transpose() @ x
        std = np.sqrt(np.dot(model.Q * x, x) + np.dot(model.D @ v, v)) / value
        return mu, std

    def import_csv(self, filepath: str|bytes|io.BytesIO) -> None:
        """
        Import portfolio from csv file

        :param filepath: the file path
        """
        # read csv
        portfolio = pd.read_csv(filepath)

        # set ticker as index
        assert 'ticker' in portfolio.columns, "Portfolio must contain a column " \
                                              "named 'ticker'"
        portfolio.set_index('ticker', inplace=True)

        # initialize shares
        if 'shares' not in portfolio:
            portfolio['shares'] = 0.0

        # initialize lower and upper
        portfolio['lower'] = -np.inf
        portfolio['upper'] = np.inf

        # invalidate models and frontier
        self.invalidate_frontier()
        self.invalidate_model()

        # remove all other columns and set portfolio
        self.portfolio = portfolio[['shares', 'lower', 'upper']]

    def retrieve_prices(self) -> float:
        """
        Retrieve the prices of the portfolio

        :return: the total portfolio value
        """

        # retrieve prices
        data = { 'symbols': self.portfolio.index.tolist() }
        prices = self.model_client.call('prices', data)
        prices = pd.DataFrame.from_dict(prices, orient='index',
                                        columns=['timestamp', 'close'])

        # update portfolio value
        self.portfolio['close ($)'] = prices['close']
        self.portfolio['value ($)'] = self.portfolio['shares'] * self.portfolio['close ($)']
        value = sum(self.portfolio['value ($)'])
        self.portfolio['value (%)'] = self.portfolio['value ($)'] / value

        return value

    def set_models_weights(self, model_weights: dict[str, float]):

        # range checking
        assert self.models.keys() == model_weights.keys(), ("Weights must have the same "
                                                            "keys as models")

        assert all(value >= 0 for value in model_weights.values()), ("Weights must be "
                                                                     "non-negative")

        # normalize
        sum_model_weights = sum(model_weights.values())
        if sum_model_weights > 0:
            self.model_weights = {rg: value/sum_model_weights for rg, value in
                                  model_weights.items()}
        else:
            self.model_weights = {rg: 1/len(model_weights) for rg in
                                  model_weights.keys()}

        # update frontier
        if self.frontier is not None:
            model = self.get_model()
            mu_std = (self.frontier['x']
                      .apply(lambda x: pd.Series(Portfolio.return_and_variance(x, model),
                                                 index=['mu', 'std'])))
            self.frontier[['mu', 'std']] = mu_std
            self.frontier_method = 'approximate'

    def retrieve_models(self,
                        market_tickers: list[str],
                        ranges: str | list[str],
                        return_model: str = 'median',
                        common_factors: bool = True,
                        model_weights: dict[str, float]=None) -> None:

        # retrieve models
        data = {
            'tickers': self.portfolio.index.tolist(),
            'market': market_tickers,
            'range': ranges,
            'options': {
                'common': common_factors,
                'return_model': return_model
            }
        }
        model = self.model_client.call('model', data)
        self.models = { rg: Model(data) for rg, data in model.items() }

        # set model weights
        model_weights = model_weights or {rg: 1.0 for rg in model.keys()}
        self.set_models_weights(model_weights)

        # reinitialize frontier
        self.invalidate_frontier()

    def _get_portfolio_query(self,
                             cashflow: float, max_sales: float,
                             short_sales: bool, buy: bool, sell: bool) -> dict:

        # get model data
        model = self.get_model()
        data = model.to_dict(('r', 'Q', 'D', 'F'), as_list=True)

        # get portfolio data
        value0 = self.portfolio['value ($)'].sum()
        value = value0 + cashflow
        x0 = self.portfolio['value (%)']
        data['x0'] = x0.tolist()

        # add cashflow and constraints
        data['cashflow'] = cashflow / value0
        data['options'] = {
            'short': short_sales,
            'buy': buy,
            'sell': sell
        }
        data['constraints'] = [
            {'label': 'sales', 'function': 'sales', 'bounds': max_sales / value0}]

        # has lower bound
        if np.isfinite(self.portfolio['lower']).any():
            xlo = self.portfolio['lower'] * self.portfolio['close ($)'] / value
            data['xlo'] = xlo.tolist()

        # has upper bound
        if np.isfinite(self.portfolio['upper']).any():
            # has lower bound
            xup = self.portfolio['upper'] * self.portfolio['close ($)'] / value
            data['xup'] = xup.tolist()

        return data

    def invalidate_model(self):
        self.models = {}
        self.model_weights = {}

    def invalidate_frontier(self):
        self.frontier = None
        self.frontier_query_params = {}
        self.frontier_method = 'none'

    def retrieve_frontier(self,
                          cashflow: float, max_sales: float,
                          short_sales: bool, buy: bool, sell: bool):

        query = self._get_portfolio_query(cashflow, max_sales,
                                          short_sales, buy, sell)
        sol = self.portfolio_client.call('frontier', query)
        if len(sol['frontier']) == 0:
            raise ValueError('Could not calculate optimal frontier; constraints likely make the problem infeasible.')
        frontier = pd.DataFrame(
            [(s['mu'], np.sqrt(s['sol']['obj']), np.array(s['sol']['x'])) for s in sol['frontier'] if
             s['sol']['status'] == 'optimal'], columns=['mu', 'std', 'x'])
        self.frontier = frontier
        # save query params
        self.frontier_query_params = query
        self.frontier_method = 'optimal'

    def has_frontier(self) -> bool:
        return self.frontier is not None

    def has_models(self) -> bool:
        return bool(self.models)

    def get_model(self) -> Model:
        model = Model({attr: sum([weight * getattr(self.models[rg], attr)
                                  for rg, weight in self.model_weights.items()])
                       for attr in ['r', 'D', 'F', 'Q']})
        return model

    def get_tickers(self) -> list[str]:

        return self.portfolio.index.tolist()

    def get_portfolio_dataframe(self):

        # copy portfolio
        portfolio_df = self.portfolio.copy()

        if self.models:

            # add return
            model = self.get_model()
            portfolio_df['return (%)'] = model.r.copy()
            portfolio_df['std (%)'] = model.std.copy()
            # portfolio_df['trade'] = ['Y' if s and b else 'S' if s else 'B' if b else 'N'
            #                          for s, b in zip(portfolio_df['sell'], portfolio_df['buy'])]
            # portfolio_df.drop(columns=['sell', 'buy'], inplace=True)
            portfolio_df['lower ($)'] = portfolio_df['close ($)'] * portfolio_df['lower']
            portfolio_df['upper ($)'] = portfolio_df['close ($)'] * portfolio_df['upper']

        return portfolio_df

    def get_frontier_range(self):
        if self.frontier is None:
            raise ValueError('Frontier has not been retrieved yet')
        mu_range = self.frontier['mu'].iloc[0], self.frontier['mu'].iloc[-1]
        std_range = self.frontier['std'].iloc[0], self.frontier['std'].iloc[-1]
        return mu_range, std_range

    def get_range(self) -> tuple[list[float], list[float]]:

        model = self.get_model()
        mu_range = [model.r.min(), model.r.max()]
        std_range = [model.std.min(), model.std.max()]

        if self.frontier is not None:
            # account for frontier
            fmu_range, fstd_range = self.get_frontier_range()
            mu_range = min(mu_range[0], fmu_range[0]), max(mu_range[1], fmu_range[1])
            std_range = min(std_range[0], fstd_range[0]), max(std_range[1], fstd_range[1])

        return mu_range, std_range

    def get_return_and_variance(self) -> tuple[float, float]:
        return Portfolio.return_and_variance(self.portfolio['value (%)'], self.get_model())

    def get_unconstrained_frontier(self, x_bar: float=1.):
        return Portfolio.unconstrained_frontier(self.get_model(), x_bar)

    @staticmethod
    def _locate_value(value: Any, column: str, df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        index = df[column].searchsorted(value)
        n = df.shape[0]
        if index == n:
            # got the last element
            return df.iloc[-1], None
        elif index == 0:
            # got the first element
            return None, df.iloc[0]
        else:
            # interpolate
            return df.iloc[index - 1], df.iloc[index]

    def get_recommendation(self, mu: Optional[float]=None,
                           method: MethodLiteral = 'approximate'):
        if self.frontier is None:
            return None

        if mu is None:
            # locate std
            _, std = self.get_return_and_variance()
            left, right = Portfolio._locate_value(std, 'std', self.frontier)
            if left is None:
                # got the first element
                mu = right['mu']
            elif right is None:
                # got the last element
                mu = left['mu']
            else:
                # interpolate
                std1, std2 = left['std'], right['std']
                eta = (std - std1)/(std2-std1)
                mu = (1 - eta) * left['mu'] + eta * right['mu']

        if method == 'approximate':
            # calculate approximate weights

            # locate mu
            left, right = Portfolio._locate_value(mu, 'mu', self.frontier)
            if left is None:
                # got the first element
                x = right['x']
                std = right['std']
            elif right is None:
                # got the last element
                x = left['x']
                std = left['std']
            else:
                # interpolate
                mu1, mu2 = left['mu'], right['mu']
                eta = (mu - mu1)/(mu2-mu1)
                x = (1 - eta) * left['x'] + eta * right['x']
                std = (1 - eta) * left['std'] + eta * right['std']

            return {'x': x, 'status': 'optimal', 'std': std, 'mu': mu}
        elif method == 'exact':
            # exact recommendation
            data = self.frontier_query_params.copy()
            data['mu'] = mu

            recs = self.portfolio_client.call('portfolio', data)
            # return approximate if optimization failed
            return {'x': np.array(recs['x']), 'status': recs['status'], 'std': np.sqrt(recs['obj']), 'mu': mu} \
                if recs['status'] == 'optimal' \
                else self.get_recommendation(mu, method='approximate')

    def remove_constraints(self, tickers: list[str]) -> None:
        """
        Remove all individual constraints on the listed tickers

        :param tickers: the list of tickers
        """

        # quick return
        if not tickers:
            return

        self.portfolio.loc[tickers, 'lower'] = -np.inf
        self.portfolio.loc[tickers, 'upper'] = np.inf

    def apply_constraint(self, tickers: list[str],
                         function: ConstraintFunctionLiteral,
                         sign: ConstraintSignLiteral,
                         value: list[float]|float,
                         unit: ConstraintUnitLiteral,
                         short_sales: bool=True, buy: bool=True, sell: bool=True) -> None:
        """
        Apply constraints to list of tickers

        :param tickers: the list of tickers
        :param function: the function to apply on the left-hand side of the inequality
        :param sign: the sign of the inequality
        :param value: the value of the right-hand side of the inequality
        :param unit: the unit in which the constraint is expressed
        :param short_sales: whether to allow short sales
        :param buy: whether to allow buying
        :param sell: whether to allow selling
        """
        # quick return
        if not tickers:
            return

        # make sure value is array
        if isinstance(value, float):
            value = [value] * len(tickers)
        value = np.array(value)

        # make sure value is in shares
        shares = self.portfolio.loc[tickers, 'shares']
        if unit == 'value':
            value /= self.portfolio.loc[tickers, 'close ($)']
        elif unit == 'percent value':
            value *= shares / self.portfolio.loc[tickers, 'close ($)']

        # initialize
        lb, ub = None, None

        # sales
        if function == 'sales':
            if sign == LESS_THAN_OR_EQUAL:
                lb = shares - value
            elif sign == GREATER_THAN_OR_EQUAL:
                ub = shares - value
        elif function == 'purchases':
            if sign == LESS_THAN_OR_EQUAL:
                ub = shares + value
            elif sign == GREATER_THAN_OR_EQUAL:
                lb = shares + value
        elif function == 'short sales':
            if sign == LESS_THAN_OR_EQUAL:
                lb = -value
            elif sign == GREATER_THAN_OR_EQUAL:
                ub = -value
        elif function == 'holdings':
            if sign == LESS_THAN_OR_EQUAL:
                ub = value
            elif sign == GREATER_THAN_OR_EQUAL:
                lb = value

        # short sales
        if not short_sales:
            if lb is not None:
                lb = np.where(lb > 0, lb, 0)
            if ub is not None:
                ub = np.where(ub > 0, ub, 0)

        # no buys
        if not buy:
            if lb is not None:
                lb = np.where(lb > shares, shares, lb)
            if ub is not None:
                ub = np.where(ub > shares, shares, ub)

        # no sells
        if not sell:
            if lb is not None:
                lb = np.where(lb < shares, shares, lb)
            if ub is not None:
                ub = np.where(ub < shares, shares, ub)

        # apply bounds
        if lb is not None:
            self.portfolio.loc[tickers, 'lower'] = np.maximum(self.portfolio.loc[tickers, 'lower'], lb)

        if ub is not None:
            self.portfolio.loc[tickers, 'upper'] = np.minimum(self.portfolio.loc[tickers, 'upper'], ub)