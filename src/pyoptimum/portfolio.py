import io
import datetime
from pathlib import Path
from typing import Optional, Literal, Any, Union, List, Tuple, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyoptimum import AsyncClient
from pyoptimum.model import Model

LESS_THAN_OR_EQUAL = "\u2264"
GREATER_THAN_OR_EQUAL = "\u2265"

class Portfolio:

    ModelMethodLiteral = Literal['linear', 'linear-fractional']

    MethodLiteral = Literal['approximate', 'optimal']
    ConstraintFunctionLiteral = Literal['purchases', 'sales', 'holdings', 'short sales']
    ConstraintSignLiteral = Literal[LESS_THAN_OR_EQUAL, GREATER_THAN_OR_EQUAL]
    ConstraintUnitLiteral = Literal['shares', 'value', 'percent value']

    def __init__(self,
                 portfolio_client: AsyncClient,
                 model_client: AsyncClient,
                 model_method: ModelMethodLiteral = 'linear',
                 follow_resource: bool = True,
                 max_retries: int = 18,
                 wait_time: Optional[float]=None):
        self.portfolio_client = portfolio_client
        self.model_client = model_client
        self.model_method = model_method
        self.models: dict = {}
        self.model_weights: Dict[str, float] = {}
        self.portfolio = None
        self.frontier = None
        self.frontier_query_params = {}
        self.frontier_method: Portfolio.MethodLiteral = 'approximate'
        self.follow_resource = follow_resource
        self.max_retries = max_retries
        self.wait_time = wait_time

    @staticmethod
    def _locate_value(value: Any, column: str, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
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

    def _update_prices(self, prices: dict) -> float:

        # add prices to dataframe
        prices = pd.DataFrame.from_dict(prices, orient='index',
                                        columns=['timestamp', 'close'])

        # update portfolio value and weights
        self.portfolio['close ($)'] = prices['close']
        self.portfolio['value ($)'] = self.portfolio['shares'] * self.portfolio['close ($)']
        value = self.get_value()
        self.portfolio['value (%)'] = self.portfolio['value ($)'] / value if value > 0.0 else 0.0

        return value

    def _get_portfolio_query(self,
                             cashflow: float, max_sales: float,
                             short_sales: bool, buy: bool, sell: bool,
                             rho: float=0.0) -> dict:

        # get model data
        model = self.get_model()
        data = model.to_dict(('r', 'Q', 'D', 'F'),
                             as_list=True,
                             normalize=True)

        # has regularization
        if rho > 0:
            data['rho'] = rho

        # get portfolio data
        value0 = self.get_value()
        value = value0 + cashflow
        x0 = self.portfolio['value (%)']
        data['x0'] = x0.tolist()

        # check bound in max_sales
        if np.isfinite(max_sales) and max_sales < 0:
            raise ValueError('max_sales must be non-negative')

        # add cashflow and constraints
        if value0 > 0.0:
            data['cashflow'] = cashflow / value0
            if np.isfinite(max_sales):
                data['constraints'] = [
                    {'label': 'sales', 'function': 'sales', 'bounds': max_sales / value0}]
        else:
            if cashflow == 0:
                raise ValueError("Cashflow cannot be zero on a portfolio with no shares")
            if np.isfinite(max_sales) and not short_sales:
                sell = False

            data['cashflow'] = 1

        data['options'] = {
            'short': short_sales,
            'buy': buy,
            'sell': sell
        }

        # has lower bound
        if np.isfinite(self.portfolio['lower']).any():
            xlo = self.portfolio['lower'] * self.portfolio['close ($)'] / value
            data['xlo'] = xlo.tolist()

        # has upper bound
        if np.isfinite(self.portfolio['upper']).any():
            xup = self.portfolio['upper'] * self.portfolio['close ($)'] / value
            data['xup'] = xup.tolist()

        return data

    def set_follow_resource(self, follow_resource: bool,
                            max_retries: int=18,
                            wait_time: Optional[float]=None) -> None:
        self.follow_resource = follow_resource
        self.max_retries = max_retries
        self.wait_time = wait_time

    def get_follow_resource(self) -> dict:
        results = {
            'follow_resource': self.follow_resource,
            'max_retries': self.max_retries
        }
        if self.wait_time is not None:
            results['wait_time'] = self.wait_time
        return results

    def invalidate_model(self):
        """
        Invalidate the current portfolio models
        """
        self.models = {}
        self.model_weights = {}

    def invalidate_frontier(self):
        """
        Invalidate the current frontier
        """
        self.frontier = None
        self.frontier_query_params = {}
        self.frontier_method = 'none'

    def has_prices(self) -> bool:
        """
        :return: True if prices have been retrieved
        """
        return self.portfolio is not None and 'close ($)' in self.portfolio

    def has_frontier(self) -> bool:
        """
        :return: True if a frontier is available
        """
        return self.frontier is not None

    def has_models(self) -> bool:
        """
        :return: True if models have been retrieved
        """
        return bool(self.models)

    def set_models(self, models: Dict[str, Union[dict, Model]],
                   model_weights: Optional[Dict[str, float]]=None) -> None:
        """
        Set portfolio models

        :param models: a dictionary with the models per range
        :param model_weights: the model weights
        """
        # add models
        self.models = {rg: Model(data) for rg, data in models.items()}

        # set model weights
        model_weights = model_weights or {rg: 1.0 for rg in models.keys()}
        self.set_models_weights(model_weights)

        # reinitialize frontier
        self.invalidate_frontier()

    def get_model(self) -> Model:
        """
        :return: the portfolio model for the current model and weights
        """
        assert self.has_models(), "Models have not yet been retrieved"

        if self.model_method == 'linear' or len(self.models) == 1:
            # linear model
            model = Model({attr: sum([weight * getattr(self.models[rg], attr)
                                      for rg, weight in self.model_weights.items()])
                           for attr in ['r', 'D', 'F', 'Q']})
        else:
            # linear-fractional model
            model = Model({attr: sum([weight * getattr(self.models[rg], attr)
                                      for rg, weight in self.model_weights.items()])
                           for attr in ['r', 'Di', 'F', 'Q']})

        return model

    def get_tickers(self) -> List[str]:
        """
        :return: the portfolio tickers
        """
        return self.portfolio.index.tolist()

    def get_value(self) -> float:
        try:
            return sum(self.portfolio['value ($)'])
        except (KeyError, TypeError):
            return 0.0

    def import_csv(self, filepath: Union[str,bytes,io.BytesIO,Path]) -> None:
        """
        Import portfolio from csv file

        :param filepath: the file path
        """
        # read csv
        portfolio = pd.read_csv(filepath)

        # sanitize column names
        portfolio.columns = [c.strip() for c in portfolio.columns]

        # set ticker as index
        assert 'ticker' in portfolio.columns, "Portfolio must contain a column " \
                                              "named 'ticker'"
        portfolio.set_index('ticker', inplace=True)

        # initialize shares
        if 'shares' not in portfolio.columns:
            portfolio['shares'] = 0.0

        # make sure shares are float
        portfolio = portfolio.astype({"shares": float})

        # initialize lower and upper
        portfolio['lower'] = -np.inf
        portfolio['upper'] = np.inf

        # invalidate models and frontier
        self.invalidate_frontier()
        self.invalidate_model()

        # remove all other columns and set portfolio
        self.portfolio = portfolio[['shares', 'lower', 'upper']]

    async def retrieve_prices(self) -> float:
        """
        Retrieve the prices of the portfolio

        :return: the total portfolio value
        """

        # retrieve prices
        data = {'symbols': self.portfolio.index.tolist()}
        prices = await self.model_client.call('prices', data,
                                              **self.get_follow_resource())

        # add prices to dataframe
        value = self._update_prices(prices)

        return value

    async def retrieve_models(self,
                              market_tickers: List[str],
                              ranges: Union[str, List[str]],
                              end: datetime.date = datetime.date.today(),
                              common_factors: bool = False,
                              include_prices: bool = False,
                              model_weights: Dict[str, float] = None) -> List[str]:
        """
        Retrieve portfolio models based on market tickers

        :param market_tickers: the market tickers
        :param ranges: the ranges to retrieve the portfolio models
        :param end: the last day to retrieve models
        :param common_factors: whether to keep factors common
        :param include_prices: whether to include prices on results
        :param model_weights: the model weights
        :return a list of messages
        """

        # retrieve models
        data = {
            'tickers': self.portfolio.index.tolist(),
            'market': market_tickers,
            'end': str(end),
            'range': ranges,
            'options': {
                'common': common_factors,
                'include_prices': include_prices
            }
        }
        models = await self.model_client.call('model', data,
                                              **self.get_follow_resource())

        if include_prices:
            # update prices
            self._update_prices(models.pop('prices'))

        # remove messages
        messages = models.pop('messages')

        # set models
        self.set_models(models, model_weights)

        return messages

    async def retrieve_frontier(self,
                                cashflow: float, max_sales: float,
                                short_sales: bool, buy: bool, sell: bool,
                                rho: float=0.0) -> None:
        """
        Retrieve the portfolio frontier

        :param cashflow: the cashflow
        :param max_sales: the max sales
        :param short_sales: whether to allow short sales
        :param buy: whether to allow buys
        :param sell: whether to allow sells
        :param rho: regularization factor
        """

        # assert models and prices are defined
        assert self.has_prices() and self.has_models(),\
            "Either prices or models are missing"

        # retrieve frontier
        query = self._get_portfolio_query(cashflow, max_sales,
                                          short_sales, buy, sell, rho)
        sol = await self.portfolio_client.call('frontier', query,
                                               **self.get_follow_resource())

        if len(sol['frontier']) == 0:
            self.invalidate_frontier()
            raise ValueError('Could not calculate optimal frontier; constraints likely make the problem infeasible.')

        # calculate variance
        model = self.get_model()
        values = []
        for s in sol['frontier']:
            if s['sol']['status'] == 'optimal':
                mu = s['mu']
                x = np.array(s['sol']['x'])
                _, std = model.return_and_variance(x)
                values.append((mu, std, x))

        # assemble return dataframe
        frontier = pd.DataFrame(values, columns=['mu', 'std', 'x'])
        self.frontier = frontier

        # save query params
        self.frontier_query_params = query
        self.frontier_method = 'optimal'

    async def retrieve_recommendation(self, mu: Optional[float]=None,
                                      method: MethodLiteral = 'approximate') -> dict:
        """
        Retrieve or calculate recommendations

        :param mu: the expected return
        :param method: whether to calculate approximate recommendations or retrieve from API
        :return:
        """

        assert self.has_frontier(), "Frontier has not been retrieved"

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

            recs = await self.portfolio_client.call('portfolio', data,
                                                    **self.get_follow_resource())
            if recs['status'] == 'optimal':
                x = np.array(recs['x'])
                _, std = self.get_model().return_and_variance(x)
                return {'x': x, 'status': recs['status'], 'std': std, 'mu': mu}
            else:
                # return approximate if optimization failed
                return await self.retrieve_recommendation(mu, method='approximate')

        else:

            raise Exception("Invalid method '%s'", method)

    def add_to_frontier(self, mu: float, std: float, x: npt.NDArray) -> None:
        """
        Add point to frontier

        :param mu: the return
        :param std: the standard deviation
        :param x: the portfolio values
        """
        assert self.has_frontier(), "Frontier has not been retrieved"

        frontier = pd.concat((self.frontier,
                              pd.DataFrame([(mu, std, x)],
                                           columns=['mu', 'std', 'x'])),
                             ignore_index=True)
        frontier.sort_values(by=['mu'], inplace=True, ignore_index=True)
        self.frontier = frontier

    def set_models_weights(self, model_weights: Dict[str, float]) -> None:
        """
        Set weights for the current portfolio models

        :param model_weights: the model weights
        """

        # range checking
        assert self.has_models(), "Models have not yet been retrieved"

        assert self.models.keys() == model_weights.keys(), ("Weights must have the same "
                                                            "keys as models")

        assert all(value >= 0 for value in model_weights.values()), ("Weights must be "
                                                                     "non-negative")

        # normalize weights
        sum_model_weights = sum(model_weights.values())
        if sum_model_weights > 0:
            self.model_weights = {rg: value/sum_model_weights for rg, value in
                                  model_weights.items()}
        else:
            self.model_weights = {rg: 1/len(model_weights) for rg in
                                  model_weights.keys()}

        # update frontier
        if self.has_frontier():
            model = self.get_model()
            mu_std = (self.frontier['x']
                      .apply(lambda x: pd.Series(model.return_and_variance(x),
                                                 index=['mu', 'std'])))
            self.frontier[['mu', 'std']] = mu_std
            self.frontier_method = 'approximate'

    def set_model_method(self, method: ModelMethodLiteral):
        self.model_method = method

    def get_portfolio_dataframe(self):
        """
        :return: the portfolio as a dataframe
        """

        # copy portfolio
        portfolio_df = self.portfolio.copy()

        if self.has_models():
            # add return
            model = self.get_model()
            portfolio_df['return (%)'] = model.r.copy()
            portfolio_df['std (%)'] = model.std.copy()

        if self.has_prices():
            # add lower and upper bounds in value
            portfolio_df['lower ($)'] = portfolio_df['close ($)'] * portfolio_df['lower']
            portfolio_df['upper ($)'] = portfolio_df['close ($)'] * portfolio_df['upper']

        return portfolio_df

    def get_frontier_range(self):
        """
        :return: the range of frontier values
        """
        if self.frontier is None:
            raise ValueError('Frontier has not been retrieved yet')
        mu_range = self.frontier['mu'].iloc[0], self.frontier['mu'].iloc[-1]
        std_range = self.frontier['std'].iloc[0], self.frontier['std'].iloc[-1]
        return mu_range, std_range

    def get_range(self) -> Tuple[List[float], List[float]]:
        """
        :return: the range of the current model
        """

        model = self.get_model()
        mu_range = [model.r.min(), model.r.max()]
        std_range = [model.std.min(), model.std.max()]

        if self.frontier is not None:
            # account for frontier
            fmu_range, fstd_range = self.get_frontier_range()
            mu_range = min(mu_range[0], fmu_range[0]), max(mu_range[1], fmu_range[1])
            std_range = min(std_range[0], fstd_range[0]), max(std_range[1], fstd_range[1])

        return mu_range, std_range

    def get_return_and_variance(self) -> Tuple[float, float]:
        """
        :return: the return and standard deviation of the current portfolio
        """
        return self.get_model().return_and_variance(self.portfolio['value (%)'])

    def get_unconstrained_frontier(self, x_bar: float=1.):
        """
        :return: the unconstrained frontier parameters
        """
        return self.get_model().unconstrained_frontier(x_bar)

    def remove_constraints(self, tickers: List[str]) -> None:
        """
        Remove all individual constraints on the listed tickers

        :param tickers: the list of tickers
        """

        # quick return
        if not tickers:
            return

        self.portfolio.loc[tickers, 'lower'] = -np.inf
        self.portfolio.loc[tickers, 'upper'] = np.inf

    def apply_constraint(self, tickers: List[str],
                         function: ConstraintFunctionLiteral,
                         sign: ConstraintSignLiteral,
                         value: Union[List[float], float],
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
        if isinstance(value, (int, float)):
            value = [value] * len(tickers)
        value = np.array(value, dtype='float64')

        # make sure value is in shares
        shares = self.portfolio.loc[tickers, 'shares'].values
        if unit == 'value':
            close = self.portfolio.loc[tickers, 'close ($)'].values
            value /= close
        elif unit == 'percent value':
            value *= shares / 100

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
