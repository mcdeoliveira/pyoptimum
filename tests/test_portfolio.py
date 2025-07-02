import typing
import unittest
import datetime
import os

import numpy as np


username = 'demo@optimize.vicbee.net'
password = 'optimize'
base_url = os.getenv('TEST_BASE_URL', 'https://optimize.vicbee.net')


class TestBasic(unittest.TestCase):

    def setUp(self):

        import pyoptimum

        self.portfolio_client = pyoptimum.AsyncClient(username=username,
                                                      password=password,
                                                      api='optimize', base_url=base_url)
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,
                                                  api='models', base_url=base_url)

    def test_constructor(self):

        from pyoptimum.portfolio import Portfolio

        portfolio = Portfolio(self.portfolio_client, self.model_client)
        self.assertIsInstance(portfolio, Portfolio)
        self.assertFalse(portfolio.has_models())
        self.assertFalse(portfolio.has_frontier())

        self.assertEqual(portfolio.get_value(), 0.0)

        from pathlib import Path
        file = Path(__file__).parent / 'test.csv'
        portfolio.import_csv(file)
        self.assertListEqual(portfolio.portfolio.columns.tolist(),['shares', 'lower', 'upper'])
        self.assertListEqual(portfolio.portfolio.index.tolist(),['AAPL', 'MSFT', 'ASML', 'TQQQ'])
        self.assertListEqual(portfolio.portfolio['shares'].tolist(), [1, 10, 0, 13])
        self.assertCountEqual(portfolio.groups['g1'], ['AAPL', 'TQQQ'])
        self.assertCountEqual(portfolio.groups['g2'], ['AAPL', 'ASML'])

        self.assertEqual(portfolio.get_value(), 0.0)


class TestPortfolio(unittest.IsolatedAsyncioTestCase):

    def setUp(self):

        import pyoptimum
        from pyoptimum.portfolio import Portfolio

        self.portfolio_client = pyoptimum.AsyncClient(username=username,
                                                      password=password,
                                                      api='optimize', base_url=base_url)
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,
                                                  api='models', base_url=base_url)
        self.portfolio = Portfolio(self.portfolio_client, self.model_client)
        from pathlib import Path
        file = Path(__file__).parent / 'test.csv'
        self.portfolio.import_csv(file)

    def test_split(self):

        tickers = ['AAPL', 'MSFT', 'ASML', 'TQQQ']
        self.assertListEqual(tickers, self.portfolio.get_tickers())
        self.assertIsNone(self.portfolio.inactive_portfolio)

        self.portfolio.split(tickers)
        self.assertListEqual(tickers, self.portfolio.get_tickers())
        self.assertIsNone(self.portfolio.inactive_portfolio)

        tickers = ['AAPL', 'MSFT', 'TQQQ']
        self.portfolio.split(tickers)

        self.assertListEqual(tickers, self.portfolio.get_tickers())
        self.assertIsNotNone(self.portfolio.inactive_portfolio)
        self.assertListEqual(self.portfolio.inactive_portfolio.index.to_list(), ['ASML'])

        tickers = ['AAPL', 'MSFT', 'TQQQ', 'PORT', 'WINE']
        with self.assertRaises(ValueError) as e:
            self.portfolio.split(tickers)
        self.assertIn("are not active in the current portfolio", str(e.exception))

        tickers = []
        with self.assertRaises(ValueError) as e:
            self.portfolio.split(tickers)
        self.assertIn("tickers cannot be empty", str(e.exception))


    async def test_prices(self):

        self.assertEqual(self.portfolio.get_value(), 0.0)

        # retrieve prices
        self.assertFalse(self.portfolio.has_prices())
        await self.portfolio.retrieve_prices()
        self.assertTrue(self.portfolio.has_prices())

        self.assertEqual(self.portfolio.get_value(), sum(self.portfolio.portfolio['value ($)']))

        self.assertIn('close ($)', self.portfolio.portfolio)
        self.assertIn('value ($)', self.portfolio.portfolio)
        self.assertIn('value (%)', self.portfolio.portfolio)
        np.testing.assert_array_equal(self.portfolio.portfolio['value ($)'], self.portfolio.portfolio['close ($)'] * self.portfolio.portfolio['shares'])
        np.testing.assert_array_equal(self.portfolio.portfolio['value (%)'], self.portfolio.portfolio['value ($)'] / sum(self.portfolio.portfolio['value ($)']))

        with self.assertRaises(AssertionError):
            await self.portfolio.retrieve_frontier(0, 0, False, True, True)

    async def test_models(self):

        # try getting model before retrieving
        with self.assertRaises(AssertionError):
            self.portfolio.get_model()

        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})

        # retrieve models
        from pyoptimum.portfolio import Portfolio
        market_tickers = list(Portfolio.BasicMarket.keys())
        ranges = Portfolio.BasicRanges
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        end = datetime.date(2024, 12, 13)
        await self.portfolio.retrieve_custom_models(market_tickers, ranges, end=end)
        self.assertFalse(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        with self.assertRaises(AssertionError) as e:
            await self.portfolio.retrieve_frontier(0, 0, False, True, True)
        self.assertIn('Either prices or models are missing', str(e.exception))

        from pyoptimum.model import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

    async def test_model_without_data(self):

        import pandas as pd

        # try getting model before retrieving
        with self.assertRaises(AssertionError):
            self.portfolio.get_model()

        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})

        # add new asset
        self.portfolio.portfolio = pd.concat([
            self.portfolio.portfolio,
            pd.DataFrame({'shares': 1, 'lower': -np.inf, 'upper': np.inf},
                         index=['NNE'])
        ])
        portfolio_tickers = self.portfolio.get_tickers()
        self.assertListEqual(portfolio_tickers, ['AAPL', 'MSFT', 'ASML', 'TQQQ', 'NNE'])
        #print(self.portfolio.portfolio)

        # retrieve models
        market_tickers = ['^DJI']
        ranges = ['1mo', '6mo', '1y']
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        messages, tickers, market = await self.portfolio.retrieve_custom_models(market_tickers, ranges)
        self.assertIsNone(self.portfolio.inactive_portfolio)
        self.assertFalse(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())
        self.assertListEqual(market, market_tickers)
        self.assertListEqual(tickers, portfolio_tickers)

        with self.assertRaises(AssertionError):
            await self.portfolio.retrieve_frontier(0, 0, False, True, True)

        from pyoptimum.model import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # retrieve model without data
        messages, tickers, market = await self.portfolio.retrieve_custom_models(market_tickers, ranges,
                                                                                end=datetime.date(2023, 12, 29),
                                                                                include_prices=True)
        self.assertCountEqual(tickers, ['AAPL', 'MSFT', 'ASML', 'TQQQ'])
        self.assertCountEqual(market, market_tickers)
        self.assertCountEqual(self.portfolio.get_tickers(), ['AAPL', 'MSFT', 'ASML', 'TQQQ'])
        self.assertIsNotNone(self.portfolio.inactive_portfolio)
        self.assertListEqual(self.portfolio.inactive_portfolio.index.to_list(), ['NNE'])
        #print(self.portfolio.portfolio)
        #print(self.portfolio.inactive_portfolio)


    async def test_models_with_prices(self):

        # try getting model before retrieving
        with self.assertRaises(AssertionError):
            self.portfolio.get_model()

        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})

        # retrieve models
        from pyoptimum.portfolio import Portfolio
        market_tickers = list(Portfolio.BasicMarket.keys())
        ranges = Portfolio.BasicRanges
        end = datetime.date(2024, 12, 13)
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_custom_models(market_tickers, ranges, end=end, include_prices=True)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        await self.portfolio.retrieve_frontier(100, 0, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        from pyoptimum.model import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # repeat with basic model
        import copy
        portfolio_copy = copy.copy(self.portfolio)

        # reinitialize
        self.setUp()
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())

        self.assertTrue(portfolio_copy.has_prices())
        self.assertTrue(portfolio_copy.has_models())

        await self.portfolio.retrieve_basic_models(end=end, include_prices=True)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        await self.portfolio.retrieve_frontier(100, 0, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # models should be equal
        for v1, v2 in zip(self.portfolio.get_model().to_dict().values(),
                          portfolio_copy.get_model().to_dict().values()):
            np.testing.assert_array_equal(v1, v2)

    async def test_frontier(self):

        from pyoptimum import PyOptimumException

        # retrieve prices
        self.assertFalse(self.portfolio.has_prices())
        await self.portfolio.retrieve_prices()
        self.assertTrue(self.portfolio.has_prices())

        # retrieve models
        market_tickers = ['^DJI']
        ranges = ['1mo', '6mo', '1y']
        self.assertTrue(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_custom_models(market_tickers, ranges)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        # retrieve frontier
        await self.portfolio.retrieve_frontier(0, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # retrieve unfeasible frontier
        with self.assertRaises(PyOptimumException) as e:
            await self.portfolio.retrieve_frontier(-100, 0, False, True, True)
        self.assertIn('constraint is not feasible', str(e.exception))

        # make sure it gets invalidated
        self.assertFalse(self.portfolio.has_frontier())

        # set model weights
        self.assertDictEqual(self.portfolio.model_weights, {rg: 1/3 for rg in ranges})
        self.portfolio.set_model_weights({rg: v for rg, v in zip(ranges, [1, 2, 3])})
        self.assertDictEqual(self.portfolio.model_weights, {rg: v/6 for rg, v in zip(ranges, [1,2,3])})
        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})
        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({rg: v for rg, v in zip(ranges, [1, -2, 3])})

    async def test_diagonal_models(self):

        from pyoptimum import PyOptimumException

        # try getting model before retrieving
        with self.assertRaises(AssertionError):
            self.portfolio.get_model()

        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})

        # retrieve models
        market_tickers = []
        ranges = ['1mo', '6mo', '1y']
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_custom_models(market_tickers, ranges, include_prices=True)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())
        self.assertEqual(self.portfolio.model_method, 'diagonal')

        await self.portfolio.retrieve_frontier(0, 0, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        from pyoptimum.model import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # retrieve frontier
        await self.portfolio.retrieve_frontier(0, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # retrieve unfeasible frontier
        with self.assertRaises(PyOptimumException) as e:
            await self.portfolio.retrieve_frontier(-100, 0, False, True, True)
        self.assertIn('constraint is not feasible', str(e.exception))

        # make sure it gets invalidated
        self.assertFalse(self.portfolio.has_frontier())

        # set model weights
        self.assertDictEqual(self.portfolio.model_weights, {rg: 1/3 for rg in ranges})
        self.portfolio.set_model_weights({rg: v for rg, v in zip(ranges, [1, 2, 3])})
        self.assertDictEqual(self.portfolio.model_weights, {rg: v/6 for rg, v in zip(ranges, [1,2,3])})
        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({})
        with self.assertRaises(AssertionError):
            self.portfolio.set_model_weights({rg: v for rg, v in zip(ranges, [1, -2, 3])})


class TestPortfolioZeroShares(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        import pyoptimum
        from pyoptimum.portfolio import Portfolio

        self.portfolio_client = pyoptimum.AsyncClient(username=username,
                                                      password=password, api='optimize',
                                                      base_url=base_url)
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,
                                                  api='models', base_url=base_url)
        self.portfolio = Portfolio(self.portfolio_client, self.model_client)
        from pathlib import Path
        file = Path(__file__).parent / 'test_zero.csv'
        self.portfolio.import_csv(file)

    async def test_prices(self):

        self.assertEqual(self.portfolio.get_value(), 0.0)

        # retrieve prices
        self.assertFalse(self.portfolio.has_prices())
        await self.portfolio.retrieve_prices()
        self.assertTrue(self.portfolio.has_prices())

        self.assertEqual(self.portfolio.get_value(),
                         sum(self.portfolio.portfolio['value ($)']))
        self.assertEqual(self.portfolio.get_value(), 0.0)

        self.assertIn('close ($)', self.portfolio.portfolio)
        self.assertIn('value ($)', self.portfolio.portfolio)
        self.assertIn('value (%)', self.portfolio.portfolio)
        np.testing.assert_array_equal(self.portfolio.portfolio['value ($)'], 0.0)
        np.testing.assert_array_equal(self.portfolio.portfolio['value (%)'], 0.0)

        with self.assertRaises(AssertionError):
            await self.portfolio.retrieve_frontier(0, 0, False, True, True)

    async def test_frontier(self):

        # retrieve prices
        self.assertFalse(self.portfolio.has_prices())
        await self.portfolio.retrieve_prices()
        self.assertTrue(self.portfolio.has_prices())

        # retrieve models
        market_tickers = ['^DJI']
        ranges = ['1mo', '6mo', '1y']
        self.assertTrue(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_custom_models(market_tickers, ranges)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        # zero cashflow without short sales
        with self.assertRaises(ValueError) as e:
            await self.portfolio.retrieve_frontier(0, 100, False, True, True)
        self.assertTrue("Cashflow cannot be zero" in str(e.exception))

        # retrieve frontier
        await self.portfolio.retrieve_frontier(1000, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # # retrieve frontier
        # with self.assertRaises(ValueError) as e:
        #     await self.portfolio.retrieve_frontier(-100, 100, True, True, True)
        # self.assertIn("Could not calculate optimal frontier; constraints likely make the problem infeasible.", str(e.exception))


class TestWithPortfolio(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):

        import pyoptimum
        from pyoptimum.portfolio import Portfolio

        self.portfolio_client = pyoptimum.AsyncClient(username=username,
                                                      password=password,
                                                      api='optimize',
                                                      base_url=base_url)
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,
                                                  api='models', base_url=base_url)
        self.portfolio = Portfolio(self.portfolio_client, self.model_client)
        from pathlib import Path
        file = Path(__file__).parent / 'test.csv'
        self.portfolio.import_csv(file)

        # retrieve models and price
        self.market_tickers = ['^DJI', '^RUT']
        self.ranges = ['1mo', '6mo', '1y']
        await self.portfolio.retrieve_custom_models(self.market_tickers, self.ranges,
                                                    include_prices=True)


class TestModelMethods(TestWithPortfolio):

    async def test_model_methods(self):

        from pyoptimum.model import Model

        weights = {
            '1mo': 3,
            '6mo': 1,
            '1y': 2
        }
        self.portfolio.set_model_weights(weights)
        weights = {
            '1mo': 3/6,
            '6mo': 1/6,
            '1y': 2/6
        }
        self.assertDictEqual(self.portfolio.model_weights, weights)

        self.assertEqual(self.portfolio.model_method, 'linear')
        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # check model correctness
        np.testing.assert_array_almost_equal(1e10 * model.Q,
                                             1e10*((3/6) * self.portfolio.models['1mo'].Q +
                                                   (1/6) * self.portfolio.models['6mo'].Q +
                                                   (2/6) * self.portfolio.models['1y'].Q))
        np.testing.assert_array_almost_equal(model.F,
                                             (3/6) * self.portfolio.models['1mo'].F +
                                             (1/6) * self.portfolio.models['6mo'].F +
                                             (2/6) * self.portfolio.models['1y'].F)
        np.testing.assert_array_almost_equal(1e10 * model.D,
                                             1e10 * ((3/6) * self.portfolio.models['1mo'].D +
                                                     (1/6) * self.portfolio.models['6mo'].D +
                                                     (2/6) * self.portfolio.models['1y'].D))
        np.testing.assert_array_almost_equal(model.r,
                                             (3/6) * self.portfolio.models['1mo'].r +
                                             (1/6) * self.portfolio.models['6mo'].r +
                                             (2/6) * self.portfolio.models['1y'].r)

        self.portfolio.set_model_method('linear-fractional')
        self.assertEqual(self.portfolio.model_method, 'linear-fractional')
        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

        # check model correctness
        np.testing.assert_array_almost_equal(1e10 * model.Q,
                                             1e10*((3/6) * self.portfolio.models['1mo'].Q +
                                                   (1/6) * self.portfolio.models['6mo'].Q +
                                                   (2/6) * self.portfolio.models['1y'].Q))
        np.testing.assert_array_almost_equal(model.F,
                                             (3/6) * self.portfolio.models['1mo'].F +
                                             (1/6) * self.portfolio.models['6mo'].F +
                                             (2/6) * self.portfolio.models['1y'].F)
        np.testing.assert_array_almost_equal(1e10 * model.Di,
                                             1e10 * ((3/6) * self.portfolio.models['1mo'].Di +
                                                     (1/6) * self.portfolio.models['6mo'].Di +
                                                     (2/6) * self.portfolio.models['1y'].Di))
        np.testing.assert_array_almost_equal(model.r,
                                             (3/6) * self.portfolio.models['1mo'].r +
                                             (1/6) * self.portfolio.models['6mo'].r +
                                             (2/6) * self.portfolio.models['1y'].r)


class TestPortfolioFunctions(TestWithPortfolio):

    async def test_apply_constraint(self):

        from pyoptimum.portfolio import LESS_THAN_OR_EQUAL, GREATER_THAN_OR_EQUAL, EQUAL, Portfolio

        tickers = ['MSFT']
        value = 1
        sign = LESS_THAN_OR_EQUAL
        for function in typing.get_args(Portfolio.ConstraintFunctionLiteral):
            for unit in typing.get_args(Portfolio.ConstraintUnitLiteral):
                self.portfolio.apply_constraint(tickers, function, sign, value, unit)
                if function == 'sales' or function == 'short sales':
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                elif function == 'purchases' or function == 'holdings':
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                self.portfolio.remove_constraints(tickers)

        sign = GREATER_THAN_OR_EQUAL
        for function in typing.get_args(Portfolio.ConstraintFunctionLiteral):
            for unit in typing.get_args(Portfolio.ConstraintUnitLiteral):
                self.portfolio.apply_constraint(tickers, function, sign, value, unit)
                if function == 'sales' or function == 'short sales':
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                elif function == 'purchases' or function == 'holdings':
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                self.portfolio.remove_constraints(tickers)

        sign = EQUAL
        for function in typing.get_args(Portfolio.ConstraintFunctionLiteral):
            for unit in typing.get_args(Portfolio.ConstraintUnitLiteral):
                self.portfolio.apply_constraint(tickers, function, sign, value, unit)
                if function == 'sales' or function == 'short sales':
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                elif function == 'purchases' or function == 'holdings':
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                self.portfolio.remove_constraints(tickers)

        tickers = ['MSFT', 'AAPL']
        value = 1
        sign = LESS_THAN_OR_EQUAL
        for function in typing.get_args(Portfolio.ConstraintFunctionLiteral):
            for unit in typing.get_args(Portfolio.ConstraintUnitLiteral):
                self.portfolio.apply_constraint(tickers, function, sign, value, unit)
                if function == 'purchases':
                    if unit == 'shares':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'upper'],
                                                      self.portfolio.portfolio.loc[tickers, 'shares'] + value)
                    elif unit == 'value':
                        np.testing.assert_array_equal(
                            self.portfolio.portfolio.loc[tickers, 'upper'],
                            self.portfolio.portfolio.loc[tickers, 'shares'] + value / self.portfolio.portfolio.loc[tickers, 'close ($)'])
                    elif unit == 'percent value':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'upper'],
                                                      (1 + value/100) * self.portfolio.portfolio.loc[tickers, 'shares'])
                elif function == 'sales':
                    if unit == 'shares':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'lower'],
                                                      self.portfolio.portfolio.loc[tickers, 'shares'] - value)
                    elif unit == 'value':
                        np.testing.assert_array_equal(
                            self.portfolio.portfolio.loc[tickers, 'lower'],
                            self.portfolio.portfolio.loc[tickers, 'shares'] - value / self.portfolio.portfolio.loc[tickers, 'close ($)'])
                    elif unit == 'percent value':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'lower'],
                                                      (1 - value/100) * self.portfolio.portfolio.loc[tickers, 'shares'])
                elif function == 'holdings':
                    if unit == 'shares':
                        self.assertTrue(np.all(self.portfolio.portfolio.loc[tickers, 'upper'] == value))
                    elif unit == 'value':
                        np.testing.assert_array_equal(
                            self.portfolio.portfolio.loc[tickers, 'upper'],
                            value / self.portfolio.portfolio.loc[tickers, 'close ($)'])
                    elif unit == 'percent value':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'upper'],
                                                      (value/100) * self.portfolio.portfolio.loc[tickers, 'shares'])
                elif function == 'short sales':
                    if unit == 'shares':
                        self.assertTrue(np.all(self.portfolio.portfolio.loc[tickers, 'lower'] == -value))
                    elif unit == 'value':
                        np.testing.assert_array_equal(
                            self.portfolio.portfolio.loc[tickers, 'lower'],
                            -value / self.portfolio.portfolio.loc[tickers, 'close ($)'])
                    elif unit == 'percent value':
                        np.testing.assert_array_equal(self.portfolio.portfolio.loc[tickers, 'lower'],
                                                      -(value/100) * self.portfolio.portfolio.loc[tickers, 'shares'])
                if function == 'sales' or function == 'short sales':
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                elif function == 'purchases' or function == 'holdings':
                    self.assertFalse(np.isfinite(self.portfolio.portfolio.loc[tickers, 'lower']).all())
                    self.assertTrue(np.isfinite(self.portfolio.portfolio.loc[tickers, 'upper']).all())
                self.portfolio.remove_constraints(tickers)


class TestPortfolioGroup(TestWithPortfolio):

    async def test_apply_constraint(self):

        from pyoptimum.portfolio import LESS_THAN_OR_EQUAL, GREATER_THAN_OR_EQUAL, EQUAL, Portfolio

        bounds = 1
        sign = LESS_THAN_OR_EQUAL
        for group in ['g1', 'g2']:
            tickers = self.portfolio.groups[group]
            for function in ['sales', 'purchases', 'short sales']:
                for unit in typing.get_args(Portfolio.GroupConstraintUnitLiteral):
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    if c is not None:
                        c['bounds'] = np.inf
                    self.portfolio.apply_group_constraint(group, function, sign, bounds, unit)
                    if unit == 'percent value':
                        value = bounds * (self.portfolio.portfolio.loc[tickers, 'close ($)'] * self.portfolio.portfolio.loc[tickers, 'shares']).sum() / 100
                    else:   # if unit == 'value':
                        value = bounds
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    self.assertIsNotNone(c)
                    self.assertEqual(c['bounds'], value)

        bounds = 1
        sign = LESS_THAN_OR_EQUAL
        for group in ['g1', 'g2']:
            tickers = self.portfolio.groups[group]
            for function in ['holdings', 'return']:
                for unit in typing.get_args(Portfolio.GroupConstraintUnitLiteral):
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    if c is not None:
                        c['bounds'] = [-np.inf, np.inf]
                    self.portfolio.apply_group_constraint(group, function, sign, bounds, unit)
                    if unit == 'percent value':
                        value = bounds * (self.portfolio.portfolio.loc[tickers, 'close ($)'] * self.portfolio.portfolio.loc[tickers, 'shares']).sum() / 100
                    else:   # if unit == 'value':
                        value = bounds
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    self.assertIsNotNone(c)
                    self.assertEqual(c['bounds'][1], value)

        bounds = -1
        sign = GREATER_THAN_OR_EQUAL
        for group in ['g1', 'g2']:
            tickers = self.portfolio.groups[group]
            for function in ['holdings', 'return']:
                for unit in typing.get_args(Portfolio.GroupConstraintUnitLiteral):
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    if c is not None:
                        c['bounds'] = [-np.inf, np.inf]
                    self.portfolio.apply_group_constraint(group, function, sign, bounds, unit)
                    if unit == 'percent value':
                        value = bounds * (self.portfolio.portfolio.loc[tickers, 'close ($)'] * self.portfolio.portfolio.loc[tickers, 'shares']).sum() / 100
                    else:   # if unit == 'value':
                        value = bounds
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    self.assertIsNotNone(c)
                    self.assertEqual(c['bounds'][0], value)

        bounds = 1
        sign = EQUAL
        for group in ['g1', 'g2']:
            tickers = self.portfolio.groups[group]
            for function in ['holdings', 'return']:
                for unit in typing.get_args(Portfolio.GroupConstraintUnitLiteral):
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    if c is not None:
                        c['bounds'] = [-np.inf, np.inf]
                    self.portfolio.apply_group_constraint(group, function, sign, bounds, unit)
                    if unit == 'percent value':
                        value = bounds * (self.portfolio.portfolio.loc[tickers, 'close ($)'] * self.portfolio.portfolio.loc[tickers, 'shares']).sum() / 100
                    else:   # if unit == 'value':
                        value = bounds
                    _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
                    self.assertIsNotNone(c)
                    self.assertEqual(c['bounds'][0], value)
                    self.assertEqual(c['bounds'][1], value)

        # test existing bound, simple
        group = 'g2'
        function = 'purchases'
        unit = 'value'
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        c['bounds'] = np.inf
        bounds = 10
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'], 10)
        bounds = 20
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'], 10)
        bounds = 5
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'], 5)

        # test existing bound, double
        group = 'g1'
        function = 'holdings'
        unit = 'value'
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        c['bounds'] = [-np.inf, np.inf]
        bounds = 10
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -np.inf)
        self.assertEqual(c['bounds'][1], 10)
        bounds = 20
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -np.inf)
        self.assertEqual(c['bounds'][1], 10)
        bounds = 5
        self.portfolio.apply_group_constraint(group, function, LESS_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -np.inf)
        self.assertEqual(c['bounds'][1], 5)
        bounds = -10
        self.portfolio.apply_group_constraint(group, function, GREATER_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -10)
        self.assertEqual(c['bounds'][1], 5)
        bounds = -20
        self.portfolio.apply_group_constraint(group, function, GREATER_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -10)
        self.assertEqual(c['bounds'][1], 5)
        bounds = -5
        self.portfolio.apply_group_constraint(group, function, GREATER_THAN_OR_EQUAL, bounds, unit)
        _, c = self.portfolio._get_group_constraint(group, Portfolio.FunctionTable[function])
        self.assertIsNotNone(c)
        self.assertEqual(c['bounds'][0], -5)
        self.assertEqual(c['bounds'][1], 5)

        # test some errors
        group = 'g2'
        function = 'purchases'
        unit = 'value'
        with self.assertRaises(ValueError) as e:
            self.portfolio.apply_group_constraint(group, function, GREATER_THAN_OR_EQUAL, bounds, unit)
        self.assertIn('greater than or equal inequality not supported', str(e.exception))
        with self.assertRaises(ValueError) as e:
            self.portfolio.apply_group_constraint(group, function, EQUAL, bounds, unit)
        self.assertIn('equality not supported', str(e.exception))

        constraint = self.portfolio.remove_group_constraint(group, function)
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint['set'], group)
        self.assertEqual(constraint['function'], 'buys')

    def test_group_df(self):

        import pandas as pd

        df = self.portfolio.get_portfolio_dataframe()
        gdf = self.portfolio.get_group_dataframe()
        self.assertIsInstance(gdf, pd.DataFrame)
        for group, tickers in self.portfolio.groups.items():
            self.assertEqual(gdf.loc[group, 'value ($)'], df.loc[tickers]['value ($)'].sum())
            self.assertEqual(gdf.loc[group, 'value (%)'], df.loc[tickers]['value (%)'].sum())
            self.assertListEqual(gdf.loc[group, 'tickers'], tickers)

    async def test_group_holdings_constraint(self):

        from pyoptimum.portfolio import LESS_THAN_OR_EQUAL

        # retrieve frontier
        cf = 0
        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # print(f'total = {self.portfolio.get_value()}')
        # for i, row in self.portfolio.frontier.iterrows():
        #     x = row['x']
        #     gdf = self.portfolio.get_group_dataframe(x, cf)
        #     print(gdf.loc[['g1']])

        # apply holdings group constraint
        x = self.portfolio.frontier.loc[3]['x']
        # df = self.portfolio.get_recommendation_dataframe(x, cf)
        # df.loc['Total'] = df.sum(numeric_only=True)
        # print(df)
        gdf = self.portfolio.get_group_dataframe()
        # print(gdf)
        value = gdf.loc['g1', 'value ($)']
        bound = .5 * value
        # print(value, bound)
        self.portfolio.apply_group_constraint('g1', 'holdings', LESS_THAN_OR_EQUAL, bound, 'value')
        self.assertEqual(len(self.portfolio.group_constraints), 1)

        await self.portfolio.retrieve_frontier(cf, 1.2*bound, False, True, True)
        # print(self.portfolio.frontier_query_params)

        for i, row in self.portfolio.frontier.iterrows():
            x = row['x']
            gdf = self.portfolio.get_group_dataframe(x, cf)
            # print(gdf.loc[['g1']])
            self.assertLessEqual(gdf.loc['g1', 'value ($)'], 1.01*bound)

        # remove group constraint
        self.portfolio.remove_group_constraint('g1', 'holdings')
        self.assertEqual(len(self.portfolio.group_constraints), 0)

        # retrieve frontier, this time with non-zero cashflow
        cf = 1000
        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # print(f'total = {self.portfolio.get_value()}')
        # for i, row in self.portfolio.frontier.iterrows():
        #     x = row['x']
        #     gdf = self.portfolio.get_group_dataframe(x, cf)
        #     print(gdf.loc[['g1']])

        # apply holdings group constraint
        x = self.portfolio.frontier.loc[3]['x']
        # df = self.portfolio.get_recommendation_dataframe(x, cf)
        # df.loc['Total'] = df.sum(numeric_only=True)
        # print(df)
        gdf = self.portfolio.get_group_dataframe()
        # print(gdf)
        value = gdf.loc['g1', 'value ($)']
        bound = .5 * value
        # print(value, bound)
        self.portfolio.apply_group_constraint('g1', 'holdings', LESS_THAN_OR_EQUAL, bound, 'value')
        self.assertEqual(len(self.portfolio.group_constraints), 1)

        await self.portfolio.retrieve_frontier(cf, 1.1 * bound, False, True, True)
        # print(self.portfolio.frontier_query_params)

        for i, row in self.portfolio.frontier.iterrows():
            x = row['x']
            gdf = self.portfolio.get_group_dataframe(x, cf)
            # print(gdf.loc[['g1']])
            self.assertLessEqual(gdf.loc['g1', 'value ($)'], 1.01*bound)

        # remove group constraint
        self.portfolio.remove_group_constraint('g1', 'holdings')
        self.assertEqual(len(self.portfolio.group_constraints), 0)

    async def test_group_sales_constraint(self):

        from pyoptimum.portfolio import LESS_THAN_OR_EQUAL

        # retrieve frontier
        cf = 0
        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())
        # print(self.portfolio.frontier_query_params)

        # print(f'total = {self.portfolio.get_value()}')
        # for i, row in self.portfolio.frontier.iterrows():
        #     x = row['x']
        #     gdf = self.portfolio.get_group_dataframe(x, cf)
        #     print(gdf.loc[['g1']])

        # apply sales group constraint
        x = self.portfolio.frontier.loc[3]['x']
        # print(self.portfolio.get_recommendation_dataframe(x, cf))
        # print(self.portfolio.get_recommendation_dataframe(x, cf).sum())
        gdf = self.portfolio.get_group_dataframe(x, cf)
        # print(gdf)
        value = gdf.loc['g1', ['purchases ($)', 'sales ($)']].values
        function, value = ('sales', value[1]) if value[1] > value[0] else ('purchases', value[0])
        bound = .8 * value
        # print(value, bound, function)
        self.portfolio.apply_group_constraint('g1', function,
                                              LESS_THAN_OR_EQUAL,
                                              bound, 'value')
        self.assertEqual(len(self.portfolio.group_constraints), 1)

        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        # print(self.portfolio.frontier_query_params)

        for i, row in self.portfolio.frontier.iterrows():
            x = row['x']
            gdf = self.portfolio.get_group_dataframe(x, cf)
            # print(gdf.loc[['g1']])
            if function == 'sales':
                self.assertLessEqual(gdf.loc['g1', 'sales ($)'], 1.01*bound)
            else:
                self.assertLessEqual(-gdf.loc['g1', 'purchases ($)'], 1.01*bound)

        # remove group constraint
        self.portfolio.remove_group_constraint('g1', function)
        self.assertEqual(len(self.portfolio.group_constraints), 0)

        # retrieve frontier, this time with non-zero cashflow
        cf = 1000
        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())
        # print(self.portfolio.frontier_query_params)

        # print(f'total = {self.portfolio.get_value()}')
        # for i, row in self.portfolio.frontier.iterrows():
        #     x = row['x']
        #     gdf = self.portfolio.get_group_dataframe(x, cf)
        #     print(gdf.loc[['g1']])

        # apply sales group constraint
        x = self.portfolio.frontier.loc[3]['x']
        # print(self.portfolio.get_recommendation_dataframe(x, cf))
        # print(self.portfolio.get_recommendation_dataframe(x, cf).sum())
        gdf = self.portfolio.get_group_dataframe(x, cf)
        # print(gdf)
        value = gdf.loc['g1', ['purchases ($)', 'sales ($)']].values
        function, value = ('sales', value[1]) if value[1] > value[0] else ('purchases', value[0])
        bound = .8 * value
        # print(value, bound, function)
        self.portfolio.apply_group_constraint('g1', function,
                                              LESS_THAN_OR_EQUAL,
                                              bound, 'value')
        self.assertEqual(len(self.portfolio.group_constraints), 1)

        await self.portfolio.retrieve_frontier(cf, 100, False, True, True)
        # print(self.portfolio.frontier_query_params)

        for i, row in self.portfolio.frontier.iterrows():
            x = row['x']
            gdf = self.portfolio.get_group_dataframe(x, cf)
            # print(gdf.loc[['g1']])
            if function == 'sales':
                self.assertLessEqual(gdf.loc['g1', 'sales ($)'], bound)
            else:
                self.assertLessEqual(-gdf.loc['g1', 'purchases ($)'], bound)
