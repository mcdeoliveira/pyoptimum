import typing
import unittest
import os

import numpy as np

username = 'demo@optimize.vicbee.net'
password = 'optimize'
base_url = os.getenv('TEST_BASE_URL', 'https://optimize.vicbee.net')


class TestBasic(unittest.TestCase):

    def setUp(self):

        import pyoptimum

        self.portfolio_client = pyoptimum.AsyncClient(username=username, password=password,api='optimize')
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,api='models')

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

        self.assertEqual(portfolio.get_value(), 0.0)


class TestModel(unittest.TestCase):

    def test_constructor_1(self):

        from pyoptimum.portfolio import Model

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'D': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'].T @ data['Q']
        data['D'] = data['D'].T @ data['D']

        # data does not have Di
        model = Model(data)
        self.assertIsNone(model._Di)

        # will calculate Di
        di = model.Di
        self.assertIsNotNone(model._Di)
        np.testing.assert_array_almost_equal(model.D @ model.Di, np.eye(3))

        # make sure it is cached
        di2 = model.Di
        self.assertIs(di, di2)

        # will set D
        D = np.random.normal(size=(3, 3))
        D = D.T @ D
        model.D = D
        self.assertIsNone(model._Di)

        # will calculate Di
        di = model.Di
        self.assertIsNotNone(model._Di)
        np.testing.assert_array_almost_equal(model.D @ model.Di, np.eye(3))

        # make sure it is cached
        di2 = model.Di
        self.assertIs(di, di2)

        # will set Di
        Di = np.random.normal(size=(3, 3))
        Di = Di.T @ Di
        model.Di = Di
        self.assertIsNone(model._D)

        # will calculate D
        d = model.D
        self.assertIsNotNone(model._D)
        np.testing.assert_array_almost_equal(model.D @ model.Di, np.eye(3))

        # make sure it is cached
        d2 = model.D
        self.assertIs(d, d2)

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'Di': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'].T @ data['Q']
        data['Di'] = data['Di'].T @ data['Di']

        # data does not have D
        model = Model(data)
        self.assertIsNone(model._D)

        # will calculate D
        d = model.D
        self.assertIsNotNone(model._D)
        np.testing.assert_array_almost_equal(model.D @ model.Di, np.eye(3))

        # make sure it is cached
        d2 = model.D
        self.assertIs(d, d2)

        # test std
        self.assertIsNone(model._std)
        s = model.std
        self.assertIsNotNone(model._std)

        # make sure it is cached
        s2 = model.std
        self.assertIs(s, s2)

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'Di': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'].T @ data['Q']
        data['Di'] = data['Di'].T @ data['Di']

        # data does not have D
        model = Model(data)
        self.assertIsNone(model._D)

        # test std
        self.assertIsNone(model._std)
        s = model.std
        self.assertIsNotNone(model._std)

        # make sure it is cached
        s2 = model.std
        self.assertIs(s, s2)

        # a D has been calculated
        self.assertIsNotNone(model._D)

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'D': np.random.normal(size=(3,3)),
            'Di': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        with self.assertRaises(AssertionError):
            Model(data)

    def test_constructor_2(self):

        from pyoptimum.portfolio import Model

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'D': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'].T @ data['Q']
        data['D'] = data['D'].T @ data['D']

        # create model
        model_1 = Model(data)

        # copy constructor
        model_2 = Model(model_1)
        np.testing.assert_array_equal(model_1.r, model_2.r)
        np.testing.assert_array_equal(model_1.Q, model_2.Q)
        np.testing.assert_array_equal(model_1.F, model_2.F)
        np.testing.assert_array_equal(model_1.D, model_2.D)
        np.testing.assert_array_equal(model_1.Di, model_2.Di)
        np.testing.assert_array_equal(model_1.std, model_2.std)

        self.assertIsNot(model_1.r, model_2.r)
        self.assertIsNot(model_1.Q, model_2.Q)
        self.assertIsNot(model_1.F, model_2.F)
        self.assertIsNot(model_1.D, model_2.D)
        self.assertIsNot(model_1.Di, model_2.Di)
        self.assertIsNot(model_1.std, model_2.std)

class TestPortfolio(unittest.IsolatedAsyncioTestCase):

    def setUp(self):

        import pyoptimum
        from pyoptimum.portfolio import Portfolio

        self.portfolio_client = pyoptimum.AsyncClient(username=username, password=password,api='optimize')
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,api='models')
        self.portfolio = Portfolio(self.portfolio_client, self.model_client)
        from pathlib import Path
        file = Path(__file__).parent / 'test.csv'
        self.portfolio.import_csv(file)

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
            self.portfolio.set_models_weights({})

        # retrieve models
        market_tickers = ['^DJI']
        ranges = ['1mo', '6mo', '1y']
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_models(market_tickers, ranges)
        self.assertFalse(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        with self.assertRaises(AssertionError):
            await self.portfolio.retrieve_frontier(0, 0, False, True, True)

        from pyoptimum.portfolio import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

    async def test_models_with_prices(self):

        # try getting model before retrieving
        with self.assertRaises(AssertionError):
            self.portfolio.get_model()

        with self.assertRaises(AssertionError):
            self.portfolio.set_models_weights({})

        # retrieve models
        market_tickers = ['^DJI']
        ranges = ['1mo', '6mo', '1y']
        self.assertFalse(self.portfolio.has_prices())
        self.assertFalse(self.portfolio.has_models())
        await self.portfolio.retrieve_models(market_tickers, ranges, include_prices=True)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        await self.portfolio.retrieve_frontier(0, 0, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        from pyoptimum.portfolio import Model

        model = self.portfolio.get_model()
        self.assertIsInstance(model, Model)

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
        await self.portfolio.retrieve_models(market_tickers, ranges)
        self.assertTrue(self.portfolio.has_prices())
        self.assertTrue(self.portfolio.has_models())

        # retrieve frontier
        await self.portfolio.retrieve_frontier(0, 100, False, True, True)
        self.assertTrue(self.portfolio.has_frontier())

        # retrieve unfeasible frontier
        with self.assertRaises(ValueError):
            await self.portfolio.retrieve_frontier(-100, 0, False, True, True)

        # make sure it gets invalidated
        self.assertFalse(self.portfolio.has_frontier())

        # set model weights
        self.assertDictEqual(self.portfolio.model_weights, {rg: 1/3 for rg in ranges})
        self.portfolio.set_models_weights({rg: v for rg, v in zip(ranges, [1,2,3])})
        self.assertDictEqual(self.portfolio.model_weights, {rg: v/6 for rg, v in zip(ranges, [1,2,3])})
        with self.assertRaises(AssertionError):
            self.portfolio.set_models_weights({})
        with self.assertRaises(AssertionError):
            self.portfolio.set_models_weights({rg: v for rg, v in zip(ranges, [1,-2,3])})


class TestPortfolioFunctions(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):

        import pyoptimum
        from pyoptimum.portfolio import Portfolio

        self.portfolio_client = pyoptimum.AsyncClient(username=username, password=password,api='optimize')
        self.model_client = pyoptimum.AsyncClient(username=username, password=password,api='models')
        self.portfolio = Portfolio(self.portfolio_client, self.model_client)
        from pathlib import Path
        file = Path(__file__).parent / 'test.csv'
        self.portfolio.import_csv(file)

        # retrieve models and price
        self.market_tickers = ['^DJI', '^RUT']
        self.ranges = ['1mo', '6mo', '1y']
        await self.portfolio.retrieve_models(self.market_tickers, self.ranges,
                                             include_prices=True)

    async def test_model_methods(self):

        from pyoptimum.portfolio import Model

        weights = {
            '1mo': 3,
            '6mo': 1,
            '1y': 2
        }
        self.portfolio.set_models_weights(weights)
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

    async def test_apply_constraint(self):

        from pyoptimum.portfolio import LESS_THAN_OR_EQUAL, Portfolio

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
