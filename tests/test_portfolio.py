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
