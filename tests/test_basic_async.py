import unittest
import os

import aiohttp
import math


from src import pyoptimum

username = 'demo@optimize.vicbee.net'
password = 'optimize'
base_url = os.getenv('TEST_BASE_URL', 'https://optimize.vicbee.net')

class TestBasic(unittest.IsolatedAsyncioTestCase):

    def test_constructor(self):

        client = pyoptimum.AsyncClient(username=username, password=password)
        self.assertIsNone(client.token)

        client = pyoptimum.AsyncClient(token='token')
        self.assertIsNone(client.username)
        self.assertIsNone(client.password)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.AsyncClient, username=username)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.AsyncClient, username=password)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.AsyncClient, token='')

    async def test_optimize(self):

        client = pyoptimum.AsyncClient(username=username, password=password,
                                  base_url=base_url)

        self.assertIsNone(client.token)
        await client.get_token()
        self.assertIsNotNone(client.token)

        # wrong password
        client = pyoptimum.AsyncClient(username=username, password=password + 'wrong')

        self.assertIsNone(client.token)
        with self.assertRaises(aiohttp.client_exceptions.ClientResponseError):
            await client.get_token()

        # wrong constructor
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.AsyncClient)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.AsyncClient,
                          username=username)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.AsyncClient,
                          password=password)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.AsyncClient,
                          token='')

    async def test_models(self):

        client = pyoptimum.AsyncClient(username=username, password=password,
                                       base_url=base_url, api='models')

        self.assertIsNone(client.token)
        await client.get_token()
        self.assertIsNotNone(client.token)

    async def test_portfolio(self):

        client = pyoptimum.AsyncClient(username=username, password=password,
                                       base_url=base_url)

        s1 = 0.06
        s2 = 0.03
        rho = 1
        data = {
            'Q': [[s1 ** 2, s1 * s2 * rho], [s1 * s2 * rho, s2 ** 2]],
            'cashflow': 1,
            'mu': 0.11,
            'r': [.14, .08],
        }
        response = await client.call('portfolio', data)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        status = response.get('status')
        self.assertEqual(status, 'optimal')

        x = response.get('x')
        self.assertAlmostEqual(x[0], .5)
        self.assertAlmostEqual(x[1], .5)

        self.assertIsNone(client.detail)

        # call with slash
        response = await client.call('/portfolio', data)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        # call with errors
        data['r'] = [.14, .08, 0]
        with self.assertRaises(pyoptimum.PyOptimumException):
            await client.call('portfolio', data)
        self.assertIn('must be an array', client.detail)

    async def test_portfolio_async(self):

        client = pyoptimum.AsyncClient(username=username, password=password,
                                       base_url=base_url)

        s1 = 0.06
        s2 = 0.03
        rho = 1
        data = {
            'Q': [[s1 ** 2, s1 * s2 * rho], [s1 * s2 * rho, s2 ** 2]],
            'cashflow': 1,
            'mu': 0.11,
            'r': [.14, .08],
            'is_async': True
        }
        response = await client.call('portfolio', data,
                                     follow_resource=True, wait_time=1)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        status = response.get('status')
        self.assertEqual(status, 'optimal')

        x = response.get('x')
        self.assertAlmostEqual(x[0], .5)
        self.assertAlmostEqual(x[1], .5)

        self.assertIsNone(client.detail)

        # call with slash
        response = await client.call('/portfolio', data,
                                     follow_resource=True, wait_time=1)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        # call with errors
        data['r'] = [.14, .08, 0]
        with self.assertRaises(pyoptimum.PyOptimumException):
            await client.call('portfolio', data,
                              follow_resource=True, wait_time=1)
        self.assertIn('must be an array', client.detail)

        # call with errors
        data['r'] = [.14, .08]
        data['options'] = { 'mu_max': -1 }
        with self.assertRaises(pyoptimum.PyOptimumException):
            await client.call('frontier', data,
                              follow_resource=True, wait_time=1)
        self.assertIn('is larger than mu_max', client.detail)

    async def test_forbidden(self):

        client = pyoptimum.AsyncClient(username=username, password=password,
                                  base_url=base_url)

        data = {
            'A': [[2]],
            'blo': [0],
            'bup': [4],
            'c': [-1],
            'xlo': [0],
            'xup': [2]
        }
        with self.assertRaises(aiohttp.client_exceptions.ClientResponseError):
            await client.call('lp', data)
