import unittest

import requests
import math

from src import pyoptimum

username = 'demo@optimize.vicbee.net'
password = 'optimize'

class TestBasic(unittest.TestCase):

    def test_client(self):

        client = pyoptimum.Client(username=username, password=password)

        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

        # wrong password
        client = pyoptimum.Client(username=username, password=password + 'wrong')

        self.assertIsNone(client.token)
        self.assertRaises(requests.exceptions.HTTPError, client.get_token)

        # wrong constructor
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.Client)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.Client,
                          username=username)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.Client,
                          password=password)
        self.assertRaises(pyoptimum.PyOptimumException, pyoptimum.Client,
                          token='')

    def test_portfolio(self):

        client = pyoptimum.Client(username=username, password=password)

        s1 = 0.06
        s2 = 0.03
        rho = 1
        data = {
            'Q': [[s1 ** 2, s1 * s2 * rho], [s1 * s2 * rho, s2 ** 2]],
            'cashflow': 1,
            'mu': 0.11,
            'r': [.14, .08],
        }
        response = client.call('portfolio', data)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        status = response.get('status')
        self.assertEqual(status, 'optimal')

        x = response.get('x')
        self.assertAlmostEqual(x[0], .5)
        self.assertAlmostEqual(x[1], .5)

        self.assertIsNone(client.detail)

        # call with errors
        data['r'] = [.14, .08, 0]
        self.assertRaises(pyoptimum.PyOptimumException, client.call, 'portfolio', data)
        self.assertIn('must be an array', client.detail)
    def test_forbidden(self):

        client = pyoptimum.Client(username=username, password=password)

        data = {
            'A': [[2]],
            'blo': [0],
            'bup': [4],
            'c': [-1],
            'xlo': [0],
            'xup': [2]
        }
        self.assertRaises(requests.exceptions.HTTPError, client.call, 'lp', data)
