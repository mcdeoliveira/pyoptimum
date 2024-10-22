import unittest
import os

import requests
import math

from src import pyoptimum

username = 'demo@optimize.vicbee.net'
password = 'optimize'
base_url = os.getenv('TEST_BASE_URL', 'https://optimize.vicbee.net')

class TestBasic(unittest.TestCase):

    def test_constructor(self):

        client = pyoptimum.Client(username=username, password=password)
        self.assertIsNone(client.token)

        client = pyoptimum.Client(token='token')
        self.assertIsNone(client.username)
        self.assertIsNone(client.password)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.Client, username=username)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.Client, username=password)

        self.assertRaises(pyoptimum.PyOptimumException,
                          pyoptimum.Client, token='')

    def test_urls(self):

        answer = 'a/b/c'
        result = pyoptimum.Client.url_join('a', 'b', 'c')
        self.assertEqual(result, answer)

        answer = 'a/b/c'
        result = pyoptimum.Client.url_join('a', '', 'b', '', 'c', '')
        self.assertEqual(result, answer)

        answer = 'a/b/c'
        result = pyoptimum.Client.url_join('a/', '', '/b', '', 'c/', '')
        self.assertEqual(result, answer)

        answer = 'a/b/c'
        result = pyoptimum.Client.url_join('//a//', '', '//b', '', 'c//', '')
        self.assertEqual(result, answer)

        answer = 'a:5000/b/c'
        result = pyoptimum.Client.url_join('//a:5000', '', '//b', '', 'c//', '')
        self.assertEqual(result, answer)

        # remove trailing slashes
        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url + '////')
        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

        # remove trailing slashes
        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url + '////', api='optimize', prefix='api')
        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

        # remove trailing slashes but not in the middle
        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url + '////', api='optimize/api', prefix='')
        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

        # wrong url that is not json
        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url, api='optimizeXY', prefix='api')
        self.assertIsNone(client.token)
        with self.assertRaises(requests.exceptions.HTTPError):
            client.get_token()

    def test_optimize(self):

        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url)

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

    def test_models(self):

        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url, api='models')

        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

    def test_auth(self):

        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url, api='models',
                                  auth_url='https://optimize.vicbee.net/auth/api')

        self.assertIsNone(client.token)
        client.get_token()
        self.assertIsNotNone(client.token)

    def test_portfolio(self):

        client = pyoptimum.Client(username=username, password=password,
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
        response = client.call('portfolio', data)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        status = response.get('status')
        self.assertEqual(status, 'optimal')

        x = response.get('x')
        self.assertAlmostEqual(x[0], .5)
        self.assertAlmostEqual(x[1], .5)

        self.assertIsNone(client.detail)

        # call with slash
        response = client.call('/portfolio', data)

        obj = response.get('obj')
        self.assertTrue(math.fabs(math.sqrt(obj) - .045) < 1e-5)

        # call with errors
        data['r'] = [.14, .08, 0]
        self.assertRaises(pyoptimum.PyOptimumException, client.call, 'portfolio', data)
        self.assertIn('must be an array', client.detail)

    def test_forbidden(self):

        client = pyoptimum.Client(username=username, password=password,
                                  base_url=base_url)

        data = {
            'A': [[2]],
            'blo': [0],
            'bup': [4],
            'c': [-1],
            'xlo': [0],
            'xup': [2]
        }
        self.assertRaises(requests.exceptions.HTTPError, client.call, 'lp', data)
