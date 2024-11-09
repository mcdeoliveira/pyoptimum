import unittest
import numpy as np


class TestModel(unittest.TestCase):

    def test_constructor_1(self):

        from pyoptimum.model import Model

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'D': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'] ** 2
        data['D'] = data['D'].T @ data['D']

        # data does not have Di
        model = Model(data)
        self.assertTrue(model.has_factors)
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
        data['Q'] = data['Q'] ** 2
        data['Di'] = data['Di'].T @ data['Di']

        # data does not have D
        model = Model(data)
        self.assertTrue(model.has_factors)
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
        data['Q'] = data['Q'] ** 2
        data['Di'] = data['Di'].T @ data['Di']

        # data does not have D
        model = Model(data)
        self.assertTrue(model.has_factors)
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

        from pyoptimum.model import Model

        data = {
            'Q': np.random.normal(size=(5,)),
            'F': np.random.normal(size=(5,3)),
            'D': np.random.normal(size=(3,3)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'] ** 2
        data['D'] = data['D'].T @ data['D']

        # create model
        model_1 = Model(data)
        self.assertTrue(model_1.has_factors)

        # copy constructor
        model_2 = Model(model_1)
        self.assertTrue(model_2.has_factors)
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

    def test_constructor_3(self):

        from pyoptimum.model import Model

        data = {
            'Q': np.random.normal(size=(5,)),
            'r': np.random.normal(size=(5,))
        }
        data['Q'] = data['Q'] ** 2

        # create model
        model_1 = Model(data)
        self.assertFalse(model_1.has_factors)
        self.assertIsNone(model_1.F)
        self.assertIsNone(model_1.D)
        self.assertIsNone(model_1.Di)

        # copy constructor
        model_2 = Model(model_1)
        self.assertFalse(model_2.has_factors)
        np.testing.assert_array_equal(model_1.r, model_2.r)
        np.testing.assert_array_equal(model_1.Q, model_2.Q)
        np.testing.assert_array_equal(model_1.F, model_2.F)
        np.testing.assert_array_equal(model_1.D, model_2.D)
        np.testing.assert_array_equal(model_1.Di, model_2.Di)
        np.testing.assert_array_equal(model_1.std, model_2.std)

        self.assertIsNot(model_1.r, model_2.r)
        self.assertIsNot(model_1.Q, model_2.Q)
        self.assertIsNot(model_1.std, model_2.std)

        with self.assertRaises(KeyError):
            data['F'] = np.random.normal(size=(5,3))
            Model(data)

        with self.assertRaises(AssertionError):
            del data['F']
            data['D'] = np.random.normal(size=(5,5))
            Model(data)

        with self.assertRaises(AssertionError):
            del data['D']
            data['Di'] = np.random.normal(size=(5,5))
            Model(data)

    def test_return_1(self):

        from pyoptimum.model import Model

        r = np.array([2,1])
        q = np.array([1,2])
        d = np.array([[1,-.1],[-.1,1]])
        f = np.array([[1,-1],[1/2,1]])
        data = {
            'Q': q,
            'F': f,
            'D': d,
            'r': r
        }
        x = np.array([1,3])
        mu = r @ x / np.sum(x)
        var0 = np.dot((q * x), x)
        var1 = np.dot(np.dot(f @ d @ f.T, x) , x)
        std = np.sqrt(var0 + var1) / np.sum(x)

        # create model
        model = Model(data)
        np.testing.assert_array_equal(r, model.r)
        np.testing.assert_array_equal(q, model.Q)
        np.testing.assert_array_equal(f, model.F)
        np.testing.assert_array_equal(d, model.D)

        mu_, std_ = model.return_and_variance(x)
        self.assertEqual(mu, mu_)
        self.assertEqual(std, std_)

        dic = model.to_dict()
        np.testing.assert_array_equal(r, dic['r'])
        np.testing.assert_array_equal(q, dic['Q'])
        np.testing.assert_array_equal(f, dic['F'])
        np.testing.assert_array_equal(d, dic['D'])

        dic = model.to_dict(as_list=True)
        self.assertListEqual(r.tolist(), dic['r'])
        self.assertListEqual(q.tolist(), dic['Q'])
        self.assertListEqual(f.tolist(), dic['F'])
        self.assertListEqual(d.tolist(), dic['D'])

        dic = model.to_dict(normalize_variance=True)
        v = np.max(model.std) ** 2
        np.testing.assert_array_equal(r, dic['r'])
        np.testing.assert_array_equal(q/v, dic['Q'])
        np.testing.assert_array_equal(f, dic['F'])
        np.testing.assert_array_equal(d/v, dic['D'])

        # check that model did not change
        np.testing.assert_array_equal(r, model.r)
        np.testing.assert_array_equal(q, model.Q)
        np.testing.assert_array_equal(f, model.F)
        np.testing.assert_array_equal(d, model.D)

    def test_return_2(self):

        from pyoptimum.model import Model

        r = np.array([2,1])
        q = np.array([1,2])
        data = {
            'Q': q,
            'r': r
        }
        x = np.array([1,3])
        mu = r @ x / np.sum(x)
        var0 = np.dot((q * x), x)
        std = np.sqrt(var0) / np.sum(x)

        # create model
        model = Model(data)
        np.testing.assert_array_equal(r, model.r)
        np.testing.assert_array_equal(q, model.Q)

        mu_, std_ = model.return_and_variance(x)
        self.assertEqual(mu, mu_)
        self.assertEqual(std, std_)

        dic = model.to_dict()
        np.testing.assert_array_equal(r, dic['r'])
        np.testing.assert_array_equal(q, dic['Q'])

        dic = model.to_dict(as_list=True)
        self.assertListEqual(r.tolist(), dic['r'])
        self.assertListEqual(q.tolist(), dic['Q'])

        dic = model.to_dict(normalize_variance=True)
        v = np.max(model.std) ** 2
        np.testing.assert_array_equal(r, dic['r'])
        np.testing.assert_array_equal(q/v, dic['Q'])

        # check that model did not change
        np.testing.assert_array_equal(r, model.r)
        np.testing.assert_array_equal(q, model.Q)

    def test_unconstrained_frontier_and_return_1(self):

        from pyoptimum.model import Model

        rng = np.random.default_rng(12345)
        data = {
            'Q': rng.normal(size=(5,)),
            'F': rng.normal(size=(5,3)),
            'D': rng.normal(size=(3,3)),
            'r': rng.normal(size=(5,))
        }
        data['Q'] = data['Q'] ** 2
        data['D'] = data['D'].T @ data['D']
        x = rng.random(size=(5,))

        # create model
        model = Model(data)

        std = [3.147443, 5.117713, 2.710299, 2.30415 , 4.837606]
        np.testing.assert_array_almost_equal(std, model.std)

        # variance and return
        mu, std = model.return_and_variance(x)
        np.testing.assert_array_almost_equal([mu, std], [0.5121689723772288, 2.9559252230222635])

        # unconstrained frontier
        vals = model.unconstrained_frontier()
        np.testing.assert_array_almost_equal(vals, [0.14766907361100318, -0.8407610288104532, 0.5613114301321959])

        # zero return model
        data = {
            'r': np.hstack(([0], model.r, [0])),
            'Q': np.hstack(([0], model.Q, [0])),
            'F': np.vstack((np.zeros((1, 3)), model.F, np.zeros((1, 3)))),
            'D': model.D,
        }
        x = np.hstack(([0], x, [0]))

        # create model
        model = Model(data)

        # variance and return
        mu, std = model.return_and_variance(x)
        np.testing.assert_array_almost_equal([mu, std], [0.5121689723772288, 2.9559252230222635])

        # unconstrained frontier
        vals = model.unconstrained_frontier()
        np.testing.assert_array_almost_equal(vals, [0.14766907361100318, -0.8407610288104532, 0.5613114301321959])

        # holding a zero-return stock
        x[0] = 0.3
        mu, std = model.return_and_variance(x)
        np.testing.assert_array_almost_equal([mu, std], [0.4744052318205996, 2.7379760709896956])

        x[1] = -10.3
        with self.assertWarns(Warning) as w:
            mu, std = model.return_and_variance(x)
        self.assertEqual(str(w.warnings[0].message), "Total portfolio is negative")
        np.testing.assert_array_almost_equal([mu, std], [-3.8480800555552026, 3.726822456664636])

        with self.assertWarns(Warning) as w:
            mu, std = model.return_and_variance(0 * x)
        self.assertEqual(str(w.warnings[0].message), "Total portfolio is zero")
        np.testing.assert_array_almost_equal([mu, std], [0, 0])

    def test_unconstrained_frontier_and_return_2(self):

        from pyoptimum.model import Model

        rng = np.random.default_rng(12345)
        data = {
            'Q': rng.normal(size=(5,)),
            'F': rng.normal(size=(5,3)),
            'D': rng.normal(size=(3,3)),
            'r': rng.normal(size=(5,))
        }
        data['Q'] = data['Q'] ** 2
        data['D'] = data['D'].T @ data['D']
        x = rng.random(size=(5,))

        F = data.pop('F')
        D = data.pop('D')

        data['Q'] += np.diag(F @ D @ F.T)

        # create model
        model = Model(data)

        std = [3.147443, 5.117713, 2.710299, 2.30415 , 4.837606]
        np.testing.assert_array_almost_equal(std, model.std)

        # variance and return
        mu, std = model.return_and_variance(x)
        np.testing.assert_array_almost_equal([mu, std], [0.512169, 1.730823])

        # unconstrained frontier
        vals = model.unconstrained_frontier()
        np.testing.assert_array_almost_equal(vals, [1.01615 , 0.34456 , 1.405324])
