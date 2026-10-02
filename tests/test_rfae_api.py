import unittest

import graphtools
import numpy as np
import inspect
from rfphate import RFPHATE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from rfae import RFAE


class RFAEAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_classification(
            n_samples=72, n_features=6, n_informative=4, random_state=7
        )

    def test_fit_signatures_match_rfphate(self):
        for name in ('fit', 'fit_transform'):
            self.assertEqual(
                inspect.signature(getattr(RFAE, name)),
                inspect.signature(getattr(RFPHATE, name)),
            )

    def test_partial_overrides_preserve_defaults(self):
        params = {'phate_params': {'n_landmark': 12, 't': 9}}
        model = RFAE(random_state=7, embedder_params=params)
        self.assertEqual(model.embedder.phate_params['n_landmark'], 12)
        self.assertEqual(model.embedder.phate_params['n_components'], 2)
        self.assertEqual(model.embedder.phate_params['t'], 9)
        self.assertEqual(model.embedder.proximity_params['weight_scheme'], 'gap')
        self.assertEqual(model.embedder.forest.random_state, 7)
        self.assertEqual(model.embedder.forest.n_jobs, -1)
        self.assertEqual(params, {'phate_params': {'n_landmark': 12, 't': 9}})

    def test_full_and_landmark_graphs(self):
        for n_landmark in (None, 12):
            with self.subTest(n_landmark=n_landmark):
                forest = RandomForestClassifier(
                    n_estimators=30, random_state=7, n_jobs=1
                )
                model = RFAE(
                    forest=forest, lam=0.2, epochs=1, batch_size=32,
                    random_state=7, device='cpu', hidden_dims=[8, 4],
                    embedder_params={
                        'n_jobs': 1,
                        'phate_params': {
                            'n_landmark': n_landmark, 'n_svd': 5, 'verbose': 0, 't': 2,
                        },
                    },
                )
                embedding = model.fit_transform(self.x[:60], self.y[:60])
                self.assertIs(model.embedder.forest, forest)
                self.assertEqual(embedding.shape, (60, 2))
                self.assertTrue(np.isfinite(embedding).all())
                self.assertTrue(np.isfinite(model.epoch_losses_recon).all())
                self.assertTrue(np.isfinite(model.epoch_losses_emb).all())
                self.assertEqual(model.lam, 0.2)
                np.testing.assert_allclose(
                    model.epoch_losses_balanced,
                    0.2 * np.array(model.epoch_losses_recon)
                    + 0.8 * np.array(model.epoch_losses_emb), rtol=1e-6,
                )
                self.assertEqual(
                    isinstance(model.embedder.phate_op_.graph,
                               graphtools.graphs.LandmarkGraph),
                    n_landmark is not None,
                )
                projected = model.transform(self.x[60:])
                self.assertEqual(projected.shape, (12, 2))
                self.assertTrue(np.isfinite(projected).all())
                reconstructed = model.inverse_transform(projected)
                self.assertEqual(reconstructed.shape, (12, model.input_shape))
                np.testing.assert_allclose(reconstructed.sum(axis=1), 1, atol=1e-6)

    def test_regression_forest_and_symmetric_kernel(self):
        forest = RandomForestRegressor(n_estimators=30, random_state=7, n_jobs=1)
        model = RFAE(
            forest=forest, lam=0.2, epochs=1, device='cpu', hidden_dims=[8],
            embedder_params={
                'forest': RandomForestClassifier(),
                'n_jobs': 1,
                'phate_params': {'n_landmark': None, 'verbose': 0, 't': 2},
            },
        )
        self.assertIs(model.embedder.forest, forest)
        model.fit(self.x, self.x[:, 0], force_symmetric=True)
        kernel = model.embedder.proximity_model_.training_proximity(
            force_symmetric=True, adjust_diagonal=True
        ).toarray()
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
        self.assertEqual(model.transform(self.x[:3]).shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
