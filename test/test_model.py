import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sit_smart_sensor.model import SitSmartModel

class TestModel(unittest.TestCase):
    def test_input_output(self):
        # get this file name using pathlib

        model = SitSmartModel()
        inp = torch.zeros(1, 3, 224, 224)
        pred = model(inp)
        expected_shape = torch.Size([1, 3])
        self.assertEqual(pred.shape, expected_shape)

    def test_predict_proba(self):
        model = SitSmartModel()
        inp = torch.zeros(1, 3, 224, 224)
        pred = model.predict_proba(inp)
        self.assertEqual(pred.shape, torch.Size([1, 3]))
        self.assertTrue(torch.allclose(pred.sum(), torch.ones(1)))

    def test_default_parameters(self):
        model = SitSmartModel()
        self.assertEqual(model.train_n_layers, 1)
        self.assertEqual(model.model_name, 'resnet34')
        self.assertEqual(model.lr, 1e-3)
        self.assertEqual(model.weight_decay, 1e-6)
        self.assertEqual(model.patience, 5)
        self.assertEqual(model.reduce_factor, 0.5)

    def test_train_n_layers(self):
        model = SitSmartModel(train_n_layers=2)
        inp = torch.zeros(1, 3, 224, 224)
        pred = model(inp)
        expected_shape = torch.Size([1, 3])
        self.assertEqual(pred.shape, expected_shape)
        self.assertEqual(len(list(model.classifier)), 5 ) # 2 blocks + global average pooling + flatten + linear


if __name__ == '__main__':
    unittest.main()
