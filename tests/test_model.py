import unittest
import torch
from src.quick_draw.model import QuickDrawModel

class TestQuickDrawModel(unittest.TestCase):
    def setUp(self):
        self.model = QuickDrawModel(num_classes=10)
        self.input_tensor = torch.randn(1, 1, 224, 224)  # Batch size 1, 1 channel (grayscale), 224x224 image size

    def test_model_output_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (1, 10))  # Batch size 1, num_classes 10

    def test_classifier_trainable(self):
        for param in self.model.base_model.classifier.parameters():
            self.assertTrue(param.requires_grad)

    def test_base_model_frozen(self):
        for name, param in self.model.base_model.named_parameters():
            if 'classifier' not in name:
                self.assertFalse(param.requires_grad)

if __name__ == '__main__':
    unittest.main()