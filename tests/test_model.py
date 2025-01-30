import unittest
import torch
from models.medical_x3d import MedicalX3D

class TestHardware(unittest.TestCase):
    def test_cpu_inference(self):
        model = MedicalX3D(device='cpu')
        dummy = torch.randn(1, 1, 16, 160, 160)
        output = model(dummy)
        self.assertEqual(output.shape, (1, 2))
        
    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available")
    def test_gpu_inference(self):
        model = MedicalX3D(device='cuda')
        dummy = torch.randn(1, 1, 32, 224, 224).cuda()
        output = model(dummy)
        self.assertEqual(output.shape, (1, 2))
