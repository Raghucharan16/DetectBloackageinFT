class GradCAMX3D(MedicalX3D):
    def __init__(self):
        super().__init__()
        self.activations = None
        self.gradients = None
        
        # Hook final conv layer
        target_layer = self.backbone.blocks[4].res_blocks[0].conv
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        # Forward pass
        output = self(input_tensor)
        self.zero_grad()
        
        # Backward for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Calculate weights
        gradients = self.gradients.mean(dim=(2,3,4), keepdims=True)
        activations = self.activations
        weights = torch.mean(gradients, dim=1, keepdim=True)
        
        # Generate heatmap
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.squeeze().cpu().numpy()
