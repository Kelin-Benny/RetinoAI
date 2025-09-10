import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles = [handle_forward, handle_backward]
    
    def generate(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros((1, output.size()[-1]), device=self.model.device)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Process gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam
    
    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

def overlay_heatmap(cam, original_image, alpha=0.5):
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    
    # Resize cam to match original image
    original_image = np.array(original_image)
    if original_image.shape[:2] != cam.shape[:2]:
        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    overlayed = cv2.addWeighted(original_image, alpha, cam, 1 - alpha, 0)
    return Image.fromarray(overlayed)

def get_grad_cam(model, image, target_layer, target_class=None):
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(image, target_class)
    heatmap = overlay_heatmap(cam, image)
    return heatmap