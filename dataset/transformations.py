import torch
from PIL import Image


class AddRandomNoise:
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            noise = torch.randn(img.size()) * self.intensity
        elif isinstance(img, Image.Image):
            noise = torch.randn(img.size[::-1]) * self.intensity
            noise = noise.permute(2, 0, 1)  # if noise is 3D, permute to match image tensor shape
        else:
            raise TypeError("img must be a PIL image or a PyTorch tensor")
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0, 1)
