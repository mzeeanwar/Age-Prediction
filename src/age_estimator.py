"""CNN-based age regression model (ResNet18 backbone)."""

import torch
import torch.nn as nn
from torchvision import models, transforms

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AgeEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model = backbone

    def forward(self, x):
        return self.model(x).squeeze(-1)

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        model = cls(pretrained=False)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, face_crop_bgr, device="cpu"):
        """face_crop_bgr: HxWx3 numpy array (OpenCV BGR crop)."""
        tensor = PREPROCESS(face_crop_bgr).unsqueeze(0).to(device)
        age = self.forward(tensor).item()
        return max(0, round(age))
