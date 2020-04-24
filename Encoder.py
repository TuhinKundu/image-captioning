import torch
import torch.nn as nn
import torchvision.models as models

class Encoder_Resnet101(nn.Module):
    def __init__(self):
        super(Encoder_Resnet101, self).__init__()
        model = models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.model(images))
        out = out.permute(0, 2, 3, 1)
        return out
