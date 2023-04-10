import torch
import torch.nn as nn

"""
U-Net works by wrapping the interior blocks so that connections can be made
across the network structure
"""
class WrappableWithOptionalSkipConnection(nn.Module):
    def __init__(self, left, middle, right, concat):
        super().__init__()

        self.concat = concat

        layers = left

        if middle is not None:
            layers += [middle]
        
        layers += right

        self.model = nn.Sequential(*layers)
    
    def forward(self, batch):
        processed = self.model(batch)

        if self.concat:
            processed = torch.cat([batch, processed], 1)
        
        return processed
    
def createEdgeUnetBlock(initial_ch, in_ch, out_ch, module_to_wrap):
    left = [
        nn.Conv2d(initial_ch, in_ch, kernel_size=4, stride=2, padding=1)
    ]

    right = [
        nn.ReLU(),
        nn.ConvTranspose2d(in_ch * 2, out_ch, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    ]

    return WrappableWithOptionalSkipConnection(left, module_to_wrap, right, concat=False)

def createIntermediateUnetBlock(in_ch, out_ch, module_to_wrap):
    left = [
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_ch, in_ch, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(in_ch)
    ]

    right = [
        nn.ReLU(),
        nn.ConvTranspose2d(in_ch * 2, out_ch, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch)
    ]

    return WrappableWithOptionalSkipConnection(left, module_to_wrap, right, concat=True)

def createCentralUnetBlock(in_ch, out_ch):
    left = [
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_ch, in_ch, kernel_size=4, stride=2, padding=1)
    ]

    right = [
        nn.ReLU(),
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch)
    ]

    return WrappableWithOptionalSkipConnection(left, None, right, concat=True)

class Predictor(nn.Module):
    def __init__(self, in_ch=6, feature_size=64, out_ch=3):
        super().__init__()

        to_wrap = createCentralUnetBlock(feature_size * 8, feature_size * 8)
        
        to_wrap = createIntermediateUnetBlock(feature_size * 8, feature_size * 8, to_wrap)
        to_wrap = createIntermediateUnetBlock(feature_size * 8, feature_size * 8, to_wrap)
        # to_wrap = createIntermediateUnetBlock(feature_size * 8, feature_size * 8, to_wrap)

        to_wrap = createIntermediateUnetBlock(feature_size * 8, feature_size * 4, to_wrap)
        to_wrap = createIntermediateUnetBlock(feature_size * 4, feature_size * 2, to_wrap)
        to_wrap = createIntermediateUnetBlock(feature_size * 2, feature_size * 1, to_wrap)

        self.model = createEdgeUnetBlock(in_ch, feature_size, out_ch, to_wrap)
    
    def forward(self, batch):
        return self.model(batch)