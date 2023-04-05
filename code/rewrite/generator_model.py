from typing import List
import torch.nn as nn

# CINR = Convolution Instance Norm ReLU
# 3x3 convolutions with stride 1/2, 1 or 2 depending on position
# Uses reflection padding
# In the paper:
# dk denotes a k filter stride 2 with 3x3 conv
# c7s1-k denotes a k filter stride 1 with 7x7 conv
# uk denotes a k filter stride 1/2 with 3x3 conv
# Instead of using a nn.Module, this has much less overhead
def createGeneratorCINRLayer(in_ch, out_ch, stride, kernel_size, reflect_pad, up_type):
    layers = []
        
    padding = 1
    
    if reflect_pad:
        layers.append(nn.ReflectionPad2d(kernel_size // 2))
        padding = 0

    if stride < 1:
        if up_type == "conv_transpose":
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=int(1 / stride), padding=padding, output_padding=padding, bias=True)
            )
        elif up_type == "upsample":
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear"), # could try with nearest neighbour instead
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
            ]
        elif up_type == "pixel_shuffle":
            # https://gist.github.com/bearpelican/a87a6140661ffbc9b97409a12a1cf45b
            layers += [
                nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
            ]
        else:
            assert False, "Invalid up_type"
    else:
        layers.append(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        )
    
    layers += [
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(True)
    ]

    return nn.Sequential(*layers)

# Contains 2 3x3 convolutional layers with the same number of filters on both layers
# Use reflect padding in these
# Don't use dropout
# Use instancenorm
class GeneratorResidualBlock(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, bias=True),
            nn.InstanceNorm2d(feature_size),
            nn.ReLU(True),
            # Dropout would go here if I want it
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, bias=True),
            nn.InstanceNorm2d(feature_size)
        ]
        
        self.seq = nn.Sequential(*layers)
    
    def forward(self, batch):
        return batch + self.seq(batch)

# For the 128x128 case:
# c7s1-64, d128, d256, R256 x 6, u128, u64, c7s1-3
class Generator(nn.Module):
    def __init__(self, resnet_blocks, up_type):
        super().__init__()
        
        layers: List[nn.Module] = [
            createGeneratorCINRLayer(in_ch=3, out_ch=64, stride=1, kernel_size=7, reflect_pad=True, up_type=None),
            createGeneratorCINRLayer(in_ch=64, out_ch=128, stride=2, kernel_size=3, reflect_pad=False, up_type=None),
            createGeneratorCINRLayer(in_ch=128, out_ch=256, stride=2, kernel_size=3, reflect_pad=False, up_type=None)
        ]
        
        # same dim all the way through
        for _ in range(resnet_blocks):
            layers.append(GeneratorResidualBlock(feature_size=256))

        layers += [ 
            createGeneratorCINRLayer(in_ch=256, out_ch=128, kernel_size=3, stride=0.5, reflect_pad=False, up_type=up_type),
            createGeneratorCINRLayer(in_ch=128, out_ch=64, kernel_size=3, stride=0.5, reflect_pad=False, up_type=up_type),

            # Last one is a bit different without the ReLU and instance norm
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, stride=1, kernel_size=7, bias=True),
            nn.Tanh()
        ]
        
        self.seq = nn.Sequential(*layers)
        
    def forward(self, batch):
        return self.seq(batch)