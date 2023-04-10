from typing import List
import torch.nn as nn

# CINR = Convolution Instance Normalisation Layer
# Denoted Ck where k = #filters
# Follows the notation of the CycleGAN paper
def createDiscriminatorCINRLayer(in_ch, out_ch, stride, apply_norm):
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
    ]

    if apply_norm:
        layers.append(
            nn.InstanceNorm2d(out_ch)
        )
    
    layers.append(
        nn.LeakyReLU(0.2, True)
    )

    return nn.Sequential(*layers)

"""
70x70 Patch Discriminator, follows the architecture set out in CycleGAN paper

References:
Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.
Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            createDiscriminatorCINRLayer(in_ch=3, out_ch=64, stride=2, apply_norm=False),
            createDiscriminatorCINRLayer(in_ch=64, out_ch=128, stride=2, apply_norm=True),
            createDiscriminatorCINRLayer(in_ch=128, out_ch=256, stride=2, apply_norm=True),
            createDiscriminatorCINRLayer(in_ch=256, out_ch=512, stride=1, apply_norm=True),
            # In the source of CycleGAN they use k=4, s=1, p=1 despite not saying this in the paper
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, batch):
        return self.seq(batch)