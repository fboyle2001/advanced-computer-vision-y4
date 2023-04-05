from typing import List

import random
import time
import os

import torch
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
        
        layers = [
            createGeneratorCINRLayer(in_ch=3, out_ch=64, stride=1, kernel_size=7, reflect_pad=True, up_type=None),
            createGeneratorCINRLayer(in_ch=64, out_ch=128, stride=2, kernel_size=3, reflect_pad=False, up_type=None),
            createGeneratorCINRLayer(in_ch=128, out_ch=256, stride=2, kernel_size=3, reflect_pad=False, up_type=None)
        ]
        
        # same dim all the way through
        for _ in range(resnet_blocks):
            layers.append(GeneratorResidualBlock(feature_size=256)) # type: ignore

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

def PredictorUnetBlock(nn.Module):
    pass

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

# CINR = Convolution Instance Normalisation Layer
# Denoted Ck where k = #filters
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

class HistoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)

    def _make_space(self, max_del_size):
        current_available_space = self.max_size - len(self)
        del_size = max(0, max_del_size - current_available_space)

        if del_size == 0:
            return

        del_indexes = random.sample(range(0, len(self)), del_size)

        for del_idx in del_indexes:
            del self.buffer[del_idx]

    def add(self, batch):
        self._make_space(len(batch))

        for item in batch:
            self.buffer.append(item.detach())

    def sample_batch(self, batch_size):
        return torch.stack(random.sample(self.buffer, batch_size))

    def randomise_existing_batch(self, existing_batch):
        if len(self) < existing_batch.shape[0] / 2:
            return existing_batch
        
        new_batch = []

        for item in existing_batch:
            if random.uniform(0, 1) < 0.5:
                new_batch.append(item.detach())
            else:
                new_batch.append(self.buffer[random.randint(0, len(self) - 1)])

        return torch.stack(new_batch)

class CycleGAN:
    def __init__(self, device, blocks, up_type, init=True, start_epoch=0, save_folder=None):
        self.device = device
        self.block_count = blocks

        self.G = Generator(blocks, up_type=up_type).to(device)
        self.F = Generator(blocks, up_type=up_type).to(device)
        
        self.D_X = PatchDiscriminator().to(device)
        self.D_Y = PatchDiscriminator().to(device)

        if init:
            self._initialise_weights()

        self.fake_X_buffer = HistoryBuffer(50)
        self.fake_Y_buffer = HistoryBuffer(50)

        self.gan_X_loss = nn.MSELoss().to(device)
        self.gan_Y_loss = nn.MSELoss().to(device)

        self.cycle_X_loss = nn.L1Loss().to(device)
        self.cycle_Y_loss = nn.L1Loss().to(device)

        self.identity_loss = nn.L1Loss().to(device)

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.F_opt = torch.optim.Adam(self.F.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.D_X_opt = torch.optim.Adam(self.D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_Y_opt = torch.optim.Adam(self.D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 99) / float(101)
            return lr_l

        self.G_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_opt, lr_lambda=lambda_rule)
        self.F_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.F_opt, lr_lambda=lambda_rule)
        self.D_X_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_X_opt, lr_lambda=lambda_rule)
        self.D_Y_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_Y_opt, lr_lambda=lambda_rule)

        self.save_folder = save_folder if save_folder is not None else f"./runs/{time.time()}"
        os.makedirs(self.save_folder, exist_ok=True)
        self.start_epoch = start_epoch

    def _initialise_weights(self):
        def applicator(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
        
        self.G.apply(applicator)
        self.F.apply(applicator)
        self.D_X.apply(applicator)
        self.D_Y.apply(applicator)

    def _step_learning_rate(self, name, scheduler, optimiser):
        previous_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimiser.param_groups[0]["lr"]
        print(f"Updated {name} learning rate from {previous_lr} to {new_lr}")

    def step_learning_rates(self):
        self._step_learning_rate("G_opt", self.G_opt_scheduler, self.G_opt)
        self._step_learning_rate("F_opt", self.F_opt_scheduler, self.F_opt)
        self._step_learning_rate("D_X_opt", self.D_X_opt_scheduler, self.D_X_opt)
        self._step_learning_rate("D_Y_opt", self.D_Y_opt_scheduler, self.D_Y_opt)

    def apply(self, tensors, x_to_y):
        model = self.G if x_to_y else self.F
        model.eval()

        with torch.no_grad():
            processed_tensors = model(tensors.to(self.device).detach())
        
        model.train()
        return processed_tensors.detach().cpu()

    def save(self, epoch):
        folder = f"{self.save_folder}/{epoch}"
        os.makedirs(folder, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "G": self.G.state_dict(),
            "F": self.F.state_dict(),
            "D_X": self.D_X.state_dict(),
            "D_Y": self.D_Y.state_dict(),
            "G_opt": self.G_opt.state_dict(),
            "F_opt": self.F_opt.state_dict(),
            "D_X_opt": self.D_X_opt.state_dict(),
            "D_Y_opt": self.D_Y_opt.state_dict(),
            "fake_X_buffer": self.fake_X_buffer.buffer,
            "fake_Y_buffer": self.fake_Y_buffer.buffer
        }, f"{folder}/checkpoint.tar")

        torch.save(self.G.state_dict(), f"{folder}/G.pth")
        torch.save(self.F.state_dict(), f"{folder}/F.pth")
    
    @staticmethod
    def load(save_folder, epoch_dir, device, blocks, up_type):
        checkpoint_path = f"{save_folder}/{epoch_dir}/checkpoint.tar"
        checkpoint = torch.load(checkpoint_path)

        cyclegan = CycleGAN(device, blocks, up_type, init=False, start_epoch=checkpoint["epoch"], save_folder=save_folder)

        cyclegan.G.load_state_dict(checkpoint["G"])
        cyclegan.D_X.load_state_dict(checkpoint["D_X"])
        cyclegan.F.load_state_dict(checkpoint["F"])
        cyclegan.D_Y.load_state_dict(checkpoint["D_Y"])

        cyclegan.G_opt.load_state_dict(checkpoint["G_opt"])
        cyclegan.G_opt_scheduler.last_epoch = cyclegan.start_epoch

        cyclegan.D_X_opt.load_state_dict(checkpoint["D_X_opt"])
        cyclegan.D_X_opt_scheduler.last_epoch = cyclegan.start_epoch

        cyclegan.F_opt.load_state_dict(checkpoint["F_opt"])
        cyclegan.F_opt_scheduler.last_epoch = cyclegan.start_epoch

        cyclegan.D_Y_opt.load_state_dict(checkpoint["D_Y_opt"])
        cyclegan.D_Y_opt_scheduler.last_epoch = cyclegan.start_epoch

        cyclegan.fake_X_buffer.buffer = [x.to(device) for x in checkpoint["fake_X_buffer"]]
        cyclegan.fake_Y_buffer.buffer = [y.to(device) for y in checkpoint["fake_Y_buffer"]]

        cyclegan.G.train()
        cyclegan.D_X.train()
        cyclegan.F.train()
        cyclegan.D_Y.train()

        print(f"Models and buffers loaded from {checkpoint_path}")

        return cyclegan