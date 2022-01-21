"""
code from https://github.com/bryandlee/animegan2-pytorch
"""
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image

import settings

logger = logging.getLogger('AnimeGANv2')

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


net = Generator()
net.load_state_dict(torch.load(settings.ANIMEGANV2_PATH, map_location="cpu"))
net.to(settings.DEVICE).eval()
logger.info(f"AnimeGANv2 model loaded: {settings.ANIMEGANV2_PATH}")


def load_image(image: Image, x32: bool = False) -> Image:
    img = image.convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def inference(
        input_images: list[Image],
        device='cuda:0',
        upsample_align=False,
        x32=None,
):
    """

    Args:
        input_images (list[Image]): list og pil images
        device: pytorch device
        upsample_align (bool): Align corners in decoder upsampling layers
        x32: Resize images to multiple of 32
    """

    output_images = []

    for image in input_images:
        image = load_image(image, x32)

        memory_start = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(image.to(device), upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)

        output_images.append(out)

        memory_user = torch.cuda.max_memory_allocated(device) - memory_start

        logger.info(f'Max memory using during converting {out.width}x{out.height} image is '
                    f'{round(memory_user / 1024 ** 3, 1)} GB')

    torch.cuda.empty_cache()

    return output_images
