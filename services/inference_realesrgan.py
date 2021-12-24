import os
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from services.fixes.facexlib import GFPGANer
from services.fixes.realesrgan_fixed import RealESRGANer

file_path = Path(__file__).parent
os.makedirs(os.path.join(file_path, 'pretrained_models'), exist_ok=True)


def chose_model(model_name):
    # determine models according to model names
    model_name = model_name.split('.')[0]
    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'

    elif model_name in ['RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x4plus_anime']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'

    elif model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

    elif model_name in [
        'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth'

    elif model_name in [
        'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth'

    else:
        raise ValueError(f'Model {model_name} does not exist.')

    return model, netscale, url


def determine_model_paths(model_name, url):
    model_file = url.split('/')[-1]
    model_path = os.path.join(file_path, 'nns', 'pretrained_models', model_file)
    if not os.path.isfile(model_path):
        try:
            print(f'Downloading model {model_name}')
            urllib.request.urlretrieve(url, os.path.join(file_path, 'pretrained_models', model_file))
        except urllib.error.HTTPError:
            raise ValueError(f'Model {model_name} does not exist.')

    return model_path


def upscale(
        input_image,
        model_name: str = 'RealESRGAN_x4plus',
        outscale: float = 4,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        alpha_upsampler: str = 'realesrgan',
        face_enhance=False,
        half=False,
):
    """
    :param input_image: Input image (must be loaded with cv2.imread(*input_file*, cv2.IMREAD_UNCHANGED))
    :param model_name: Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4
    :param outscale: The final upsampling scale of the image
    :param tile: Tile size, 0 for no tile during testing
    :param tile_pad: Tile padding
    :param pre_pad: Pre padding size at each border
    :param face_enhance: Use GFPGAN to enhance face
    :param half: Use half precision during inference
    :param alpha_upsampler: The upsampler for the alpha channels. Options: realesrgan | bicubic

    :return: np.ndarray image with BGR format
    """

    # determine model paths
    model, netscale, url = chose_model(model_name)
    model_path = determine_model_paths(model_name, url)

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half)

    if face_enhance:  # Use GFPGAN for face enhancement
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(input_image, has_aligned=False, only_center_face=False,
                                                 paste_back=True)
        else:
            output, _ = upsampler.enhance(input_image, outscale=outscale)

    except RuntimeError as error:
        print('Error', error)
        raise RuntimeError(f'{error}.\nIf you encounter CUDA out of memory, try to set --tile with a smaller number.')

    else:
        return output


def upscale_pil(
        input_pil_image,
        model_name: str = 'RealESRGAN_x4plus',
        outscale: float = 4,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        alpha_upsampler: str = 'realesrgan',
        face_enhance=False,
        half=False,
):
    cv2_image = cv2.cvtColor(np.array(input_pil_image), cv2.COLOR_RGB2BGR)

    upscaled_image = upscale(
        input_image=cv2_image,
        model_name=model_name,
        outscale=outscale,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        alpha_upsampler=alpha_upsampler,
        face_enhance=face_enhance,
        half=half,
    )

    return Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
