import os

import realesrgan
import torch
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils import load_file_from_url

import settings


class RealESRGANer(realesrgan.RealESRGANer):
    def __init__(self, scale, model_path, model=None, tile=0, tile_pad=10, pre_pad=10, half=False,
                 device=settings.DEVICE):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = device
        # if the model_path starts with https, it will first download models to the folder: realesrgan/weights
        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(realesrgan.ROOT_DIR, 'realesrgan/weights'), progress=True,
                file_name=None)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()
