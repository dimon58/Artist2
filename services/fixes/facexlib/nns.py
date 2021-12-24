import os
from copy import deepcopy

import facexlib.utils.face_restoration_helper
import gfpgan
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils import load_file_from_url
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

import settings
from .retinaface import RetinaFace


def init_detection_model(model_name, half=False, device='cuda'):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='facexlib/weights', progress=True, file_name=None)
    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


class FaceRestoreHelper(facexlib.utils.face_restoration_helper.FaceRestoreHelper):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=True,
                 pad_blur=False,
                 device=None):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = upscale_factor
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = device

        # init face detection model
        self.face_det = init_detection_model(det_model, half=False, device=device)


class GFPGANer(gfpgan.GFPGANer):

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None,
                 device=settings.DEVICE):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        else:
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device)

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path,
                model_dir=os.path.join(gfpgan.ROOT_DIR, 'gfpgan/weights'),
                progress=True,
                file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)
