# -*- coding: utf-8 -*-
import os

try:
    from path import Path
except ModuleNotFoundError:
    from pathlib import Path

from huggingface_hub import hf_hub_url, cached_download

from rudalle.dalle import MODELS


def download_rudalle_model(name, cache_dir='/tmp/rudalle', **model_kwargs):
    assert name in MODELS

    config = MODELS[name].copy()
    config['model_params'].update(model_kwargs)

    cache_dir = os.path.join(cache_dir, name)
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    print('rudalle --> ready')


def download_tokenizer(cache_dir='/tmp/rudalle'):
    repo_id = 'shonenkov/rudalle-utils'
    filename = 'bpe.model'
    cache_dir = os.path.join(cache_dir, 'tokenizer')
    config_file_url = hf_hub_url(repo_id=repo_id, filename=filename)
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)
    print('tokenizer --> ready')


def download_vae(dwt=False, cache_dir='/tmp/rudalle'):
    repo_id = 'shonenkov/rudalle-utils'
    if dwt:
        filename = 'vqgan.gumbelf8-sber-dwt.model.ckpt'
    else:
        filename = 'vqgan.gumbelf8-sber.model.ckpt'

    cache_dir = os.path.join(cache_dir, 'vae')
    config_file_url = hf_hub_url(repo_id=repo_id, filename=filename)
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)
    print('vae --> ready')


pretrained_path = Path(__file__).parent.parent / 'services' / 'nns' / 'pretrained_models' / 'rudalle'
pretrained_path = '/tmp/rudalle'

download_rudalle_model('Malevich', cache_dir=pretrained_path)
download_tokenizer(cache_dir=pretrained_path)
download_vae(dwt=True, cache_dir=pretrained_path)