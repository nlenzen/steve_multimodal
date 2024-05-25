# Imports
import os
import torch
import yaml
from .model import ClipModel


def load_model(chkpt, config, device):
    if isinstance(chkpt, str):
        chkpt = torch.load(chkpt, map_location=device)
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
    state_dict = chkpt['state_dict']
    model = ClipModel(**config)
    model.vision_encoder.resize_pos_embed([160, 256])

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def save_model(model, path):
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, path)


def init_model(video_enc_path, audio_enc_path, config, device):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

    video_enc_dict, pool_dict = load_video_statedict(video_enc_path)
    audio_enc_dict = load_audio_statedict(audio_enc_path)

    model = ClipModel(**config)
    model.vision_encoder.resize_pos_embed([160, 256])
    model.vision_encoder.load_state_dict(video_enc_dict)
    model.audio_encoder.load_state_dict(audio_enc_dict)
    if config['pooling']:
        model.temporal_encoder.load_state_dict(pool_dict)

    model.to(device)
    model.eval()
    return model


def load_video_statedict(path):
    chkpt_dict = torch.load(path)
    state_dict = chkpt_dict['state_dict']
    video_enc_dict = {k.replace('model.clip_model.vision_model.', ''): state_dict[k] for k in state_dict.keys() if k.startswith('model.clip_model.vision_model.')}
    pool_dict = {k.replace('model.temporal_encoder.', ''): state_dict[k] for k in state_dict.keys() if k .startswith('model.temporal_encoder.')}

    return video_enc_dict, pool_dict


def load_audio_statedict(path):
    chkpt_dict = torch.load(path)
    audio_enc_dict = {k.replace('module.', ''): chkpt_dict[k] for k in chkpt_dict.keys()}

    return audio_enc_dict


