# Imports
from typing import List

from minerl.herobraine.hero.handler import Handler

from config import DEVICE
import gym
import torch
import torchaudio
import pickle
import cv2
import numpy as np

from steve1.MineRLConditionalAgent import MineRLConditionalAgent
from steve1.VPT.agent import ENV_KWARGS

import sys
import os

# sys.path.append(os.getcwd() + '/src')
sys.path.insert(0, os.getcwd())

from src.source.load import load_model
from src.source.preprocess import make_features
from src.source.cvae.cvae import load_cvae

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
import minerl.herobraine.hero.handlers as handlers


# Environment ==========================================================================================================


class WrapperEnv(HumanSurvival):
    def __init__(self, **kwargs):
        super(WrapperEnv, self).__init__(**kwargs)

    def create_agent_start(self) -> List[Handler]:
        print('Setting starting position')
        retval = super().create_agent_start()
        # retval.append(handlers.AgentStartPlacement(0, 30, 0, 0., 0.))
        return retval


"""
    def create_server_world_generators(self) -> List[Handler]:
        return [handlers.DefaultWorldGenerator(force_reset=True, generator_options="{\"seed\":1337}")]
"""


# Model loading ========================================================================================================


def load_prior(model_path, model_cfg):
    return load_cvae(model_cfg, model_path, DEVICE)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


# preffered_spawn_biome="forest"
def make_env(seed):
    print('Loading MineRL...')
    env = WrapperEnv(**ENV_KWARGS).make()
    print('Starting new env...')
    env.reset()
    if seed is not None:
        print(f'Setting seed to {seed}...')
        env.seed(seed)
    return env


def make_agent(in_model, in_weights, cond_scale):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device='cuda', policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_clip_agent_env(clip_path, clip_config, in_model, in_weights, seed, cond_scale):
    print('Loading Clip model...')
    clip_model = load_model(clip_path, clip_config, DEVICE)
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed)

    return agent, clip_model, env


# embed_utils ==========================================================================================================


def get_prior_embed(path, clip, prior, device):
    """Get the embed processed by the prior."""
    # Load audio file
    data, sr = torchaudio.load(path)
    features = make_features(data, sr).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        embed = clip.encode_audio(features).detach()
        audio_embed = clip.project_audio_embeddings(embed).detach().cpu().numpy()
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio_prompt_embed = prior(torch.tensor(audio_embed).float().to(device)).cpu().detach().numpy()
    return audio_prompt_embed


# Video ================================================================================================================


def save_frames_as_video(frames: list, savefile_path: str, fps: int = 20, to_bgr: bool = False,
                         fx: float = 1.0, fy: float = 1.0):
    """Save a list of frames as a video to savefile_path"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first = cv2.resize(frames[0], None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    out = cv2.VideoWriter(savefile_path, fourcc, fps, (first.shape[1], first.shape[0]))
    for frame in frames:
        frame = np.uint8(frame)
        if to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        out.write(frame)
    out.release()
