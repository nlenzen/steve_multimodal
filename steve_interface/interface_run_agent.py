import os
import sys

import cv2
import torch
from tqdm import tqdm
import argparse

from config import DEVICE
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator

from interface_util import load_clip_agent_env, get_prior_embed, load_prior, save_frames_as_video

FPS = 20    # 30

SEEDS = [0, 87400, 1184, 87630, 15405, 3254, 41745, 37, 654336, 8362362]

def run_agent(prompt_embed, gameplay_length, save_video_filepath, seed, cond_scale, agent, env):
    assert cond_scale is not None
    # Make sure seed is set if specified
    obs = env.reset()
    if seed is not None:
        env.seed(seed)
        print("Setting seed to {}".format(seed))

    # Setup
    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)

    # Run agent in MineRL env
    for _ in tqdm(range(gameplay_length)):
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)
        frame = obs['pov']
        frame = cv2.resize(frame, (256, 160))
        gameplay_frames.append(frame)

        prog_evaluator.update(obs)

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()


def generate_audio_prompt_videos(prompt_embeds, in_model, in_weights, cond_scale, gameplay_length, save_dirpath, agent, env):
    for name, prompt_embed, seed in prompt_embeds:
        print(f'\nGenerating video for audio prompt from: {name}  with seed {seed}')
        # prompt_embed = item[0]
        # seed = item[1]
        name = os.path.splitext(name)[0]
        save_video_filepath = os.path.join(save_dirpath, name, f'seed-{seed}.mp4')
        # save_video_filepath = os.path.join(save_dirpath, f'{name}-audio_prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath, seed, cond_scale, agent, env)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_path', type=str, default='checkpoints/model/avclip.pth')
    parser.add_argument('--clip_cfg', type=str, default='src/configs/avclip.yaml')
    parser.add_argument('--in_model', type=str, default='STEVE-1/data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='STEVE-1/data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_path', type=str, default='checkpoints/cvae/audio_prior.pth')
    parser.add_argument('--prior_cfg', type=str, default='src/configs/cvae/audio_prior.yaml')
    parser.add_argument('--audio_cond_scale', type=float, default=6.0)
    parser.add_argument('--gameplay_length', type=int, default=3000)
    parser.add_argument('--seed', type=float, default=None)
    parser.add_argument('--save_dirpath', type=str, default='videos/generated_videos/')
    parser.add_argument('--custom_audio_path', type=str, default='datasets/audio_prompts')
    args = parser.parse_args()

    if args.custom_audio_path is not None:
        # Generate a video for the audio prompt
        agent, clip, env = load_clip_agent_env(args.clip_path, args.clip_cfg, args.in_model, args.in_weights, args.seed, args.audio_cond_scale)
        prior = load_prior(args.prior_path, args.prior_cfg)
        if os.path.isdir(args.custom_audio_path):
            custom_prompt_embeds = []
            entries = os.listdir(args.custom_audio_path)
            dirs = [name for name in entries if os.path.isdir(os.path.join(args.custom_audio_path, name))]
            for directory in dirs:
                dirname = os.path.join(args.custom_audio_path, directory)
                files = os.listdir(dirname)
                files = sorted(files)
                for file in files:
                    if file.endswith('.wav'):
                        name = os.path.join(dirname, file)
                        prompt_embed = get_prior_embed(name, clip, prior, DEVICE)
                        for seed in SEEDS:
                            name = os.path.join(directory, os.path.splitext(file)[0])
                            custom_prompt_embeds.append((name, prompt_embed, seed))

        else:
            prompt_embed = get_prior_embed(args.custom_audio_prompt, clip, prior, DEVICE)
            custom_prompt_embeds = {args.custom_audio_prompt: prompt_embed}
        print('Num embeds: {}'.format(len(custom_prompt_embeds)))
        generate_audio_prompt_videos(custom_prompt_embeds, args.in_model, args.in_weights, args.audio_cond_scale,
                                    args.gameplay_length, args.save_dirpath, agent, env)
