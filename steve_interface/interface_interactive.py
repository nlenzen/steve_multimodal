import argparse
import json
import os

import cv2
import numpy as np
import torch

from interface_util import load_clip_agent_env, get_prior_embed, load_prior
from steve1.utils.text_overlay_utils import created_fitted_text_image
from steve1.utils.video_utils import save_frames_as_video

from config import PRIOR_INFO, DEVICE

FPS = 20

def create_video_frame(gameplay_pov, prompt):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    prompt_section = created_fitted_text_image(frame.shape[1] // 2, prompt,
                                               background_color=(0, 0, 0),
                                               text_color=(255, 255, 255))
    pad_top_height = (frame.shape[0] - prompt_section.shape[0]) // 2
    pad_top = np.zeros((pad_top_height, prompt_section.shape[1], 3), dtype=np.uint8)
    pad_bottom_height = frame.shape[0] - pad_top_height - prompt_section.shape[0]
    pad_bottom = np.zeros((pad_bottom_height, prompt_section.shape[1], 3), dtype=np.uint8)
    prompt_section = np.vstack((pad_top, prompt_section, pad_bottom))
    frame = np.hstack((frame, prompt_section))
    return frame


def run_interactive(clip_path, clip_cfg, in_model, in_weights, cond_scale, seed, prior_path, prior_cfg, output_video_dirpath):
    """Runs the agent in the MineRL env and allows the user to enter prompts to control the agent.
    Clicking on the gameplay window will pause the gameplay and allow the user to enter a new prompt.

    Typing 'reset agent' will reset the agent's state.
    Typing 'reset env' will reset the environment.
    Typing 'save video' will save the video so far (and ask for a video name). It will also save a json storing
        the active prompt at each frame of the video.
    """
    agent, clip, env = load_clip_agent_env(clip_path, clip_cfg, in_model, in_weights, seed, cond_scale)
    prior = load_prior(prior_path, prior_cfg)
    window_name = 'STEVE-1 Gameplay (Click to Enter Prompt)'

    state = {'obs': None}
    os.makedirs(output_video_dirpath, exist_ok=True)
    video_frames = []
    frame_prompts = []

    def handle_prompt():
        # Pause the gameplay and ask for a new prompt
        # prompt = input('\n\nEnter a prompt:\n>').strip().lower()
        prompt = input('\n\nEnter a prompt:\n>').strip()

        # Reset the agent or env if prompted
        if prompt == 'reset agent':
            print('\n\nResetting agent...')
            agent.reset(cond_scale)
            print(f'Done. Continuing gameplay with previous prompt...')
            return
        elif prompt == 'reset env':
            reset_env()
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Save the video so far if prompted
        if prompt == 'save video':
            # Ask for a video name
            video_name = input('Enter a video name:\n>').strip().lower()

            # Save both the video and the prompts for each frame
            output_video_filepath = os.path.join(output_video_dirpath, f'{video_name}.mp4')
            prompts_for_frames_filepath = os.path.join(output_video_dirpath, f'{video_name}.json')
            print(f'Saving video to {output_video_filepath}...')
            save_frames_as_video(video_frames, output_video_filepath, fps=FPS)
            print(f'Saving prompts for frames to {prompts_for_frames_filepath}...')
            with open(prompts_for_frames_filepath, 'w') as f:
                json.dump(frame_prompts, f)
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Use prior to get the prompt embed
        prompt_embed = get_prior_embed(prompt, clip, prior, DEVICE)

        with torch.cuda.amp.autocast():
            while True:
                minerl_action = agent.get_action(state['obs'], prompt_embed)
                state['obs'], _, _, _ = env.step(minerl_action)

                frame = create_video_frame(state['obs']['pov'], prompt)
                video_frames.append(frame)
                frame_prompts.append(prompt)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    def reset_env():
        print('\nResetting environment...')
        state['obs'] = env.reset()
        if seed is not None:
            print(f'Setting seed to {seed}...')
            env.seed(seed)
    reset_env()
    print('Success')

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_prompt()

    initial_frame = create_video_frame(state['obs']['pov'], 'Click to Enter a Prompt')
    cv2.imshow(window_name, initial_frame)
    cv2.setMouseCallback(window_name, on_click)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window when 'q' is pressed
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_path', type=str, default='steve_multimodal/checkpoints/avclip.pth')
    parser.add_argument('--clip_cfg', type=str, default='steve_multimodal/configs/avclip.yaml')
    parser.add_argument('--in_model', type=str, default='STEVE-1/data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='STEVE-1/data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_path', type=str, default='steve_multimodal/checkpoints/audio_prior.pth')
    parser.add_argument('--prior_cfg', type=str, default='steve_multimodal/configs/audio_prior.yaml')
    parser.add_argument('--output_video_dirpath', type=str, default='videos/generated_videos/interactive_videos')
    parser.add_argument('--minecraft_seed', type=float, default=1337)  # None for random seed
    parser.add_argument('--cond_scale', type=float, default=6.0)
    args = parser.parse_args()

    run_interactive(
        clip_path=args.clip_path,
        clip_cfg=args.clip_cfg,
        in_model=args.in_model,
        in_weights=args.in_weights,
        cond_scale=args.cond_scale,
        seed=args.minecraft_seed,
        prior_path=args.prior_path,
        prior_cfg=args.prior_cfg,
        output_video_dirpath=args.output_video_dirpath,
    )
