# Extending the Prompting Modalities of STEVE-1-based Generative Agents

## Installation
We recomend using python 3.10 when running this code.
1. Follow the instructions to install the [STEVE-1 agent](https://github.com/Shalev-Lifshitz/STEVE-1)
2. Download this repository and place it in the same directory as the STEVE-1 installation: `git clone https://github.com/nlenzen/steve_multimodal`
   The directory structure should look like this:
   ```
    parent_dir
    ├── STEVE-1
    └── steve_multimodal

   ```
4. Change into the steve_multimodal directory: `cd steve_multimodal`
5. Run `pip install requirements.txt`
6. All done!

## Training the model
### Downloading the datasets

### Training the Audio-Video CLIP model

### Training the audio prior

## Generating gameplay video conditioned on audio
The audio prompts used in our experiments can be found in the directory `audio_prompts`.
1. To use you own audio prompts, place the them in the form of `.wav` files in this folder. For organization, the directory can have one layer of subdirectories in which the `.wav` files can be placed.
2. Start the generation by running `python steve_multimodal/steve_interfaces/interface_run_agent.py` inside `parent_dir`. This will generate a gameplay video for each audio prompt in the `audio_prompts` directory.

3. To run an interactive session run `python steve_multimodal/steve_interfaces/interface_interactive.py` inside `parent_dir`. Here you can specify a path to an audio prompt by clicking inside the video window and entering it into the console.
