# Imports
import torch
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRIOR_INFO = {
    'mineclip_dim': 512,
    'latent_dim': 512,
    'hidden_dim': 512,
    'model_path': 'STEVE-1/data/weights/steve1/steve1_prior.pt',
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
