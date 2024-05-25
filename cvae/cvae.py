import torch.nn as nn
import torch
import yaml
import os


def load_cvae(cfg, model_chkpt=None, device=None):
    if isinstance(cfg, str):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
    model = CVAE(**config)
    if model_chkpt is not None:
        if isinstance(model_chkpt, str):
            model_chkpt = torch.load(model_chkpt)
        chkpt = model_chkpt['state_dict']
        model.load_state_dict(chkpt)
    if device:
        model = model.to(device)

    return model


def save_vae(savepath, model):
    dir_path = os.path.dirname(savepath)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, savepath)


class CVAE(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim)
        )

    def encode(self, visual_embeddings, audio_embeddings):
        x = torch.cat([visual_embeddings, audio_embeddings], dim=1)
        return self.encoder(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, visual_embeddings, audio_embeddings):
        x = torch.cat([visual_embeddings, audio_embeddings], dim=1)
        return self.decoder(x)

    def forward(self, audio_embeddings):
        visual_embeddings = torch.randn(audio_embeddings.shape[0], self.latent_dim, device=audio_embeddings.device)

        return self.decode(visual_embeddings, audio_embeddings)
