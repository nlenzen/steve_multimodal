# Imports
import torch
import torch.nn as nn
import numpy as np

from .attention import *
from .pooling import *
from .vision_encoder import VisionTransformer, basic_transform, img_transform
from .audio_encoder import ASTModel

# Projections ==========================================================================================================


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 bias=True,
                 activation=True,
                 lr_multiplier=1.,
                 weight_init=1.,
                 bias_init=0.,
                 slope=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.randn([output_dim, input_dim]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [output_dim])
        self.bias = nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.activation = nn.LeakyReLU(negative_slope=slope) if activation else None
        self.weight_gain = lr_multiplier / np.sqrt(input_dim)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weights.to(x.dtype) * self.weight_gain
        x = x.matmul(w.t())
        if self.bias is not None:
            b = self.bias.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
            x = x + b
        if self.activation:
            x = self.activation(x)

        return x


class MappingNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 final_activation=True,
                 hidden_dim=-1,
                 num_layers=8,
                 lr_multiplier=0.01,
                 w_avg_beta=0.998):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim < 0:
            hidden_dim = output_dim
        self.num_layers = num_layers
        self.lr_multiplier = lr_multiplier
        self.w_avg_beta = w_avg_beta

        features = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        features = zip(range(num_layers), features[:-1], features[1:])
        for idx, input_dim, output_dim in features:
            activation = True if idx < num_layers - 2 else final_activation
            layer = FullyConnectedLayer(input_dim, output_dim, activation, lr_multiplier=self.lr_multiplier)
            setattr(self, 'linear_{}'.format(idx), layer)
        self.register_buffer('w_avg', torch.zeros([output_dim]))
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        # normalize input
        # x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        x = self.bn(x)

        for idx in range(self.num_layers):
            x = getattr(self, 'linear_{}'.format(idx))(x)

        return x


class Projection(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, embed_dim)
        self.RELU = nn.ReLU()

    def forward(self, x):
        x = self.RELU(self.linear1(x))
        x = self.RELU(self.linear2(x))
        x = self.RELU(self.linear3(x))
        x = self.RELU(self.linear4(x))
        return x

# CLIP model ===========================================================================================================


class ClipModel(nn.Module):
    def __init__(self,
                 # Projection
                 video_projection=True,
                 final_activation=True,
                 proj_type='mapping',    # Type of projection in ['mapping', 'projection']
                 embed_dim=512,
                 hidden_dim=-1,
                 num_layers=8,
                 # Video
                 vision_embed_dim=512,
                 vision_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=16,
                 # Pooling params
                 pooling=True,
                 pool_type='attn.d2.nh8.glusw',
                 # Audio
                 audio_label_dim=527,
                 audio_fstride=10,
                 audio_tstride=10,
                 audio_input_fdim=128,
                 audio_input_tdim=1024,
                 # Base Parameters
                 image_adapter_layers=0,
                 video_seq_len=32
                 ):
        super().__init__()
        heads = vision_width // 64
        self.vision_encoder = VisionTransformer(resolution=vision_resolution,
                                                patch_size=vision_patch_size,
                                                width=vision_width,
                                                layers=vision_layers,
                                                heads=heads,
                                                output_dim=vision_embed_dim)
        if pooling:
            self.temporal_encoder = TemporalPooling(pool_type=pool_type,
                                                    input_dim=vision_embed_dim,
                                                    hidden_dim=hidden_dim,
                                                    output_dim=vision_embed_dim,
                                                    layers_before_pool=image_adapter_layers,
                                                    max_seq_len=video_seq_len)
            if not isinstance(self.temporal_encoder.mlp_before_pool, nn.Identity):
                for module in self.temporal_encoder.mlp_before_pool:
                    # initialize linear layers as identity matrices
                    if isinstance(module, nn.Linear):
                        module.weight.data.copy_(torch.eye(module.weight.shape[0]))
                        module.bias.data.zero_()
        else:
            self.temporal_encoder = None

        self.audio_encoder = ASTModel(label_dim=audio_label_dim,
                                      fstride=audio_fstride,
                                      tstride=audio_tstride,
                                      input_fdim=audio_input_fdim,
                                      input_tdim=audio_input_tdim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Projection Networks
        self.proj_type = proj_type
        if proj_type == 'projection':
            if hidden_dim < 0:
                hidden_dim = embed_dim * 2
            self.audio_projection = Projection(527, hidden_dim, embed_dim)
            self.video_projection = Projection(vision_embed_dim, hidden_dim, embed_dim) if video_projection else None
            self.init()
        elif proj_type == 'mapping':
            self.audio_projection = MappingNetwork(audio_label_dim, embed_dim, final_activation, hidden_dim, num_layers=num_layers)
            self.video_projection = MappingNetwork(vision_embed_dim, embed_dim, final_activation, hidden_dim, num_layers=num_layers) if video_projection else None

    def init(self):
        def init_weights(l):
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, std=0.02)
                nn.init.normal_(l.bias, std=0.02)

        self.audio_projection.apply(init_weights)
        self.video_projection.apply(init_weights)

    def encode_images(self, images):
        imgs = images.permute(0, 3, 1, 2)
        imgs = img_transform(imgs)
        x = self.vision_encoder(imgs)

        return x

    def encode_video(self, frames):
        frames = frames.permute(0, 1, 4, 2, 3)
        leading_dims = frames.size()[:-3]
        c, h, w = frames.size()[-3:]
        frames = frames.view(-1, c, h, w)
        frames = basic_transform(frames)

        x = self.vision_encoder(frames)
        x = x.view(*leading_dims, x.size(-1))
        if self.temporal_encoder is not None:
            x = self.temporal_encoder(x)

        # x = self.video_projection(x)
        return x

    def project_video_embeddings(self, embeddings):
        if self.video_projection is not None:
            return self.video_projection(embeddings)
        else:
            return embeddings

    def encode_audio(self, audio):
        x = self.audio_encoder(audio)

        # x = self.audio_projection(x)
        return x

    def project_audio_embeddings(self, embeddings):
        return self.audio_projection(embeddings)

    def forward_audio(self, audio):
        embeddings = self.encode_audio(audio)
        embeddings = self.project_audio_embeddings(embeddings)

        return embeddings

    def forward_video(self, frames):
        embeddings = self.encode_video(frames)
        embeddings = self.project_video_embeddings(embeddings)

        return embeddings

    def forward_embeddings(self, audio_embeddings, video_embeddings):

        video_features = self.project_video_embeddings(video_embeddings)
        audio_features = self.project_audio_embeddings(audio_embeddings)

        logits_per_video, logits_per_audio = self.compute_logits(video_features, audio_features)

        return logits_per_video, logits_per_audio

    def forward(self, audio, frames):
        video_features = self.project_video_embeddings(self.encode_video(frames))
        audio_features = self.project_audio_embeddings(self.encode_audio(audio))

        logits_per_video, logits_per_audio = self.compute_logits(video_features, audio_features)

        return logits_per_video, logits_per_audio

    def compute_logits(self, video_features, audio_features):
        # normalizing features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ audio_features.t()
        logits_per_audio = logits_per_video.t()

        return logits_per_video, logits_per_audio
