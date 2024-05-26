import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from cvae.cvae import save_vae, load_cvae
from dataloading.datasets import EmbeddingsDataset


def train_step(model, audio_embeddings, visual_embeddings, optimizer, beta):
    optimizer.zero_grad()

    mu, logvar = model.encode(visual_embeddings, audio_embeddings).chunk(2, dim=1)
    latent = model.sample(mu, logvar)
    prediction = model.decode(latent, audio_embeddings)

    # Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = torch.nn.functional.mse_loss(prediction, visual_embeddings)
    loss = beta * kl_loss + recon_loss
    # Backpropagate the loss.
    loss.backward()
    # Update the parameters.
    optimizer.step()

    return loss.item(), kl_loss.item(), recon_loss.item()


def val_step(model, audio_embeddings, visual_embeddings, beta):
    with torch.no_grad():
        mu, logvar = model.encode(visual_embeddings, audio_embeddings).chunk(2, dim=1)
        latent = model.sample(mu, logvar)
        prediction = model.decode(latent, audio_embeddings)

        # Loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.nn.functional.mse_loss(prediction, visual_embeddings)
        loss = beta * kl_loss + recon_loss

    return loss.item(), kl_loss.item(), recon_loss.item()


def train_cvae(model, train_dataloader, test_dataloader, savepath, lr, epochs, beta=1.):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = np.inf
    best_epoch = 0
    best_model = None

    for epoch in range(epochs):
        # Train step
        model.train()
        train_losses = []
        train_kl_losses = []
        train_recon_losses = []

        for audio_embeddings, visual_embeddings in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            loss, kl_loss, recon_loss = train_step(model, audio_embeddings.to(device), visual_embeddings.to(device), optimizer, beta)
            train_losses.append(loss)
            train_kl_losses.append(kl_loss)
            train_recon_losses.append(recon_loss)

        # Validation step
        model.eval()
        model.eval()
        val_losses = []
        val_kl_losses = []
        val_recon_losses = []
        for audio_embeddings, visual_embeddings in tqdm(test_dataloader, desc=f'Epoch {epoch}'):
            loss, kl_loss, recon_loss = val_step(model, audio_embeddings.to(device), visual_embeddings.to(device), beta)
            val_losses.append(loss)
            val_kl_losses.append(kl_loss)
            val_recon_losses.append(recon_loss)

        # Save the best val loss
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            best_epoch = epoch
            best_model = copy.deepcopy(model)

            # Print the epoch metrics
            print(f'Epoch {epoch} train loss: {np.mean(train_losses):.4f} val loss: {np.mean(val_losses):.4f}')
            print(f'   train kl loss: {np.mean(train_kl_losses):.4f} val kl loss: {np.mean(val_kl_losses):.4f}')
            print(f'   train recon loss: {np.mean(train_recon_losses):.4f} val recon loss: {np.mean(val_recon_losses):.4f}')

        # Print the best val loss and the epoch where this occurs
    print(f'Best val loss: {best_val_loss:.4f} at epoch {best_epoch}')

    model = best_model

    print('Saving model...')
    save_vae(savepath, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, default='configs/audio_prior.yaml')
    parser.add_argument('--train_data_path', type=str, default='datasets/train_dataset_prior.npz')
    parser.add_argument('--test_data_path', type=str, default='datasets/test_dataset_prior.npz')
    parser.add_argument('--savepath', type=str, default='checkpoints/audio_prior.pth')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_cvae(args.model_cfg, device=device)

    train_dataset = EmbeddingsDataset(args.train_data_path)
    test_dataset = EmbeddingsDataset(args.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    train_cvae(model, train_dataloader, test_dataloader, args.savepath, args.lr, args.num_epochs, args.beta)
