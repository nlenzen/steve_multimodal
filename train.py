# Imports
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .load import load_model, save_model
from .preprocess import make_features


def train_model(model,
                dataloader,
                savepath,
                cfg_path=None,
                param_path=None,
                lr=0.00025,
                test_size=1200,
                num_tests=10,
                num_epochs=100):
    train_losses = []
    test_losses = []
    train_accuracies_audio = []
    train_accuracies_video = []
    test_accuracies_audio = []
    test_accuracies_video = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, str):
        chkpt = torch.load(model, map_location=device)
        model = load_model(chkpt, cfg_path, device)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    if model.temporal_encoder is not None:
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fnc = nn.CrossEntropyLoss()

    # Load parameters of previous training if training should be continued
    last_epoch = 0
    if param_path is not None:
        print("Continue training from checkpoint")
        dic = torch.load(param_path, map_location=device)
        last_epoch = dic['last_epoch']
        optimizer.load_state_dict(dic['optimizer'])
        train_losses = dic['train_losses']
        test_losses = dic['test_losses']
        train_accuracies_video = dic['train_accuracies_video']
        train_accuracies_audio = dic['train_accuracies_audio']
        test_accuracies_video = dic['test_accuracies_video']
        test_accuracies_audio = dic['test_accuracies_audio']
        if 'scheduler' in dic.keys():
            scheduler.load_state_dict(dic['scheduler'])

    for epoch in range(num_epochs):
        model.train()
        dataloader.randomize_sample_order()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("Training...")
        for num_batch in tqdm(range(len(dataloader)), desc="Processing batch: "):
            batch_loss = train_batch(num_batch, model, dataloader, optimizer, loss_fnc, device)
            train_losses.append(batch_loss)

            # Testing
            optimizer.step()
            scheduler.step()

        # Evaluate epoch performance
        print("Average training loss: {}".format(sum(train_losses[-len(dataloader):]) / len(dataloader)))

        loss, train_acc_v, train_acc_a, test_acc_v, test_acc_a = evaluate(model, dataloader, loss_fnc, test_size, num_tests, device)
        test_losses.append(loss)
        train_accuracies_video.append(train_acc_v)
        train_accuracies_audio.append(train_acc_a)
        test_accuracies_video.append(test_acc_v)
        test_accuracies_audio.append(test_acc_a)

        # Save progress every 10 epochs
        if (epoch + 1) % 10 == 0 and epoch != num_epochs - 1:
            save_model_params(model, optimizer,
                       train_losses, test_losses,
                       train_accuracies_video, train_accuracies_audio,
                       test_accuracies_video, test_accuracies_audio,
                       last_epoch + epoch, savepath,
                       scheduler)

    print("Done training saving model")
    save_model_params(model, optimizer,
               train_losses, test_losses,
               train_accuracies_video, train_accuracies_audio,
               test_accuracies_video, test_accuracies_audio,
               last_epoch + num_epochs, savepath,
               scheduler)

    return model


def evaluate(model, dataloader, loss_fnc, test_size, num_tests, device):
    print("Validating...")
    model.eval()
    train_acc_a = 0
    train_acc_v = 0
    loss = 0
    test_acc_a = 0
    test_acc_v = 0

    for _ in range(num_tests):
        # Evaluate model performance in train set
        video_embeddings, audio_embeddings = dataloader.get_random_train_samples(test_size)
        video_embeddings = torch.tensor(video_embeddings).to(device)
        audio_embeddings = torch.tensor(audio_embeddings).to(device)
        acc_v, acc_a = evaluate_model(model, video_embeddings, audio_embeddings)
        train_acc_a += acc_a
        train_acc_v += acc_v

        # Evaluate model performance on evaluation set
        video_embeddings, audio_embeddings = dataloader.get_random_test_samples(test_size)
        video_embeddings = torch.tensor(video_embeddings).to(device)
        audio_embeddings = torch.tensor(audio_embeddings).to(device)

        # Loss of test set
        with torch.no_grad():
            video_probs, audio_probs = model.forward_embeddings(audio_embeddings, video_embeddings)

        labels = torch.arange(len(video_embeddings)).to(device)
        loss_video = loss_fnc(video_probs, labels)
        loss_audio = loss_fnc(audio_probs, labels)
        partial_loss = (loss_video + loss_audio) / 2
        loss += partial_loss.item()

        # Accuracy of test set
        acc_v, acc_a = evaluate_model(model, video_embeddings, audio_embeddings)
        test_acc_a += acc_a
        test_acc_v += acc_v

    # Average accuracies and test-loss
    train_acc_a /= num_tests
    train_acc_v /= num_tests
    loss /= num_tests
    test_acc_a /= num_tests
    test_acc_v /= num_tests
    print("  Average train accuracy video:", train_acc_v)
    print("  Average train accuracy audio:", train_acc_a)
    print("  Average test loss:", loss)
    print("  Average test accuracy video:", test_acc_v)
    print("  Average test accuracy audio:", test_acc_a)

    return loss, train_acc_v, train_acc_a, test_acc_v, test_acc_a


def evaluate_model(model, video_embeddings, audio_embeddings):
    with torch.no_grad():
        video_encs = model.project_video_embeddings(video_embeddings)
        audio_encs = model.project_audio_embeddings(audio_embeddings)
        video_encs /= video_encs.norm(dim=-1, keepdim=True)
        audio_encs /= audio_encs.norm(dim=-1, keepdim=True)

    video_probs = (video_encs @ audio_encs.T).softmax(dim=-1).cpu()
    audio_probs = (audio_encs @ video_encs.T).softmax(dim=-1).cpu()

    video_labels = torch.argmax(video_probs, dim=-1)
    audio_labels = torch.argmax(audio_probs, dim=-1)

    labels = torch.arange(len(video_labels))

    correct_v = torch.sum(video_labels == labels).item()
    correct_a = torch.sum(audio_labels == labels).item()

    acc_v = correct_v / len(video_labels)
    acc_a = correct_a / len(audio_labels)

    return acc_v, acc_a


def train_batch(batch_num, model, dataloader, optimizer, loss_fnc, device):
    optimizer.zero_grad()
    video_embeddings, audio_embeddings = dataloader.get_batch(batch_num)
    video_embeddings = torch.tensor(video_embeddings).to(device)
    audio_embeddings = torch.tensor(audio_embeddings).to(device)

    video_probs, audio_probs = model.forward_embeddings(audio_embeddings, video_embeddings)

    labels = torch.arange(len(video_embeddings)).to(device)

    loss_video = loss_fnc(video_probs, labels)
    loss_audio = loss_fnc(audio_probs, labels)
    loss = (loss_video + loss_audio) / 2
    loss.backward()
    optimizer.step()

    return loss.item()


def save_model_params(model,
                      optimizer,
                      train_losses,
                      test_losses,
                      train_accuracies_video,
                      train_accuracies_audio,
                      test_accuracies_video,
                      test_accuracies_audio,
                      epoch,
                      savepath,
                      scheduler=None):
    dir_path = os.path.dirname(savepath)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    name, extension = os.path.splitext(savepath)
    param_path = name + '_train_params' + extension
    if os.path.isfile(param_path):
        os.remove(param_path)
    if os.path.isfile(savepath):
        os.remove(savepath)
    dic = {
        'last_epoch': epoch,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'optimizer': optimizer.state_dict(),
        'train_accuracies_audio': train_accuracies_audio,
        'train_accuracies_video': train_accuracies_video,
        'test_accuracies_audio': test_accuracies_audio,
        'test_accuracies_video': test_accuracies_video,
    }
    if scheduler is not None:
        dic['scheduler'] = scheduler.state_dict()
    # Save training parameters and model seperately
    torch.save(dic, param_path)
    save_model(model, savepath)
