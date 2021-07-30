                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from data import HydraDataset


def load_data(train_imgs, train_masks, val_imgs, val_masks, batch_size=4, augmentation=True):
    if augmentation:
        train_augs, val_augs = data_augmentors()
    else:
        train_augs =  None
        val_augs = None

    train_ds = HydraDataset(train_imgs, train_masks, train_augs)
    val_ds = HydraDataset(val_imgs, val_masks, val_augs)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True),
    val_dl =  DataLoader(val_ds, batch_size, shuffle=True)

    return train_dl, val_dl

def data_augmentors():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

 
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    return train_transform, val_transform

def train_model(model, device, criterion, optimizer, train_loader, val_loader, epochs):
    scaler = torch.cuda.amp.GradScaler()
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.float().unsqueeze(1).to(device)
            
            # forward pass
            preds = model(images)
            loss = criterion(preds, masks)

            # backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())
        
        # get train loss and test loss
        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, \
              Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def get_accuracy(model, train_loader, val_loader, device):
    model.eval()

    n_correct = 0.
    n_total = 0.

    for images, masks in train_loader:
        images, masks = images['image'].to(device), masks['mask'].to(device)
        # forward pass
        preds = model(images)
        _, preds = torch.max(preds, 1)
        # update counts
        n_correct += (preds == masks).sum().item()
        n_total += masks.shape[0]
    train_acc = n_correct / n_total

    n_correct = 0.
    n_total = 0.
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        # forward pass
        preds = model(images)
        _, preds = torch.max(preds, 1)

        # update counts
        n_correct += (preds == masks).sum().item()
        n_total += masks.shape[0]
    val_acc = n_correct / n_total

    print(f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}')

    return train_acc, val_acc


def main():
    # define data paths
    train_imgs = 'hydra_png_data/images_train'
    train_masks = 'hydra_png_data/masks_train'
    val_imgs = 'hydra_png_data/images_test'
    val_masks = 'hydra_png_data/masks_test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train, val = load_data(train_imgs, train_masks, val_imgs, val_masks)
    
    for image, target in train:
        print(image.shape)
        print(target.shape)
        break





    #model = UNet(in_channels=3, out_channels=1).to(device)
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(model.parameters())


    #train_loss, val_loss = train_model(model, device, criterion, optimizer, \
    #                                    train_ldr, val_ldr, epochs=50)


if __name__ == "__main__":
    main()