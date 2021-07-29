                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from data import Dataset
from torchvision.utils import make_grid


def data_augmentors():
    train_transform = A.Compose([
        A.Resize(height=160, width=240),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2()
    ])

    #train_transform = transforms.Compose([
    #    transforms.RandomRotation(10),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.Resize(224),
    #    transforms.CenterCrop(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                std=[0.229, 0.224, 0.225])
    #])

    val_transform = A.Compose([
        A.Resize(height=160, width=240),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2()
    ])

    #val_transform = transforms.Compose([
    #    transforms.Resize(224),
    #    transforms.CenterCrop(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                std=[0.229, 0.224, 0.225])
    #])

    return train_transform, val_transform

def train_model(model, device, criterion, optimizer, train_loader, val_loader, epochs):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # forward pass
            preds = model(images)
            loss = criterion(preds, masks)

            # backward and optimize
            loss.backward()
            optimizer.step()
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
        images, masks = images.to(device), masks.to(device)
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
    train_imgs = 'data/train/images'
    train_masks = 'data/train/masks'
    val_imgs = 'data/val/images'
    val_masks = 'data/val/masks'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_augs, val_augs = data_augmentors()
    train = Dataset(train_imgs, train_masks, train_augs)
    val = Dataset(val_imgs, val_masks, val_augs)

    print(train)
    print(val)

    # create data loaders
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=4)

    for images, labels in train_loader:
        break

    print(images)
    print(labels)


    #image, mask = train[0]['image'], train[0]['mask']
    #print(image.shape)
    #print(mask.shape)

    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())


    #train_loss, val_loss = train_model(model, device, criterion, optimizer, \
    #                                    train, val, epochs=50)


if __name__ == "__main__":
    main()