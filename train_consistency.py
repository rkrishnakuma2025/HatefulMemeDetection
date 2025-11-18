
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

from dataset import HatefulMemesDataset, train_transform, eval_transform
from edits import random_train_edit


# === CONFIG ===
DATA_ROOT = r"C:\Users\resmi\Downloads\Hateful_Meme_Detection\meme_robustness_data\data" 
BATCH_SIZE = 32
NUM_EPOCHS = 8
LR = 1e-4
NUM_WORKERS = 2  
OUTPUT_DIR = "outputs_consistency"
LAMBDA_CONSISTENCY = 1.0  # weight for the consistency loss term


def create_dataloaders(): # create the dataloaders for the training and development sets
    train_ds = HatefulMemesDataset(DATA_ROOT, split="train", transform=train_transform)
    dev_ds = HatefulMemesDataset(DATA_ROOT, split="dev", transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, dev_loader


def create_model(num_classes=2): # create the model
    # ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, data_loader, device): # evaluate the model
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, ids in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def build_id_to_path_map(train_loader): # build the map of image ids to image paths
     
    ds = train_loader.dataset
    mapping = {}
    for sample in ds.samples:
        mapping[int(sample["id"])] = sample["img_path"]
    return mapping


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, dev_loader = create_dataloaders()
    id_to_path = build_id_to_path_map(train_loader)

    model = create_model(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_dev_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_cons_loss = 0.0
        running_corrects = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, (images, labels, ids) in enumerate(train_loader):
            # images are already transformed tensors from the dataset
            images = images.to(device)
            labels = labels.to(device)

            # Build edited version for each image in the batch
            ids_list = [int(i) for i in ids]
            pil_clean_list = []
            for mid in ids_list:
                img_path = id_to_path[mid]
                pil_img = Image.open(img_path).convert("RGB")
                pil_clean_list.append(pil_img)

            # Apply a random in-distribution edit and transform
            edited_tensors = []
            for pil_img in pil_clean_list:
                edited_pil = random_train_edit(pil_img)
                edited_tensor = train_transform(edited_pil)
                edited_tensors.append(edited_tensor)
            edited_batch = torch.stack(edited_tensors).to(device)

            optimizer.zero_grad()

            # Forward passes
            outputs_clean = model(images)
            outputs_edit = model(edited_batch)

            # Classification loss on clean images
            ce_loss = criterion(outputs_clean, labels)

            # Consistency loss between clean and edited predictions
            cons_loss = F.mse_loss(outputs_edit, outputs_clean.detach())

            loss = ce_loss + LAMBDA_CONSISTENCY * cons_loss # calculate the total loss
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item() * images.size(0)
            running_ce_loss += ce_loss.item() * images.size(0)
            running_cons_loss += cons_loss.item() * images.size(0)

            _, preds = torch.max(outputs_clean, dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / total_samples
                avg_ce = running_ce_loss / total_samples
                avg_cons = running_cons_loss / total_samples
                avg_acc = running_corrects / total_samples
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.4f} "
                    f"(CE: {avg_ce:.4f}, Cons: {avg_cons:.4f}) "
                    f"Train Acc (clean): {avg_acc:.4f}"
                )

        epoch_loss = running_loss / total_samples
        epoch_ce = running_ce_loss / total_samples
        epoch_cons = running_cons_loss / total_samples
        epoch_acc = running_corrects / total_samples
        elapsed = time.time() - start_time

        dev_loss, dev_acc = evaluate(model, dev_loader, device)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"(CE: {epoch_ce:.4f}, Cons: {epoch_cons:.4f}) "
            f"Train Acc (clean): {epoch_acc:.4f} | "
            f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model by dev accuracy
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_path = os.path.join(OUTPUT_DIR, "consistency_resnet18_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >> New best dev acc: {best_dev_acc:.4f}. Saved model to {save_path}")

    print(f"Training finished. Best dev accuracy: {best_dev_acc:.4f}")


if __name__ == "__main__":
    train()
