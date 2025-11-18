
import os
import torch
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import torch.nn.functional as F

from dataset import HatefulMemesDataset, eval_transform
from edits import (
    minor_crop,
    jpeg_recompress,
    color_adjust,
    blur,
    add_border,
    add_emoji_overlay,
)

# ==== CONFIG ====
DATA_ROOT = r"C:\Users\resmi\Downloads\Hateful_Meme_Detection\meme_robustness_data\data"
BATCH_SIZE = 32
NUM_WORKERS = 2

BASELINE_CKPT = os.path.join("outputs_baseline", "baseline_resnet18_best.pth")
CONSIST_CKPT = os.path.join("outputs_consistency", "consistency_resnet18_best.pth")


def create_model(num_classes: int = 2) -> torch.nn.Module:
    """Create a ResNet-18 model with a 2-class output head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module: # load the model weights from the checkpoint path
    model = create_model(num_classes=2)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_id_to_path_map(dataset: HatefulMemesDataset): # build the map of image ids to image paths  
    mapping = {}
    for sample in dataset.samples:
        mapping[int(sample["id"])] = sample["img_path"]
    return mapping


def compute_ece(logits: torch.Tensor,
                labels: torch.Tensor,
                n_bins: int = 10) -> float: # compute the expected calibration error   
    probs = F.softmax(logits, dim=1)
    confidences, preds = torch.max(probs, dim=1)
    labels = labels.to(preds.device)
    correct = (preds == labels).float()

    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        # include lower, exclude upper (except last bin)
        if i < n_bins - 1:
            mask = (confidences >= lower) & (confidences < upper)
        else:
            mask = (confidences >= lower) & (confidences <= upper)

        if mask.sum() == 0:
            continue

        conf_avg = confidences[mask].mean()
        acc_avg = correct[mask].mean()
        bin_frac = mask.float().mean()  # proportion of samples in this bin

        ece += (bin_frac * torch.abs(acc_avg - conf_avg))

    return float(ece.item())


@torch.no_grad()
def eval_clean_with_ece(model: torch.nn.Module,
                        data_loader: DataLoader,
                        device: torch.device): # evaluate the accuracy and the expected calibration error on the clean development images
    all_logits = []
    all_labels = []
    correct = 0
    total = 0
    for images, labels, ids in data_loader:
        images = images.to(device)
        labels_dev = labels.to(device)
        outputs = model(images)
        all_logits.append(outputs.cpu())
        all_labels.append(labels_dev.cpu())
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels_dev).sum().item()
        total += labels_dev.size(0)

    if total == 0:
        return 0.0, 0.0

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc = correct / total
    ece = compute_ece(logits_cat, labels_cat, n_bins=10)
    return acc, ece


@torch.no_grad()
def eval_with_edit(model: torch.nn.Module,
                   data_loader: DataLoader,
                   id_to_path,
                   edit_fn,
                   device: torch.device) -> float: 
    # evaluate the accuracy when each development image is transformed by the edit function
    correct = 0
    total = 0
    for _, labels, ids in data_loader:
        labels_dev = labels.to(device)
        edited_tensors = []
        ids_list = [int(i) for i in ids]
        for mid in ids_list:
            img_path = id_to_path[mid]
            pil_img = Image.open(img_path).convert("RGB")
            edited_pil = edit_fn(pil_img)
            edited_tensor = eval_transform(edited_pil)
            edited_tensors.append(edited_tensor)
        edited_batch = torch.stack(edited_tensors).to(device)

        outputs = model(edited_batch)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels_dev).sum().item()
        total += labels_dev.size(0)
    return correct / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load dev dataset
    dev_ds = HatefulMemesDataset(DATA_ROOT, split="dev", transform=eval_transform)
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    id_to_path = build_id_to_path_map(dev_ds)

    # Load models
    print("Loading models...")
    baseline = load_model(BASELINE_CKPT, device)
    consist = load_model(CONSIST_CKPT, device)

    # Define edits
    edits = {
        "clean": None,
        "minor_crop": minor_crop,
        "jpeg_recompress": jpeg_recompress,
        "color_adjust": color_adjust,
        "blur": blur,
        "add_border": add_border,
        "add_emoji_overlay": add_emoji_overlay,
    }

    # --- Compute clean accuracy + ECE ---
    print("=== Clean dev performance (accuracy + ECE) ===")
    base_clean_acc, base_clean_ece = eval_clean_with_ece(baseline, dev_loader, device)
    cons_clean_acc, cons_clean_ece = eval_clean_with_ece(consist, dev_loader, device)

    print(
        f"Baseline   | Acc: {base_clean_acc:.4f}, ECE: {base_clean_ece:.4f}\n"
        f"Consistency| Acc: {cons_clean_acc:.4f}, ECE: {cons_clean_ece:.4f}\n"
    )

    # --- Compute per-edit accuracies (robustness) ---
    print("=== Accuracy on dev set (per edit) ===")
    results = {}

    # Clean row
    results["clean"] = {"baseline": base_clean_acc, "consistency": cons_clean_acc}
    print(f"{'clean':15s} | Baseline: {base_clean_acc:.4f} | Consistency: {cons_clean_acc:.4f}")

    # Edited versions
    for name, fn in edits.items():
        if name == "clean":
            continue
        base_acc = eval_with_edit(baseline, dev_loader, id_to_path, fn, device)
        cons_acc = eval_with_edit(consist, dev_loader, id_to_path, fn, device)
        results[name] = {"baseline": base_acc, "consistency": cons_acc}
        print(f"{name:15s} | Baseline: {base_acc:.4f} | Consistency: {cons_acc:.4f}")

    print("\nFinished evaluation.")
    print("You can copy these numbers into Excel or another tool to create bar plots.")


if __name__ == "__main__":
    main()
