import os
import math
import random
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Data / Augmentations
# ---------------------------------------------------------

class TwoCropsTransform:
    """
    Take an image and return two augmented views.
    """
    def __init__(self, image_size: int = 224):
        self.base_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class FolderMoCoDataset(Dataset):
    """
    Wrap ImageFolder dataset: returns two augmented crops of each image.
    """
    def __init__(self, folder_dataset, transform):
        self.ds = folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]   # ignore dummy label
        q, k = self.transform(img)
        return q, k


class FolderEvalDataset(Dataset):
    """Single transform for kNN sanity check"""
    def __init__(self, folder_dataset, image_size=224):
        self.ds = folder_dataset
        self.transform = T.Compose([
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        return self.transform(img)


# ---------------------------------------------------------
# MoCo-v3 ViT Model
# ---------------------------------------------------------

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class MoCoV3ViT(nn.Module):
    def __init__(self,
                 vit_name="vit_small_patch16_224",
                 proj_dim=256,
                 m=0.99,
                 tau=0.2):
        super().__init__()
        self.m = m
        self.tau = tau

        self.encoder_q = timm.create_model(
            vit_name, pretrained=False, num_classes=0, global_pool="token"
        )
        embed_dim = self.encoder_q.num_features
        self.proj_q = ProjectionMLP(embed_dim, 2048, proj_dim)

        # Momentum encoder
        self.encoder_k = timm.create_model(
            vit_name, pretrained=False, num_classes=0, global_pool="token"
        )
        self.proj_k = ProjectionMLP(embed_dim, 2048, proj_dim)

        # initialize momentum params
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        for q, k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data = k.data * self.m + q.data * (1 - self.m)

        for q, k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            k.data = k.data * self.m + q.data * (1 - self.m)

    def forward(self, im_q, im_k):
        q1 = self.proj_q(self.encoder_q(im_q))
        q2 = self.proj_q(self.encoder_q(im_k))

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = self.proj_k(self.encoder_k(im_q))
            k2 = self.proj_k(self.encoder_k(im_k))

        logits_12 = (q1 @ k2.T) / self.tau
        logits_21 = (q2 @ k1.T) / self.tau
        labels = torch.arange(im_q.size(0), device=im_q.device)

        return 0.5 * (F.cross_entropy(logits_12, labels) +
                      F.cross_entropy(logits_21, labels))

    @torch.no_grad()
    def encode(self, x):
        f = self.encoder_q(x)
        return F.normalize(f, dim=-1)


# ---------------------------------------------------------
# kNN Self-check
# ---------------------------------------------------------

@torch.no_grad()
def knn_self_accuracy(model, loader, device, max_samples=1024):
    feats = []
    for imgs in loader:
        imgs = imgs.to(device)
        feats.append(model.encode(imgs).cpu())
        if sum(f.size(0) for f in feats) >= max_samples:
            break

    feats = torch.cat(feats, dim=0)
    N = feats.size(0)
    sim = feats @ feats.T
    sim.fill_diagonal_(-1)
    nn_idx = sim.argmax(dim=1)
    labels = torch.arange(N)
    return ((nn_idx - labels).abs() <= 5).float().mean().item()


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset from folder:", args.data_dir)

    full_ds = ImageFolder(args.data_dir)
    print("Total images found:", len(full_ds))

    # Subsample
    if args.subset_size is not None:
        full_ds = torch.utils.data.Subset(full_ds, list(range(args.subset_size)))
        print("Using subset of size:", len(full_ds))

    # Train/val split
    n_total = len(full_ds)
    n_val = min(args.val_size, n_total // 10)
    n_train = n_total - n_val

    train_ds = torch.utils.data.Subset(full_ds, range(n_train))
    val_ds = torch.utils.data.Subset(full_ds, range(n_train, n_total))

    train_dataset = FolderMoCoDataset(train_ds, TwoCropsTransform(args.image_size))
    val_dataset = FolderEvalDataset(val_ds, args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    print("Building MoCoV3 ViT-S model…")
    model = MoCoV3ViT(
        vit_name=args.vit_name,
        proj_dim=args.proj_dim,
        m=args.momentum,
        tau=args.temperature
    ).to(device)

    optim = torch.optim.AdamW(
        list(model.encoder_q.parameters()) + list(model.proj_q.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        for im_q, im_k in pbar:
            im_q, im_k = im_q.to(device), im_k.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model(im_q, im_k)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        knn_acc = knn_self_accuracy(model, val_loader, device, args.knn_max_samples)

        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} | knn_acc={knn_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict()
        }, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"))


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./images",
                   help="Folder containing your extracted images")

    p.add_argument("--subset_size", type=int, default=100000)
    p.add_argument("--val_size", type=int, default=2000)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--vit_name", type=str, default="vit_small_patch16_224")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--temperature", type=float, default=0.2)

    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--knn_max_samples", type=int, default=1024)

    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)