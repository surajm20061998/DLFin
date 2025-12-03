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
from datasets import load_dataset
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
    This is the SimCLR / MoCo style "siamese" transform.
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


class HFMoCoDataset(Dataset):
    """
    Wrap a HuggingFace dataset: returns two augmented crops of each image.
    """

    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        img = sample["image"]
        q, k = self.transform(img)
        return q, k


class HFEvalDataset(Dataset):
    """
    Eval dataset: single deterministic transform, used for kNN sanity checks.
    """

    def __init__(self, hf_dataset, image_size: int = 224):
        self.ds = hf_dataset
        self.transform = T.Compose([
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img = self.ds[idx]["image"]
        img = self.transform(img)
        return img


# ---------------------------------------------------------
# MoCo v3–style ViT model
# ---------------------------------------------------------

class ProjectionMLP(nn.Module):
    """
    Simple 3-layer projection head as used in many SSL methods.
    """

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
        x = F.normalize(x, dim=-1)
        return x


class MoCoV3ViT(nn.Module):
    """
    MoCo v3–style ViT-S:
    - Online encoder (encoder_q + proj_q)
    - Momentum encoder (encoder_k + proj_k, EMA of online)
    - InfoNCE loss using only current batch (no queue), symmetric.
    """

    def __init__(self,
                 vit_name: str = "vit_small_patch16_224",
                 proj_dim: int = 256,
                 m: float = 0.99,
                 tau: float = 0.2):
        super().__init__()
        self.m = m
        self.tau = tau

        # Online encoder
        self.encoder_q = timm.create_model(
            vit_name, pretrained=False, num_classes=0, global_pool="token"
        )

        embed_dim = self.encoder_q.num_features

        self.proj_q = ProjectionMLP(embed_dim, hidden_dim=2048, out_dim=proj_dim)

        # Momentum encoder (initialized as copy)
        self.encoder_k = timm.create_model(
            vit_name, pretrained=False, num_classes=0, global_pool="token"
        )
        self.proj_k = ProjectionMLP(embed_dim, hidden_dim=2048, out_dim=proj_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the momentum encoder (encoder_k & proj_k)
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> torch.Tensor:
        """
        Compute MoCo v3 symmetric loss given two crops.
        im_q: (B, 3, H, W)
        im_k: (B, 3, H, W)
        """
        # Online branch
        q1 = self.encoder_q(im_q)  # (B, D)
        q1 = self.proj_q(q1)       # (B, C)
        q2 = self.encoder_q(im_k)
        q2 = self.proj_q(q2)

        # Momentum branch - no grad
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = self.encoder_k(im_q)
            k1 = self.proj_k(k1)
            k2 = self.encoder_k(im_k)
            k2 = self.proj_k(k2)

        # Compute InfoNCE logits
        # q1 vs k2
        logits_12 = (q1 @ k2.T) / self.tau
        # q2 vs k1
        logits_21 = (q2 @ k1.T) / self.tau

        labels = torch.arange(im_q.size(0), device=im_q.device)

        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)

        loss = (loss_12 + loss_21) * 0.5
        return loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction with the online encoder (for eval).
        """
        feat = self.encoder_q(x)
        feat = F.normalize(feat, dim=-1)
        return feat


# ---------------------------------------------------------
# k-NN sanity check (unsupervised)
# ---------------------------------------------------------

@torch.no_grad()
def knn_self_accuracy(model: MoCoV3ViT,
                      loader: DataLoader,
                      device: torch.device,
                      max_samples: int = 1024) -> float:
    """
    Simple unsupervised sanity metric:
    - Extract features for a subset of images
    - Compute cosine sim matrix
    - For each row, check if highest-sim index (excluding self) is near-diagonal
      (here: "self-consistency")
    This is NOT a real downstream metric, just a collapse check.
    """
    model.eval()
    feats = []

    for imgs in loader:
        imgs = imgs.to(device)
        f = model.encode(imgs)
        feats.append(f.cpu())
        if sum(x.size(0) for x in feats) >= max_samples:
            break

    feats = torch.cat(feats, dim=0)  # (N, D)
    N = feats.size(0)

    sim = feats @ feats.T   # cosine since already normalized
    # Ignore diagonal
    sim.fill_diagonal_(-1.0)
    # Nearest neighbor index
    nn_idx = sim.argmax(dim=1)

    # If model is not collapsed, we expect non-trivial structure (NN index != random).
    # As a dummy "accuracy", count how often NN index is within +/- 5 positions.
    labels = torch.arange(N)
    acc = ((nn_idx - labels).abs() <= 5).float().mean().item()
    return acc


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading HF dataset: {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, split="train")

    # Subsample if needed
    if args.subset_size is not None and len(ds) > args.subset_size:
        ds = ds.shuffle(seed=args.seed).select(range(args.subset_size))
        print(f"Using subset of size {len(ds)}")

    # Train / val split for sanity eval
    n_total = len(ds)
    n_val = min(args.val_size, n_total // 10)
    n_train = n_total - n_val
    ds_train = ds.select(range(n_train))
    ds_val = ds.select(range(n_train, n_train + n_val))

    train_transform = TwoCropsTransform(image_size=args.image_size)
    train_dataset = HFMoCoDataset(ds_train, train_transform)

    val_dataset = HFEvalDataset(ds_val, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("Building MoCo v3 ViT-S model...")
    model = MoCoV3ViT(
        vit_name=args.vit_name,
        proj_dim=args.proj_dim,
        m=args.momentum,
        tau=args.temperature
    ).to(device)

    # Optimizer & LR
    optim = torch.optim.AdamW(
        model.encoder_q.parameters(),  # backbone
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # You can optionally add proj_q params to optimizer too:
    optim.add_param_group({"params": model.proj_q.parameters()})

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for (im_q, im_k) in pbar:
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model(im_q, im_k)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)

        # Simple k-NN sanity metric
        knn_acc = knn_self_accuracy(model, val_loader, device, max_samples=args.knn_max_samples)

        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} | knn_self_acc={knn_acc:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scaler_state": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)

    print("Training finished.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MoCo v3-style ViT-S on HF images")

    p.add_argument("--hf_dataset", type=str, default="sm12377/trImgs",
                   help="HuggingFace dataset name")
    p.add_argument("--subset_size", type=int, default=100000,
                   help="Max number of training samples to use (None for all)")
    p.add_argument("--val_size", type=int, default=2000,
                   help="Validation subset size for kNN sanity eval")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--vit_name", type=str, default="vit_small_patch16_224",
                   help="timm ViT backbone name")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--momentum", type=float, default=0.99,
                   help="EMA momentum for key encoder")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="Softmax temperature for contrastive logits")

    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--amp", action="store_true", help="Use mixed precision")

    p.add_argument("--output_dir", type=str, default="./checkpoints_mocov3")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)