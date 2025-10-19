#!/usr/bin/env python3
import os, json, math, argparse, random, re
from pathlib import Path
from typing import List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------- Basics -----------------------
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Determinism (you may switch for speed on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------- Class Scanning & Dataset -----------------------
CANON = ["Clear","Rain","Snow","Fog","Sand"]

def normalize_class(name: str) -> str | None:
    n = name.lower()
    if re.search(r"(clear|sunny)", n): return "Clear"
    if re.search(r"(rain|rainy|storm|drizzle|thunder)", n): return "Rain"
    if re.search(r"(snow|snowy|sleet|blizzard)", n): return "Snow"
    if re.search(r"(fog|haze|mist|smog)", n): return "Fog"
    if re.search(r"(sand|dust|sandstorm|duststorm)", n): return "Sand"
    return None

def list_leaf_class_folders(root: str):
    """Yield (leaf_path, canonical_label) for layouts:
       root/<class>/*  or  root/*/<class>/*"""
    found = []
    p = Path(root)
    if not p.exists(): return found
    # level-1
    for d in sorted([x for x in p.iterdir() if x.is_dir()]):
        c = normalize_class(d.name)
        if c: found.append((d, c))
    if not found:
        # level-2
        for mid in sorted([x for x in p.iterdir() if x.is_dir()]):
            for d2 in sorted([x for x in mid.iterdir() if x.is_dir()]):
                c = normalize_class(d2.name)
                if c: found.append((d2, c))
    return found

def union_classes(roots: List[str]) -> List[str]:
    seen = set()
    for r in roots:
        for _, c in list_leaf_class_folders(r):
            seen.add(c)
    classes = [c for c in CANON if c in seen]
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes across sources. Check folder names (Clear/Rain/Snow/Fog/Sand).")
    return classes

def build_source_dataset(roots: List[str], classes: List[str], tf, max_per_class: int = 0):
    """Concat sources by scanning leaf class folders and loading their images directly.
       Each leaf folder (e.g., .../DAWN/Fog) yields samples with a GLOBAL class label."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    datasets_list = []

    # allowed file extensions
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.jfif'}

    def list_images(root: Path):
        return [str(p) for p in root.rglob('*') if p.suffix.lower() in exts]

    class LeafFolderDataset(torch.utils.data.Dataset):
        def __init__(self, files, y_idx, canonical, transform):
            self.files = files
            self.y = int(y_idx)
            self.canonical_class = canonical
            self.transform = transform
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            fp = self.files[i]
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, self.y

    for r in roots:
        leaves = list_leaf_class_folders(r)
        if len(leaves) == 0:
            print(f"[WARN] No class folders under: {r}")
            continue
        for leaf, canon in leaves:
            y_idx = class_to_idx[canon]
            files = list_images(Path(leaf))
            if len(files) == 0:
                print(f"[WARN] No images in leaf: {leaf}")
                continue
            if max_per_class > 0 and len(files) > max_per_class:
                files = files[:max_per_class]
            ds = LeafFolderDataset(files, y_idx, canon, tf)
            datasets_list.append(ds)

    if len(datasets_list) == 0:
        raise RuntimeError("No valid source data found after scanning.")

    class Wrapped(torch.utils.data.Dataset):
        def __init__(self, subs, classes):
            self.subs = subs
            self.classes = classes
            self.cum = np.cumsum([len(s) for s in subs]).tolist()
            self.len = self.cum[-1]
        def __len__(self): return self.len
        def __getitem__(self, i):
            j = np.searchsorted(self.cum, i, side="right")
            i0 = i - (self.cum[j-1] if j>0 else 0)
            return self.subs[j][i0]

    return Wrapped(datasets_list, classes)

def collect_targets(ds) -> np.ndarray:
    """Return numeric labels for a (possibly wrapped/subset) dataset."""
    t = []
    for i in range(len(ds)):
        _, y = ds[i]
        t.append(int(y))
    return np.array(t, dtype=np.int64)

# ----------------------- Transforms -----------------------
def make_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    to_rgb = transforms.Lambda(lambda im: im.convert("RGB"))
    train_tf = transforms.Compose([
        to_rgb,
        transforms.Resize(256),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        to_rgb,
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf

# ----------------------- Models -----------------------
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*(list(m.children())[:-1]))  # up to avgpool
        self.out_dim = 2048
    def forward(self, x): return torch.flatten(self.features(x), 1)

class EfficientNetB0Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 1280
    def forward(self, x):
        z = self.features(x); z = self.pool(z); return torch.flatten(z, 1)

class DenseNet121Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 1024
    def forward(self, x):
        z = self.features(x); z = F.relu(z, inplace=True); z = self.pool(z); return torch.flatten(z, 1)

def make_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet50":          enc = ResNet50Encoder(pretrained)
    elif name == "efficientnet_b0": enc = EfficientNetB0Encoder(pretrained)
    elif name == "densenet121":     enc = DenseNet121Encoder(pretrained)
    else: raise ValueError(f"Unsupported backbone: {name}")
    return enc, enc.out_dim

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__(); self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, z): return self.fc(z)

# ----------------------- Training -----------------------
def evaluate_acc(device, Ms, C, loader):
    Ms.eval(); C.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = C(Ms(x))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return (correct/total) if total>0 else 0.0

def train_source(device, Ms, C, loader_tr, loader_val, epochs, lr, class_weights_ce=None):
    Ms.train(); C.train()
    opt = torch.optim.Adam(list(Ms.parameters()) + list(C.parameters()), lr=lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss(weight=(class_weights_ce.to(device) if class_weights_ce is not None else None))

    best_acc = -1.0
    best = {"Ms": None, "C": None}
    history = []

    for ep in range(1, epochs+1):
        Ms.train(); C.train()
        losses=[]
        pbar = tqdm(loader_tr, desc=f"[SRC] epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = C(Ms(x))
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_acc = evaluate_acc(device, Ms, C, loader_val) if loader_val is not None else 0.0
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_acc": float(val_acc)})
        if val_acc >= best_acc:
            best_acc = val_acc
            best["Ms"] = {k: v.cpu() for k, v in Ms.state_dict().items()}
            best["C"]  = {k: v.cpu() for k, v in C.state_dict().items()}
        print(f"[VAL] acc={val_acc:.4f}")

    if best["Ms"] is not None:
        Ms.load_state_dict(best["Ms"]); C.load_state_dict(best["C"])
    return Ms, C, history

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser("Source Pretraining for WA-ADDA (Step 1 only)")
    ap.add_argument("--sources", nargs="+", required=True, help="List of non–street-view source roots (class-per-folder).")
    ap.add_argument("--backbone", choices=["resnet50","efficientnet_b0","densenet121"], default="resnet50")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--pretrained_imagenet", action="store_true")
    ap.add_argument("--balance_source_sampler", action="store_true",
                    help="Use WeightedRandomSampler for class-balanced batches.")
    ap.add_argument("--class_weighted_ce", action="store_true",
                    help="Use class-weighted cross-entropy.")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Cap max samples per class across sources (0 disables).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="runs/source_pretrain")
    ap.add_argument("--scan_only", action="store_true",
                    help="Scan sources and print detected classes & counts, then exit.")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # classes across all sources
    classes = union_classes(args.sources)
    print(f"[INFO] classes ({len(classes)}): {classes}")

    # transforms & datasets
    train_tf, val_tf = make_transforms(args.img_size)
    src_ds_train_tf = build_source_dataset(args.sources, classes, train_tf, max_per_class=args.max_per_class)
    src_ds_val_tf   = build_source_dataset(args.sources, classes, val_tf,   max_per_class=args.max_per_class)

    # optional scan-only mode
    if args.scan_only:
        counts = [0]*len(classes)
        for _, y in DataLoader(src_ds_train_tf, batch_size=512, shuffle=False, num_workers=0):
            for v in y.tolist(): counts[v] += 1
        print("[SCAN] counts (train transforms):")
        for i,c in enumerate(classes): print(f"  {c}: {counts[i]}")
        return

    # ---- split indices ONCE, then mirror to both TF variants
    full_len = len(src_ds_train_tf)
    n_val = int(math.ceil(full_len * args.val_split))
    n_tr  = full_len - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(full_len, generator=g).tolist()
    idx_tr  = perm[:n_tr]
    idx_val = perm[n_tr:]
    src_train = Subset(src_ds_train_tf, idx_tr)   # uses train_tf
    src_val   = Subset(src_ds_val_tf,   idx_val)  # uses val_tf

    # sampler (optional) + class-weighted CE
    sampler = None
    class_weights_ce = None
    if args.balance_source_sampler or args.class_weighted_ce:
        targets_tr = collect_targets(src_train)
        counts = np.bincount(targets_tr, minlength=len(classes)).astype(np.float64)
        print(f"[INFO] source class counts (train split): {counts.tolist()}")
        if args.balance_source_sampler:
            inv = 1.0 / np.where(counts==0, 1.0, counts)
            weights = inv[targets_tr]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                            num_samples=len(targets_tr), replacement=True)
        if args.class_weighted_ce:
            invc = 1.0 / np.where(counts==0, 1.0, counts)
            class_weights_ce = torch.tensor(invc, dtype=torch.float32)

    # dataloaders
    loader_tr  = DataLoader(src_train, batch_size=args.batch_size,
                            shuffle=(sampler is None), sampler=sampler,
                            num_workers=args.workers, pin_memory=True)
    loader_val = DataLoader(src_val,   batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # models
    Ms, feat_dim = make_backbone(args.backbone, pretrained=args.pretrained_imagenet)
    C = ClassifierHead(feat_dim, len(classes))
    Ms, C = Ms.to(device), C.to(device)

    # train
    Ms, C, history = train_source(device, Ms, C, loader_tr, loader_val,
                                  epochs=args.epochs, lr=args.lr,
                                  class_weights_ce=class_weights_ce)

    # save artifacts
    torch.save(Ms.state_dict(), os.path.join(args.out, "Ms_src.pth"))
    torch.save(C.state_dict(),  os.path.join(args.out, "C_src.pth"))
    with open(os.path.join(args.out, "labelmap.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] Saved to {args.out}")
    print(" Next: use Ms_src.pth and C_src.pth for WA-ADDA adaptation (Step 2).")

if __name__ == "__main__":
    main()
