#!/usr/bin/env python3
import argparse, json, os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

CANON = ["Clear","Rain","Snow","Fog","Sand"]

# ---------- backbones ----------
class ResNet50Encoder(nn.Module):
    def __init__(self): super().__init__(); m=models.resnet50(weights=None); self.features=nn.Sequential(*(list(m.children())[:-1])); self.out_dim=2048
    def forward(self,x): return torch.flatten(self.features(x),1)

class EfficientNetB0Encoder(nn.Module):
    def __init__(self): super().__init__(); m=models.efficientnet_b0(weights=None); self.features=m.features; self.pool=nn.AdaptiveAvgPool2d((1,1)); self.out_dim=1280
    def forward(self,x): return torch.flatten(self.pool(self.features(x)),1)

class DenseNet121Encoder(nn.Module):
    def __init__(self): super().__init__(); m=models.densenet121(weights=None); self.features=m.features; self.pool=nn.AdaptiveAvgPool2d((1,1)); self.out_dim=1024
    def forward(self,x): z=self.features(x); z=F.relu(z,inplace=True); z=self.pool(z); return torch.flatten(z,1)

def make_backbone(name:str):
    n=name.lower()
    if n=="resnet50": enc=ResNet50Encoder()
    elif n=="efficientnet_b0": enc=EfficientNetB0Encoder()
    elif n=="densenet121": enc=DenseNet121Encoder()
    else: raise ValueError(f"unknown backbone {name}")
    return enc, enc.out_dim

class ClassifierHead(nn.Module):
    def __init__(self,in_dim,num_classes): super().__init__(); self.fc=nn.Linear(in_dim,num_classes)
    def forward(self,z): return self.fc(z)

# ---------- dataset (force canonical order) ----------
def make_eval_loader(root, img_size=224, batch_size=128, workers=8):
    tf = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root, transform=tf)
    # enforce class names to be exactly our CANON (any missing class will raise)
    if sorted(ds.classes) != sorted(CANON):
        raise RuntimeError(f"Eval set classes {ds.classes} != expected {CANON}. "
                           f"Folders under {root} must be {CANON}.")
    # remap targets to CANON indices
    name2idx = {name:i for i,name in enumerate(ds.classes)}
    map2canon = {name2idx[c]: CANON.index(c) for c in ds.classes}
    class Wrap(torch.utils.data.Dataset):
        def __init__(self, base): self.base=base
        def __len__(self): return len(self.base)
        def __getitem__(self,i):
            x,y = self.base[i]
            return x, map2canon[y]
    dl = torch.utils.data.DataLoader(Wrap(ds), batch_size=batch_size, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    return dl

@torch.no_grad()
def eval_per_class(device, enc, clf, loader):
    enc.eval(); clf.eval()
    nC=len(CANON)
    cm = np.zeros((nC,nC), dtype=int)  # rows=true, cols=pred
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = clf(enc(x))
        pred = logits.argmax(1)
        for t,p in zip(y.cpu().tolist(), pred.cpu().tolist()):
            cm[t,p]+=1
    # per-class recall (accuracy per true class)
    per_acc = []
    for c in range(nC):
        tot = cm[c].sum()
        acc = (cm[c,c]/tot) if tot>0 else 0.0
        per_acc.append(acc)
    overall = np.trace(cm)/np.sum(cm)
    macro = float(np.mean(per_acc))
    return cm, per_acc, overall, macro

def main():
    ap = argparse.ArgumentParser("Per-class eval on street_view_test")
    ap.add_argument("--backbone", required=True, choices=["resnet50","efficientnet_b0","densenet121"])
    ap.add_argument("--ckpt_mt", required=True, help="Path to Mt_best.pth (or Ms_src.pth for baseline)")
    ap.add_argument("--ckpt_c",  required=True, help="Path to C_tgt.pth (or C_src.pth for baseline)")
    ap.add_argument("--eval_root", required=True, help="data/target/street_view_test")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dim = make_backbone(args.backbone)
    clf = ClassifierHead(dim, len(CANON))

    enc.load_state_dict(torch.load(args.ckpt_mt, map_location="cpu"))
    clf.load_state_dict(torch.load(args.ckpt_c,  map_location="cpu"))
    enc, clf = enc.to(device), clf.to(device)

    loader = make_eval_loader(args.eval_root, img_size=args.img_size,
                              batch_size=args.batch_size, workers=args.workers)
    cm, per_acc, overall, macro = eval_per_class(device, enc, clf, loader)

    print("\nPer-class accuracy (recall by true class):")
    for c,acc in zip(CANON, per_acc):
        print(f"  {c:>5}: {acc*100:.2f}%")
    print(f"\nOverall Acc: {overall*100:.2f}%   Macro: {macro*100:.2f}%")
    print("\nConfusion matrix (rows=true, cols=pred, class order:", CANON, ")")
    print(cm)

    if args.out_csv:
        import csv
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w=csv.writer(f)
            w.writerow(["Class","Accuracy"])
            for c,acc in zip(CANON, per_acc): w.writerow([c, f"{acc:.6f}"])
            w.writerow(["Overall", f"{overall:.6f}"])
            w.writerow(["Macro",   f"{macro:.6f}"])
        print(f"\nSaved per-class CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
