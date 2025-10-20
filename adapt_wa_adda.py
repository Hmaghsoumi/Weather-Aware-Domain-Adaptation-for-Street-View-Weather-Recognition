#!/usr/bin/env python3
# WA-ADDA: Weather-Aware ADDA (Step 2: target adaptation, final stable)

import os, json, argparse, random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------- Constants -----------------------
CANON = ["Clear", "Rain", "Snow", "Fog", "Sand"]
EXTS  = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp','.gif','.jfif'}

# ----------------------- Utils -----------------------
def set_seed(seed: int = 1337):
    import torch.backends.cudnn as cudnn
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def make_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    to_rgb = transforms.Lambda(lambda im: im.convert("RGB"))
    weak = transforms.Compose([
        to_rgb, transforms.Resize(256),
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    strong = transforms.Compose([
        to_rgb, transforms.Resize(256),
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.ColorJitter(0.2,0.3),
        transforms.GaussianBlur(3, sigma=(0.1,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    eval_tf = transforms.Compose([
        to_rgb, transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    return weak, strong, eval_tf

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

# ----------------------- Backbones & Heads -----------------------
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*(list(m.children())[:-1]))
        self.out_dim = 2048
    def forward(self, x): return torch.flatten(self.features(x), 1)

class EfficientNetB0Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 1280
    def forward(self, x): return torch.flatten(self.pool(self.features(x)), 1)

class DenseNet121Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 1024
    def forward(self, x):
        z = self.features(x); z = F.relu(z, inplace=True); z = self.pool(z)
        return torch.flatten(z, 1)

def make_backbone(name: str):
    name = name.lower()
    if name == "resnet50":          enc = ResNet50Encoder(False)
    elif name == "efficientnet_b0": enc = EfficientNetB0Encoder(False)
    elif name == "densenet121":     enc = DenseNet121Encoder(False)
    else: raise ValueError(f"Unsupported backbone: {name}")
    return enc, enc.out_dim

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes): super().__init__(); self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, z): return self.fc(z)

class Discriminator(nn.Module):
    """Conditional discriminator: input concat([z, p]) where p = softmax(C(z))."""
    def __init__(self, in_dim, num_classes, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden//2), nn.ReLU(True),
            nn.Linear(hidden//2, 1)  # logits
        )
    def forward(self, z, p): return self.net(torch.cat([z, p], dim=1))

# ----------------------- Datasets -----------------------
def list_images(root: Path):
    return [str(p) for p in root.rglob('*') if p.suffix.lower() in EXTS]

class UnlabeledFolder(torch.utils.data.Dataset):
    def __init__(self, root, tf):
        self.files = list_images(Path(root))
        if len(self.files)==0: raise RuntimeError(f"No images found in {root}")
        self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = Image.open(self.files[i]).convert("RGB")
        return self.tf(x)

class UnlabeledMultiRoot(torch.utils.data.Dataset):
    def __init__(self, roots, tf):
        self.files = []
        for r in roots: self.files += list_images(Path(r))
        if len(self.files)==0: raise RuntimeError(f"No images found in sources: {roots}")
        self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = Image.open(self.files[i]).convert("RGB")
        return self.tf(x)

class LabeledFolder(torch.utils.data.Dataset):
    def __init__(self, root, tf):
        self.ds = datasets.ImageFolder(root, transform=tf)
        for c in self.ds.classes:
            if c not in CANON:
                raise RuntimeError(f"Found class '{c}' in eval set; expected one of {CANON}")
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y_local = self.ds[i]
        name = self.ds.classes[y_local]
        y = CANON.index(name)
        return x, y

# ----------------------- Loss helpers -----------------------
def entropy_minimization(logits):
    p = F.softmax(logits, dim=1).clamp_min(1e-8)
    return -(p * torch.log(p)).sum(dim=1).mean()

def consistency_kl(logits_w, logits_s):
    pw = F.softmax(logits_w, dim=1).clamp_min(1e-8)
    log_ps = F.log_softmax(logits_s, dim=1)
    return F.kl_div(log_ps, pw, reduction="batchmean")

# ----------------------- Evaluation -----------------------
@torch.no_grad()
def evaluate(device, Mt, C, loader):
    Mt.eval(); C.eval()
    correct = 0; tot = 0
    per = {i: {"tp":0,"tot":0} for i in range(len(CANON))}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = C(Mt(x)).argmax(1)
        correct += (pred==y).sum().item()
        tot += y.numel()
        for i in range(len(CANON)):
            m = (y==i)
            per[i]["tp"] += (pred[m]==i).sum().item()
            per[i]["tot"]+= m.sum().item()
    acc = correct / max(1, tot)
    macro = float(np.mean([per[i]["tp"]/per[i]["tot"] if per[i]["tot"]>0 else 0.0 for i in range(len(CANON))]))
    return acc, macro

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser("WA-ADDA Target Adaptation (final stable)")
    ap.add_argument("--backbone", choices=["resnet50","efficientnet_b0","densenet121"], required=True)
    ap.add_argument("--src_run", type=str, required=True, help="Folder with Ms_src.pth, C_src.pth, labelmap.json")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Source roots (same non–street-view folders used in Step 1).")
    ap.add_argument("--target_unlabeled", type=str, required=True, help="Target unlabeled images root")
    ap.add_argument("--target_eval", type=str, required=True, help="Target labeled test root (evaluation only)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs_tgt", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_mt", type=float, default=5e-5)
    ap.add_argument("--lr_d", type=float, default=5e-5)
    ap.add_argument("--lambda_adv", type=float, default=0.2)
    ap.add_argument("--lambda_ent", type=float, default=0.0)
    ap.add_argument("--lambda_cons", type=float, default=0.0)
    ap.add_argument("--adv_warmup", type=int, default=3, help="epochs with lambda_adv=0; also skip D updates")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="runs/wa_adda_stable")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = ap.parse_args()

    set_seed(args.seed); ensure_dir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ---------- load source encoder/classifier ----------
    with open(os.path.join(args.src_run, "labelmap.json"), "r") as f:
        lm = json.load(f)
    src_classes = lm.get("classes", CANON)
    if set(src_classes) != set(CANON):
        raise RuntimeError(f"Source classes {src_classes} != expected {CANON}")

    Ms, feat_dim = make_backbone(args.backbone)
    C  = ClassifierHead(feat_dim, len(CANON))
    Ms.load_state_dict(torch.load(os.path.join(args.src_run, "Ms_src.pth"), map_location="cpu"))
    C.load_state_dict(torch.load(os.path.join(args.src_run, "C_src.pth"),  map_location="cpu"))
    Ms, C = Ms.to(device), C.to(device)
    Ms.eval(); C.eval()  # frozen

    # init Mt from Ms and freeze BN running stats
    Mt, _ = make_backbone(args.backbone)
    Mt.load_state_dict(Ms.state_dict()); Mt = Mt.to(device)
    Mt.apply(set_bn_eval)

    D = Discriminator(feat_dim, len(CANON)).to(device)
    bce = nn.BCEWithLogitsLoss()

    # ---------- data ----------
    weak_tf, strong_tf, eval_tf = make_transforms(args.img_size)

    # target: paired weak/strong views
    tgt_ds_w = UnlabeledFolder(args.target_unlabeled, weak_tf)
    tgt_ds_s = UnlabeledFolder(args.target_unlabeled, strong_tf)
    class Pair(torch.utils.data.Dataset):
        def __init__(self, A, B): self.A, self.B = A, B
        def __len__(self): return len(self.A)
        def __getitem__(self, i): return self.A[i], self.B[i]
    tgt_pair = Pair(tgt_ds_w, tgt_ds_s)
    tgt_loader = DataLoader(tgt_pair, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True)

    # source: single-view weak/eval aug (labels unused)
    src_ds = UnlabeledMultiRoot(args.sources, transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize(256), transforms.CenterCrop(args.img_size),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]))
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True)
    src_iter = iter(src_loader)

    eval_loader = DataLoader(LabeledFolder(args.target_eval, eval_tf),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # ---------- optim & AMP ----------
    opt_mt = torch.optim.Adam(Mt.parameters(), lr=args.lr_mt, weight_decay=1e-4)
    opt_d  = torch.optim.Adam(D.parameters(),  lr=args.lr_d,  weight_decay=1e-4)
    use_amp = args.amp
    if use_amp:
        from torch.cuda.amp import autocast, GradScaler
        sc_mt = GradScaler()
        sc_d  = GradScaler()

    # ---------- training ----------
    best_macro = -1.0
    for ep in range(1, args.epochs_tgt+1):
        Mt.train(); D.train()
        lam_adv_eff = 0.0 if ep <= args.adv_warmup else args.lambda_adv

        pbar = tqdm(tgt_loader, desc=f"[ADAPT] epoch {ep}/{args.epochs_tgt}")
        for step, (xw, xs) in enumerate(pbar):
            xw = xw.to(device); xs = xs.to(device)

            # pull a SOURCE batch
            try:
                x_src = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                x_src = next(src_iter)
            x_src = x_src.to(device)

            # ---- 1) Train D: SOURCE(Ms) vs TARGET(Mt) ----
            with torch.no_grad():
                z_s = Ms(x_src); p_s = F.softmax(C(z_s), dim=1)
                z_t = Mt(xw);    p_t = F.softmax(C(z_t), dim=1)

            # label smoothing
            y_s = torch.full((z_s.size(0), 1), 0.9, device=device)
            y_t = torch.full((z_t.size(0), 1), 0.1, device=device)

            opt_d.zero_grad(set_to_none=True)
            if ep > args.adv_warmup:
                if use_amp:
                    with autocast():
                        loss_d = bce(D(z_s, p_s), y_s) + bce(D(z_t, p_t), y_t)
                    sc_d.scale(loss_d).backward()
                    sc_d.step(opt_d); sc_d.update()
                else:
                    loss_d = bce(D(z_s, p_s), y_s) + bce(D(z_t, p_t), y_t)
                    loss_d.backward(); opt_d.step()
            else:
                # compute for logging only
                if use_amp:
                    with autocast():
                        loss_d = bce(D(z_s, p_s), y_s) + bce(D(z_t, p_t), y_t)
                else:
                    loss_d = bce(D(z_s, p_s), y_s) + bce(D(z_t, p_t), y_t)
                loss_d = loss_d.detach()

            # ---- 2) Train Mt: confidence-masked adversarial + optional regs ----
            opt_mt.zero_grad(set_to_none=True)

            def masked_mean(v, mask):
                num = (v * mask).sum()
                den = mask.sum().clamp_min(1.0)
                return num / den

            if use_amp:
                with autocast():
                    z_t = Mt(xw)
                    logits_w = C(z_t)
                    p_t = F.softmax(logits_w, dim=1)

                    # confidence mask (max prob >= 0.8)
                    with torch.no_grad():
                        p_conf, _ = p_t.max(dim=1, keepdim=True)  # [B,1]
                    mask = (p_conf >= 0.8).float()

                    adv_logits = D(z_t, p_t)
                    adv_vec = nn.BCEWithLogitsLoss(reduction='none')(adv_logits, y_s)  # [B,1]
                    adv_loss = masked_mean(adv_vec, mask)

                    if args.lambda_ent > 0:
                        ent_vec = -(p_t.clamp_min(1e-8) * torch.log(p_t.clamp_min(1e-8))).sum(dim=1, keepdim=True)
                        ent_loss = masked_mean(ent_vec, mask)
                    else:
                        ent_loss = torch.tensor(0.0, device=device)

                    if args.lambda_cons > 0:
                        z_saug   = Mt(xs); logits_s = C(z_saug)
                        cons_vec = F.kl_div(F.log_softmax(logits_s, dim=1),
                                            p_t.detach(), reduction='none').sum(dim=1, keepdim=True)
                        cons_loss = masked_mean(cons_vec, mask)
                    else:
                        cons_loss = torch.tensor(0.0, device=device)

                    total = lam_adv_eff*adv_loss + args.lambda_ent*ent_loss + args.lambda_cons*cons_loss
                sc_mt.scale(total).backward()
                sc_mt.step(opt_mt); sc_mt.update()
            else:
                z_t = Mt(xw)
                logits_w = C(z_t)
                p_t = F.softmax(logits_w, dim=1)
                with torch.no_grad():
                    p_conf, _ = p_t.max(dim=1, keepdim=True)
                mask = (p_conf >= 0.8).float()

                adv_logits = D(z_t, p_t)
                adv_vec = nn.BCEWithLogitsLoss(reduction='none')(adv_logits, y_s)
                adv_loss = masked_mean(adv_vec, mask)

                if args.lambda_ent > 0:
                    ent_vec = -(p_t.clamp_min(1e-8) * torch.log(p_t.clamp_min(1e-8))).sum(dim=1, keepdim=True)
                    ent_loss = masked_mean(ent_vec, mask)
                else:
                    ent_loss = torch.tensor(0.0, device=device)

                if args.lambda_cons > 0:
                    z_saug   = Mt(xs); logits_s = C(z_saug)
                    cons_vec = F.kl_div(F.log_softmax(logits_s, dim=1),
                                        p_t.detach(), reduction='none').sum(dim=1, keepdim=True)
                    cons_loss = masked_mean(cons_vec, mask)
                else:
                    cons_loss = torch.tensor(0.0, device=device)

                total = lam_adv_eff*adv_loss + args.lambda_ent*ent_loss + args.lambda_cons*cons_loss
                total.backward(); opt_mt.step()

            pbar.set_postfix(
                d=f"{float(loss_d):.3f}",
                adv=f"{float(adv_loss):.3f}",
                ent=f"{float(ent_loss):.3f}",
                cons=f"{float(cons_loss):.3f}",
                lam_adv=lam_adv_eff
            )

        # eval each epoch
        acc, macro = evaluate(device, Mt, C, eval_loader)
        print(f"[EVAL] tgt-acc={acc:.4f}  tgt-macro={macro:.4f}")
        if macro >= best_macro:
            best_macro = macro
            torch.save(Mt.state_dict(), os.path.join(args.out, "Mt_best.pth"))
            torch.save(C.state_dict(),  os.path.join(args.out, "C_tgt.pth"))

    # final save
    torch.save(Mt.state_dict(), os.path.join(args.out, "Mt_last.pth"))
    with open(os.path.join(args.out, "labelmap.json"), "w") as f:
        json.dump({"classes": CANON}, f, indent=2)
    print(f"[DONE] Saved to {args.out}  (best macro={best_macro:.4f})")

if __name__ == "__main__":
    main()
