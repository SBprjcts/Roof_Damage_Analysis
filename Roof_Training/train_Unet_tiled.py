print("✅ START train_Unet_tiled.py", flush=True)
# I import standard libs for paths, parsing args, progress bars, and tables.
from pathlib import Path                               # cross-platform paths
import argparse                                         # command-line flags
from tqdm import tqdm                                   # pretty progress bars
import json, csv                                        # save summaries

# I import numeric + imaging libs.
import numpy as np                                      # fast array math
import numpy as _np                                     # for ascontiguousarray in augmentation
from PIL import Image                                   # simple image I/O

# I import geospatial I/O to keep CRS/transform when working with GeoTIFFs.
import rasterio                                          # read/write GeoTIFF
from rasterio.transform import Affine                    # for pixel area

# I import PyTorch to build, train, and evaluate a small U-Net model.
import torch                                             # tensors + autograd
import torch.nn as nn                                    # neural net layers
from torch.utils.data import Dataset, DataLoader         # datasets/loaders
import torchvision.transforms.functional as TF           # tensor helpers

# ----------------------------
# Small U-Net model
# ----------------------------

class DoubleConv(nn.Module):
    # I define two Conv+BN+ReLU blocks used throughout U-Net.
    def __init__(self, c_in, c_out):
        super().__init__()                                # init parent nn.Module
        self.seq = nn.Sequential(                         # stack layers neatly
            nn.Conv2d(c_in, c_out, 3, padding=1),         # 3x3 conv
            nn.BatchNorm2d(c_out),                        # normalize activations
            nn.ReLU(inplace=True),                        # nonlinearity
            nn.Conv2d(c_out, c_out, 3, padding=1),        # another 3x3 conv
            nn.BatchNorm2d(c_out),                        # normalize again
            nn.ReLU(inplace=True),                        # nonlinearity
        )

    def forward(self, x):
        return self.seq(x)                                # run both conv blocks

class UNetSmall(nn.Module):
    # I assemble a compact U-Net encoder-decoder with skip connections.
    def __init__(self, in_ch=3, out_ch=1, f=32):
        super().__init__()                                # init parent
        self.d1 = DoubleConv(in_ch, f);   self.p1 = nn.MaxPool2d(2)          # down 1
        self.d2 = DoubleConv(f, f*2);     self.p2 = nn.MaxPool2d(2)          # down 2
        self.d3 = DoubleConv(f*2, f*4);   self.p3 = nn.MaxPool2d(2)          # down 3
        self.b  = DoubleConv(f*4, f*8)                                         # bottleneck
        self.u3 = nn.ConvTranspose2d(f*8, f*4, 2, 2); self.c3 = DoubleConv(f*8, f*4)  # up 3
        self.u2 = nn.ConvTranspose2d(f*4, f*2, 2, 2); self.c2 = DoubleConv(f*4, f*2)  # up 2
        self.u1 = nn.ConvTranspose2d(f*2, f,   2, 2); self.c1 = DoubleConv(f*2, f)    # up 1
        self.out = nn.Conv2d(f, out_ch, 1)                                            # logits

    def forward(self, x):
        c1 = self.d1(x); p1 = self.p1(c1)                                      # enc1
        c2 = self.d2(p1); p2 = self.p2(c2)                                     # enc2
        c3 = self.d3(p2); p3 = self.p3(c3)                                     # enc3
        b  = self.b(p3)                                                        # bottleneck
        u3 = self.u3(b);  u3 = torch.cat([u3, c3], 1); u3 = self.c3(u3)        # dec3
        u2 = self.u2(u3); u2 = torch.cat([u2, c2], 1); u2 = self.c2(u2)        # dec2
        u1 = self.u1(u2); u1 = torch.cat([u1, c1], 1); u1 = self.c1(u1)        # dec1
        return self.out(u1)                                                    # raw logits

# ----------------------------
# I/O helpers
# ----------------------------

def read_any_image(path: Path):
    """I read the roof image; if it's GeoTIFF I keep transform+crs."""
    try:
        with rasterio.open(path) as src:                          # try GeoTIFF
            if src.count >= 3:                                    # if RGB present
                arr = src.read([1,2,3])                           # read R,G,B
            else:
                b1 = src.read(1)                                  # single band
                arr = np.repeat(b1[None,...], 3, axis=0)          # make 3-band
            arr = np.clip(arr, 0, 255).astype(np.uint8)           # clamp to 8-bit
            return arr, src.transform, src.crs                    # data + geo
    except Exception:
        im = Image.open(path).convert("RGB")                      # PNG/JPG path
        arr = np.array(im).transpose(2,0,1)                       # HWC→CHW
        return arr, None, None                                    # no geo info

def read_mask_binary(path: Path):
    """I read a 0/255 mask and return HxW array with values 0/1."""
    try:
        with rasterio.open(path) as src:                          # GeoTIFF path
            m = src.read(1)                                       # first band
    except Exception:
        m = np.array(Image.open(path).convert("L"))               # grayscale
    return (m > 127).astype(np.uint8)                             # 0/1 threshold

def pixel_area_m2(transform: Affine):
    """I compute pixel area in square meters from the GeoTransform."""
    return abs(transform.a) * abs(transform.e)                    # |dx|*|dy|

def write_mask_outputs(mask_bool, outdir: Path, base: str, transform=None, crs=None):
    """I save a PNG (always) and a GeoTIFF (if geo info is available)."""
    outdir.mkdir(parents=True, exist_ok=True)                     # ensure folder
    png_path = outdir / f"{base}_pred_mask.png"                   # PNG output
    Image.fromarray((mask_bool*255).astype(np.uint8)).save(png_path)  # write PNG
    tif_path = None                                               # default none
    if transform is not None and crs is not None:                 # if georeferenced
        tif_path = outdir / f"{base}_pred_mask.tif"               # GeoTIFF path
        with rasterio.open(                                       # open writer
            tif_path, "w", driver="GTiff",
            height=mask_bool.shape[0], width=mask_bool.shape[1],
            count=1, dtype="uint8", crs=crs, transform=transform, nodata=0
        ) as dst:
            dst.write(mask_bool.astype(np.uint8), 1)              # write band
    return png_path, tif_path                                     # return paths

# ----------------------------
# Dataset with tiling + aug
# ----------------------------

class TiledRoofDataset(Dataset):
    # I cut the big CHW image + HW mask into many tiles for training.
    def __init__(self, img_chw, mask_hw, tile=256, stride=256, augment=False):
        self.img = img_chw                                        # C,H,W image
        self.mask = mask_hw                                       # H,W 0/1 mask
        self.tile = tile                                          # tile size
        self.stride = stride                                      # step size
        self.augment = augment                                    # enable aug?
        C,H,W = img_chw.shape                                     # dimensions
        self.tiles = []                                           # tile list
        for y in range(0, max(H - tile + 1, 1), stride):          # all rows
            for x in range(0, max(W - tile + 1, 1), stride):      # all cols
                self.tiles.append((y, x))                         # save origin
        if not self.tiles:                                        # tiny image fallback
            self.tiles = [(0,0)]                                  # single tile

    def __len__(self):
        return len(self.tiles)                                    # number of tiles

    def __getitem__(self, i):
        y, x = self.tiles[i]                                      # tile origin
        C,H,W = self.img.shape                                    # dims
        img_t = np.zeros((C, self.tile, self.tile), np.uint8)     # padded img
        msk_t = np.zeros((self.tile, self.tile), np.uint8)        # padded mask
        y2 = min(y+self.tile, H); x2 = min(x+self.tile, W)        # crop bounds
        h = y2 - y; w = x2 - x                                    # actual size
        img_t[:, :h, :w] = self.img[:, y:y2, x:x2]                # copy patch
        msk_t[:h, :w]   = self.mask[y:y2, x:x2]                   # copy labels

        if self.augment:                                          # simple aug
            if np.random.rand() < 0.5:                            # random h-flip
                img_t = img_t[:,:,::-1].copy(); msk_t = msk_t[:,::-1].copy()
            if np.random.rand() < 0.5:                            # random v-flip
                img_t = img_t[:,::-1,:].copy(); msk_t = msk_t[::-1,:].copy()
            if np.random.rand() < 0.25:                           # 90° rotate
                img_t = np.rot90(img_t, k=1, axes=(1,2)).copy()
                msk_t = np.rot90(msk_t, k=1, axes=(0,1)).copy()

        # make arrays contiguous to avoid negative-stride issue
        img_t = _np.ascontiguousarray(img_t)
        msk_t = _np.ascontiguousarray(msk_t)
        img_t = torch.from_numpy(img_t).float()/255.0             # to float [0,1]
        msk_t = torch.from_numpy(msk_t>0).float().unsqueeze(0)    # to 0/1, add C
        return img_t, msk_t                                       # sample pair

# ----------------------------
# Metrics (IoU)
# ----------------------------

def iou_score(pred_prob, target, thresh=0.5):
    """I compute IoU on a batch: threshold probs, then intersection/union."""
    pred = (pred_prob >= thresh).float()                          # binarize
    inter = (pred*target).sum(dim=(1,2,3))                        # overlap
    union = (pred+target - pred*target).sum(dim=(1,2,3))          # union area
    iou = (inter / (union + 1e-6)).mean().item()                  # mean IoU
    return iou                                                    # scalar

# ----------------------------
# Training + validation
# ----------------------------

def train_validate(img_chw, mask_hw, epochs=10, tile=256, stride=256, device=None):
    """I split tiles into train/val, train the model, and report val IoU."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick device
    # I make all tiles (no aug) just to get the list for splitting.
    all_ds = TiledRoofDataset(img_chw, mask_hw, tile=tile, stride=stride, augment=False)
    n = len(all_ds.tiles)                                         # number of tiles
    idx = np.arange(n)                                            # indices 0..n-1
    rng = np.random.RandomState(42)                               # reproducible split
    rng.shuffle(idx)                                              # shuffle
    cut = max(1, int(0.9 * n))                                    # 90% train / 10% val
    train_idx, val_idx = idx[:cut], idx[cut:]                     # split indices

    # I build two datasets: train with aug, val without aug.
    train_ds = TiledRoofDataset(img_chw, mask_hw, tile=tile, stride=stride, augment=True)
    val_ds   = TiledRoofDataset(img_chw, mask_hw, tile=tile, stride=stride, augment=False)
    # I restrict each to its subset of tile positions.
    train_ds.tiles = [all_ds.tiles[i] for i in train_idx]         # train tiles
    val_ds.tiles   = [all_ds.tiles[i] for i in val_idx]           # val tiles

    # I wrap them in data loaders.
    train_ld = DataLoader(train_ds, batch_size=8, shuffle=True)   # train loader
    val_ld   = DataLoader(val_ds,   batch_size=8, shuffle=False)  # val loader

    # I create the model, loss, and optimizer.
    model = UNetSmall(in_ch=3, out_ch=1).to(device)               # small U-Net
    loss_fn = nn.BCEWithLogitsLoss()                              # stable binary loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)           # Adam optimizer

    # I train for a few epochs and compute val IoU each time.
    for ep in range(1, epochs+1):
        model.train()                                             # training mode
        running = 0.0                                             # loss accumulator
        for imgs, msks in tqdm(train_ld, desc=f"Train {ep}/{epochs}"):
            imgs, msks = imgs.to(device), msks.to(device)         # move to device
            opt.zero_grad()                                       # reset grads
            logits = model(imgs)                                  # forward pass
            loss = loss_fn(logits, msks)                          # compute loss
            loss.backward()                                       # backprop
            opt.step()                                            # update weights
            running += loss.item()*imgs.size(0)                   # accumulate
        train_loss = running / max(1, len(train_ds))              # epoch train loss

        # I evaluate on validation tiles (no grad).
        model.eval()                                              # eval mode
        vloss, viou, vcount = 0.0, 0.0, 0                         # val stats
        with torch.no_grad():
            for imgs, msks in tqdm(val_ld, desc=f"Val   {ep}/{epochs}"):
                imgs, msks = imgs.to(device), msks.to(device)     # to device
                logits = model(imgs)                              # forward
                loss = loss_fn(logits, msks)                      # val loss
                vloss += loss.item()*imgs.size(0)                 # accumulate
                prob = torch.sigmoid(logits)                      # to probs
                viou += iou_score(prob, msks) * imgs.size(0)      # batch IoU
                vcount += imgs.size(0)                            # batch count
        val_loss = vloss / max(1, len(val_ds))                    # mean val loss
        val_iou  = viou / max(1, vcount)                          # mean val IoU
        print(f"Epoch {ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_IoU={val_iou:.3f}")

    return model                                                  # return trained model

# ----------------------------
# Full-image inference (stitch)
# ----------------------------

@torch.no_grad()
def infer_full_image(model, img_chw, device=None, tile=256):
    """I predict the whole image by sliding tiles and averaging overlaps."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
    C,H,W = img_chw.shape                                       # dimensions
    xfull = torch.from_numpy(img_chw).float()/255.0             # to float [0,1]
    xfull = xfull.unsqueeze(0).to(device)                       # add batch dim

    prob = np.zeros((H,W), dtype=np.float32)                    # prob accumulator
    count = np.zeros((H,W), dtype=np.float32)                   # coverage count
    step = tile                                                 # non-overlap for speed
    for y in range(0, H, step):                                 # iterate rows
        for x0 in range(0, W, step):                            # iterate cols
            y2 = min(y+tile, H); x2 = min(x0+tile, W)           # tile bounds
            tile_t = torch.zeros((1,3,tile,tile), device=device)# padded tile tensor
            tile_t[:,:,:y2-y,:x2-x0] = xfull[:,:,y:y2,x0:x2]    # copy image patch
            logits = model(tile_t)                              # forward pass
            p = torch.sigmoid(logits)[0,0].cpu().numpy()        # to prob map
            prob[y:y2, x0:x2] += p[:y2-y, :x2-x0]               # accumulate probs
            count[y:y2, x0:x2] += 1.0                           # accumulate count
    prob = prob / np.maximum(count, 1e-6)                       # average overlaps
    return prob                                                 # full prob map

# ----------------------------
# Main: glue everything
# ----------------------------

def main():
    # I parse command-line flags so you can control key settings.
    ap = argparse.ArgumentParser(description="Roof damage segmentation with proper tiling (PyTorch U-Net).")
    ap.add_argument("--image", required=True, help="Roof image (GeoTIFF/PNG/JPG)")
    ap.add_argument("--mask",  required=True, help="Binary mask (0=undamaged, 255=damaged)")
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs")
    ap.add_argument("--tile",   type=int, default=256, help="Tile size (pixels)")
    ap.add_argument("--stride", type=int, default=256, help="Stride for tile placement")
    ap.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for damage")
    ap.add_argument("--outdir", default="outputs_tiled", help="Folder for outputs")
    args = ap.parse_args()                                       # parse args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick device
    print(f"Using device: {device}")                              # show device

    # I read the image and keep geo info if available (GeoTIFF).
    img_chw, transform, crs = read_any_image(Path(args.image))    # C,H,W + geo
    C,H,W = img_chw.shape                                         # dims

    # I read the binary mask (expects 0/255), convert to 0/1.
    mask_hw = read_mask_binary(Path(args.mask))                   # H,W 0/1

    # I ensure image and mask sizes match perfectly (critical).
    assert mask_hw.shape == (H,W), "Image and mask must have identical size."

    # I train + validate on tiled patches with augmentations for generalization.
    model = train_validate(img_chw, mask_hw, epochs=args.epochs, tile=args.tile, stride=args.stride, device=device)

    # I infer on the full image and threshold to a boolean damage mask.
    prob = infer_full_image(model, img_chw, device=device, tile=args.tile)  # prob map
    pred_bool = (prob >= args.thresh)                                       # True=damage

    # I compute pixel stats (always available).
    total_px   = pred_bool.size                                             # total pixels
    damaged_px = int(pred_bool.sum())                                       # damaged pixels
    undamaged  = int(total_px - damaged_px)                                 # undamaged
    percent    = (damaged_px / total_px) * 100.0                            # % damaged

    # I compute area in m² if georeferenced in meters (optional).
    per_px_m2 = tot_m2 = dmg_m2 = None                                      # defaults
    if transform is not None and crs is not None:                           # if geo exists
        try:
            per_px_m2 = pixel_area_m2(transform)                            # m² per pixel
            tot_m2 = total_px * per_px_m2                                   # total area
            dmg_m2 = damaged_px * per_px_m2                                 # damaged area
        except Exception:
            pass                                                            # skip if not metric

    # I save predicted mask as PNG (always) and GeoTIFF (if geo info).
    outdir = Path(args.outdir)                                              # outputs folder
    base = Path(args.image).stem                                            # base name
    png_path, tif_path = write_mask_outputs(pred_bool, outdir, base, transform, crs)

    # I write a JSON and CSV summary for reporting.
    summary = {
        "image": str(args.image),                     # input image path
        "mask_training_labels": str(args.mask),       # label path
        "epochs": int(args.epochs),                   # epochs used
        "tile": int(args.tile),                       # tile size
        "stride": int(args.stride),                   # stride used
        "threshold": float(args.thresh),              # prob threshold
        "total_pixels": int(total_px),                # pixel count
        "damaged_pixels": int(damaged_px),            # damaged count
        "undamaged_pixels": int(undamaged),           # undamaged count
        "percent_damaged": float(round(percent, 6)),  # % damaged
        "per_pixel_m2": None if per_px_m2 is None else float(per_px_m2),   # m²/px
        "total_area_m2": None if tot_m2 is None else float(tot_m2),        # total m²
        "damaged_area_m2": None if dmg_m2 is None else float(dmg_m2),      # damaged m²
        "pred_mask_png": str(png_path),                                     # PNG path
        "pred_mask_tif": None if tif_path is None else str(tif_path),       # GeoTIFF path
        "device": str(device),                                              # cpu/gpu
    }
    json_path = outdir / f"{base}_summary.json"                              # JSON path
    csv_path  = outdir / f"{base}_summary.csv"                               # CSV path
    with open(json_path, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)  # write JSON
    with open(csv_path, "w", newline="", encoding="utf-8") as f:                          # write CSV
        w = csv.DictWriter(f, fieldnames=list(summary.keys())); w.writeheader(); w.writerow(summary)

    # I print a short console summary for quick confirmation.
    print(f"Percent damaged: {percent:.2f}%")                                # % damaged
    if dmg_m2 is not None:                                                   # area if known
        print(f"Damaged area: {dmg_m2:.2f} m² (pixel area {per_px_m2:.4f} m²)")
    print(f"Wrote: {png_path}")                                              # PNG out
    if tif_path: print(f"Wrote: {tif_path}")                                  # GeoTIFF out
    print(f"Wrote: {json_path}")                                             # JSON out
    print(f"Wrote: {csv_path}")                                              # CSV out

# I run main() only when the file is executed directly.
if __name__ == "__main__":
    main()
