"""Stitch overlapping square tiles into one image with feathered seams.
usage: python stitch.py <tile_dir> <out.png> [scale]
"""
import sys, json, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

MAP = json.load(open('/home/claude/g3d/map.json'))
T, ST = MAP['tile_px'], MAP['stride_px']
src, out = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
tiles = json.load(open('/home/claude/g3d/tiles_all.json'))
R = max(t['row'] for t in tiles) + 1; C = max(t['col'] for t in tiles) + 1
W = int(((C - 1) * ST + T) * scale); H = int(((R - 1) * ST + T) * scale)
t = int(round(T * scale)); st = ST * scale
acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W, 1), np.float32)
ov = T - ST
BL = int(os.environ.get('BLEND', '64'))  # width of the feathered seam inside the overlap
ramp = np.ones(T, np.float32)
a0 = (ov - BL) // 2
ramp[:a0] = 1e-4; ramp[a0:a0 + BL] = np.linspace(1e-4, 1, BL)
ramp[T - a0:] = 1e-4; ramp[T - a0 - BL:T - a0] = np.linspace(1, 1e-4, BL)
w2 = np.outer(ramp, ramp)
if scale != 1.0:
    w2 = np.array(Image.fromarray(w2).resize((t, t), Image.BILINEAR))
w2 = w2[..., None]
for tl in tiles:
    p = f"{src}/{tl['id']}.png"
    if not os.path.exists(p): continue
    im = Image.open(p).convert('RGB')
    if im.size != (t, t): im = im.resize((t, t), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    y = int(round(tl['row'] * st)); x = int(round(tl['col'] * st))
    hh = min(t, H - y); ww = min(t, W - x)
    acc[y:y + hh, x:x + ww] += a[:hh, :ww] * w2[:hh, :ww]; wsum[y:y + hh, x:x + ww] += w2[:hh, :ww]
img = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 230).clip(0, 255).astype(np.uint8)
Image.fromarray(img).save(out)
print('stitched', W, H)
