"""Assemble final styled tiles, synthesize pure-sea tiles (no AI), stitch the full map."""
import json, os, shutil, sys
import numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
G = '/home/claude/g3d'
MAP = json.load(open(f'{G}/map.json')); T, ST = MAP['tile_px'], MAP['stride_px']
tiles = json.load(open(f'{G}/tiles_all.json'))
WATER = ['r03_c06', 'r08_c00', 'r09_c00', 'r09_c01', 'r08_c01', 'r08_c02']
FIN = f'{G}/final_tiles'; os.makedirs(FIN, exist_ok=True)
src = {}
for t in tiles:
    i = t['id']
    if i in WATER: continue
    p = f'{G}/styled/{i}.png' if os.path.exists(f'{G}/styled/{i}.png') else f'{G}/nb/{i}.png'
    im = Image.open(p).convert('RGB')
    if im.size != (T, T): im = im.resize((T, T), Image.LANCZOS)
    im.save(f'{FIN}/{i}.png'); src[i] = os.path.basename(os.path.dirname(p))
json.dump(src, open(f'{G}/tile_sources.json', 'w'))

# stitch land tiles (narrow seams)
os.environ['BLEND'] = '64'
sys.argv = ['stitch', FIN, f'{G}/stage_land.png']
exec(open('/home/claude/scripts/stitch.py').read())
img = np.asarray(Image.open(f'{G}/stage_land.png'), np.float32)
H, W = img.shape[:2]
covered = np.zeros((H, W), bool)
for t in tiles:
    if t['id'] in WATER: continue
    y, x = t['row'] * ST, t['col'] * ST
    covered[y:y + T, x:x + T] = True

# push-pull fill of the uncovered sea, using only water-coloured known pixels near the holes
known = covered.copy()
r, g, b = img[..., 0], img[..., 1], img[..., 2]
waterish = (b > r + 25) & (b > g)
seed = known & waterish


def pushpull(col, m):
    levels = []
    c, w = col * m[..., None], m.astype(np.float32)
    while min(c.shape[:2]) > 4:
        levels.append((c, w))
        h2, w2 = c.shape[0] // 2 * 2, c.shape[1] // 2 * 2
        c = c[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).sum((1, 3)); w = w[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
    est = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels):
        up = np.asarray(Image.fromarray(est.astype(np.float32)[..., 0]).resize((c.shape[1], c.shape[0]), Image.BILINEAR))
        ups = np.stack([np.asarray(Image.fromarray(est[..., k].astype(np.float32)).resize((c.shape[1], c.shape[0]), Image.BILINEAR)) for k in range(3)], -1)
        cur = c / np.maximum(w, 1e-6)[..., None]
        a = np.clip(w, 0, 1)[..., None]
        est = cur * a + ups * (1 - a)
    return est

small = 8
hs, ws = H // small, W // small
img_s = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((ws, hs), Image.BOX), np.float32)
seed_s = np.asarray(Image.fromarray((seed * 255).astype(np.uint8)).resize((ws, hs), Image.BOX), np.float32) / 255 > 0.6
fill_s = pushpull(img_s, seed_s.astype(np.float32))
fill = np.stack([np.asarray(Image.fromarray(fill_s[..., k].astype(np.float32)).resize((W, H), Image.BICUBIC)) for k in range(3)], -1)
# gentle pixel-art ripples: short lighter dashes on a 4 px grid
rng = np.random.default_rng(5)
rip = np.zeros((H, W), np.float32)
n = int((~covered).sum() / 1400)
ys = rng.integers(0, H, n); xs = rng.integers(0, W, n); ls = rng.integers(6, 18, n)
for y, x, l in zip(ys, xs, ls):
    y = y // 2 * 2; x = x // 2 * 2
    rip[y:y + 2, x:x + l] = 1
rip = rip * (~covered)
fill = fill + rip[..., None] * np.array([26, 30, 30], np.float32)
# soft transition at the hole border (land tiles already end with their own water)
cov_soft = np.asarray(Image.fromarray((covered * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24)), np.float32)[..., None] / 255
cov_soft = np.where(covered[..., None], np.maximum(cov_soft, 0.0), 0.0)
out = img * cov_soft + fill * (1 - cov_soft)
out = out.clip(0, 255).astype(np.uint8)
Image.fromarray(out).save(f'{G}/tinynagasaki_full.png')
for t in tiles:
    if t['id'] in WATER:
        y, x = t['row'] * ST, t['col'] * ST
        Image.fromarray(out[y:y + T, x:x + T]).save(f'{FIN}/{t["id"]}.png')
print('done', out.shape)
