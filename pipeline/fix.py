"""Free (non-AI) fixes: tile colour harmonisation, unified procedural water from the 3D render's water mask,
forest re-synthesis for tiles where the AI invented towns, and a cartoon placeholder for tiles awaiting a redo."""
import json, os, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
G = '/home/claude/g3d'
MAP = json.load(open(f'{G}/map.json')); T, ST = MAP['tile_px'], MAP['stride_px']
tiles = json.load(open(f'{G}/tiles_all.json'))
WATER_T = ['r03_c06', 'r08_c00', 'r09_c00', 'r09_c01', 'r08_c01', 'r08_c02']
FOREST_FIX = ['r00_c00', 'r01_c01', 'r02_c00', 'r02_c01', 'r04_c02']
REDO = ['r06_c07']
FOREST_SRC = ['r03_c01', 'r01_c03', 'r00_c02']   # Nano Banana Pro forest tiles used as texture source
rng = np.random.default_rng(11)

render = np.asarray(Image.open(f'{G}/render_full.png').convert('RGB'), np.float32)
H, W = render.shape[:2]
R, Gc, B = render[..., 0], render[..., 1], render[..., 2]

# ---------- masks from the real 3D render ----------
water = np.unpackbits(np.load(f'{G}/water_mask.npy'))[:H * W].reshape(H, W).astype(bool)
water_core = ndimage.binary_erosion(water, iterations=4)          # keep the AI's own shoreline/quays
lum = render.mean(-1); sat = render.max(-1) - render.min(-1)
veg = (Gc >= R - 6) & (lum < 125) & ~water
veg = ndimage.binary_opening(ndimage.binary_closing(veg, iterations=2), iterations=2)
print('masks ready', water.mean().round(3), veg.mean().round(3), flush=True)

# ---------- 1. per-tile colour harmonisation (gains solved over overlaps, land only) ----------
land_ok = ~water
ids = [t['id'] for t in tiles if t['id'] not in WATER_T]
pos = {t['id']: (t['row'], t['col']) for t in tiles}
imgs = {i: np.asarray(Image.open(f'{G}/final_tiles/{i}.png').convert('RGB'), np.float32) for i in ids}
k = {i: n for n, i in enumerate(ids)}
rows, rhs = [], []
for i in ids:
    r, c = pos[i]
    for (dr, dc) in [(0, 1), (1, 0)]:
        j = f'r{r + dr:02d}_c{c + dc:02d}'
        if j not in imgs: continue
        y0, x0 = r * ST, c * ST
        if dc: ys, xs = slice(0, T), slice(ST, T); ys2, xs2 = slice(0, T), slice(0, T - ST)
        else: ys, xs = slice(ST, T), slice(0, T); ys2, xs2 = slice(0, T - ST), slice(0, T)
        gy, gx = y0 + (ST if dr else 0), x0 + (ST if dc else 0)
        m = land_ok[gy:gy + (T if dc else T - ST), gx:gx + (T - ST if dc else T)]
        if m.shape != (imgs[i][ys, xs].shape[:2]) or m.sum() < 5000: continue
        a = imgs[i][ys, xs][m].mean(0) + 1; b = imgs[j][ys2, xs2][m].mean(0) + 1
        row = np.zeros(len(ids)); row[k[i]] = 1; row[k[j]] = -1
        rows.append(row); rhs.append(np.log(b) - np.log(a))
A = np.vstack(rows + [np.eye(len(ids)) * 0.5]); bb = np.vstack(rhs + [np.zeros((len(ids), 3))])
logg = np.linalg.lstsq(A, bb, rcond=None)[0]
logg -= logg.mean(0)
gains = np.clip(np.exp(0.75 * logg), 0.85, 1.18)
HT = f'{G}/harmonised_tiles'; os.makedirs(HT, exist_ok=True)
for i in ids:
    Image.fromarray((imgs[i] * gains[k[i]]).clip(0, 255).astype(np.uint8)).save(f'{HT}/{i}.png')
print('gains range', gains.min(0).round(2), gains.max(0).round(2), flush=True)
del imgs
os.environ['BLEND'] = '64'
sys.argv = ['stitch', HT, f'{G}/stage_h.png']
exec(open('/home/claude/scripts/stitch.py').read())
out = np.asarray(Image.open(f'{G}/stage_h.png'), np.float32).copy()

# ---------- helper: tile-dominant region (inset) with feather ----------
def region_weight(tid, inset=96, feather=64):
    r, c = pos[tid]
    w = np.zeros((H, W), np.float32)
    y0, x0 = r * ST + inset, c * ST + inset; y1, x1 = r * ST + T - inset, c * ST + T - inset
    if r == 0: y0 = 0
    if c == 0: x0 = 0
    if r == max(p[0] for p in pos.values()): y1 = H
    if c == max(p[1] for p in pos.values()): x1 = W
    w[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] = 1
    return np.asarray(Image.fromarray((w * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather / 2)), np.float32) / 255

def cartoon(img):
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).filter(ImageFilter.UnsharpMask(2, 80, 2))
    a = np.asarray(im, np.float32)
    m = a.mean(-1, keepdims=True); a = m + (a - m) * 1.6
    a = 255 * (np.clip(a, 0, 255) / 255) ** 0.7
    a = (a // 16) * 16 + 8
    return a.clip(0, 255)

# ---------- 2. forest re-synthesis ----------
P = 96; OV = 32
lib = []
for tid in FOREST_SRC:
    r, c = pos[tid]
    for _ in range(140):
        y = r * ST + rng.integers(96, T - 96 - P); x = c * ST + rng.integers(96, T - 96 - P)
        pt = out[y:y + P, x:x + P]
        green = (pt[..., 1] > pt[..., 0] + 12) & (pt[..., 1] > pt[..., 2] + 5)
        if green.mean() > 0.9:
            lib.append(pt.copy())
print('forest patches', len(lib), flush=True)
ramp = np.ones(P, np.float32); ramp[:OV] = np.linspace(0.05, 1, OV); ramp[-OV:] = np.linspace(1, 0.05, OV)
pw = np.outer(ramp, ramp)[..., None]

def forest_fill(y0, y1, x0, x1):
    acc = np.zeros((y1 - y0 + P, x1 - x0 + P, 3), np.float32); ws = np.zeros(acc.shape[:2] + (1,), np.float32)
    for yy in range(0, y1 - y0, P - OV):
        for xx in range(0, x1 - x0, P - OV):
            p = lib[rng.integers(len(lib))]
            if rng.random() < 0.5: p = p[:, ::-1]
            acc[yy:yy + P, xx:xx + P] += p * pw; ws[yy:yy + P, xx:xx + P] += pw
    tex = (acc / np.maximum(ws, 1e-6))[:y1 - y0, :x1 - x0]
    rel = ndimage.gaussian_filter(lum[y0:y1, x0:x1], 5)
    rel = (rel - rel.mean()) / (rel.std() + 1e-6)
    return tex * np.clip(1 + 0.12 * rel, 0.8, 1.2)[..., None]

for tid in FOREST_FIX + REDO:
    w = region_weight(tid, inset=96 if tid in FOREST_FIX else 200, feather=64 if tid in FOREST_FIX else 48)
    ys, xs = np.nonzero(w > 0.001); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = forest_fill(y0, y1, x0, x1)
    car = cartoon(render[y0:y1, x0:x1])
    v = ndimage.gaussian_filter(veg[y0:y1, x0:x1].astype(np.float32), 1.5)[..., None]
    rep = sub * v + car * (1 - v) if tid in FOREST_FIX else car
    ww = w[y0:y1, x0:x1, None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - ww) + rep * ww
    print('fixed', tid, flush=True)

# ---------- 3. unified procedural water (strip-wise to save memory) ----------
del render, lum, sat, veg, Gc, B
import gc; gc.collect()
base = np.array([46, 136, 204], np.float32); deep = np.array([34, 112, 186], np.float32); shallow = np.array([96, 182, 226], np.float32)
dist = ndimage.distance_transform_edt(water).astype(np.float32)
noise = ndimage.gaussian_filter(rng.standard_normal((H // 8 + 1, W // 8 + 1)).astype(np.float32), 3)
noise = np.array(Image.fromarray(noise).resize((W, H), Image.BICUBIC), np.float32); noise /= (np.abs(noise).max() + 1e-6)
rip = np.zeros((H, W), np.uint8)
nrip = int(water.sum() / 900)
for y, x, l in zip(rng.integers(0, H, nrip), rng.integers(0, W, nrip), rng.integers(5, 16, nrip)):
    y = y // 2 * 2; x = x // 2 * 2; rip[y:y + 2, x:x + l] = 1
wm_full = ndimage.gaussian_filter(water_core.astype(np.float32), 1.0)
for y0 in range(0, H, 512):
    y1 = min(H, y0 + 512)
    d = dist[y0:y1]
    deepness = np.clip((d - 30) / 160, 0, 1)[..., None]
    wcol = base * (1 - deepness) + deep * deepness + noise[y0:y1, :, None] * 5
    sh = np.clip(1 - d / 14, 0, 1)[..., None] * 0.75
    wcol = wcol * (1 - sh) + shallow * sh
    wcol += (rip[y0:y1] * water_core[y0:y1])[..., None] * np.array([34, 34, 26], np.float32)
    wm = wm_full[y0:y1, :, None]
    out[y0:y1] = out[y0:y1] * (1 - wm) + wcol * wm
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f'{G}/tinynagasaki_full.png')
np.save(f'{G}/geo_water_mask.npy', np.packbits(water))
print('saved', out.shape)
