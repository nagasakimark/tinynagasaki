"""Water mask in map pixels from the 3D surface: flat cells at sea level, z-buffered in the map projection."""
import json, math
import numpy as np
from scipy import ndimage
G = '/home/claude/g3d'
MAP = json.load(open(f'{G}/map.json'))
S = MAP['m_per_px']; P = math.radians(MAP['pitch_deg']); sp, cp = math.sin(P), math.cos(P)
D = np.load(f'{G}/geo/dsm.npz'); dsm = D['dsm']; E0 = float(D['e0']); N1 = float(D['n1']); CELL = float(D['cell'])
SEA = 33.15
Hc, Wc = dsm.shape
loc_std = np.sqrt(np.maximum(ndimage.uniform_filter(dsm ** 2, 3) - ndimage.uniform_filter(dsm, 3) ** 2, 0))
isw = (dsm > 26.5) & (dsm < 35.0) & (loc_std < 0.35)
isw = ndimage.binary_opening(isw, iterations=1)
isw = ndimage.binary_fill_holes(isw)
i, j = np.mgrid[0:Hc, 0:Wc]
e = E0 + (j + 0.5) * CELL; n = N1 - (i + 0.5) * CELL; z = dsm
x = (e - MAP['x0']) / S; y = (MAP['v1'] - (n * sp + z * cp)) / S
depth = n * cp - z * sp            # along the view direction; smaller = nearer the camera
W = 8704; H = 7936
order = np.argsort(-depth.ravel())  # far first, near last (near overwrites)
xs = x.ravel()[order]; ys = y.ravel()[order]; ws = isw.ravel()[order]
lab = np.full((H, W), 255, np.uint8)
step_x = CELL / S; step_y = CELL * sp / S
for dy in np.arange(-step_y / 2, step_y / 2 + 0.01, 1.0):
    for dx in np.arange(-step_x / 2, step_x / 2 + 0.01, 1.0):
        xi = np.round(xs + dx).astype(np.int64); yi = np.round(ys + dy).astype(np.int64)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        lab[yi[ok], xi[ok]] = ws[ok]
water = lab == 1
water = ndimage.binary_closing(water, iterations=2)
lb, nlab = ndimage.label(water); sz = ndimage.sum(water, lb, range(1, nlab + 1))
water = np.isin(lb, 1 + np.nonzero(sz > 3000)[0])
np.save(f'{G}/water_mask.npy', np.packbits(water))
print('water fraction', water.mean().round(3))
from PIL import Image
Image.fromarray((water * 255).astype(np.uint8)).resize((1088, 992)).save('/tmp/watermask.png')

# ---- refine with colour: drop grey roads, fill enclosed gaps (mesh seams) ----
from PIL import Image as _I
_I.MAX_IMAGE_PIXELS = None
r = np.asarray(_I.open(f'{G}/render_full.png').convert('RGB'), np.int16)
R_, G_, B_ = r[..., 0], r[..., 1], r[..., 2]; L = r.sum(-1) // 3
colorish = (B_ >= R_ + 2) & (L < 135)
strict = (B_ > R_ + 8) & (B_ >= G_ - 2) & (L < 118)
del r
filled = ndimage.binary_fill_holes(ndimage.binary_closing(water, iterations=6))
water2 = (water & colorish) | (filled & strict)
water2 = ndimage.binary_opening(ndimage.binary_closing(water2, iterations=2), iterations=2)
lb, nlab = ndimage.label(water2); sz = ndimage.sum(water2, lb, range(1, nlab + 1))
water2 = np.isin(lb, 1 + np.nonzero(sz > 6000)[0])
np.save(f'{G}/water_mask.npy', np.packbits(water2))
print('refined water fraction', water2.mean().round(3))
