"""Build a Deep Zoom (DZI) tile pyramid from the full map PNG."""
import math, os, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
src, outdir = sys.argv[1], sys.argv[2]
TS, OV, Q = 256, 1, 86
img = Image.open(src).convert('RGB')
W, H = img.size
maxl = math.ceil(math.log2(max(W, H)))
base = os.path.join(outdir, 'map_files'); os.makedirs(base, exist_ok=True)
lvl_img = img
for level in range(maxl, -1, -1):
    s = 2 ** (maxl - level)
    w, h = max(1, math.ceil(W / s)), max(1, math.ceil(H / s))
    if lvl_img.size != (w, h):
        lvl_img = lvl_img.resize((w, h), Image.LANCZOS)
    d = os.path.join(base, str(level)); os.makedirs(d, exist_ok=True)
    for ty in range(math.ceil(h / TS)):
        for tx in range(math.ceil(w / TS)):
            x0 = max(tx * TS - OV, 0); y0 = max(ty * TS - OV, 0)
            x1 = min((tx + 1) * TS + OV, w); y1 = min((ty + 1) * TS + OV, h)
            lvl_img.crop((x0, y0, x1, y1)).save(os.path.join(d, f'{tx}_{ty}.jpg'), quality=Q)
open(os.path.join(outdir, 'map.dzi'), 'w').write(
    f'<?xml version="1.0" encoding="UTF-8"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="jpg" Overlap="{OV}" TileSize="{TS}"><Size Width="{W}" Height="{H}"/></Image>')
print('levels', maxl + 1, W, H)
