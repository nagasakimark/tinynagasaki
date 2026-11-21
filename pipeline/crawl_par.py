"""Crawl Google Photorealistic 3D Tiles for a small area and save the .glb tiles at a chosen detail level.
usage: python crawl.py <max_geometric_error> [outdir]
"""
import json, math, os, sys, urllib.parse, subprocess, hashlib
import numpy as np
sys.path.insert(0, '/home/claude/scripts')
from keys import keys

KEY = keys()['google']
BASE = 'https://tile.googleapis.com'
MAXERR = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
OUT = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/g3d/tiles'
os.makedirs(OUT, exist_ok=True)
AREA = json.load(open(sys.argv[3] if len(sys.argv) > 3 else '/home/claude/g3d/area.json'))  # {lat0, lon0, half_e, half_n}


def ecef(lat, lon, h=0.0):
    a = 6378137.0; e2 = 6.69437999014e-3
    la, lo = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(N + h) * math.cos(la) * math.cos(lo), (N + h) * math.cos(la) * math.sin(lo), (N * (1 - e2) + h) * math.sin(la)])

sys.path.insert(0, '/home/claude/scripts')
from glbload import enu_to_latlon
CS = []
_e0, _e1, _n0, _n1 = AREA['rect']  # metres relative to lat0/lon0
for _e in np.arange(_e0 + 250, _e1, 500):
    for _n in np.arange(_n0 + 250, _n1, 500):
        _la, _lo = enu_to_latlon(_e, _n, AREA['lat0'], AREA['lon0'])
        CS.append(ecef(_la, _lo, 40))
RADIUS = 360 + 80
session = None
nreq = 0


def get(uri):
    global session, nreq
    url = uri if uri.startswith('http') else BASE + uri
    p = urllib.parse.urlparse(url); q = dict(urllib.parse.parse_qsl(p.query))
    if 'session' in q: session = q['session']
    elif session: q['session'] = session
    q['key'] = KEY
    url = urllib.parse.urlunparse(p._replace(query=urllib.parse.urlencode(q)))
    cache = f'/home/claude/g3d/cache/{hashlib.md5(p.path.encode()).hexdigest()}'
    os.makedirs('/home/claude/g3d/cache', exist_ok=True)
    if os.path.exists(cache): return open(cache, 'rb').read(), p.path
    for _ in range(3):
        r = subprocess.run(['curl', '-s', '-f', '--max-time', '60', url], capture_output=True)
        nreq += 1
        if r.returncode == 0:
            open(cache, 'wb').write(r.stdout); return r.stdout, p.path
    raise RuntimeError('fetch failed ' + p.path)


def hits(box):
    return any(hit1(box, C) for C in CS)


def hit1(box, C):
    c = np.array(box[:3]); axes = np.array(box[3:]).reshape(3, 3)
    d = C - c; dist2 = 0.0
    for ax in axes:
        L = np.linalg.norm(ax)
        if L == 0: continue
        t = d @ ax / L
        if abs(t) > L: dist2 += (abs(t) - L) ** 2
    return dist2 <= RADIUS ** 2

saved = []
import threading
from concurrent.futures import ThreadPoolExecutor
lock = threading.Lock()
pool = ThreadPoolExecutor(16)
pending = []


def visit(tile, base_uri):
    with lock: pending.append(pool.submit(_visit, tile, base_uri))


def _visit(tile, base_uri):
    bv = tile.get('boundingVolume', {})
    if 'box' in bv and not hits(bv['box']): return
    err = tile.get('geometricError', 0)
    kids = tile.get('children', [])
    content = tile.get('content', {}).get('uri')
    if content:
        full = urllib.parse.urljoin(base_uri, content)
        if content.split('?')[0].endswith('.json'):
            data, path = get(full)
            sub = json.loads(data)
            _visit(sub['root'], full)
            return
        if err <= MAXERR or not kids:
            data, path = get(full)
            fn = f'{OUT}/{hashlib.md5(path.encode()).hexdigest()}.glb'
            open(fn, 'wb').write(data)
            with lock: saved.append((fn, err))
            return
    for ch in kids:
        visit(ch, base_uri)

root, _ = get('/v1/3dtiles/root.json')
visit(json.loads(root)['root'], BASE + '/v1/3dtiles/root.json')
import time
while True:
    with lock: ps = list(pending)
    for p in ps: p.result()
    with lock:
        if len(pending) == len(ps): break
print('saved', len(saved), 'requests', nreq, 'errors', sorted({round(e, 2) for _, e in saved})[:10])
json.dump(saved, open(f'{OUT}/index.json', 'w'))
