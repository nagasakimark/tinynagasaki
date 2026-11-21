"""Export location data for interactivity: surface heights, roads/rail/labels projected into map pixels.
Outputs in /home/claude/g3d/geo/:
  dsm.npz            surface height grid (ENU metres) from the Google mesh, 2 m cells
  roads.json         road centrelines: [{cls, width_m, pts:[[x_px,y_px,lat,lon],...]}]
  rail.json          railway / tram lines, same format
  labels.json        GSI place labels: [{name, kana, cat, x_px, y_px, lat, lon}]
"""
import json, glob, math, os, sys, subprocess
import numpy as np
sys.path.insert(0, '/home/claude/scripts')
from glbload import enu_to_latlon
import mapbox_vector_tile as mvt

MAP = json.load(open('/home/claude/g3d/map.json'))
LAT0, LON0 = MAP['lat0'], MAP['lon0']
S = MAP['m_per_px']; P = math.radians(MAP['pitch_deg']); sp, cp = math.sin(P), math.cos(P)
OUT = '/home/claude/g3d/geo'; os.makedirs(OUT, exist_ok=True)
E0, E1, N0, N1, CELL = -2400, 2400, -2950, 2950, 2.0
SEA_Z = 33.15

# ---------- 1. DSM from mesh vertices ----------
dsm_path = f'{OUT}/dsm.npz'
if not os.path.exists(dsm_path):
    W = int((E1 - E0) / CELL); H = int((N1 - N0) / CELL)
    dsm = np.full((H, W), -1e9, np.float32)
    for f in glob.glob('/home/claude/g3d/mesh_big/*.npz'):
        v = np.load(f)['v']
        i = ((N1 - v[:, 1]) / CELL).astype(np.int64); j = ((v[:, 0] - E0) / CELL).astype(np.int64)
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        np.maximum.at(dsm, (i[ok], j[ok]), v[ok, 2])
    # fill holes from neighbours
    from scipy import ndimage
    hole = dsm < -1e8
    if hole.any():
        idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        dsm = dsm[tuple(idx)]
    np.savez_compressed(dsm_path, dsm=dsm, e0=E0, n1=N1, cell=CELL)
D = np.load(dsm_path); dsm = D['dsm']
from scipy import ndimage
ground = ndimage.minimum_filter(dsm, size=5)  # strips building edges / trees next to roads


def z_at(e, n, grid):
    i = int((N1 - n) / CELL); j = int((e - E0) / CELL)
    i = min(max(i, 0), grid.shape[0] - 1); j = min(max(j, 0), grid.shape[1] - 1)
    return float(grid[i, j])


def to_px(e, n, z):
    v = n * sp + z * cp
    return round((e - MAP['x0']) / S, 1), round((MAP['v1'] - v) / S, 1)

# ---------- 2. GSI vector tiles ----------
def gtile(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

la_lo = enu_to_latlon(E0, N0, LAT0, LON0); la_hi = enu_to_latlon(E1, N1, LAT0, LON0)
x0, y0 = gtile(la_hi[0], la_lo[1], 16); x1, y1 = gtile(la_lo[0], la_hi[1], 16)
os.makedirs('/home/claude/g3d/gsi', exist_ok=True)
feats = {'road': [], 'railway': [], 'label': [], 'symbol': []}
a = 6378137.0; e2 = 6.69437999014e-3; la0 = math.radians(LAT0)
Mr = a * (1 - e2) / (1 - e2 * math.sin(la0) ** 2) ** 1.5; Nr = a / math.sqrt(1 - e2 * math.sin(la0) ** 2)


def ll_to_en(lat, lon):
    return math.radians(lon - LON0) * Nr * math.cos(la0), math.radians(lat - LAT0) * Mr

for tx in range(int(x0), int(x1) + 1):
    for ty in range(int(y0), int(y1) + 1):
        p = f'/home/claude/g3d/gsi/{tx}_{ty}.pbf'
        if not os.path.exists(p):
            subprocess.run(['curl', '-s', '-o', p, f'https://cyberjapandata.gsi.go.jp/xyz/experimental_bvmap/16/{tx}/{ty}.pbf'])
        try:
            dec = mvt.decode(open(p, 'rb').read())
        except Exception:
            continue
        for lname in feats:
            if lname not in dec: continue
            for ft in dec[lname]['features']:
                g = ft['geometry']
                def conv(c):
                    if isinstance(c[0], (int, float)):
                        X = tx + c[0] / 4096; Y = ty + 1 - c[1] / 4096
                        lon = X / 2 ** 16 * 360 - 180
                        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * Y / 2 ** 16))))
                        return (lat, lon)
                    return [conv(q) for q in c]
                feats[lname].append((g['type'], conv(g['coordinates']), ft['properties']))

WIDTH = {0: 3.0, 1: 4.5, 2: 8.0, 3: 13.0, 4: 18.0}


def lines(kind):
    out = []
    for gt, coords, prop in feats[kind]:
        parts = [coords] if gt == 'LineString' else (coords if gt == 'MultiLineString' else [])
        for part in parts:
            pts = []
            for lat, lon in part:
                e, n = ll_to_en(lat, lon)
                if not (E0 < e < E1 and N0 < n < N1): continue
                z = z_at(e, n, ground if kind == 'road' else dsm)
                z = max(z, SEA_Z)
                x, y = to_px(e, n, z)
                pts.append([x, y, round(lat, 6), round(lon, 6)])
            if len(pts) >= 2:
                out.append({'cls': prop.get('rdCtg', prop.get('ftCode')), 'ftCode': prop.get('ftCode'),
                            'width_m': WIDTH.get(prop.get('rnkWidth'), 4.0), 'pts': pts,
                            **({'rail_kind': prop.get('railState'), 'tram': prop.get('ftCode') in (8202, 8203)} if kind == 'railway' else {})})
    return out

json.dump(lines('road'), open(f'{OUT}/roads.json', 'w'))
json.dump(lines('railway'), open(f'{OUT}/rail.json', 'w'))
labels = []
seen = set()
for gt, c, prop in feats['label'] + feats['symbol']:
    if gt != 'Point': continue
    name = prop.get('knj') or ''
    lat, lon = c; e, n = ll_to_en(lat, lon)
    if not (E0 < e < E1 and N0 < n < N1): continue
    key = (name, prop.get('ftCode'), round(lat, 4), round(lon, 4))
    if key in seen: continue
    seen.add(key)
    z = max(z_at(e, n, ground), SEA_Z); x, y = to_px(e, n, z)
    labels.append({'name': name, 'kana': prop.get('kana', ''), 'ftCode': prop.get('ftCode'), 'annoCtg': prop.get('annoCtg'),
                   'x_px': x, 'y_px': y, 'lat': round(lat, 6), 'lon': round(lon, 6)})
json.dump(labels, open(f'{OUT}/labels.json', 'w'), ensure_ascii=False)
print('roads', len(json.load(open(f'{OUT}/roads.json'))), 'rail', len(json.load(open(f'{OUT}/rail.json'))), 'labels', len(labels))
