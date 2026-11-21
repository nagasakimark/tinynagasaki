"""Convert crawled .glb tiles into npz meshes in local ENU + jpg textures, with a bbox index."""
import sys, os, json, glob
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/claude/scripts')
from glbload import load, enu_matrix

A = json.load(open('/home/claude/g3d/area.json'))
R, O = enu_matrix(A['lat0'], A['lon0'])
SRC = sys.argv[1]; DST = sys.argv[2]
os.makedirs(DST, exist_ok=True)


def conv(fn):
    base = os.path.basename(fn)[:-4]
    out = []
    try:
        parts = load(fn, R, O)
    except Exception as e:
        return []
    for k, (P, UV, I, jpg) in enumerate(parts):
        name = f'{base}_{k}'
        npz = f'{DST}/{name}.npz'
        if not os.path.exists(npz):
            np.savez(npz, v=P.astype(np.float32), uv=UV.astype(np.float32), f=I.astype(np.int32))
            open(f'{DST}/{name}.jpg', 'wb').write(jpg)
        out.append([name, P.min(0).tolist(), P.max(0).tolist()])
    return out

if __name__ == '__main__':
    files = glob.glob(f'{SRC}/*.glb')
    with Pool(2) as p:
        res = p.map(conv, files, chunksize=64)
    idx = [r for rs in res for r in rs]
    json.dump(idx, open(f'{DST}/index.json', 'w'))
    print('parts', len(idx))
