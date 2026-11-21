"""Load Google 3D tile .glb files into local East-North-Up metres (float64 math)."""
import json, struct, math
import numpy as np

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
YUP2ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def ecef(lat, lon, h=0.0):
    a = 6378137.0; e2 = 6.69437999014e-3
    la, lo = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(N + h) * math.cos(la) * math.cos(lo), (N + h) * math.cos(la) * math.sin(lo), (N * (1 - e2) + h) * math.sin(la)])


def enu_matrix(lat0, lon0, h0=0.0):
    la, lo = math.radians(lat0), math.radians(lon0)
    R = np.array([[-math.sin(lo), math.cos(lo), 0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    return R, ecef(lat0, lon0, h0)


def enu_to_latlon(e, n, lat0, lon0):
    # small-area approximation, good to centimetres over a few km
    a = 6378137.0; e2 = 6.69437999014e-3
    la = math.radians(lat0)
    M = a * (1 - e2) / (1 - e2 * math.sin(la) ** 2) ** 1.5
    N = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return lat0 + math.degrees(n / M), lon0 + math.degrees(e / (N * math.cos(la)))


def load(fn, R, O):
    b = open(fn, 'rb').read()
    L = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + L])
    bin_off = 20 + L + 8
    binb = b[bin_off:]

    def acc(i):
        a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        n = NC[a['type']]; dt = np.dtype(CT[a['componentType']])
        stride = bv.get('byteStride', 0)
        if stride and stride != n * dt.itemsize:
            raw = np.frombuffer(binb, np.uint8, count=stride * a['count'], offset=off).reshape(a['count'], stride)
            arr = raw[:, :n * dt.itemsize].copy().view(dt).reshape(a['count'], n)
        else:
            arr = np.frombuffer(binb, dt, count=a['count'] * n, offset=off).reshape(a['count'], n)
        if a.get('normalized'):
            arr = arr.astype(np.float64) / np.iinfo(dt).max
        return arr
    out = []
    for node in j['nodes']:
        if 'mesh' not in node: continue
        M = np.array(node.get('matrix', np.eye(4).T.ravel().tolist()), float).reshape(4, 4).T
        T = YUP2ZUP @ M
        for prim in j['meshes'][node['mesh']]['primitives']:
            P = acc(prim['attributes']['POSITION']).astype(np.float64)
            Pe = (T[:3, :3] @ P.T).T + T[:3, 3]
            Pl = (R @ (Pe - O).T).T
            UV = acc(prim['attributes']['TEXCOORD_0']).astype(np.float64)
            I = acc(prim['indices']).astype(np.int64).reshape(-1, 3)
            mat = j['materials'][prim['material']]
            ti = mat['pbrMetallicRoughness']['baseColorTexture']['index']
            img = j['images'][j['textures'][ti]['source']]
            bv = j['bufferViews'][img['bufferView']]
            jpg = binb[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
            out.append((Pl, UV, I, jpg))
    return out
