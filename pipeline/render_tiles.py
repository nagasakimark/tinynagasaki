"""Render the isometric map from prepared Google 3D tile meshes, as overlapping square tiles.
usage: python render_tiles.py <mesh_dir> <out_dir> <tile_list.json>
Projection (shared by every tile, defined in map.json):
  camera looks north, tilted 45 deg down, orthographic.
  screen x = east, screen up = (north + height) * sin45
"""
import bpy, sys, json, math, os
import numpy as np

MESH, OUT, TL = sys.argv[1], sys.argv[2], sys.argv[3]
MAP = json.load(open('/home/claude/g3d/map.json'))
S = MAP['m_per_px']; T = MAP['tile_px']
PITCH = math.radians(MAP['pitch_deg'])
sp, cp = math.sin(PITCH), math.cos(PITCH)
idx = json.load(open(f'{MESH}/index.json'))
tiles = json.load(open(TL))
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sh = sc.display.shading
sh.light = 'FLAT'; sh.color_type = 'TEXTURE'
sh.show_shadows = False; sh.show_cavity = False; sh.show_object_outline = False
sc.display.render_aa = '8'
sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x = sc.render.resolution_y = T
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new('w'); sc.world.color = (0.62, 0.72, 0.8)
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'

cam = bpy.data.cameras.new('c'); cam.type = 'ORTHO'; cam.ortho_scale = T * S; cam.clip_end = 20000
co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
# camera rotation: x tilt so it looks north and down by pitch
co.rotation_euler = (math.pi / 2 - PITCH, 0, 0)
fwd = np.array([0, cp, -sp]); up = np.array([0, sp, cp])

loaded = {}


def ensure(name):
    if name in loaded: return
    d = np.load(f'{MESH}/{name}.npz')
    v, uv, f = d['v'], d['uv'], d['f']
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v)); me.vertices.foreach_set('co', v.ravel())
    me.loops.add(f.size); me.loops.foreach_set('vertex_index', f.ravel())
    me.polygons.add(len(f)); me.polygons.foreach_set('loop_start', np.arange(0, f.size, 3)); me.polygons.foreach_set('loop_total', np.full(len(f), 3))
    me.update(); me.validate()
    uvl = me.uv_layers.new(name='UV')
    uvl.data.foreach_set('uv', uv[f.ravel()].astype(np.float32).ravel() * np.tile([1, -1], f.size).astype(np.float32) + np.tile([0, 1], f.size).astype(np.float32))
    img = bpy.data.images.load(f'{MESH}/{name}.jpg')
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes.get('Principled BSDF')
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; tex.interpolation = 'Linear'
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    loaded[name] = ob


def drop(name):
    ob = loaded.pop(name)
    me = ob.data; mat = me.materials[0]; img = mat.node_tree.nodes['Image Texture'].image
    bpy.data.objects.remove(ob); bpy.data.meshes.remove(me); bpy.data.materials.remove(mat); bpy.data.images.remove(img)

# screen coords of each part's bbox
parts = []
for name, mn, mx in idx:
    xs = (mn[0], mx[0]); vs = [n * sp + z * cp for n in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    parts.append((name, xs[0], xs[1], min(vs), max(vs)))

for t in tiles:
    out = f"{OUT}/{t['id']}.png"
    if os.path.exists(out): continue
    # tile screen window in metres
    x0 = MAP['x0'] + t['col'] * MAP['stride_px'] * S; x1 = x0 + T * S
    v1 = MAP['v1'] - t['row'] * MAP['stride_px'] * S; v0 = v1 - T * S
    need = {p[0] for p in parts if p[2] > x0 - 5 and p[1] < x1 + 5 and p[4] > v0 - 5 and p[3] < v1 + 5}
    for n in list(loaded):
        if n not in need: drop(n)
    for n in need: ensure(n)
    cx = (x0 + x1) / 2; cv = (v0 + v1) / 2
    # point on screen-centre ray at height 0 plane is irrelevant for ortho; place camera far back along -fwd
    centre = np.array([cx, 0, 0]) + up * cv
    co.location = tuple(centre - fwd * 6000)
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print('TILE', t['id'], len(need), flush=True)
