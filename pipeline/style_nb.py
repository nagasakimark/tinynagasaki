"""Restyle render tiles with Nano Banana Pro (fal). usage: python style_nb.py id1 id2 ...  (max budget guarded)"""
import sys, os, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/claude/scripts')
import fal

PROMPT = """Redraw the FIRST image (an aerial view of part of Nagasaki, Japan) as a detailed isometric pixel-art illustration in the style of a modern HD city-builder game. Match the art style, colour palette, outline weight and level of detail of the SECOND image exactly. Use the first image for all content and layout; use the second image only for style.

KEEP EXACTLY: the camera angle, framing and composition; every building's position, footprint, height and shape; the road layout, rivers, coastline, hills and parks; the real colours of roofs and walls (keep red roofs red, white buildings white, etc.). Do not add, remove or move any buildings or roads. The output must line up pixel-for-pixel with the first image.

STYLE:
- Crisp pixel-art look at high resolution: clean hard edges, no blur, no painterly brushstrokes, no photo textures left.
- Bright, cheerful, saturated daylight colours. Warm sunlight from the upper left, soft clean shadows falling to the lower right.
- Buildings simplified into clean shapes with neat regular grids of windows; flat rooftops show small details (AC units, water tanks, solar panels, rooftop gardens).
- Trees as rounded puffy clusters of green with 3-4 shades and light pixel dithering; dense forest on hillsides becomes a lush bumpy carpet of these tree clusters.
- Roads: clean grey asphalt with crisp white lane markings and crosswalks; tiny pixel cars, buses and trucks in varied colours.
- Pavements, plazas and paving patterns drawn clearly and neatly.
- Water: clean flat blue with subtle lighter ripples.

DO NOT include: any text, labels, readable signs, logos, watermarks, borders, people's faces, or fisheye distortion. Do not leave any area blank or grey: fill the entire frame edge to edge."""

STYLE_REF = '/mnt/user-data/uploads/ace11ab8-image.png'
SRC = '/home/claude/g3d/render'; DST = '/home/claude/g3d/nb'
BUDGET = float(os.environ.get('NB_BUDGET', '5.0'))
os.makedirs(DST, exist_ok=True)
ref = fal.data_uri(STYLE_REF, max_side=1536)


def one(tid):
    out = f'{DST}/{tid}.png'
    if os.path.exists(out): return tid, 'cached'
    if fal.spent() + 0.15 > BUDGET: return tid, 'budget'
    res = fal.run('fal-ai/nano-banana-pro/edit', {
        'prompt': PROMPT, 'image_urls': [fal.data_uri(f'{SRC}/{tid}.png'), ref],
        'num_images': 1, 'aspect_ratio': '1:1', 'resolution': '1K', 'output_format': 'png'}, 0.15)
    fal.download(res['images'][0]['url'], out)
    return tid, 'ok'

if __name__ == '__main__':
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, sys.argv[1:]): print(*r, flush=True)
    print('spent so far $%.2f' % fal.spent())
