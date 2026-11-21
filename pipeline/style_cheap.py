"""Restyle tiles with a cheaper fal edit model, using Mark's approved image + nearest Nano Banana Pro example as style refs."""
import sys, os, json, glob
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/claude/scripts')
import fal
from style_nb import PROMPT, STYLE_REF
MODEL = os.environ.get('MODEL', 'fal-ai/nano-banana-2/edit'); COST = float(os.environ.get('COST', '0.08'))
DST = os.environ.get('DST', '/home/claude/g3d/styled'); BUDGET = float(os.environ.get('BUDGET', '9.7'))
os.makedirs(DST, exist_ok=True)
PRO = {os.path.basename(f)[:-4]: f for f in glob.glob('/home/claude/g3d/nb/*.png')}
P2 = PROMPT.replace('Match the art style, colour palette, outline weight and level of detail of the SECOND image exactly.',
  'Match the art style, colour palette, water colour, outline weight and level of detail of the SECOND and THIRD images exactly (they are finished tiles of the same map; the new tile must look like part of the same map).').replace('use the second image only for style', 'use the other images only for style')
ref0 = fal.data_uri(STYLE_REF, max_side=1536)

def nearest(tid):
    r, c = int(tid[1:3]), int(tid[5:7])
    cands = sorted(PRO, key=lambda k: (int(k[1:3]) - r) ** 2 + (int(k[5:7]) - c) ** 2 + (0.5 if k == tid else 0))
    return [k for k in cands if k != tid][0]

def one(tid):
    out = f'{DST}/{tid}.png'
    if os.path.exists(out): return tid, 'cached'
    if fal.spent() + COST > BUDGET: return tid, 'budget'
    nb = nearest(tid)
    try:
      res = fal.run(MODEL, {'prompt': P2, 'image_urls': [fal.data_uri(f'/home/claude/g3d/render/{tid}.png'), ref0, fal.data_uri(PRO[nb], q=90)],
                          'num_images': 1, 'aspect_ratio': '1:1', 'output_format': 'png', 'resolution': '1K'}, COST)
    except Exception as e:
      return tid, 'FAILED ' + str(e)[:120]
    fal.download(res['images'][0]['url'], out)
    return tid, 'ok ref=' + nb

if __name__ == '__main__':
    with ThreadPoolExecutor(int(os.environ.get('PAR', '4'))) as ex:
        for r in ex.map(one, sys.argv[1:]): print(*r, flush=True)
    print('spent so far $%.2f' % fal.spent())
