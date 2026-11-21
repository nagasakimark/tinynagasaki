"""Minimal fal.ai queue client (no SDK). Keys never printed."""
import json, time, base64, io, sys, urllib.request, urllib.error
sys.path.insert(0, '/home/claude/scripts')
from keys import keys
KEY = keys()['fal']
LOG = '/home/claude/g3d/fal_spend.jsonl'


def _req(url, data=None, method=None):
    h = {'Authorization': f'Key {KEY}', 'Content-Type': 'application/json'}
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=h, method=method or ('POST' if data is not None else 'GET'))
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} {url.split("?")[0]}: {e.read()[:800]}')


def data_uri(path, fmt='JPEG', q=92, max_side=None):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, fmt, quality=q) if fmt == 'JPEG' else im.save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def run(model, args, est_cost, poll=3, timeout=900):
    sub = _req(f'https://queue.fal.run/{model}', args)
    rid = sub['request_id']
    open('/home/claude/g3d/fal_requests.log', 'a').write(f'{time.time()} {model} {rid}\n')
    status_url = sub.get('status_url') or f'https://queue.fal.run/{model}/requests/{rid}/status'
    resp_url = sub.get('response_url') or f'https://queue.fal.run/{model}/requests/{rid}'
    t0 = time.time()
    while True:
        st = _req(status_url + ('&' if '?' in status_url else '?') + 'logs=1')
        if st.get('status') == 'COMPLETED': break
        if time.time() - t0 > timeout: raise TimeoutError(rid)
        time.sleep(poll)
    out = _req(resp_url)
    with open(LOG, 'a') as f:
        f.write(json.dumps({'t': time.time(), 'model': model, 'cost': est_cost, 'rid': rid}) + '\n')
    return out


def spent():
    try:
        return sum(json.loads(l)['cost'] for l in open(LOG))
    except FileNotFoundError:
        return 0.0


def download(url, path):
    urllib.request.urlretrieve(url, path)
