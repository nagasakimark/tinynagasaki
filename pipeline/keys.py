"""Reads API keys from the project's .env (lines like 'FAL API Key: ...'). Never commit .env."""
import os
ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
def keys():
    d = {}
    for l in open(ENV, encoding='utf-8'):
        if ':' in l:
            k, v = l.split(':', 1); k = k.lower(); v = v.strip()
            if 'fal' in k: d['fal'] = v
            elif 'kaggle' in k: d['kaggle'] = v
            elif 'maps' in k or 'google' in k: d['google'] = v
    return d
