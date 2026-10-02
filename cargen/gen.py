"""Auto-Generator (läuft in GitHub Actions).
1) Konzeptbild mit FLUX.1 [schnell] (Apache-2.0)
2) 3D-Modell mit TRELLIS (MIT) bzw. Ersatz-Spaces
Ergebnisse: cargen/out/<id>/concept.png, model.glb, info.json; Protokoll: cargen/out/log.txt
"""
import json, os, sys, time, shutil, traceback
from gradio_client import Client, handle_file

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, 'log.txt'), 'a')
TOKEN = os.environ.get('HF_TOKEN') or None

def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()

def connect(space):
    try:
        if not TOKEN: c = Client(space, verbose=False)
        else:
            try: c = Client(space, token=TOKEN, verbose=False)
            except TypeError: c = Client(space, hf_token=TOKEN, verbose=False)
        return c
    except Exception as e:
        log('connect failed', space, repr(e)[:300]); return None

def spec_of(c):
    try: return c.view_api(return_format='dict', print_info=False)
    except Exception as e:
        log('view_api failed', repr(e)[:200]); return {'named_endpoints': {}}

def dump_api(c, space):
    sp = spec_of(c)
    lines = []
    for name, ep in sp.get('named_endpoints', {}).items():
        ps = ['%s:%s=%s' % (p.get('parameter_name') or p.get('label'), p.get('component'), p.get('parameter_default') if p.get('parameter_has_default') else '!') for p in ep.get('parameters', [])]
        rs = [r.get('component') for r in ep.get('returns', [])]
        lines.append('  %s (%s) -> %s' % (name, ', '.join(ps), rs))
    log('API', space, '\n' + '\n'.join(lines))
    return sp

def call(c, sp, api, **kw):
    """Ruft einen Endpunkt auf; fehlende Parameter mit Standardwerten, Bilder/Listen automatisch."""
    ep = sp['named_endpoints'][api]
    args = []
    for p in ep.get('parameters', []):
        n = p.get('parameter_name') or p.get('label')
        comp = (p.get('component') or '').lower()
        if n in kw: v = kw[n]
        elif comp == 'image' and '_image' in kw: v = handle_file(kw['_image'])
        elif comp == 'gallery': v = []
        elif p.get('parameter_has_default'): v = p.get('parameter_default')
        elif comp == 'image' or comp == 'file': v = handle_file(kw['_image'])
        else: raise ValueError('kein Wert für %s (%s)' % (n, comp))
        args.append(v)
    return c.predict(*args, api_name=api)

def first_path(res, ext=None):
    """Sucht im Ergebnis (Tupel/Dict/str) den ersten existierenden Dateipfad."""
    stack = [res]
    while stack:
        r = stack.pop(0)
        if isinstance(r, str) and os.path.exists(r) and (not ext or r.lower().endswith(ext)): return r
        if isinstance(r, dict):
            for k in ('path', 'value', 'name', 'video', 'image'):
                if k in r: stack.append(r[k])
        if isinstance(r, (list, tuple)): stack.extend(r)
    return None

def make_image(prompt, dst, seed, w=1024, h=768):
    for space in ('black-forest-labs/FLUX.1-schnell',):
        c = connect(space)
        if not c: continue
        sp = dump_api(c, space)
        api = '/infer' if '/infer' in sp.get('named_endpoints', {}) else next(iter(sp.get('named_endpoints', {})), None)
        if not api: continue
        try:
            res = call(c, sp, api, prompt=prompt, seed=seed, randomize_seed=False, width=w, height=h, num_inference_steps=4)
            p = first_path(res)
            if p:
                shutil.copy(p, dst); log('image ok', space, dst); return space
            log('image: no path in result', str(res)[:300])
        except Exception as e:
            log('image failed', space, repr(e)[:400])
    return None

def make_3d(img, dst, seed):
    for space in ('trellis-community/TRELLIS', 'JeffreyXiang/TRELLIS', 'microsoft/TRELLIS'):
        c = connect(space)
        if not c: continue
        sp = dump_api(c, space)
        eps = sp.get('named_endpoints', {})
        try:
            if '/start_session' in eps:
                try: call(c, sp, '/start_session')
                except Exception as e: log('start_session', repr(e)[:200])
            pre = img
            if '/preprocess_image' in eps:
                r = call(c, sp, '/preprocess_image', _image=img)
                pre = first_path(r) or img
            if '/generate_and_extract_glb' in eps:
                r = call(c, sp, '/generate_and_extract_glb', _image=pre, seed=seed, mesh_simplify=0.95, texture_size=1024)
                p = first_path(r, '.glb')
                if p:
                    shutil.copy(p, dst); log('glb ok', space, os.path.getsize(dst)); return space
                log('generate_and_extract_glb: no glb', str(r)[:400]); continue
            gen = '/image_to_3d' if '/image_to_3d' in eps else None
            if not gen: log('no /image_to_3d'); continue
            r = call(c, sp, gen, _image=pre, seed=seed, is_multiimage=False, ss_guidance_strength=7.5, ss_sampling_steps=12,
                     slat_guidance_strength=3.0, slat_sampling_steps=12)
            log('image_to_3d ok', str(r)[:200])
            ex = '/extract_glb' if '/extract_glb' in eps else None
            if not ex: log('no /extract_glb'); continue
            r = call(c, sp, ex, mesh_simplify=0.9, texture_size=1024)
            p = first_path(r, '.glb')
            if p:
                shutil.copy(p, dst); log('glb ok', space, os.path.getsize(dst)); return space
            log('glb: no path', str(r)[:300])
        except Exception as e:
            log('3d failed', space, repr(e)[:500])
    # Ersatz: TripoSR (MIT)
    for space in ('stabilityai/TripoSR',):
        c = connect(space)
        if not c: continue
        sp = dump_api(c, space); eps = sp.get('named_endpoints', {})
        try:
            pre = img
            if '/preprocess' in eps:
                r = call(c, sp, '/preprocess', _image=img, do_remove_background=True, foreground_ratio=0.85)
                pre = first_path(r) or img
            api = '/generate' if '/generate' in eps else None
            if not api: continue
            r = call(c, sp, api, _image=pre, mc_resolution=320)
            p = first_path(r, '.glb')
            if p:
                shutil.copy(p, dst); log('glb ok', space, os.path.getsize(dst)); return space
            log('triposr: no glb', str(r)[:300])
        except Exception as e:
            log('3d failed', space, repr(e)[:500])
    return None

def main():
    jobs = json.load(open(os.path.join(ROOT, 'prompts.json')))
    log('=== run', len(jobs), 'jobs, token:', bool(TOKEN))
    for j in jobs:
        d = os.path.join(OUT, j['id']); os.makedirs(d, exist_ok=True)
        info_p = os.path.join(d, 'info.json')
        info = json.load(open(info_p)) if os.path.exists(info_p) else {}
        same = info.get('prompt') == j['prompt']
        seed = int(j.get('seed', 7))
        # Kandidaten-Modus: erst mehrere Konzeptbilder (cand_<k>.png), Auswahl per "pick": k, dann 3D
        if j.get('cands'):
            if not same:
                for fn in os.listdir(d):
                    if fn.startswith('cand_') or fn in ('concept.png', 'model.glb'): os.remove(os.path.join(d, fn))
                info = {}
            if 'pick' not in j:
                for k in range(int(j['cands'])):
                    cp = os.path.join(d, 'cand_%d.png' % k)
                    if not os.path.exists(cp): make_image(j['prompt'], cp, seed + k, j.get('w', 1024), j.get('h', 768))
                info['prompt'] = j['prompt']; info['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                json.dump(info, open(info_p, 'w'), indent=1); continue
            pk = int(j['pick'])
            if info.get('picked') == pk and info.get('glb'): log('skip', j['id']); continue
            cp = os.path.join(d, 'cand_%d.png' % pk)
            if not os.path.exists(cp): make_image(j['prompt'], cp, seed + pk, j.get('w', 1024), j.get('h', 768))
            if os.path.exists(cp):
                shutil.copy(cp, os.path.join(d, 'concept.png'))
                info['model_space'] = make_3d(cp, os.path.join(d, 'model.glb'), seed + pk)
                info['glb'] = bool(info['model_space']); info['picked'] = pk
            info['prompt'] = j['prompt']; info['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            json.dump(info, open(info_p, 'w'), indent=1); continue
        if same and (info.get('glb') or (j.get('image_only') and os.path.exists(os.path.join(d, 'concept.png')))) and not j.get('redo'): log('skip', j['id']); continue
        if not same:
            for fn in ('concept.png', 'model.glb'):
                if os.path.exists(os.path.join(d, fn)): os.remove(os.path.join(d, fn))
            info = {}
        img = os.path.join(d, 'concept.png')
        if not os.path.exists(img) or j.get('redo'):
            info['image_space'] = make_image(j['prompt'], img, seed, j.get('w', 1024), j.get('h', 768))
        if os.path.exists(img) and not j.get('image_only'):
            info['model_space'] = make_3d(img, os.path.join(d, 'model.glb'), seed)
            info['glb'] = bool(info['model_space'])
        info['prompt'] = j['prompt']; info['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        json.dump(info, open(info_p, 'w'), indent=1)
    log('=== done')

if __name__ == '__main__':
    try: main()
    except Exception:
        log('FATAL', traceback.format_exc()[:2000])
