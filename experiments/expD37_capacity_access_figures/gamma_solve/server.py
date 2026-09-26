"""Local-only interactive gamma_solve viewer. No external services or CDN."""
import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ.setdefault('MPLCONFIGDIR', '/tmp/precisionmlps-mpl')
import numpy as np
from threadpoolctl import threadpool_limits
import plotly
from core import Geometry, TARGETS, kernel_view, sweep, comparison

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT/'results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve'
LOCK = threading.RLock()


def clean(value):
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def configuration(payload):
    return Geometry(N=int(payload.get('N',128)),m=int(payload.get('m',263)),
                    center_jitter=float(payload.get('center_jitter',0)),
                    data_jitter=float(payload.get('data_jitter',0)),seed=int(payload.get('seed',42)))


def calculation(payload):
    geom = configuration(payload)
    count = int(payload.get('count',41))
    if not 9 <= count <= 101:
        raise ValueError('Use 9–101 gamma samples.')
    data = sweep(geom,float(payload.get('gamma_min',.25)),float(payload.get('gamma_max',128)),
                 count,float(payload.get('gamma0',8)))
    result = comparison(data,payload.get('target','mixed'),int(payload.get('steps',1000000)),payload.get('matching','rank'))
    return data,result


def save(payload, data, result, name=None):
    RESULTS.joinpath('data').mkdir(parents=True,exist_ok=True)
    config = clean(payload)
    stem = name or hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    np.savez_compressed(RESULTS/'data'/f'{stem}.npz',
                        gammas=data['gammas'], lambdas=data['lambdas'], rates=data['rates'],
                        eigenvalues=data['eigenvalues'], rank_resolved=data['rank_resolved'],
                        permutations=data['permutations'], adjacent_overlap=data['adjacent_overlap'],
                        **{'p_'+k:v for k,v in data['p'].items()})
    (RESULTS/'data'/f'{stem}.json').write_text(json.dumps(clean({'config':config,'comparison':result}),indent=2))
    return stem


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ('200','304'):
            super().log_message(fmt,*args)

    def send(self, body, kind='application/json', code=200):
        self.send_response(code)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-cache' if kind=='application/json' or kind.startswith('text/html') else 'public, max-age=3600')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/info':
            return self.send(json.dumps({'targets':TARGETS}).encode())
        if path in ('/','/index.html'):
            return self.send((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
        if path == '/plotly.js':
            asset = Path(plotly.__file__).parent/'package_data/plotly.min.js'
            return self.send(asset.read_bytes(),'text/javascript')
        if path.startswith('/saved/'):
            file = (RESULTS/path.removeprefix('/saved/')).resolve()
            if file.is_relative_to(RESULTS.resolve()) and file.is_file():
                return self.send(file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send(b'Not found','text/plain',404)

    def do_POST(self):
        try:
            if int(self.headers.get('Content-Length',0)) > 20000:
                raise ValueError('Request too large.')
            payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            with LOCK:
                if self.path == '/api/kernel':
                    gamma=float(payload.get('gamma',16))
                    if not .03 <= gamma <= 512:
                        raise ValueError('Use .03 ≤ gamma ≤ 512.')
                    result=kernel_view(configuration(payload),gamma)
                elif self.path in ('/api/sweep','/api/save'):
                    data,result=calculation(payload)
                    if self.path == '/api/save':
                        stem=save(payload,data,result)
                        result={'json':f'/saved/data/{stem}.json','npz':f'/saved/data/{stem}.npz'}
                else:
                    return self.send(b'Not found','text/plain',404)
            self.send(json.dumps(clean(result),separators=(',',':'),allow_nan=False).encode())
        except Exception as exc:
            self.send(json.dumps({'error':str(exc)}).encode(),code=400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8067)
    parser.add_argument('--snapshot',action='store_true',help='Save default calculations and PNG figures, then exit.')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.snapshot:
            from figures import make_figures
            payload={'N':128,'m':263,'gamma':16,'gamma0':8,'gamma_min':.25,'gamma_max':128,
                     'count':41,'steps':1000000,'target':'mixed','matching':'rank','seed':42,
                     'center_jitter':0,'data_jitter':0}
            data,result=calculation(payload)
            save(payload,data,result,'default')
            make_figures(RESULTS,kernel_view(configuration(payload),16),result)
            print(RESULTS,flush=True)
            return
        print(f'Gamma solve: http://127.0.0.1:{args.port}',flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':
    main()
